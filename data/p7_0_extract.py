"""Phase 7, step 0: what the retained graphs hold beyond their 49 features.
PREREGISTRATION_CLOSING.md §2 (P, P_layer, L) and §3 (gate G2). Label-free:
every quantity is computed per member or symmetrically in a pair's two members,
and no survivor is compared with a killed member.

Each retained graph is read once, in full, then dropped from the page cache. No
graph is built, moved, linked or deleted. Run from data/, so that the installed
circuit_tracer (the one that wrote the graphs) is imported, not the repository's.

Per member:
  posmean     every active transcoder feature's activation summed over positions
              and divided by the position count (index layer * 16384 + feature),
              stored sparse in WORK: P's raw material
  P_layer     per layer, log(1 + active features) and log(1 + summed activation)
  L_act       D = the positions where the pair's two token sequences differ
              (1 or 2); W = the positions 1-3 after a position of D, not in D and
              not the last
                a1 log(1 + active features in D)   a2 log(1 + summed activation in D)
                a3 log(1 + mean active features per position of W)
                a4 log(1 + mean summed activation per position of W)
  L_attr      influence by circuit_tracer.graph.compute_node_influence on the full
              adjacency, logit nodes weighted by their probabilities, no pruning
                b1 D's share of the influence of all non-logit nodes
                b2 error influence in D / (feature + error influence in D); 0 if
                   both are 0 (counted)
                b3 log(1 + sum |w|) over edges from D's nodes to nodes at later
                   positions and to the logits
                b4 the share of D's active features the attribution selected
Checks (G2; any failure stops the run with exit 2): the graph counts; each file
name's row and expr_id; input tokens equal the tokenizer's ids; no active
feature at position 0; features 0, 3 and 4 recomputed exactly (bf16, as
graph_dataset computed them); |D| in {1, 2} and equal lengths; each member's
edit span (its edit's characters) meets D; W not empty; the node layout.
Recorded, never used to drop a pair: whether a pair's prefix activations are
bit-identical; how often its attribution differs on the shared prefix; whether
p5_2's edit token lies in D; the logit counts.

  MODE=tokens   tokenizer checks on every pair, no graph read
  LIMIT=k       the first k pairs of each phase only (tests)
  WORKERS=2  OUT_DIR=results/closing  WORK=~/closing_work
"""
import gc
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7_common as pc  # noqa: E402

MODE = os.environ.get('MODE', 'full')
LIMIT = int(os.environ.get('LIMIT', 0))
WORKERS = int(os.environ.get('WORKERS', 2))
TAG = {5: 'matched', 4: 'mutant'}
N_LAYERS, D_SAE = 26, 16384
ORPHANS = {4: {63300}}                       # Phase 4: graph_146_63300, its partner failed


class Gate(Exception):
    pass


def check(ok, msg):
    if not ok:
        raise Gate(msg)


def graph_files(phase, P):
    """expr_id -> path; every name must match its row in the row file."""
    keep = pc.KEEP[phase]
    out = {}
    for name in os.listdir(keep):
        if not name.endswith('.pt'):
            continue
        _, row, eid, _ = name[:-3].split('_')
        row, eid = int(row), int(eid)
        if P['row_index'].get(eid) != row:
            raise Gate(f"{name}: row {row} is not expr {eid}'s row {P['row_index'].get(eid)}")
        out[eid] = os.path.join(keep, name)
    want = {5: 518, 4: 197}[phase]
    check(len(out) == want, f"Phase {phase}: {len(out)} graphs, expected {want}")
    members = set(P['ids'].tolist())
    check(set(out) - members == ORPHANS.get(phase, set()),
          f"Phase {phase}: unexpected graphs {sorted(set(out) - members)}")
    check(members <= set(out), f"Phase {phase}: missing graphs {sorted(members - set(out))}")
    return out


def token_jobs(P, tok):
    """Per pair: both members' ids, edit spans, p5_2 edit tokens; D and W."""
    jobs = []
    for k in range(P['n']):
        a, b = int(P['ids'][2 * k]), int(P['ids'][2 * k + 1])
        mem = []
        for i in (a, b):
            r = P['rows'][i]
            enc = tok(r['text'], return_offsets_mapping=True)
            offs = enc['offset_mapping']
            mem.append({'expr_id': i, 'ids': enc['input_ids'], 'span': pc.edit_span(r, offs),
                        'p_e': pc.edit_token(r, offs),
                        'stored': [P['feats'][i]['feature'][j] for j in (0, 3, 4)]})
        ia, ib = mem[0]['ids'], mem[1]['ids']
        check(len(ia) == len(ib), f"pair {a}/{b}: lengths {len(ia)} and {len(ib)}")
        D = [q for q, (u, v) in enumerate(zip(ia, ib)) if u != v]
        n_pos = len(ia)
        W = sorted({q for d in D for q in (d + 1, d + 2, d + 3)} - set(D))
        W = [q for q in W if q <= n_pos - 2]
        check(len(D) in (1, 2), f"pair {a}/{b}: |D| = {len(D)}")
        for m in mem:
            check(set(m['span']) & set(D), f"pair {a}/{b}: member {m['expr_id']}'s edit span "
                                           f"{m['span']} misses D {D}")
        check(len(W) > 0, f"pair {a}/{b}: W is empty (D {D}, n_pos {n_pos})")
        jobs.append({'k': k, 'mem': mem, 'D': D, 'W': W, 'n_pos': n_pos,
                     'pe_in_D': [m['p_e'] in D for m in mem]})
    return jobs


def _member(path, m, D, W, n_pos, q0):
    import torch
    from circuit_tracer.graph import compute_node_influence
    d = torch.load(path, weights_only=False, map_location='cpu')
    af = d['active_features']
    av = d['activation_values']
    sel = d['selected_features'].numpy()
    lp = d['logit_probabilities']
    toks = d['input_tokens'].tolist()
    check(toks == m['ids'], f"{path}: input tokens differ from the tokenizer's")
    f0 = float(av.shape[0])
    f3 = lp[0].item()
    f4 = (-torch.sum(lp * torch.log(lp + 1e-8))).item()          # bf16, as graph_dataset
    check([f0, f3, f4] == m['stored'], f"{path}: features 0/3/4 {[f0, f3, f4]} != stored {m['stored']}")
    f4_32 = float(-(lp.float() * torch.log(lp.float() + 1e-8)).sum())
    lay, pos, fid = (af[:, j].numpy() for j in range(3))
    check(not (pos == 0).any(), f"{path}: active features at position 0")
    val = av.float().numpy().astype(np.float64)
    nnz = len(val)
    count = np.bincount(pos, minlength=n_pos)
    actsum = np.bincount(pos, weights=val, minlength=n_pos)
    is_sel = np.zeros(nnz, bool)
    is_sel[sel] = True
    selcount = np.bincount(pos[is_sel], minlength=n_pos)
    p_layer = np.r_[np.log1p(np.bincount(lay, minlength=N_LAYERS)),
                    np.log1p(np.bincount(lay, weights=val, minlength=N_LAYERS))]
    key = lay.astype(np.int64) * D_SAE + fid
    uk, inv = np.unique(key, return_inverse=True)
    pm_val = (np.bincount(inv, weights=val) / n_pos).astype(np.float32)
    cD = int(count[D].sum())
    out = {'expr_id': m['expr_id'], 'n_pos': n_pos, 'n_logits': int(lp.numel()), 'nnz': nnz,
           'f4_float32_rel_dev': abs(f4_32 - f4) / abs(f4) if f4 else 0.0,
           'P_layer': p_layer.tolist(),
           'a1': float(np.log1p(cD)), 'a2': float(np.log1p(actsum[D].sum())),
           'a3': float(np.log1p(count[W].mean())), 'a4': float(np.log1p(actsum[W].mean())),
           'b4': float(selcount[D].sum() / cD) if cD else 0.0, 'b4_empty': cD == 0}
    # prefix activations, for the pair's bit-identity record
    pre = pos < q0
    pkey = (lay[pre].astype(np.int64) * 200 + pos[pre]) * D_SAE + fid[pre]
    o = np.argsort(pkey, kind='stable')
    prefix_act = (pkey[o], av[torch.from_numpy(np.flatnonzero(pre)[o])].view(torch.int16).numpy())

    # attribution
    A = d['adjacency_matrix']
    n_sel, n_log = len(sel), int(lp.numel())
    N = A.shape[0]
    check(N == n_sel + (N_LAYERS + 1) * n_pos + n_log, f"{path}: node count {N}")
    e0, t0 = n_sel, n_sel + N_LAYERS * n_pos
    l0 = t0 + n_pos
    check(float(A[e0:l0].abs().sum()) == 0.0, f"{path}: error or token rows are not empty")
    node_pos = np.empty(N, np.int64)
    node_pos[:n_sel] = pos[sel]
    node_pos[e0:t0] = np.arange(N_LAYERS * n_pos) % n_pos
    node_pos[t0:l0] = np.arange(n_pos)
    node_pos[l0:] = n_pos - 1
    lw = torch.zeros(N, dtype=A.dtype)
    lw[l0:] = lp.to(A.dtype)
    infl = compute_node_influence(A, lw).double().numpy()
    inD = np.isin(node_pos[:l0], D)
    tot = infl[:l0].sum()
    fi = infl[:n_sel][inD[:n_sel]].sum()
    ei = infl[e0:t0][inD[e0:t0]].sum()
    out['b1'] = float(infl[:l0][inD].sum() / tot)
    out['b2'] = float(ei / (fi + ei)) if fi + ei > 0 else 0.0
    out['b2_zero'] = not (fi + ei > 0)
    b3 = 0.0
    later_or_logit = lambda q: torch.from_numpy((node_pos > q) | (np.arange(N) >= l0))
    for q in D:
        cols = torch.from_numpy(np.flatnonzero(node_pos[:l0] == q))
        b3 += float(A[:, cols][later_or_logit(q)].abs().sum())
    out['b3'] = float(np.log1p(b3))
    # the shared prefix's attribution block (selected prefix features as targets)
    fpre = np.flatnonzero(node_pos[:n_sel] < q0)
    fk = (lay[sel][fpre].astype(np.int64) * 200 + pos[sel][fpre]) * D_SAE + fid[sel][fpre]
    of = np.argsort(fk, kind='stable')
    fpre, fk = fpre[of], fk[of]
    eidx = np.array([e0 + l_ * n_pos + p for l_ in range(N_LAYERS) for p in range(q0)], np.int64)
    tidx = np.arange(t0, t0 + q0)
    blk = A[torch.from_numpy(fpre)][:, torch.from_numpy(np.r_[fpre, eidx, tidx])].numpy().copy()
    del d, A, infl, af, av, lp
    gc.collect()
    return out, (uk.astype(np.int32), pm_val), (count, actsum, selcount), prefix_act, (fk, blk)


def _pair(job):
    import torch
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 4)))
    t = time.time()
    try:
        D, W, n_pos = job['D'], job['W'], job['n_pos']
        q0 = min(D)
        res = []
        for m in job['mem']:
            res.append(_member(job['paths'][m['expr_id']], m, D, W, n_pos, q0))
            pc.drop_cache(job['paths'][m['expr_id']])
        (ka, va), (kb, vb) = res[0][3], res[1][3]
        prefix_identical = bool(np.array_equal(ka, kb) and np.array_equal(va, vb))
        # attribution on the shared prefix: common selected features x common sources
        (fa, Ba), (fb, Bb) = res[0][4], res[1][4]
        common, xa, xb = np.intersect1d(fa, fb, return_indices=True)
        n_fa, n_fb = len(fa), len(fb)
        cols_a = np.r_[xa, n_fa + np.arange(Ba.shape[1] - n_fa)]
        cols_b = np.r_[xb, n_fb + np.arange(Bb.shape[1] - n_fb)]
        sa, sb = Ba[np.ix_(xa, cols_a)], Bb[np.ix_(xb, cols_b)]
        nz = (sa != 0) | (sb != 0)
        rs_a, rs_b = np.abs(sa).sum(1), np.abs(sb).sum(1)
        attr = {'common_prefix_features': int(len(common)),
                'edges_compared': int(nz.sum()),
                'edge_mismatch_share': float(((sa != sb) & nz).sum() / nz.sum()) if nz.any() else 0.0,
                'rowsum_rel_diff_median': float(np.median(np.abs(rs_a - rs_b) / np.maximum(rs_a, 1e-30)))
                if len(common) else 0.0}
        out = []
        for (o, pm, arrs, _, _), m, pe in zip(res, job['mem'], job['pe_in_D']):
            o.update({'k': job['k'], 'D': D, 'W': W, 'q0': q0, 'pe_in_D': pe,
                      'span': m['span'], 'prefix_identical': prefix_identical, 'prefix_attr': attr})
            out.append((o, pm, arrs))
        return {'ok': True, 'k': job['k'], 'members': out, 'seconds': time.time() - t}
    except Gate as e:
        return {'ok': False, 'k': job['k'], 'error': str(e)}


def main():
    t_start = time.time()
    import circuit_tracer
    loc = os.path.dirname(circuit_tracer.__file__)
    if 'site-packages' not in loc:
        raise SystemExit(f"circuit_tracer resolves to {loc}; run from data/, not the repository root")
    if os.environ.get('CUDA_VISIBLE_DEVICES', None) != '':
        raise SystemExit("set CUDA_VISIBLE_DEVICES='' (no GPU in Phase 7)")
    os.makedirs(pc.OUT, exist_ok=True)
    os.makedirs(pc.WORK, exist_ok=True)
    tok = pc.tokenizer()
    summary = {'mode': MODE, 'limit': LIMIT, 'circuit_tracer': loc, 'phases': {}}
    try:
        for phase in (5, 4):
            P = pc.load_pairs(phase)
            jobs = token_jobs(P, tok)
            sizes = {s: sum(len(j['D']) == s for j in jobs) for s in (1, 2)}
            ps = {'pairs': P['n'], 'D_sizes': sizes,
                  'pe_in_D_both': sum(all(j['pe_in_D']) for j in jobs),
                  'pe_not_in_D': [(int(P['pprob'][j['k']]), j['D'], [m['p_e'] for m in j['mem']])
                                  for j in jobs if not all(j['pe_in_D'])],
                  'W_size_median': float(np.median([len(j['W']) for j in jobs])),
                  'distance_D_to_end_min': int(min(j['n_pos'] - 1 - max(j['D']) for j in jobs))}
            print(f"Phase {phase}: {P['n']} pairs; |D| {sizes}; p5_2 edit token in D for both members "
                  f"in {ps['pe_in_D_both']}; not: {ps['pe_not_in_D']}; |W| median {ps['W_size_median']}; "
                  f"min distance from D to the last position {ps['distance_D_to_end_min']}", flush=True)
            summary['phases'][phase] = ps
            if MODE == 'tokens':
                continue
            files = graph_files(phase, P)
            if LIMIT:
                jobs = jobs[:LIMIT]
            for j in jobs:
                j['paths'] = {m['expr_id']: files[m['expr_id']] for m in j['mem']}
            got = {}
            with Pool(WORKERS) as pool:
                for n, r in enumerate(pool.imap_unordered(_pair, jobs)):
                    if not r['ok']:
                        pool.terminate()
                        raise Gate(r['error'])
                    got[r['k']] = r
                    if (n + 1) % 25 == 0:
                        print(f"  Phase {phase}: {n + 1}/{len(jobs)} pairs, {time.time() - t_start:.0f} s",
                              flush=True)
            mems = [m for k in sorted(got) for m in got[k]['members']]
            ids = [m[0]['expr_id'] for m in mems]
            check(ids == P['ids'][:2 * len(jobs)].tolist(), "member order differs from load_pairs'")
            tag = TAG[phase]
            with open(os.path.join(pc.OUT, f'local_{tag}.jsonl.tmp'), 'w', encoding='utf-8') as f:
                for o, _, _ in mems:
                    f.write(json.dumps(o) + '\n')
            os.replace(os.path.join(pc.OUT, f'local_{tag}.jsonl.tmp'), os.path.join(pc.OUT, f'local_{tag}.jsonl'))
            indptr = np.r_[0, np.cumsum([len(pm[0]) for _, pm, _ in mems])]
            np.savez(os.path.join(pc.WORK, f'posmean_{tag}.tmp.npz'), ids=np.array(ids),
                     indices=np.concatenate([pm[0] for _, pm, _ in mems]),
                     data=np.concatenate([pm[1] for _, pm, _ in mems]), indptr=indptr,
                     shape=np.array([len(mems), N_LAYERS * D_SAE]))
            os.replace(os.path.join(pc.WORK, f'posmean_{tag}.tmp.npz'), os.path.join(pc.WORK, f'posmean_{tag}.npz'))
            off = np.r_[0, np.cumsum([len(a[0]) for _, _, a in mems])]
            np.savez(os.path.join(pc.OUT, f'positions_{tag}.npz'), ids=np.array(ids), offsets=off,
                     count=np.concatenate([a[0] for _, _, a in mems]),
                     actsum=np.concatenate([a[1] for _, _, a in mems]),
                     selcount=np.concatenate([a[2] for _, _, a in mems]))
            o = [m[0] for m in mems]
            pairs_o = o[0::2]
            ps.update({
                'graphs_read': len(mems),
                'prefix_identical_share': float(np.mean([x['prefix_identical'] for x in pairs_o])),
                'n_logits_differ_share': float(np.mean([a['n_logits'] != b['n_logits']
                                                        for a, b in zip(o[0::2], o[1::2])])),
                'b2_zero': int(sum(x['b2_zero'] for x in o)), 'b4_empty': int(sum(x['b4_empty'] for x in o)),
                'prefix_attr_edge_mismatch_median': float(np.median([x['prefix_attr']['edge_mismatch_share']
                                                                     for x in pairs_o])),
                'prefix_attr_rowsum_rel_diff_median': float(np.median([x['prefix_attr']['rowsum_rel_diff_median']
                                                                      for x in pairs_o])),
                'f4_float32_rel_dev_max': float(max(x['f4_float32_rel_dev'] for x in o)),
                'seconds_per_pair_median': float(np.median([got[k]['seconds'] for k in got])),
                'sha256': {f'local_{tag}.jsonl': pc.sha256(os.path.join(pc.OUT, f'local_{tag}.jsonl')),
                           f'positions_{tag}.npz': pc.sha256(os.path.join(pc.OUT, f'positions_{tag}.npz')),
                           f'posmean_{tag}.npz': pc.sha256(os.path.join(pc.WORK, f'posmean_{tag}.npz'))}})
            print(f"  Phase {phase}: {len(mems)} graphs read; prefix activations bit-identical in "
                  f"{ps['prefix_identical_share']:.3f} of pairs; attribution on the shared prefix differs in "
                  f"{ps['prefix_attr_edge_mismatch_median']:.4f} of edges (median); logit counts differ in "
                  f"{ps['n_logits_differ_share']:.3f}; b2 = 0 in {ps['b2_zero']}", flush=True)
    except Gate as e:
        summary['gate_G2'] = f'FAILED: {e}'
        pc.write_json(os.path.join(pc.OUT, 'extract.json' if MODE != 'tokens' else 'extract_tokens.json'), summary)
        print(f"G2 FAILED: {e}", flush=True)
        sys.exit(2)
    summary['gate_G2'] = 'passed'
    summary['seconds'] = time.time() - t_start
    pc.write_json(os.path.join(pc.OUT, 'extract.json' if MODE != 'tokens' else 'extract_tokens.json'), summary)
    print(f"G2 passed ({MODE}); {time.time() - t_start:.0f} s", flush=True)


if __name__ == '__main__':
    main()
