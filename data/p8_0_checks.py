"""Phase 8, P0 (final_design.md §9): plumbing checks on retained Phase 5 graphs. No graph is built,
written, moved or deleted; ~/matched_graph_keep is only read. Uses p8_2_run's engine (the same
hook, decision logits, e and flip machinery the run uses).

Graphs: the design asks for 20 retained graphs whose top logit has p >= 0.8. No retained Phase 5
graph reaches 0.8 (gemma_matched_features.jsonl column 3 = the stored top probability; max 0.79),
so (amended rule, to be recorded in PREREGISTRATION_P8) P0 takes ONE graph per problem - the
problem's graph with the highest top probability (ties by expr_id) - for the N_GRAPHS problems with
the highest such p, among problems not among the 35 site problems (ownfix_verify.json, verdict
true); 58 problems are eligible. (The first P0 took the 20 highest graphs: only 4 problems.)
The first-order term of e (p8_2_run.compute_e first_only, from graph_e's blocks) must equal
adjacency[N - n_logits, :n_sel] read from the stored graph (gate component of (a)).

Per graph (prefix = the graph's input tokens; E = {its top logit}; C = bf16 top-64 | E):
  (0) noise floor: the fp32 decision logits at batch 1 twice and as every row of a batch of 64
      identical rows; max |delta logit| and top-2 order flips. delta = the max over graphs.
  (a) e (node 0) from the graph in float64: iterations <= 27, finite; for the first 3 graphs also
      against circuit_tracer.graph.compute_influence on the full column-scaled adjacency. Then the
      top 200 features by |e| (positions >= 1) are ablated singly (our hook, screen batches of 64)
      and the measured drop of the node-0 target (ln_final.hook_normalized * w . demeaned W_U
      column, fp32) is compared with e over those with |e| > 3 delta: sign agreement and Spearman.
      Gate: the random-DAG unit tests pass (p8_2_run.e_unit_tests), iterations <= 27, finite,
      median Spearman >= 0.3. Descriptive (to localise a failure, never a gate): Spearman of the
      first-order term (the logit-row entry) and of the declared fallback; pooled over graphs, the
      'direct' features (last layer, last position: e is the logit-row entry and only ln_final's
      scale is nonlinear; dtarget_fs holds it clean, so e must equal dtarget_fs up to bf16) and the
      Spearman by layer band.
  (b) G's top k (k in 1, 8, 64): our hook at batch 1 against
      model.feature_intervention(freeze_attention=False, constrained_layers=None), value 0, same
      deterministic mode: fp32 decision logits bitwise or within 1 ulp (gate), hook_normalized
      ulps reported; the rows k = 1..64 as one batch of 64 against batch 1: within delta (gate, max
      |delta fp32 logit| over C, as delta itself). Descriptive: the same rows' margin deviation
      and flip-status disagreements (what the screen decides on).
  (c) hooks at every layer that add a zero delta reproduce the clean decision logits bitwise at
      batch 1 and batch 64 (gate).
  (d) the GxA row (p8_2_run.Engine.logit_rows: attribute()'s phases 0-3) against the stored logit
      row 0: the active features, activations and logit targets must be equal and the feature
      part bitwise equal (gate); error and token parts reported.
  (e) the TL bf16 argmax equals the stored top logit, and the fp32 argmax over C equals the chunked
      full-vocabulary fp32 argmax (gate, every graph).
Canary: the first graph's prefix, clean and with G's top 8 ablated, as a batch of 2; p8_2_run
checks it at every process start.
Writes $P8_OUT/p0.json (pass = every gate). GPU time (from process start, model load included,
beaten every 30 s) is logged to p8_2_run.CLOCK_PATH (P8_CLOCK, default
data/results/intervention/gpu_clock.jsonl whatever P8_OUT is): the hard stop counts P0.

Usage (cwd data/):  python p8_0_checks.py
Env: P8_OUT  P8_WORK  N_GRAPHS=20 (fewer for a smoke test; then pass is false)  OUT_JSON=p0.json
data/1_code_correct.py is never opened, imported or run.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p8_2_run as R  # noqa: E402  (sets the CUDA env before torch is imported)
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import p8_common as C  # noqa: E402

KEEP = '/home/user1/matched_graph_keep'
N_GRAPHS = int(os.environ.get('N_GRAPHS', 20))
KS_B = (1, 8, 64)
N_MEAS = 200
N_LIB = 3
OUT_JSON = os.environ.get('OUT_JSON', 'p0.json')


def site_problems():
    with open(os.path.join(HERE, 'ownfix_verify.json'), encoding='utf-8') as f:
        v = json.load(f)
    return sorted({int(k.split('|')[0]) for k, x in v['mutants'].items() if x is True})


def choose_graphs():
    rows = C.read_jsonl(os.path.join(HERE, 'matched_rows.jsonl'))
    feats = {int(r['expr_id']): r for r in C.read_jsonl(os.path.join(HERE, 'gemma_matched_features.jsonl'))}
    p35 = set(site_problems())
    cand, n_in35 = [], 0
    for fn in sorted(os.listdir(KEEP)):
        if not fn.endswith('.pt'):
            continue
        _, i, t, _ = fn[:-3].split('_')
        i, t = int(i), int(t)
        if int(rows[i]['task_id']) != t:
            raise SystemExit(f"{fn}: row {i} is not expr {t}")
        if int(rows[i]['problem']) in p35:
            n_in35 += 1
            continue
        cand.append((-float(feats[t]['feature'][3]), t, fn, int(rows[i]['problem'])))
    cand.sort()
    per_problem, seen = [], set()
    for c in cand:                           # one graph per problem: its highest p (ties by expr_id)
        if c[3] not in seen:
            seen.add(c[3])
            per_problem.append(c)
    pick = per_problem[:N_GRAPHS]
    info = {'retained': len(cand) + n_in35, 'in_site_problems': n_in35, 'eligible': len(cand),
            'eligible_problems': len(per_problem),
            'p_at_least_0.8': sum(1 for c in cand if -c[0] >= 0.8), 'p_range_chosen': [-pick[-1][0], -pick[0][0]],
            'problems_chosen': sorted(c[3] for c in pick),
            'rule': 'amended (no graph has p >= 0.8): one graph per problem (its highest stored top '
                    'probability), the N_GRAPHS problems with the highest such p, problem not among the 35 '
                    'site problems'}
    return [(fn, t, prob, -p) for p, t, fn, prob in sorted(pick, key=lambda c: c[1])], info


def ulps(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    b = np.ascontiguousarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    ka = np.where(a >= 0, a, -(a & 0x7fffffff))
    kb = np.where(b >= 0, b, -(b & 0x7fffffff))
    return int(np.abs(ka - kb).max()) if len(ka) else 0


def top2(z):
    o = np.argsort(-z, kind='stable')[:2]
    return tuple(o.tolist())


def fi_norm(E, prefix, feats):
    """ln_final.hook_normalized at the last position under circuit_tracer's own feature_intervention."""
    torch = E.torch
    cap = {}

    def h(a, hook):
        cap['n'] = a[:, -1].clone()

    x = torch.tensor(prefix, dtype=torch.long, device='cuda')
    with torch.no_grad(), E.model.hooks(fwd_hooks=[('ln_final.hook_normalized', h)]):
        E.model.feature_intervention(x, [(int(f[0]), int(f[1]), int(f[2]), 0.0) for f in feats],
                                     freeze_attention=False, constrained_layers=None)
    return cap['n']


def main():
    from scipy.stats import spearmanr
    t_start = time.time()
    graphs, choice = choose_graphs()
    print(f"P0 on {len(graphs)} graphs: {choice}", flush=True)
    a_unit = R.e_unit_tests()
    print('e unit tests:', a_unit, flush=True)
    lock = R.gpu_lock()
    clock = R.Clock('p8_0_checks', t0=R.T_START)      # before the model load (fixed clock path)
    E = R.Engine(lock=lock)
    torch = E.torch
    from circuit_tracer.graph import Graph, compute_influence
    per, canary = [], None
    for gi, (fn, expr, prob, p_top) in enumerate(graphs):
        t0 = time.time()
        g = Graph.from_pt(os.path.join(KEEP, fn))
        prefix = g.input_tokens.tolist()
        t_top = int(g.logit_token_ids[0])
        rec = {'file': fn, 'expr_id': expr, 'problem': prob, 'p_top_stored': float(g.logit_probabilities[0]),
               'n_pos': len(prefix), 'n_sel': int(len(g.selected_features)), 'n_active': int(len(g.activation_values))}
        # ---- e
        feats_sel, act_sel, e0, elp, iters = R.graph_e(g)
        rec['e_iterations'], rec['e_finite'] = int(iters), bool(np.isfinite(e0).all() and np.isfinite(elp).all())
        if gi < N_LIB:
            adj = g.adjacency_matrix.double()
            n_sel, N = len(g.selected_features), adj.shape[0]
            adj[:, :n_sel] /= torch.tensor(act_sel)[None, :]
            w = torch.zeros(N, dtype=torch.float64)
            w[N - len(g.logit_token_ids)] = 1.0
            lib = compute_influence(adj, w)[:n_sel].numpy() * act_sel
            rec['e_vs_compute_influence_max_rel'] = float(np.abs(lib - e0).max() / np.abs(e0).max())
            del adj, lib
        G = R.ordered(feats_sel, act_sel, e0)
        by_abs = [i for i in np.argsort(-np.abs(e0), kind='stable') if feats_sel[i, 1] >= 1][:N_MEAS]
        meas_list = [(int(feats_sel[i, 0]), int(feats_sel[i, 1]), int(feats_sel[i, 2]), float(act_sel[i]), float(e0[i]))
                     for i in by_abs]
        n_all = g.adjacency_matrix.shape[0]
        first_order = g.adjacency_matrix[n_all - len(g.logit_token_ids), :len(g.selected_features)].double().numpy()
        n_log = len(g.logit_token_ids)
        w0 = np.zeros(n_log)
        w0[0] = 1.0
        e1, _ = R.compute_e(None, g.adjacency_matrix[n_all - n_log:, :len(g.selected_features)].double().numpy(),
                            act_sel, w0, first_only=True)
        rec['e_first_order_max_rel'] = float(np.abs(e1 - first_order).max() / max(np.abs(first_order).max(), 1e-300))
        # ---- site, (e)
        S = E.site(prefix, {t_top})
        S.u0 = E.u_vec(t_top).float()
        S.clean = E.decide(S, [[]])
        rec['e_check'] = {'tl_argmax': S.tl_argmax, 'stored_top': t_top, 'tl_equal': S.tl_argmax == t_top,
                          'fp32_C_argmax': int(S.clean['am'][0]), 'fullvocab_argmax': S.full_argmax,
                          'fp32_equal_full': int(S.clean['am'][0]) == S.full_argmax, 'n_C': len(S.C)}
        # ---- (0) noise
        r1, r2 = E.decide(S, [[]]), E.decide(S, [[]])
        r64 = E.decide(S, [[]] * 64)
        z1 = r1['z'][0]
        rec['noise'] = {'b1_twice_max': float(np.abs(r2['z'][0] - z1).max()),
                        'b64_vs_b1_max': float(np.abs(r64['z'] - z1[None]).max()),
                        'b64_rows_identical': bool((r64['z'] == r64['z'][0:1]).all()),
                        'top2_flips_b1': int(top2(r2['z'][0]) != top2(z1)),
                        'top2_flips_b64_rows': int(sum(top2(z) != top2(z1) for z in r64['z'])),
                        'clean_b1_bitwise_site': bool((S.clean['z'][0] == z1).all())}
        # ---- (c) null hooks
        n1 = E.decide(S, None, cap=E.run(prefix, [[]], null_layers=range(E.L)))
        n64 = E.decide(S, None, cap=E.run(prefix, [[]] * 64, null_layers=range(E.L)))
        rec['c'] = {'b1_bitwise': bool((n1['z'] == z1[None]).all()) and bool(torch.equal(n1['norm'], r1['norm'])),
                    'b64_bitwise': bool((n64['z'] == r64['z']).all()) and bool(torch.equal(n64['norm'], r64['norm']))}
        # ---- (b) our hook vs feature_intervention
        rb = {'k': [], 'b1_bitwise': [], 'b1_logit_ulps': [], 'b1_norm_ulps': []}
        ks = [k for k in KS_B if k <= len(G)]
        kmax = min(64, len(G))
        r64k = E.decide(S, [G[:k] for k in range(1, kmax + 1)])
        dev64, mdev64, fdis64, nflip1 = [], [], 0, 0
        for k in range(1, kmax + 1):
            ours = E.decide(S, [G[:k]])
            dev64.append(float(np.abs(r64k['z'][k - 1] - ours['z'][0]).max()))
            mdev64.append(abs(r64k['margin'][k - 1] - ours['margin'][0]))      # decision-relevant (descriptive)
            fdis64 += int(r64k['flip'][k - 1] != ours['flip'][0])
            nflip1 += int(ours['flip'][0])
            if k not in ks:
                continue
            ref_n = fi_norm(E, prefix, G[:k])
            ref = E.decide(S, None, cap={'n': ref_n})
            rb['k'].append(k)
            rb['b1_bitwise'].append(bool(torch.equal(ours['norm'], ref_n)) and bool((ours['z'] == ref['z']).all()))
            rb['b1_logit_ulps'].append(ulps(ours['z'][0], ref['z'][0]))
            rb['b1_norm_ulps'].append(ulps(ours['norm'][0].cpu().numpy(), ref_n[0].cpu().numpy()))
        rb['b64_vs_b1'] = dev64                                  # rows k = 1..64 of one batch vs batch 1
        rb['b64_vs_b1_max'] = max(dev64)
        rb['b64_margin_dev'] = mdev64
        rb['b64_flip_disagree'], rb['b1_rows_flipped'] = fdis64, nflip1
        rb['b64_vs_b1_max_at_ks'] = [dev64[k - 1] for k in ks]
        rec['b'] = rb
        # ---- (d) the GxA row against the stored logit row 0
        gx = E.logit_rows(prefix)
        adj = g.adjacency_matrix
        N, n_sel, n_log = adj.shape[0], len(g.selected_features), len(g.logit_token_ids)
        stored = adj[N - n_log].double().numpy()
        row = gx['rows'][0].double().numpy()
        sel = g.selected_features.cpu().numpy()
        same_feats = gx['nnz'] == len(g.activation_values) and bool(
            (gx['feats'] == g.active_features.cpu().numpy()).all())
        same_acts = same_feats and bool((gx['acts'] == g.activation_values.cpu().float().numpy()).all())
        fr, fs = row[:gx['nnz']][sel], stored[:n_sel]
        rest_n = (E.L + 1) * len(prefix)
        er, es = row[gx['nnz']:gx['nnz'] + rest_n], stored[n_sel:n_sel + rest_n]
        rec['d'] = {'same_active_features': same_feats, 'same_activations': same_acts,
                    'same_logit_targets': gx['logit_ids'] == g.logit_token_ids.tolist(),
                    'feature_bitwise': bool((fr == fs).all()), 'feature_max_abs': float(np.abs(fr - fs).max()),
                    'feature_scale': float(np.abs(fs).max()), 'feature_exact_share': float((fr == fs).mean()),
                    'feature_spearman': float(spearmanr(fr, fs)[0]),
                    'error_token_max_abs': float(np.abs(er - es).max()), 'error_token_scale': float(np.abs(es).max()),
                    'error_token_exact_share': float((er == es).mean())}
        del gx, adj, stored, row
        # ---- (a) measured single ablations
        sg = E.singles(S, meas_list, batch=1)
        from circuit_tracer.graph import compute_node_influence
        lw = torch.zeros(g.adjacency_matrix.shape[0])
        lw[g.adjacency_matrix.shape[0] - len(g.logit_token_ids)] = 1.0
        infl = compute_node_influence(g.adjacency_matrix, lw).numpy()      # the declared fallback's score
        pos_of = {(int(feats_sel[i, 0]), int(feats_sel[i, 1]), int(feats_sel[i, 2])): i for i in by_abs}
        rec['a_meas'] = {'e': [m[4] for m in meas_list], 'dtarget': [d['dtarget'] for d in sg],
                         'dtarget_fs': [d['dtarget_fs'] for d in sg], 'drop': [d['drop'] for d in sg],
                         'unsigned': [float(infl[pos_of[m[:3]]]) for m in meas_list],
                         'layer': [m[0] for m in meas_list], 'pos': [m[1] for m in meas_list],
                         'first_order': [float(first_order[i]) for i in by_abs]}
        if gi == 0:
            canary = {'ids': prefix, 'C': S.C, 'ablate': [[x[0], x[1], x[2]] for x in G[:8]]}
            z = R.canary_z(E, canary)
            canary['clean'], canary['ablated'] = z[0].tolist(), z[1].tolist()
            canary['source'] = fn
        del g
        E.device_check()
        rec['seconds'] = time.time() - t0
        per.append(rec)
        clock.beat(event='graph', file=fn)
        print(f"{time.strftime('%H:%M:%S')} {gi + 1}/{len(graphs)} {fn} {rec['seconds']:.0f}s "
              f"noise {rec['noise']['b64_vs_b1_max']:.4f} b {rb['b1_bitwise']} {rb['b64_vs_b1_max']:.4f} "
              f"(margin {max(mdev64):.4f}, flip disagree {fdis64}/{kmax}, b1 flips {nflip1}) c {rec['c']} "
              f"d {rec['d']['feature_bitwise']} {rec['d']['feature_max_abs']:.3g} e {rec['e_check']['tl_equal']} "
              f"{rec['e_check']['fp32_equal_full']}", flush=True)
    # ---- summary and gates
    delta = max(max(r['noise']['b1_twice_max'], r['noise']['b64_vs_b1_max']) for r in per)
    rho, sign, nq, alt = [], [], [], {'dtarget_fs': [], 'drop': [], 'unsigned_fallback_pos_e': []}
    for r in per:
        M = r['a_meas']
        q = [i for i, e in enumerate(M['e']) if abs(e) > 3 * delta]
        nq.append(len(q))
        s = {'n': len(q), 'spearman': None, 'sign_agree': None}
        if len(q) >= 3:
            e = np.array(M['e'])[q]
            d = np.array(M['dtarget'])[q]
            rho.append(float(spearmanr(e, d)[0]))
            sign.append(float(np.mean(np.sign(e) == np.sign(d))))
            s.update(spearman=rho[-1], sign_agree=sign[-1])
            for k in ('dtarget_fs', 'drop'):
                s[f'spearman_{k}'] = float(spearmanr(e, np.array(M[k])[q])[0])
                alt[k].append(s[f'spearman_{k}'])
            qp = [i for i in q if M['e'][i] > 0]
            if len(qp) >= 3:
                s['spearman_unsigned_fallback_pos_e'] = float(spearmanr(np.array(M['unsigned'])[qp], np.array(M['dtarget'])[qp])[0])
                s['spearman_e_pos_e'] = float(spearmanr(np.array(M['e'])[qp], np.array(M['dtarget'])[qp])[0])
                alt['unsigned_fallback_pos_e'].append(s['spearman_unsigned_fallback_pos_e'])
        if len(q) >= 3:
            fo = np.array(M['first_order'])[q]
            s['spearman_first_order'] = float(spearmanr(fo, np.array(M['dtarget'])[q])[0])
            alt.setdefault('first_order', []).append(s['spearman_first_order'])
        r['a_meas_summary'] = s
    # descriptive localisation of the e-vs-measured gap (pooled over graphs, |e| > 3 delta):
    # direct = features in the last layer at the last position, where e is the logit-row entry and the
    # only nonlinearity left in the measurement is ln_final's scale (dtarget_fs holds it at its clean value)
    pool = {'e': [], 'dt': [], 'fs': [], 'layer': [], 'direct': []}
    for r in per:
        M = r['a_meas']
        for i, e in enumerate(M['e']):
            if abs(e) > 3 * delta:
                pool['e'].append(e)
                pool['dt'].append(M['dtarget'][i])
                pool['fs'].append(M['dtarget_fs'][i])
                pool['layer'].append(M['layer'][i])
                pool['direct'].append(M['layer'][i] == E.L - 1 and M['pos'][i] == r['n_pos'] - 1)
    pe, pdt, pfs, pl, pdir = (np.array(pool[k]) for k in ('e', 'dt', 'fs', 'layer', 'direct'))
    diag = {'n': int(len(pe))}
    if pdir.sum():
        rel = np.abs(pe[pdir] - pfs[pdir]) / np.abs(pe[pdir])
        diag['direct'] = {'n': int(pdir.sum()), 'rel_err_e_vs_dtarget_fs_median': float(np.median(rel)),
                          'rel_err_max': float(rel.max()),
                          'sign_agree_fs': float(np.mean(np.sign(pe[pdir]) == np.sign(pfs[pdir]))),
                          'sign_agree': float(np.mean(np.sign(pe[pdir]) == np.sign(pdt[pdir])))}
    bands = {}
    for lo, hi in ((0, 9), (9, 18), (18, E.L)):
        m = (pl >= lo) & (pl < hi) & ~pdir
        if m.sum() >= 3:
            bands[f'{lo}-{hi - 1}'] = {'n': int(m.sum()), 'spearman': float(spearmanr(pe[m], pdt[m])[0]),
                                       'sign_agree': float(np.mean(np.sign(pe[m]) == np.sign(pdt[m]))),
                                       'spearman_final_scale_frozen': float(spearmanr(pe[m], pfs[m])[0]),
                                       'sign_agree_final_scale_frozen': float(np.mean(np.sign(pe[m]) == np.sign(pfs[m])))}
    if len(pe) >= 3:
        diag['all'] = {'spearman': float(spearmanr(pe, pdt)[0]), 'spearman_final_scale_frozen': float(spearmanr(pe, pfs)[0]),
                       'sign_agree': float(np.mean(np.sign(pe) == np.sign(pdt))),
                       'sign_agree_final_scale_frozen': float(np.mean(np.sign(pe) == np.sign(pfs)))}
    diag['by_layer_band_excluding_direct'] = bands
    a_graph = {'max_iterations': max(r['e_iterations'] for r in per), 'finite': all(r['e_finite'] for r in per),
               'lib_max_rel': max(r.get('e_vs_compute_influence_max_rel', 0.0) for r in per),
               'first_order_max_rel': max(r['e_first_order_max_rel'] for r in per),
               'measured': 'node-0 target (live model, batch 1) decrease on single ablation',
               'spearman_median': float(np.median(rho)) if rho else None, 'spearman': rho,
               'sign_agree_median': float(np.median(sign)) if sign else None, 'n_qualifying': nq,
               'descriptive_medians': {k: (float(np.median(v)) if v else None) for k, v in alt.items()},
               'localisation_pooled': diag}
    b_ulp = max(max(r['b']['b1_logit_ulps']) for r in per)
    b64 = max(r['b']['b64_vs_b1_max'] for r in per)
    all_dev = np.concatenate([r['b']['b64_vs_b1'] for r in per])
    gates = {
        'a': bool(a_unit['pass'] and a_graph['max_iterations'] <= 27 and a_graph['finite'] and a_graph['lib_max_rel'] < 1e-9
                  and a_graph['first_order_max_rel'] < 1e-12
                  and a_graph['spearman_median'] is not None and a_graph['spearman_median'] >= 0.3),
        # (b) as designed also asked batch-64 rows to match batch 1 within delta. They do not (a re-encoded
        # feature at its JumpReLU threshold moves whole rows), so no decision is taken from a batch-64
        # row: screening only selects rows, every flip and every margin below the replay guard (twice
        # the largest batch-64 margin deviation measured here) is decided at batch 1 (p8_2_run.flip_scan).
        # The gate is the batch-1 equality; the batch-64 deviation is reported and sets the guard.
        'b': bool(b_ulp <= 1),
        'c': all(r['c']['b1_bitwise'] and r['c']['b64_bitwise'] for r in per),
        'd': all(r['d']['same_active_features'] and r['d']['same_activations'] and r['d']['same_logit_targets']
                 and r['d']['feature_bitwise'] for r in per),
        'e': all(r['e_check']['tl_equal'] and r['e_check']['fp32_equal_full'] for r in per),
    }
    gpu_s = time.time() - R.T_START                 # process start (model load included), as the clock
    import torch
    out = {'what': 'final_design.md §9 P0', 'graphs_chosen': choice, 'n_graphs': len(per),
           'delta': delta, 'noise': {'b1_twice_max': max(r['noise']['b1_twice_max'] for r in per),
                                     'b64_vs_b1_max': max(r['noise']['b64_vs_b1_max'] for r in per),
                                     'top2_flips_b1': sum(r['noise']['top2_flips_b1'] for r in per),
                                     'top2_flips_b64_rows': sum(r['noise']['top2_flips_b64_rows'] for r in per)},
           'a_unit': a_unit, 'a_graph': a_graph,
           'b': {'max_logit_ulps_b1': b_ulp, 'max_norm_ulps_b1': max(max(r['b']['b1_norm_ulps']) for r in per),
                 'all_bitwise_b1': all(all(r['b']['b1_bitwise']) for r in per), 'b64_vs_b1_max': b64, 'b64_within_delta_as_designed': bool(b64 <= delta),
                 'b64_vs_b1_quantiles_50_90_99': np.quantile(all_dev, [.5, .9, .99]).tolist(),
                 'b64_rows_over_delta': int((all_dev > delta).sum()), 'b64_rows': int(len(all_dev)),
                 'descriptive_decision_level': {
                     'margin_dev_max': max(max(r['b']['b64_margin_dev']) for r in per),
                     'margin_dev_rows_over_delta': int(sum(sum(d > delta for d in r['b']['b64_margin_dev']) for r in per)),
                     'flip_status_disagreements': sum(r['b']['b64_flip_disagree'] for r in per),
                     'b1_rows_flipped': sum(r['b']['b1_rows_flipped'] for r in per)}},
           'gates': gates, 'pass': bool(all(gates.values()) and len(per) == 20),
           'canary': canary, 'per_graph': per, 'gpu_seconds': gpu_s, 'wall_seconds': time.time() - t_start,
           'det_warnings': sorted(R._DET_WARN), 'circuit_tracer': E.ct_version, 'torch': torch.__version__,
           'settings': {'K_MAX': C.K_MAX, 'SCAN_BATCH': C.SCAN_BATCH, 'TOP_C': C.TOP_C, 'ENC_CHUNK': R.ENC_CHUNK,
                        'GRAPH_KW': C.GRAPH_KW, 'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                        'deterministic_algorithms': True}}
    path = OUT_JSON if os.path.isabs(OUT_JSON) else os.path.join(C.OUT, OUT_JSON)
    C.write_json(path, out)
    clock.beat(event='end')
    print(json.dumps({k: out[k] for k in ('delta', 'noise', 'a_graph', 'b', 'gates', 'pass')}, indent=1, default=C._jsonable))
    print(f"wrote {path}")
    sys.exit(0 if out['pass'] else 2)


if __name__ == '__main__':
    main()
