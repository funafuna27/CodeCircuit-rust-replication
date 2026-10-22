"""Phase 9, part ①: the probes and the cheap signals (descriptive; PREREGISTRATION_P9_MUTSTATE.md §2).

Two contrasts, each at two positions of S's token sequence:
  probe 1   V against A (x's `mut` removed or kept; both carry y's edit)
  probe 2   V against Pz (x's or another necessary binding z's `mut` removed; both carry y's edit)
  q         the position of q_place (x's first token at the first mutation site)
  bg        one background token per row: among the item's background tokens at least 20 tokens after
            the later edit (for probe 2, after z's edit too), the one whose S-side entropy is closest to
            q_place's on a log scale (ties: the one nearer to q); rows without one are left out
At each position (forward only; circuit-tracer's ReplacementModel, no attribution):
  acts      every layer's transcoder activations at that position, stored as the difference (sparse)
  cheap     from the output distribution at that position: log p of the next token, entropy, top
            probability (each as the difference), and KL(other||V) - KL(V||other)
Probe: the 45 leading right singular vectors of the rows' activation differences (a row's sign does not
change them), the projections divided by one overall RMS (no per-column scaling: some directions vary in
held-out problems only); each row's orientation drawn at random (stream [20261022, 61]); logistic
regression without intercept on the oriented projections, 5 folds grouped by problem; a row counts as
oriented correctly when its held-out score has the sign of its orientation. A = the mean over problems
(rows -> (source, x) -> source -> problem) of that share. The same for the 4 cheap signals. Read:
A(q) - A(bg), 5,000 bootstrap draws of problems.

  ROLE=primary|pilot  STEP=acts|fit|all  WORK=/home/user1/p9_work/forward  OUT=results/p9_state/b_probe_<role>.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_common as C  # noqa: E402
import p9_b_analysis as A  # noqa: E402

ROLE = os.environ.get('ROLE', 'pilot')
STEP = os.environ.get('STEP', 'all')
WORK = os.environ.get('WORK', '/home/user1/p9_work/forward')
OUT = os.environ.get('OUT', os.path.join(C.OUT_DIR, f'b_probe_{ROLE}.json'))
K_SVD, N_FOLDS, N_BOOT = 45, 5, 5000
BG_MIN_DIST = 20
CHEAP = ('logp_next', 'entropy', 'top_prob', 'kl_diff')


def pick(r, cands):
    """cands: (h, distance, payload); the payload of the entropy-nearest candidate (ties: nearer to q)."""
    hq = r['span_h'][0]
    c = [(abs(np.log(h + 1e-6) - np.log(hq + 1e-6)), -d, k) for k, (h, d, _) in enumerate(cands)]
    return cands[min(c)[2]][2] if c else None


def select(rows, items, prose, F):
    """Rows of probe 1 and probe 2: dicts with the texts and the token indices of q and bg in each."""
    p1, p2 = [], []
    for r in rows:
        it = items[r['item_key']]
        p = prose[tuple(it['key'])]
        base = len(p) + 1
        meta = {'problem': r['problem'], 'source': r['source'], 'x': r['x'], 'item_key': r['item_key']}
        iq, jvq, jaq = r['q_j']
        bgs = [(h, d, (i, jv, ja)) for (_, h, d), (i, jv, ja) in zip(r['bg'], r['bg_j'])]
        b = pick(r, [t for t in bgs if t[1] >= BG_MIN_DIST])
        if b is not None:
            p1.append(dict(meta, texts=(p + ' ' + it['V'], p + ' ' + it['A']), pos=((jvq, jaq), (b[1], b[2]))))
        S, L, y = it['S'], it['piece'], it['y']
        offS = F[A.sha(p + ' ' + S)]['off']
        idsS = F[A.sha(p + ' ' + S)]['ids']
        for z in it['placebos']:
            tz = p + ' ' + z['text']
            rz = F[A.sha(tz)]
            mz = A.smap(S, [('del', z['z_mut_pos'], L), ('ins', y['name_pos'], L)])

            def jz(i):
                j = A.tok_at(rz, base, mz[int(offS[i][0]) - base])
                return j if j is not None and rz['ids'][j] == idsS[i] else None
            iz = A.tok_at({'off': offS}, base, z['z_mut_pos'])
            if jz(iq) is None:
                continue
            cand = [(h, d, (i, jv)) for h, d, (i, jv, _) in bgs if i - max(i - d, iz) >= BG_MIN_DIST and jz(i) is not None]
            b = pick(r, cand)
            if b is not None:
                p2.append(dict(meta, texts=(p + ' ' + it['V'], tz), pos=((jvq, jz(iq)), (b[1], jz(b[0])))))
    return p1, p2


def acts_step(sets, F):
    import torch
    from scipy.sparse import csr_matrix
    from circuit_tracer import ReplacementModel
    model = ReplacementModel.from_pretrained('google/gemma-2-2b-it', 'gemma', dtype=torch.bfloat16).to('cuda')
    cache = {}
    for l in range(model.cfg.n_layers):
        hp = model.blocks[l].get_submodule(model.feature_input_hook)
        hp.register_forward_hook(lambda m, i, o, l=l: cache.__setitem__(l, o.detach()))

    def run(text, pos):
        ids = torch.tensor(F[A.sha(text)]['ids'], dtype=torch.long, device='cuda')[None]
        logits = model(ids)[0].float()
        logq = torch.log_softmax(logits[pos], dim=-1)
        acts = torch.cat([model.transcoders.encode_layer(cache[l][0, pos], l).float() for l in range(model.cfg.n_layers)], dim=-1)
        return acts, logq, ids[0, [q + 1 for q in pos]]

    summary, secs = {}, []
    with torch.inference_mode():
        for name, rows in sets.items():
            data = {'q': ([], [], [0]), 'bg': ([], [], [0])}
            cheap = {'q': [], 'bg': []}
            D = None
            for n, row in enumerate(rows, 1):
                t0 = time.time()
                a1, l1, n1 = run(row['texts'][0], [row['pos'][0][0], row['pos'][1][0]])
                a2, l2, n2 = run(row['texts'][1], [row['pos'][0][1], row['pos'][1][1]])
                D = a1.shape[1]
                for k, arm in enumerate(('q', 'bg')):
                    d = (a1[k] - a2[k]).cpu().numpy()
                    nz = np.flatnonzero(d)
                    data[arm][0].append(d[nz].astype(np.float32))
                    data[arm][1].append(nz.astype(np.int32))
                    data[arm][2].append(data[arm][2][-1] + len(nz))
                    pa, pb = l1[k].exp(), l2[k].exp()
                    kl_ba = float((pb * (l2[k] - l1[k])).sum())
                    kl_ab = float((pa * (l1[k] - l2[k])).sum())
                    cheap[arm].append([float(l1[k][n1[k]] - l2[k][n2[k]]),
                                       float(-(pa * l1[k]).sum() + (pb * l2[k]).sum()),
                                       float(pa.max() - pb.max()), kl_ba - kl_ab])
                secs.append(time.time() - t0)
                if n % 250 == 0 or n == len(rows):
                    print(f'  {name} {n}/{len(rows)}  median {np.median(secs):.2f} s per row', flush=True)
            mats = {arm: csr_matrix((np.concatenate(v[0]), np.concatenate(v[1]), np.array(v[2])), shape=(len(rows), D))
                    for arm, v in data.items()}
            np.savez_compressed(os.path.join(WORK, f'probe_{ROLE}_{name}.npz'),
                                **{f'{arm}_{f}': getattr(m, f) for arm, m in mats.items() for f in ('data', 'indices', 'indptr')},
                                shape=np.array([len(rows), D]), cheap_q=np.array(cheap['q']), cheap_bg=np.array(cheap['bg']),
                                keys=np.array([row['item_key'] for row in rows]))
            summary[name] = len(rows)
    summary.update(seconds_total=round(float(np.sum(secs)), 1), max_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
    return summary


def oriented_share(X, groups, rng_o):
    """Per-row correctness (1/0) of a held-out logistic probe on randomly oriented rows of X."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    o = rng_o.choice([-1.0, 1.0], size=len(X))
    Xo = X * o[:, None]
    y = (o > 0).astype(int)
    correct = np.zeros(len(X))
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(Xo, y, groups):
        m = LogisticRegression(fit_intercept=False, max_iter=5000).fit(Xo[tr], y[tr])
        correct[te] = ((m.decision_function(Xo[te]) > 0) == (o[te] > 0)).astype(float)
    return correct


def boot(per_problem, stream):
    v = np.array([per_problem[k] for k in sorted(per_problem)])
    rb = np.random.default_rng([20261022, stream])
    b = np.array([v[rb.integers(len(v), size=len(v))].mean() for _ in range(N_BOOT)])
    return {'problems': int(len(v)), 'mean': round(float(v.mean()), 4),
            'ci95': [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]}


def fit_one(name, rows, stream0):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    z = np.load(os.path.join(WORK, f'probe_{ROLE}_{name}.npz'))
    if z['keys'].tolist() != [row['item_key'] for row in rows]:
        raise SystemExit(f'probe_{ROLE}_{name}.npz: row order differs from the selection')
    groups = np.array([row['problem'] for row in rows])
    out, corr = {'rows': len(rows)}, {}
    for arm in ('q', 'bg'):
        X = csr_matrix((z[f'{arm}_data'], z[f'{arm}_indices'], z[f'{arm}_indptr']), shape=tuple(z['shape'])).astype(np.float64)
        k = min(K_SVD, min(X.shape) - 1)
        v0 = np.random.default_rng([20261022, 60]).standard_normal(min(X.shape))
        _, s, Vt = svds(X, k=k, v0=v0, solver='arpack')
        Zp = np.asarray(X @ Vt.T)
        Zp /= np.sqrt((Zp ** 2).mean())
        out[f'svd_share_{arm}'] = round(float((s ** 2).sum() / X.multiply(X).sum()), 4)
        corr[f'probe_{arm}'] = oriented_share(Zp, groups, np.random.default_rng([20261022, 61]))
        ch = z[f'cheap_{arm}'] / np.sqrt((z[f'cheap_{arm}'] ** 2).mean(axis=0))
        corr[f'cheap_{arm}'] = oriented_share(ch, groups, np.random.default_rng([20261022, 61]))
        for j, nm in enumerate(CHEAP):
            corr[f'cheap_{arm}_{nm}'] = oriented_share(ch[:, [j]], groups, np.random.default_rng([20261022, 61]))
    pp = {k: A.hier([dict(row, v=c) for row, c in zip(rows, v)], lambda r: r['v']) for k, v in corr.items()}
    for j, k in enumerate(('probe_q', 'probe_bg', 'cheap_q', 'cheap_bg')):
        out[k] = boot(pp[k], stream0 + j)
    out['probe_q_minus_bg'] = boot({p: pp['probe_q'][p] - pp['probe_bg'][p] for p in pp['probe_q']}, stream0 + 4)
    out['cheap_q_minus_bg'] = boot({p: pp['cheap_q'][p] - pp['cheap_bg'][p] for p in pp['cheap_q']}, stream0 + 5)
    out['cheap_single'] = {k: round(float(np.mean(list(pp[k].values()))), 4) for k in pp if k.count('_') >= 2}
    return out


def main():
    items = [i for i in C.read_jsonl(os.path.join(C.OUT_DIR, 'b_items.jsonl.gz')) if i['role'] == ROLE]
    prose = {(int(r['task_id']), int(r['sample'])): r['text'] for r in C.read_jsonl(os.path.join(C.OUT_DIR, 'pool_samples.jsonl.gz'))}
    F = A.load(ROLE)
    rows = []
    for n, it in enumerate(items):
        r = A.item_tokens(it, F, prose)
        if r is not None:
            r['item_key'] = n
            rows.append(r)
    p1, p2 = select(rows, items, prose, F)
    sets = {'VA': p1, 'VPz': p2}
    out = {'role': ROLE, 'items_kept': len(rows)}
    for name, s in sets.items():
        out[f'{name}_rows'] = len(s)
        out[f'{name}_problems'] = len({row['problem'] for row in s})
    if STEP in ('acts', 'all'):
        if ROLE == 'primary':
            from p9_b_forward import prereg_guard
            out['guard'] = prereg_guard()
        out['acts'] = acts_step(sets, F)
    if STEP in ('fit', 'all'):
        out['VA'] = fit_one('VA', p1, 62)
        out['VPz'] = fit_one('VPz', p2, 72)
        with open(OUT, 'w') as fh:
            json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
