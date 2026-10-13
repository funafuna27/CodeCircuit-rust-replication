"""Phase 7, step 2: the reproduction gate, the blocks, P's basis and the
label-free nuisances. PREREGISTRATION_CLOSING.md §2-§3 (gate G1).

  G1     before any new number: Phase 3's forest on Phase 5's centred G, E, GE and
         S2+ and on Phase 4's G, E, GE must return the published A exactly
         (results/matched, results/mutant), and the seeded copy of the forest
         must equal w3.oof bit for bit; exit 2 otherwise. For C+ (sensitivity
         only): c, beta_5, its CR2 interval and S_U must reproduce
         results/ownfix/reference.json; if not, C+ is not used.
  P      the 45 leading right singular vectors of the pairs' differences of
         position-mean activations (canonical orientation; a pair's sign does
         not change them), each member's centred vector projected on them
  blocks member matrices in canonical order, no label: blocks_{matched,mutant}.npz
  m-hat  E[dB_j | Z] for every (conditioning set, tested block) of §3, version 1
         (forest) and version 2's linear part, and the shadow block of G5b:
         nuisance_{matched,mutant}.npz

  WORKERS=16 (N_JOBS=1 inside them)  OUT_DIR=results/closing  WORK=~/closing_work
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7_common as pc  # noqa: E402
import w3_within_analysis as w3  # noqa: E402

WORKERS = int(os.environ.get('WORKERS', 16))
MHAT_COLS = int(os.environ.get('MHAT_COLS', 0))      # tests only: fit the first k columns per block
TAG = {5: 'matched', 4: 'mutant'}
PUBLISHED = {5: {'G': ('results/matched/beyond_edit.json', ('A', 'G')),
                 'E': ('results/matched/beyond_edit.json', ('A', 'E')),
                 'GE': ('results/matched/beyond_edit.json', ('A', 'GE')),
                 'S2P': ('results/matched/analysis.json', ('S2pc', 'A'))},
             4: {'G': ('results/mutant/beyond_edit_posthoc.json', ('A', 'G')),
                 'E': ('results/mutant/beyond_edit_posthoc.json', ('A', 'E')),
                 'GE': ('results/mutant/beyond_edit_posthoc.json', ('A', 'GE'))}}
K_SVD = 45
L_NAMES = ('b1', 'b2', 'b3', 'b4', 'a1', 'a2', 'a3', 'a4')          # attribution first
P5_MEMBER = ('code_logprob', 'edit_logprob', 'min_logprob', 'mean_entropy', 'tokens_to_end')
# (conditioning set, tested blocks) whose nuisances are fitted; §3
REGISTRY = {5: {'C': ('G43', 'L', 'z', 'N45'), 'CF': ('GATTR', 'z'), 'PP': ('GATTR',),
                'C63': ('G45', 'L', 'z'), 'CPL': ('G43', 'L', 'z'), 'CL': ('G43',), 'CG': ('L',)},
            4: {'C': ('G43', 'L', 'z')}}


def gate_g1(phase, P, B):
    o = pc.orientation(P)
    perm = pc.member_perm(o)
    want = [i for p in P['pairs'] for i in (p['pass_id'], p['fail_id'])]
    if P['ids'][perm].tolist() != want:
        raise SystemExit("survivor-first order differs from the pair file's")
    y = np.tile([1, 0], P['n'])
    g = P['groups'][perm]
    sets = {'G': B['G49'], 'E': B['E'], 'GE': np.column_stack([B['G49'], B['E']])}
    if phase == 5:
        sets['S2P'] = B['S2P']
    out = {}
    for name, M in sets.items():
        Xc = pc.centred(M[perm])
        a = float(pc.correct(w3.oof(Xc, y, g, P['problems'])).mean())
        f, keys = PUBLISHED[phase][name]
        pub = json.load(open(pc.path(f), encoding='utf-8'))
        for k in keys:
            pub = pub[k]
        out[name] = {'A': a, 'published': pub, 'equal': a == pub}
        print(f"  G1 Phase {phase} {name}: A {a:.6f}, published {pub:.6f}, equal {a == pub}", flush=True)
    if phase == 5:
        Xc = pc.centred(B['G49'][perm])
        out['self_test'] = pc.self_test(Xc, y, g, P['problems'])
        print(f"  G1 seeded forest equals w3.oof: {out['self_test']}", flush=True)
    ok = all(v['equal'] for k, v in out.items() if k != 'self_test') and out.get('self_test', True)
    return ok, out


def gate_cplus(P, B, tte, ctx):
    """C+ reproduces results/ownfix/reference.json (labels: survivor first)."""
    import p6_stats as st
    from p6_stats import CONTEXTS
    ref = json.load(open(pc.path('results/ownfix/reference.json'), encoding='utf-8'))
    o = pc.orientation(P)
    perm = pc.member_perm(o)
    rows = []
    for m, i in enumerate(P['ids'][perm]):
        r = dict(P['rows'][int(i)])
        j = perm[m]
        r['surprise'] = dict(r['surprise'], tokens_to_end=float(tte[j, 0]))
        r['context'] = CONTEXTS[int(np.argmax(ctx[j]))]
        rows.append(r)
    rs, ks = rows[0::2], rows[1::2]
    f1 = B['G49'][perm][:, pc.NODES]
    raw = np.log1p(f1[0::2]) - np.log1p(f1[1::2])
    c = float(raw.std(ddof=1))
    z = st.graph_z(f1[0::2], f1[1::2], c)
    Bm, pen, names, dropped = st.baseline(rs, ks, member=P5_MEMBER)
    beta, ci = st.cr2_interval(z, Bm, pen, P['pprob'])
    S_U, _ = st.score_test(z, Bm, pen, P['pprob'], pc.rng(16), n_boot=1)
    out = {'c': c, 'beta_5': beta, 'beta_5_ci': list(ci), 'S_U': S_U, 'columns': len(names)}
    ok = (c == ref['c'] and abs(beta - ref['beta_5']) < 1e-9 and abs(S_U - ref['beta_5_S_U']) < 1e-6
          and all(abs(a - b) < 1e-9 for a, b in zip(ci, ref['beta_5_ci'])) and names == ref['baseline_columns'])
    print(f"  C+ gate: c {c:.6g} beta_5 {beta:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] S_U {S_U:+.3f}; "
          f"{len(names)} columns; reproduces reference.json: {ok}", flush=True)
    return ok, out


def svd_basis(phase, P):
    tag = TAG[phase]
    f = np.load(os.path.join(pc.WORK, f'posmean_{tag}.npz'))
    if f['ids'].tolist() != P['ids'].tolist():
        raise SystemExit(f"posmean_{tag}: member order differs from load_pairs'")
    X = csr_matrix((f['data'], f['indices'], f['indptr']), shape=tuple(f['shape'])).astype(np.float64)
    dX = (X[0::2] - X[1::2]).tocsr()
    v0 = pc.rng(12 if phase == 5 else 13).standard_normal(min(dX.shape))
    U, s, Vt = svds(dX, k=K_SVD, v0=v0, solver='arpack')
    order = np.argsort(-s)
    s, Vt = s[order], Vt[order]
    sign = np.sign(Vt[np.arange(K_SVD), np.argmax(np.abs(Vt), axis=1)])
    Vt = Vt * sign[:, None]
    half = np.asarray(dX @ Vt.T) / 2                 # a centred member row is +-dX/2

    Pm = np.empty((2 * P['n'], K_SVD))
    Pm[0::2], Pm[1::2] = half, -half
    tot = float(dX.multiply(dX).sum())
    path = os.path.join(pc.WORK, f'svd_basis_{tag}.npy')
    np.save(path, Vt.astype(np.float32))
    return Pm, {'singular_values': s.tolist(), 'share_of_sum_of_squares': float((s ** 2).sum() / tot),
                'basis_sha256': pc.sha256(path), 'columns': int(X.shape[1])}


def local_blocks(phase, P):
    rows = pc.read_jsonl(os.path.join(pc.OUT, f'local_{TAG[phase]}.jsonl'))
    if [r['expr_id'] for r in rows] != P['ids'].tolist():
        raise SystemExit(f"local_{TAG[phase]}: member order differs from load_pairs'")
    L = np.array([[r[k] for k in L_NAMES] for r in rows], float)
    Pl = np.array([r['P_layer'] for r in rows], float)
    same_site = np.array([len(r['D']) == 1 for r in rows[0::2]])
    return L, Pl, same_site


def build(phase, P, tok):
    B = pc.base_blocks(P)
    L, Pl, same_site = local_blocks(phase, P)
    B.update({'L': L, 'Lattr': L[:, :4], 'Lact': L[:, 4:], 'Ln': pc.normal_scores(L), 'P_layer': Pl,
              'same_site': same_site})
    B['P'], svd = svd_basis(phase, P)
    if phase == 5:
        B['tte'], B['ctx'] = pc.cplus_columns(P, tok)
    return B, svd


def cond_set(B, name):
    return {'C': lambda: B['C'], 'CF': lambda: np.column_stack([B['C'], B['f0']]),
            'PP': lambda: np.column_stack([B['P'], B['f0'], B['O']]), 'C63': lambda: B['C63'],
            'CPL': lambda: np.column_stack([B['C'], B['tte'], B['ctx']]),
            'CL': lambda: np.column_stack([B['C'], B['Ln']]),
            'CG': lambda: np.column_stack([B['C'], B['G43n']])}[name]()


def tested(B, name):
    return {'G43': lambda: pc.dpair(B['G43n']), 'L': lambda: pc.dpair(B['Ln']), 'z': lambda: B['z'][:, None],
            'N45': lambda: pc.dpair(B['N45']), 'GATTR': lambda: pc.dpair(B['GATTRn']),
            'G45': lambda: pc.dpair(B['G45n']), 'G43sh': lambda: B['G43sh']}[name]()


def _col(args):
    w3.N_JOBS = 1
    key, j, dZ, lev, db, groups, problems = args
    rf, lin = pc.mhat_column(dZ, lev, db, groups, problems)
    return key, j, rf, lin


def fit_nuisances(P, B, pairs, pool):
    """m-hat for every (set, block) in pairs; columns with dB identically 0 are skipped."""
    tasks, meta = [], {}
    for s, b in pairs:
        Z = cond_set(B, s)
        dZ, lev = pc.dpair(Z), pc.live_levels(Z)
        dB = tested(B, b)
        live = np.any(dB != 0, axis=0)
        meta[(s, b)] = (dB, live)
        cols = np.flatnonzero(live)
        if MHAT_COLS:                                  # tests only
            cols = cols[:MHAT_COLS]
        for j in cols:
            tasks.append(((s, b), int(j), dZ, lev, dB[:, j], P['groups'], P['problems']))
    out = {k: (np.zeros_like(m[0]), np.zeros_like(m[0])) for k, m in meta.items()}
    for key, j, rf, lin in pool.imap_unordered(_col, tasks, chunksize=1):
        out[key][0][:, j], out[key][1][:, j] = rf, lin
    return out, meta


def shadow(P, B, rf):
    """G5b's shadow block: m-hat(C) of G43 plus another problem's residual,
    randomly signed per pair. Y is independent of it given C by construction,
    and it depends on C as G does."""
    dB = pc.dpair(B['G43n'])
    res = dB - rf
    n, pp = P['n'], P['pprob']
    g = pc.rng(11, 1)
    perm = g.permutation(n)
    for _ in range(1000):
        clash = np.flatnonzero(pp[perm] == pp)
        if not len(clash):
            break
        for i in clash:
            k = g.integers(n)
            perm[i], perm[k] = perm[k], perm[i]
    else:
        raise SystemExit("no cross-problem permutation found")
    sign = pc.rng(11, 2).choice([-1.0, 1.0], size=n)
    return rf + sign[:, None] * res[perm]


def main():
    t0 = time.time()
    os.makedirs(pc.OUT, exist_ok=True)
    if not MHAT_COLS:
        ex = json.load(open(os.path.join(pc.OUT, 'extract.json'), encoding='utf-8'))
        if ex.get('gate_G2') != 'passed' or ex.get('mode') != 'full' or ex.get('limit'):
            raise SystemExit("extract.json: G2 has not passed on the full extraction")
    tok = pc.tokenizer()
    gates = {}
    built = {}
    for phase in (5, 4):
        P = pc.load_pairs(phase)
        B, svd = build(phase, P, tok)
        ok, g = gate_g1(phase, P, B)
        gates[f'G1_phase{phase}'] = g
        if not ok:
            pc.write_json(os.path.join(pc.OUT, 'gates.json'), {**gates, 'G1': 'FAILED'})
            print("G1 FAILED", flush=True)
            sys.exit(2)
        built[phase] = (P, B, svd)
    P5, B5, _ = built[5]
    cplus_ok, cp = gate_cplus(P5, B5, B5['tte'], B5['ctx'])
    gates.update({'G1': 'passed', 'cplus': cp, 'cplus_ok': cplus_ok})
    pc.write_json(os.path.join(pc.OUT, 'gates.json'), gates)
    print(f"G1 passed; C+ {'usable' if cplus_ok else 'NOT usable'}; {time.time() - t0:.0f} s", flush=True)

    manifest = {'gates_sha256': pc.sha256(os.path.join(pc.OUT, 'gates.json')), 'mhat_cols_limit': MHAT_COLS}
    with Pool(WORKERS) as pool:
        for phase in (5, 4):
            P, B, svd = built[phase]
            reg = REGISTRY[phase]
            pairs = [(s, b) for s, bs in reg.items() for b in bs if cplus_ok or s != 'CPL']
            nu, meta = fit_nuisances(P, B, pairs, pool)
            if phase == 5:
                B['G43sh'] = shadow(P, B, nu[('C', 'G43')][0])
                nu2, meta2 = fit_nuisances(P, B, [('C', 'G43sh')], pool)
                nu.update(nu2)
                meta.update(meta2)
            tag = TAG[phase]
            arrays = {k: v for k, v in B.items() if isinstance(v, np.ndarray)}
            arrays.update({'ids': P['ids'], 'groups': P['groups'], 'pprob': P['pprob'], 'problems': P['problems']})
            bpath = os.path.join(pc.OUT, f'blocks_{tag}.npz')
            np.savez(bpath + '.tmp.npz', **arrays)
            os.replace(bpath + '.tmp.npz', bpath)
            nd = {}
            for (s, b), (rf, lin) in nu.items():
                nd[f'{s}|{b}|rf'], nd[f'{s}|{b}|lin'] = rf, lin
                nd[f'{s}|{b}|db'], nd[f'{s}|{b}|live'] = meta[(s, b)]
            npath = os.path.join(pc.OUT, f'nuisance_{tag}.npz')
            np.savez(npath + '.tmp.npz', **nd)
            os.replace(npath + '.tmp.npz', npath)
            manifest[tag] = {'svd': svd, 'blocks_sha256': pc.sha256(bpath), 'nuisance_sha256': pc.sha256(npath),
                             'local_sha256': pc.sha256(os.path.join(pc.OUT, f'local_{tag}.jsonl')),
                             'posmean_sha256': pc.sha256(os.path.join(pc.WORK, f'posmean_{tag}.npz')),
                             'pairs_fitted': [f'{s}|{b}' for s, b in nu]}
            print(f"  Phase {phase}: blocks and {len(nu)} nuisance sets saved; P's 45 directions hold "
                  f"{svd['share_of_sum_of_squares']:.3f} of the differences' sum of squares; "
                  f"{time.time() - t0:.0f} s", flush=True)
    pc.write_json(os.path.join(pc.OUT, 'prepare.json'), manifest)
    print(f"done; {time.time() - t0:.0f} s", flush=True)


if __name__ == '__main__':
    main()
