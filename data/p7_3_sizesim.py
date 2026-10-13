"""Phase 7, step 3: size, null bands and power, on synthetic labels only.
PREREGISTRATION_CLOSING.md §6 (gate G3). Never reads which member passed: it
loads p7_2's blocks and nuisances (canonical orientation, no label) and draws
its own labels.

A synthetic dataset: the canonical first member passes with probability
sigmoid(k * eta), eta an odd function of the pair's differences (a pair's swap
flips its sign) and of its levels; k sets the oracle accuracy mean
sigmoid(|k eta|). Pairs are then put passer first, p-hat is refitted, and every
test on that conditioning set is run with the real m-hat (it uses no label).

  sets     C (H1 G43, H2 L, S1 z, controls N45 and the shadow G43sh, and the
           primary family's weighted Holm), PP = P + f0 + O (S2 GATTR),
           CF = C + f0 (S3 GATTR)
  designs  L  linear in the RMS-scaled differences, each block an equal share
           Q  differences x standardised levels, d_j |d_k| for 10 random pairs,
              u|u| along a random direction, and (C, CF) a random effect of each
              member's (replaced, new) token pair, differenced
           W  adversarial, one per tested block: the first principal direction
              of the tested block's differences (uncentred), predicted from the
              5 columns of dZ most correlated with it by OLS without intercept
              (fitted once on all pairs), m -> m + m^3 / 2
  cells    C: L, Q, W_G43, W_L at oracle 0.68 and 0.80; PP and CF: L, Q, W at 0.72
  gate G3  a test fails on a set if a cell rejects at 0.05 in more than 9% of its
           datasets, or the pooled rejections exceed the binomial 99.5% point at
           size 0.05; S1 also needs each cell's mean T within +-0.15 and sd in
           [0.85, 1.20]. A set with a failing test is rerun with version 2; a
           test failing version 2 too is withdrawn (T only, never rejected).
  bands    200 datasets of design L per set: the 2.5/97.5% points and mean (c0)
           of each increment of §3
  power    z at beta_5's coefficient (and half of it) for H1 and S1; the first
           direction of L's residuals at +0.04 oracle accuracy for H2

  WORKERS=18 N_CELL=400 N_BAND=200 N_POWER=200 B_SIGNS=9999  MODE=selftest
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import stats
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7_common as pc  # noqa: E402
import w3_within_analysis as w3  # noqa: E402

WORKERS = int(os.environ.get('WORKERS', 18))
N_CELL = int(os.environ.get('N_CELL', 400))
N_BAND = int(os.environ.get('N_BAND', 200))
N_POWER = int(os.environ.get('N_POWER', 200))
B_SIGNS = int(os.environ.get('B_SIGNS', 9999))
MODE = os.environ.get('MODE', 'run')
DEFAULTS = (400, 200, 200, 9999)
SET_ID = {'C': 1, 'PP': 2, 'CF': 3}
DES_ID = {'L': 1, 'Q': 2, 'WG': 3, 'WL': 4, 'W': 5}
TESTS = {'C': {'H1': 'G43', 'H2': 'L', 'S1': 'z', 'N45': 'N45', 'SH': 'G43sh'},
         'PP': {'S2': 'GATTR'}, 'CF': {'S3': 'GATTR'}}
CELLS = [('C', d, o) for d in ('L', 'Q', 'WG', 'WL') for o in (0.68, 0.80)] + \
        [(s, d, 0.72) for s in ('PP', 'CF') for d in ('L', 'Q', 'W')]
W_TARGET = {('C', 'WG'): 'G43', ('C', 'WL'): 'L', ('PP', 'W'): 'GATTR', ('CF', 'W'): 'GATTR'}
BAND_FORESTS = {'C': ['C', 'G+C', 'L+C', 'N45+C', 'L+G+C', 'Lact+C', 'P+f0+C', 'G+P+f0+C'],
                'PP': ['PP', 'G+PP'], 'CF': ['G+C', 'Gfree+C']}
INCREMENTS = {'C': {'G|C': ('G+C', 'C'), 'L|C': ('L+C', 'C'), 'N45|C': ('N45+C', 'C'),
                    'L|G+C': ('L+G+C', 'G+C'), 'Lattr|Lact+C': ('L+C', 'Lact+C'),
                    'G|P+f0+C': ('G+P+f0+C', 'P+f0+C')},
              'PP': {'G|PP': ('G+PP', 'PP')}, 'CF': {'Gattr|Gfree+C': ('G+C', 'Gfree+C')}}
STRENGTH = {'C': 0.68, 'PP': 0.72, 'CF': 0.72}
BETA5 = -0.461

S = {}                                      # per-worker state


def load():
    b = np.load(os.path.join(pc.OUT, 'blocks_matched.npz'))
    nz = np.load(os.path.join(pc.OUT, 'nuisance_matched.npz'))
    B = {k: b[k] for k in b.files}
    sets = {'C': B['C'], 'PP': np.column_stack([B['P'], B['f0'], B['O']]),
            'CF': np.column_stack([B['C'], B['f0']])}
    nu = {k: nz[k] for k in nz.files}
    return B, sets, nu


def forest_block(B, name):
    G, C = B['G49'], B['C']
    PP = np.column_stack([B['P'], B['f0'], B['O']])
    return {'C': C, 'G+C': np.column_stack([G, C]), 'L+C': np.column_stack([B['L'], C]),
            'N45+C': np.column_stack([B['N45'], C]), 'L+G+C': np.column_stack([B['L'], G, C]),
            'Lact+C': np.column_stack([B['Lact'], C]), 'P+f0+C': np.column_stack([B['P'], B['f0'], C]),
            'G+P+f0+C': np.column_stack([G, B['P'], B['f0'], C]), 'PP': PP,
            'G+PP': np.column_stack([G, PP]), 'Gfree+C': np.column_stack([B['GFREE'], C])}[name]


def rms(x):
    r = np.sqrt(np.mean(np.asarray(x) ** 2, axis=0))
    return np.where(r > 0, r, 1.0)


def unit(x):
    return x / rms(x)


def design_parts(B, Z, sname):
    """Label-free ingredients of the designs for set sname (canonical)."""
    dZ = pc.dpair(Z)
    n_e = 58 if sname in ('C', 'CF') else 0
    n_p = 45 if sname == 'PP' else 0
    nonzero = (dZ != 0).sum(axis=0)
    live = nonzero >= 10
    X = dZ / rms(dZ)
    cont = [j for j in range(Z.shape[1]) if j >= n_e + n_p and live[j]]
    head = 58 if sname in ('C', 'CF') else 45            # E's one-hots, or P's directions
    blocks = [[j for j in range(head) if live[j]], cont]
    lev = pc.mpair(Z)
    levs = (lev - lev.mean(axis=0)) / np.where(lev.std(axis=0) > 0, lev.std(axis=0), 1.0)
    E = B['E']
    tp = None
    if sname in ('C', 'CF'):
        frm, to = E[:, :25].argmax(axis=1), E[:, 25:50].argmax(axis=1)
        tp = frm * 25 + to                       # each member's (replaced, new) token pair
    return {'X': X, 'live': np.flatnonzero(live), 'blocks': blocks, 'cont': cont, 'levs': levs, 'tp': tp}


def eta_design(parts, design, g, sname, target=None, B=None, nu=None):
    X, live = parts['X'], parts['live']
    if design == 'L':
        u = [unit(X[:, b] @ g.standard_normal(len(b))) for b in parts['blocks'] if len(b)]
        return unit(np.sum(u, axis=0))
    if design == 'Q':
        cont, levs = parts['cont'], parts['levs']
        terms = [unit(np.sum([g.standard_normal() * X[:, j] * levs[:, j] for j in cont], axis=0))]
        pick = g.choice(live, size=(10, 2))
        terms.append(unit(np.sum([g.standard_normal() * X[:, j] * np.abs(X[:, k]) for j, k in pick], axis=0)))
        v = g.standard_normal(len(live))
        u = X[:, live] @ (v / np.linalg.norm(v))
        terms.append(unit(u * np.abs(u)))
        if parts['tp'] is not None:
            b = g.standard_normal(25 * 25)
            terms.append(unit(b[parts['tp'][0::2]] - b[parts['tp'][1::2]]))
        return unit(np.sum(terms, axis=0))
    # W: deterministic given the set and its target block
    dB = nu[f"{sname}|{target}|db"]
    Xb = dB / rms(dB)
    t = Xb @ first_direction(Xb)
    corr = np.abs(X[:, live].T @ t) / (np.linalg.norm(X[:, live], axis=0) * np.linalg.norm(t))
    top = live[np.argsort(-corr, kind='stable')[:5]]
    beta = np.linalg.lstsq(X[:, top], t, rcond=None)[0]
    m = unit(X[:, top] @ beta)
    return unit(m + 0.5 * m ** 3)


def first_direction(Xb):
    """Xb's first right singular vector (uncentred), its largest entry positive."""
    v = np.linalg.svd(Xb, full_matrices=False)[2][0]
    return v * np.sign(v[np.argmax(np.abs(v))])


def solve_k(eta, target):
    lo, hi = 0.0, 100.0
    if np.mean(expit(hi * np.abs(eta))) < target:
        return hi
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.mean(expit(mid * np.abs(eta))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mh(nu, sname, block, version):
    rf, lin = nu[f"{sname}|{block}|rf"], nu[f"{sname}|{block}|lin"]
    return rf if version == 1 else (rf + lin) / 2


def _init():
    w3.N_JOBS = 1
    B, sets, nu = load()
    S.update(B=B, sets=sets, nu=nu, parts={s: design_parts(B, sets[s], s) for s in sets},
             levels={s: pc.live_levels(sets[s]) for s in sets},
             pidx=np.searchsorted(B['problems'], B['pprob']))


def labels(eta, k, g):
    yp = g.random(len(eta)) < expit(k * eta)
    return np.where(yp, 1, -1)


def run_tests(sname, o, version, g_eps, tests=None):
    B, nu = S['B'], S['nu']
    perm = pc.member_perm(o)
    Z = S['sets'][sname]
    ph, _, _ = pc.phat(pc.centred(Z[perm]), S['levels'][sname], B['groups'][perm], B['problems'],
                       version=version, n_jobs=1)
    eps = g_eps.choice([-1.0, 1.0], size=(B_SIGNS, len(B['problems'])))
    out = {}
    for name, block in (tests or TESTS[sname]).items():
        dB = o[:, None] * nu[f"{sname}|{block}|db"]
        T, Sx, p = pc.gcm_fast(ph, dB, o[:, None] * mh(nu, sname, block, version), S['pidx'],
                               len(B['problems']), eps)
        out[name] = {'p': p, 'T': float(T[0]) if len(T) == 1 else None, 'S': Sx}
    return out


def _size(task):
    sname, design, strength, r, version, stream = task
    t0 = time.time()
    si, di, oi = SET_ID[sname], DES_ID[design], int(round(strength * 100))
    g_design = pc.rng(4, si, di, r)
    eta = eta_design(S['parts'][sname], design, g_design, sname, W_TARGET.get((sname, design)), S['B'], S['nu'])
    k = solve_k(eta, strength)
    o = labels(eta, k, pc.rng(stream, si, di, oi, r, version))
    res = run_tests(sname, o, version, pc.rng(stream, si, di, oi, r, version, 1))
    return {'cell': (sname, design, strength), 'r': r, 'version': version, 'k': k, 'res': res,
            'seconds': time.time() - t0}


def _band(task):
    sname, r = task
    si = SET_ID[sname]
    eta = eta_design(S['parts'][sname], 'L', pc.rng(7, si, r, 0), sname)
    o = labels(eta, solve_k(eta, STRENGTH[sname]), pc.rng(7, si, r, 1))
    B = S['B']
    perm = pc.member_perm(o)
    y = np.tile([1, 0], len(o))
    A = {}
    for f in BAND_FORESTS[sname]:
        s = pc.oof_seeded(pc.centred(forest_block(B, f)[perm]), y, B['groups'][perm], B['problems'], n_jobs=1)
        A[f] = float(pc.correct(s).mean())
    return {'set': sname, 'r': r, 'inc': {k: A[a] - A[b] for k, (a, b) in INCREMENTS[sname].items()}}


def _power(task):
    kind, coef, r, version = task
    B, nu, parts = S['B'], S['nu'], S['parts']['C']
    eta_l = eta_design(parts, 'L', pc.rng(6, 1, r, 0), 'C')
    k_l = solve_k(eta_l, 0.68)
    if kind == 'z':
        lin = k_l * eta_l + coef * B['z']
    else:
        res = nu['C|L|db'] - mh(nu, 'C', 'L', version)
        Xr = res / rms(res)
        t = unit(Xr @ first_direction(Xr))
        base = np.mean(expit(np.abs(k_l * eta_l)))
        lo, hi = 0.0, 20.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if np.mean(expit(np.abs(k_l * eta_l + mid * t))) < base + 0.04:
                lo = mid
            else:
                hi = mid
        lin = k_l * eta_l + (lo + hi) / 2 * t
    o = labels(lin, 1.0, pc.rng(6, 2 if kind == 'z' else 3, int(round(abs(coef) * 1000)), r))
    res = run_tests('C', o, version, pc.rng(6, 4, int(round(abs(coef) * 1000)), r))
    return {'kind': kind, 'coef': coef, 'r': r, 'res': res}


def cell_summary(rows, tests):
    out = {}
    for name in tests:
        ps = np.array([x['res'][name]['p'] for x in rows])
        d = {'n': len(ps), 'rej05': int((ps <= 0.05).sum()), 'rej01': int((ps <= 0.01).sum())}
        if name == 'H1':
            d['rej_at_0.035'] = int((ps <= 0.035).sum())
        if name == 'H2':
            d['rej_at_0.015'] = int((ps <= 0.015).sum())
        if name in ('S1', 'S2', 'S3'):
            d['rej_at_0.0167'] = int((ps <= 0.05 / 3).sum())
        if name == 'S1':
            T = np.array([x['res'][name]['T'] for x in rows])
            d.update({'T_mean': float(T.mean()), 'T_sd': float(T.std(ddof=1))})
        out[name] = d
    if 'H1' in tests:
        fw = [any(pc.weighted_holm2(x['res']['H1']['p'], x['res']['H2']['p'])) for x in rows]
        out['FWER'] = {'n': len(fw), 'rej05': int(sum(fw))}
    return out


def gate(sname, cells):
    """G3 for one set: {test: passed?} with the reasons."""
    names = list(next(iter(cells.values())).keys())
    verdict = {}
    for name in names:
        bad = []
        tot = sum(c[name]['n'] for c in cells.values())
        rej = sum(c[name]['rej05'] for c in cells.values())
        lim = int(stats.binom.ppf(0.995, tot, 0.05))
        if rej > lim:
            bad.append(f'pooled {rej}/{tot} > {lim}')
        for cell, c in cells.items():
            if c[name]['rej05'] > int(0.09 * c[name]['n']):
                bad.append(f'{cell}: {c[name]["rej05"]}/{c[name]["n"]}')
            if name == 'S1' and not (abs(c[name]['T_mean']) <= 0.15 and 0.85 <= c[name]['T_sd'] <= 1.20):
                bad.append(f'{cell}: T mean {c[name]["T_mean"]:+.3f} sd {c[name]["T_sd"]:.3f}')
        verdict[name] = {'passed': not bad, 'why': bad}
    return verdict


def selftest():
    """Every design is odd: swapping a random half of the pairs flips eta there."""
    B, sets, nu = load()
    g = np.random.default_rng(1)
    worst = 0.0
    for sname, Z in sets.items():
        n = len(Z) // 2
        flip = np.where(g.random(n) < 0.5, -1, 1)
        perm = pc.member_perm(flip)
        Bf = dict(B)
        Bf['E'] = B['E'][perm]
        nuf = dict(nu)
        for key in nu:
            if key.endswith('|db') or key.endswith('|rf') or key.endswith('|lin'):
                nuf[key] = nu[key] * flip[:, None]
        p0, p1 = design_parts(B, Z, sname), design_parts(Bf, Z[perm], sname)
        for d in ('L', 'Q') + (('WG', 'WL') if sname == 'C' else ('W',)):
            e0 = eta_design(p0, d, pc.rng(4, 99, DES_ID[d], 0), sname, W_TARGET.get((sname, d)), B, nu)
            e1 = eta_design(p1, d, pc.rng(4, 99, DES_ID[d], 0), sname, W_TARGET.get((sname, d)), Bf, nuf)
            worst = max(worst, float(np.max(np.abs(e1 - flip * e0))))
    print(f"selftest: designs odd under pair swaps, max |eta(flip) - flip * eta| = {worst:.2e}")
    if worst > 1e-9:
        sys.exit(1)


def main():
    if MODE == 'selftest':
        selftest()
        return
    t0 = time.time()
    prep = json.load(open(os.path.join(pc.OUT, 'prepare.json'), encoding='utf-8'))
    test_run = (N_CELL, N_BAND, N_POWER, B_SIGNS) != DEFAULTS
    if prep.get('mhat_cols_limit') and not test_run:
        raise SystemExit("prepare.json comes from a test run (mhat_cols_limit); refusing a real size run")
    log = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'params': {'N_CELL': N_CELL, 'N_BAND': N_BAND,
           'N_POWER': N_POWER, 'B_SIGNS': B_SIGNS, 'test_run': test_run},
           'inputs': {k: pc.sha256(os.path.join(pc.OUT, k)) for k in
                      ('blocks_matched.npz', 'nuisance_matched.npz', 'prepare.json', 'local_matched.jsonl')},
           'script_sha256': pc.sha256(os.path.abspath(__file__)),
           'common_sha256': pc.sha256(os.path.join(HERE, 'p7_common.py'))}
    n_cell = N_CELL
    _init()                     # fail here, not in a Pool that would respawn failing workers forever
    with Pool(WORKERS, initializer=_init) as pool:
        # time the first 20 datasets; the fallback (300 per cell) is fixed in advance
        first = pool.map(_size, [(c[0], c[1], c[2], r, 1, 3) for c in CELLS[:1] for r in range(min(20, n_cell))])
        per = np.median([x['seconds'] for x in first])
        n_total = len(CELLS) * n_cell + 12 * N_BAND + 3 * N_POWER       # forest fits
        proj = per * n_total / 12                                       # 12 effective workers
        log['projection_hours'] = proj / 3600
        log['seconds_per_dataset_first20'] = float(per)
        if proj > 6 * 3600 and not test_run:
            n_cell = 300
        log['n_cell'] = n_cell
        print(f"first 20 datasets: median {per:.1f} s each; projected {proj / 3600:.2f} h; {n_cell} per cell",
              flush=True)
        rows = {c: [] for c in CELLS}
        for x in first:
            if x['r'] < n_cell:
                rows[tuple(x['cell'])].append(x)
        todo = [(c[0], c[1], c[2], r, 1, 3) for c in CELLS for r in range(n_cell)
                if not (c == CELLS[0] and r < len(first))]
        for x in pool.imap_unordered(_size, todo, chunksize=1):
            rows[tuple(x['cell'])].append(x)
        print(f"version 1 size runs done: {time.time() - t0:.0f} s", flush=True)
        versions, cells_out, withdrawn = {}, {}, {}
        for sname in TESTS:
            cs = {f'{d}@{o}': cell_summary(rows[(s, d, o)], TESTS[sname]) for s, d, o in CELLS if s == sname}
            g1 = gate(sname, cs)
            cells_out[sname] = {'v1': cs, 'gate_v1': g1}
            versions[sname] = 1
            if not all(v['passed'] for v in g1.values()):
                versions[sname] = 2
                rows2 = {c: [] for c in CELLS if c[0] == sname}
                for x in pool.imap_unordered(_size, [(c[0], c[1], c[2], r, 2, 3) for c in rows2
                                                     for r in range(n_cell)], chunksize=1):
                    rows2[tuple(x['cell'])].append(x)
                cs2 = {f'{d}@{o}': cell_summary(rows2[(s, d, o)], TESTS[sname]) for s, d, o in CELLS if s == sname}
                g2 = gate(sname, cs2)
                cells_out[sname].update({'v2': cs2, 'gate_v2': g2})
                withdrawn[sname] = [t for t, v in g2.items() if not v['passed']]
                fresh = {c: [] for c in rows2}
                for x in pool.imap_unordered(_size, [(c[0], c[1], c[2], r, 2, 15) for c in rows2
                                                     for r in range(min(200, n_cell))], chunksize=1):
                    fresh[tuple(x['cell'])].append(x)
                cells_out[sname]['v2_fresh'] = {f'{d}@{o}': cell_summary(fresh[(s, d, o)], TESTS[sname])
                                                for s, d, o in CELLS if s == sname}
            else:
                withdrawn[sname] = []
            print(f"  set {sname}: version {versions[sname]}; withdrawn {withdrawn[sname]}", flush=True)
        # null bands and power
        bands = {s: [] for s in BAND_FORESTS}
        for x in pool.imap_unordered(_band, [(s, r) for s in BAND_FORESTS for r in range(N_BAND)], chunksize=1):
            bands[x['set']].append(x['inc'])
        band_out = {}
        for s, xs in bands.items():
            for k in INCREMENTS[s]:
                v = np.array([x[k] for x in xs])
                band_out[k] = {'set': s, 'n': len(v), 'c0': float(v.mean()), 'sd': float(v.std(ddof=1)),
                               'lo': float(np.percentile(v, 2.5)), 'hi': float(np.percentile(v, 97.5))}
        print(f"null bands done: {time.time() - t0:.0f} s", flush=True)
        ptasks = [('z', c, r, versions['C']) for c in (BETA5, BETA5 / 2) for r in range(N_POWER)] + \
                 [('L', 0.04, r, versions['C']) for r in range(N_POWER)]
        prow = {}
        for x in pool.imap_unordered(_power, ptasks, chunksize=1):
            prow.setdefault(f"{x['kind']}@{x['coef']}", []).append(x)
    power = {}
    for key, xs in prow.items():
        ps = {t: np.array([x['res'][t]['p'] for x in xs]) for t in ('H1', 'H2', 'S1')}
        power[key] = {'n': len(xs), 'H1@0.035': float(np.mean(ps['H1'] <= 0.035)), 'H1@0.05': float(np.mean(ps['H1'] <= 0.05)),
                      'H2@0.015': float(np.mean(ps['H2'] <= 0.015)), 'H2@0.05': float(np.mean(ps['H2'] <= 0.05)),
                      'S1@0.0167': float(np.mean(ps['S1'] <= 0.05 / 3)), 'S1@0.05': float(np.mean(ps['S1'] <= 0.05))}
    log.update({'versions': versions, 'withdrawn': withdrawn, 'cells': cells_out, 'bands': band_out,
                'power': power, 'seconds': time.time() - t0, 'finished': time.strftime('%Y-%m-%d %H:%M:%S')})
    path = os.path.join(pc.OUT, 'sizesim.json')
    pc.write_json(path, log)
    print(json.dumps({'versions': versions, 'withdrawn': withdrawn,
                      'bands': {k: [round(v['c0'], 4), round(v['lo'], 4), round(v['hi'], 4)] for k, v in band_out.items()},
                      'power': power}, indent=1))
    print(f"sizesim sha256: {pc.sha256(path)}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == '__main__':
    main()
