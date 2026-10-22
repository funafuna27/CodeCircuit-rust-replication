"""Phase 9, part ③ (M1): the analysis fixed by PREREGISTRATION_P9_M1.md §3, §6 (run once).

Builds C (9 columns) and the probes P and P_code (45 centred SVD directions per population) from the
extractions, then: levels; the primary Δ(G | C) with synthetic-label null bands (designs L and F,
200 draws each) and the reading table; S0-S3; the step-0 contrast for every signal (all rows, and the
one-row form, compared across signals only in that form); the step-0 Δ; seed pairs; the logistic learner;
MBPP+ labels; Rust loose; Python 240 (primary with a design-L band) and Python 256.

  WORKERS=16  N_JOBS=4  REHEARSAL=1 (stored G only: labels permuted across problems, C and P stand-ins
  built from G without labels, 4 null draws; writes to M1_OUT, never to results/p9_m1)
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, svds

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_m1_common as M  # noqa: E402

REHEARSAL = os.environ.get('REHEARSAL', '0') == '1'
WORKERS = int(os.environ.get('WORKERS', 16))
N_JOBS = int(os.environ.get('N_JOBS', 4))
S_PRIMARY = 4 if REHEARSAL else 200
S_SECONDARY = 4 if REHEARSAL else 100
K_SVD = 45
C_NAMES = ['top1', 'entropy', 'margin', 'code_logp_sum', 'code_logp_mean', 'code_logp_min',
           'code_entropy_mean', 'n_code', 'n_prose']
C_DIST = [0, 1, 2, 4, 6]          # descriptive: no counts, no sum
C_LEN = [7, 8]
SMALL = 'a3_v2_check.json'
ARTEFACTS = [593, 806, 790, 771, 559, 109, 633, 564, 119, 142, 753, 409, 274, 435]


def log(*a):
    print(*a, flush=True)


# ---------------- signals ----------------

def c_and_posmean(arm, rows, need_p, guard=None):
    """C rows (9 columns) and, if need_p, the posmean / posmean_code CSR matrices, aligned to `rows`.
    The extraction is read once (an NpzFile re-reads an array on every access)."""
    z, meta = M.load_extract(arm, guard)
    pos_of = {int(k): i for i, k in enumerate(z['keys'].tolist())}
    idx = [pos_of[r['row_index']] for r in rows]
    tptr, off_all, lp_all, en_all = z['tptr'], z['off'], z['logp'], z['ent']
    C = []
    for r, i in zip(rows, idx):
        a, b = tptr[i], tptr[i + 1]
        off, lp, en = off_all[a:b], lp_all[a:b], en_all[a:b]
        code = [t for t in range(1, b - a) if off[t][0] >= len(r['prose'])]
        if not code or len(code) != int(z['n_code'][i]):
            sys.exit(f"{arm} row {r['row_index']}: {len(code)} code tokens, extraction counted {z['n_code'][i]}")
        top1, top2, ent_f = z['final'][i]
        C.append([top1, ent_f, top1 - top2, lp[code].sum(), lp[code].mean(), lp[code].min(), en[code].mean(),
                  z['n_code'][i], z['n_prose'][i]])
    C = np.array(C, dtype=float)
    if not np.isfinite(C).all():
        sys.exit(f'{arm}: C has non-finite entries')
    if not need_p:
        return C, None, None, meta['sha256']

    def mat(field):
        full = csr_matrix((z[f'{field}_data'].astype(np.float64), z[f'{field}_indices'], z[f'{field}_indptr']),
                          shape=(len(z['keys']), int(z['shape'][1])))
        return full[idx]
    return C, mat('pm'), mat('pm_code'), meta['sha256']


def centred_svd(X, stream, key):
    """45 leading right singular vectors of X centred on its rows' mean; rows projected on them."""
    n = X.shape[0]
    mu = np.asarray(X.mean(axis=0)).ravel()
    A = LinearOperator(X.shape, dtype=np.float64,
                       matvec=lambda v: X @ v - mu @ v,
                       rmatvec=lambda u: X.T @ u - mu * u.sum(),
                       matmat=lambda V: X @ V - np.outer(np.ones(n), mu @ V),
                       rmatmat=lambda U: X.T @ U - np.outer(mu, U.sum(axis=0)))
    v0 = M.rng(stream, key).standard_normal(min(X.shape))
    _, s, Vt = svds(A, k=K_SVD, v0=v0, solver='arpack')
    o = np.argsort(-s)
    s, Vt = s[o], Vt[o]
    Vt = Vt * np.sign(Vt[np.arange(K_SVD), np.argmax(np.abs(Vt), axis=1)])[:, None]
    P = np.asarray(X @ Vt.T) - (mu @ Vt.T)[None, :]
    tot = float(X.multiply(X).sum() - n * (mu @ mu))
    return P, {'share_of_centred_ss': round(float((s ** 2).sum() / tot), 4), 'singular_values': s[:5].round(3).tolist()}


def stand_ins(P_):
    """REHEARSAL only: label-free C, P, P_code built from stored G and the texts."""
    G = P_['G']
    g = M.rng(99, 1)
    n_code = np.array([len(r['text']) - len(r['prose']) for r in P_['rows']], float)
    n_prose = np.array([len(r['prose']) for r in P_['rows']], float)
    C = np.column_stack([G[:, 3], G[:, 4], G[:, 3] - g.random(len(G)) * 0.1, -n_code / 3, g.standard_normal(len(G)),
                         g.standard_normal(len(G)), G[:, 4] * 0.5, n_code, n_prose])
    W = g.standard_normal((G.shape[1], K_SVD))
    Gs = (G - G.mean(0)) / np.where(G.std(0) > 0, G.std(0), 1)
    return C, Gs @ W, Gs @ (g.standard_normal((G.shape[1], K_SVD)))


# ---------------- population analysis ----------------

class Pop:
    def __init__(self, pop, data, signals, pool):
        self.pop, self.d, self.sig, self.pool = pop, data, signals, pool
        self.D = M.draws(pop, len(data['problems']))
        self.cache = {}

    def X(self, names, mask=None):
        X = np.column_stack([self.sig[n] for n in names])
        return X if mask is None else X[mask]

    def scores(self, names, mask=None, tag='all', y_row=None, problems=None, **kw):
        key = (tuple(names), tag, kw.get('perm_base', 100), kw.get('rf_seed', 42), kw.get('learner', 'rf'),
               None if y_row is None else hash(y_row.tobytes()))
        if key not in self.cache:
            d = self.d
            g = d['groups'] if mask is None else d['groups'][mask]
            y = (d['y_row'] if y_row is None else y_row) if mask is None else (d['y_row'] if y_row is None else y_row)[mask]
            pr = d['problems'] if problems is None else problems
            self.cache[key] = M.oof(self.X(names, mask), g, y, pr, n_jobs=N_JOBS, **kw)
        return self.cache[key]

    def level(self, names, **kw):
        return M.summarise_level(self.d['y_prob'], self.scores(names, **kw), self.D)

    def diff(self, a, b, y_prob=None, y_row=None, **kw):
        y = self.d['y_prob'] if y_prob is None else y_prob
        return M.summarise_diff(y, self.scores(a, y_row=y_row, **kw), self.scores(b, y_row=y_row, **kw), self.D)

    def band(self, full, base, designs, S, kid):
        d = self.d
        key = M.POP_KEY[self.pop] * 100 + kid
        ctx = M.band_context(self.X(base), d['groups'], d['problems'], d['y_prob'], self.scores(base), key)
        return [M.run_band(self.pool, des, S, ctx, self.X(full), self.X(base)) for des in designs]

    def increment(self, full, base, designs, S, kid):
        t0 = time.time()
        diff = self.diff(full, base)
        bands = self.band(full, base, designs, S, kid)
        x = '+'.join(n for n in full if n not in base)
        out = {'full': full, 'base': base, 'diff': diff, 'bands': bands,
               'reading': M.reading(diff, [b['c0'] for b in bands], x, '+'.join(base)),
               'seconds': round(time.time() - t0)}
        log(f"  {self.pop} {'+'.join(full)} vs {'+'.join(base)}: {diff['plugin']:+.4f} {diff['ci95']}; "
            f"c0 {[b['c0'] for b in bands]} -> {out['reading']['row']} [{out['reading']['lower_bracket']:+.4f}, "
            f"{out['reading']['upper_bracket']:+.4f}]")
        return out

    def step0(self, names):
        d = self.d
        m0 = d['step'] == 0
        p0 = np.array(sorted(set(d['groups'][m0].tolist())))
        assert np.array_equal(p0, d['problems']), 'a problem has no step-0 row'
        s0 = self.scores(names, mask=m0, tag='step0')
        return s0

    def last_row_mask(self):
        d = self.d
        last = {}
        for i, (g, s) in enumerate(zip(d['groups'], d['step'])):
            if g not in last or s > d['step'][last[g]]:
                last[g] = i
        m = np.zeros(len(d['groups']), bool)
        m[list(last.values())] = True
        return m


def mbpp_plus_labels(problems, y_prob, exclude):
    """PREREGISTRATION_P9_MUTSTATE.md §7 and P9_M1 §6: a passing greedy program that flips under MBPP+ -> 0."""
    import gzip
    flips = set()
    with gzip.open(os.path.join(HERE, 'results', 'p9_state', 'a3_relabel_v2.jsonl.gz'), 'rt') as fh:
        for l in fh:
            r = json.loads(l)
            if any(ph == '2' for _, ph in r['members']) and r['O']['stage'] == 'ok' and \
                    r['O']['n_fail'] + r['O']['n_panic'] > 0 and r['task_id'] not in exclude:
                flips.add(int(r['task_id']))
    y = np.array([0 if (int(p) in flips and yy == 1) else yy for p, yy in zip(problems, y_prob)])
    return y, sorted(int(p) for p in problems if int(p) in flips)


def analyse_rust_strict(A, have_p):
    out = {'levels': {}}
    G, C, Gf = ['G'], ['C'], ['G_free']
    sets = {'G': G, 'C': C, 'G_free': Gf, 'G+C': G + C, 'G_free+C': Gf + C}
    if have_p:
        sets.update({'P': ['P'], 'P_code': ['P_code'], 'C+P': ['P'] + C, 'G+C+P': G + C + ['P'],
                     'G_free+C+P': Gf + C + ['P'], 'P+G_free': ['P'] + Gf, 'P_code+C': ['P_code'] + C})
    for k, v in sets.items():
        out['levels'][k] = A.level(v)
        log(f"  level {k}: {out['levels'][k]}")
    out['primary'] = A.increment(G + C, C, ('L', 'F'), S_PRIMARY, 0)
    out['S0_attribution_beyond_C_and_forward_columns'] = A.increment(G + C, Gf + C, ('L',), S_SECONDARY, 1)
    if have_p:
        out['S1_graph_beyond_C_P_forward'] = A.increment(G + C + ['P'], Gf + C + ['P'], ('L',), S_SECONDARY, 2)
        out['S2_probe_beyond_C'] = A.increment(['P'] + C, C, ('L',), S_SECONDARY, 3)
        out['S2_probe_code_beyond_C'] = A.increment(['P_code'] + C, C, ('L',), S_SECONDARY, 4)
        out['S3_probe_plus_forward_vs_G'] = A.diff(['P'] + Gf, G)
        out['S3_descriptive_P_vs_G'] = A.diff(['P'], G)
        out['S3_descriptive_P_code_vs_G'] = A.diff(['P_code'], G)
    out.update(step0_block(A, have_p))
    # seed pairs (stream 85): spread of the primary's plug-in, and whether the reading depends on it
    seeds = M.rng(85).integers(1000, 10 ** 6, size=(10, 2))
    sd = []
    for pb, rs in seeds:
        sd.append(M.auc(A.d['y_prob'], A.scores(G + C, perm_base=int(pb), rf_seed=int(rs)))
                  - M.auc(A.d['y_prob'], A.scores(C, perm_base=int(pb), rf_seed=int(rs))))
    p10, p90 = np.percentile(sd, [10, 90])
    pr = out['primary']
    rows = set()
    for q in (p10, p90):
        sh = q - pr['diff']['plugin']
        rows.add(M.reading({'plugin': q, 'ci95': [pr['diff']['ci95'][0] + sh, pr['diff']['ci95'][1] + sh]},
                           [b['c0'] for b in pr['bands']], 'G', 'C')['row'].split(' (')[0].split(':')[0])
    own = pr['reading']['row'].split(' (')[0].split(':')[0]
    out['seed_pairs'] = {'deltas': [round(x, 4) for x in sd], 'p10_p90': [round(p10, 4), round(p90, 4)],
                         'seed_dependent': len(rows | {own}) > 1}
    out['logistic_learner'] = A.diff(G + C, C, learner='lr')
    out['mbpp_plus'] = {}
    for name, (y, flipped) in A.mbpp.items():
        idx = {int(p): i for i, p in enumerate(A.d['problems'])}
        y_row = y[[idx[int(g)] for g in A.d['groups']]]
        out['mbpp_plus'][name] = {'positive': int(y.sum()), 'flipped': flipped,
                                  'level_G+C': M.summarise_level(y, A.scores(G + C, y_row=y_row), A.D),
                                  'level_C': M.summarise_level(y, A.scores(C, y_row=y_row), A.D),
                                  'primary_raw': A.diff(G + C, C, y_prob=y, y_row=y_row)}
    out['descriptive_C_dist'] = {'level': A.level(['C_dist']), 'G_beyond': A.diff(G + ['C_dist'], ['C_dist'])}
    out['descriptive_length'] = {'level': A.level(['C_len']), 'G_beyond': A.diff(G + ['C_len'], ['C_len'])}
    return out


def step0_block(A, have_p):
    names = {'G': ['G'], 'C': ['C'], 'G+C': ['G', 'C']}
    if have_p:
        names.update({'P': ['P'], 'C+P': ['P', 'C'], 'G+C+P': ['G', 'C', 'P']})
    y = A.d['y_prob']
    last = A.last_row_mask()
    m0 = A.d['step'] == 0
    res = {'step0_all_rows_C1': {}, 'step0_one_row': {}, 'one_row_vs_C': {}}
    one = {}
    for k, v in names.items():
        s_all, s0 = A.scores(v), A.step0(v)
        sl = A.scores(v, mask=last, tag='last')
        res['step0_all_rows_C1'][k] = {'all': round(M.auc(y, s_all), 4), 'step0': round(M.auc(y, s0), 4),
                                       'C1': M.summarise_diff(y, s_all, s0, A.D)}
        res['step0_one_row'][k] = {'last': round(M.auc(y, sl), 4), 'step0': round(M.auc(y, s0), 4),
                                   'last_minus_step0': M.summarise_diff(y, sl, s0, A.D)}
        one[k] = (sl, s0)
    a, _ = M.boot(y, [one['C'][0], one['C'][1]], A.D)
    for k in names:
        if k == 'C':
            continue
        b, _ = M.boot(y, [one[k][0], one[k][1]], A.D)
        dd = (b[:, 0] - b[:, 1]) - (a[:, 0] - a[:, 1])
        pt = (M.auc(y, one[k][0]) - M.auc(y, one[k][1])) - (M.auc(y, one['C'][0]) - M.auc(y, one['C'][1]))
        res['one_row_vs_C'][k] = {'plugin': round(pt, 4), 'boot_mean': round(float(dd.mean()), 4),
                                  'ci95': [round(float(np.percentile(dd, 2.5)), 4),
                                           round(float(np.percentile(dd, 97.5)), 4)]}
    s0_gc, s0_c = A.step0(['G', 'C']), A.step0(['C'])
    res['primary_at_step0_raw'] = M.summarise_diff(y, s0_gc, s0_c, A.D)
    return res


def without_flagged(A, flagged, primary):
    """§5: f0 differed on a few rows of the arm; the primary repeated without their problems (raw paired
    difference, read with the population's own null centres; draws [20261022, 81, 10 + population])."""
    d, sig = A.d, A.sig
    keep, kp = ~np.isin(d['groups'], flagged), ~np.isin(d['problems'], flagged)
    y_k = d['y_prob'][kp]
    s1 = M.oof(np.column_stack([sig['G'], sig['C']])[keep], d['groups'][keep], d['y_row'][keep], d['problems'][kp],
               n_jobs=N_JOBS)
    s2 = M.oof(sig['C'][keep], d['groups'][keep], d['y_row'][keep], d['problems'][kp], n_jobs=N_JOBS)
    D = M.rng(81, 10 + M.POP_KEY[A.pop]).integers(0, len(y_k), size=(M.N_BOOT, len(y_k)))
    raw = M.summarise_diff(y_k, s1, s2, D)
    return {'dropped': flagged, 'raw': raw, 'reading': M.reading(raw, [b['c0'] for b in primary['bands']], 'G', 'C')}


def mbpp_sets(problems, y_prob):
    """Both MBPP+ label sets of Rust strict, built and checked before any fit (§6: 10 flips / 85 positive under
    the §7 rule, 18 / 77 without exclusions)."""
    small = json.load(open(os.path.join(HERE, 'results', 'p9_state', SMALL)))['tasks_below_min_cases']['tasks']
    out = {}
    for name, excl in (('rule_section7', set(map(int, small)) | set(ARTEFACTS)), ('no_exclusions', set())):
        y, flipped = mbpp_plus_labels(problems, y_prob, excl)
        want = {'rule_section7': (10, 85), 'no_exclusions': (18, 77)}[name]
        if not REHEARSAL and (len(flipped), int(y.sum())) != want:
            sys.exit(f'MBPP+ labels ({name}): {len(flipped)} flips, {int(y.sum())} positive; registered {want}')
        out[name] = (y, flipped)
    return out


def save(out):
    os.makedirs(M.OUT, exist_ok=True)
    path = os.path.join(M.OUT, 'analysis.json')
    with open(path + '.tmp', 'w') as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    os.replace(path + '.tmp', path)
    return path


def main():
    t_start = time.time()
    if REHEARSAL and os.path.abspath(M.OUT) == os.path.abspath(os.path.join(HERE, 'results', 'p9_m1')):
        sys.exit('a rehearsal must write elsewhere: set M1_OUT')
    out = {'rehearsal': REHEARSAL}
    guard = None
    if not REHEARSAL:
        guard = out['guard'] = M.guard()
        g1 = json.load(open(os.path.join(M.OUT, 'gates_x1.json')))
        g2 = json.load(open(os.path.join(M.OUT, 'gates_x2.json')))
        if not (g1['extractor_ok'] and g2['R']['r_ok']):
            sys.exit('refused: gate X1 (C) or R has not passed')
        out['gates'] = {'p_ok': g1['p_ok'], 'c_ok': g1['c_ok'], 'x1_f_ok': g1['f_ok'], 'r_ok': g2['R']['r_ok']}
        for arm in M.ARMS:
            F = g2[f'F_{arm}']
            out['gates'][f'{arm}_ok'] = F['arm_ok']
            out['gates'][f'{arm}_detail'] = {k: F[k] for k in ('f0_differ', 'f34_differ', 'flagged_problems')}
    tok = M.tokenizer()
    arm_rows = {a: M.load_arm(a) for a in M.ARMS}
    pool = Pool(WORKERS)
    for pop in ('rust_strict', 'rust_loose', 'python_240', 'python_256'):
        arm = M.POPS[pop][0]
        if not REHEARSAL and not out['gates'][f'{arm}_ok']:
            out[pop] = {'skipped': 'gate X2 failed for this arm'}
            continue
        d = M.load_pop(pop, tok, arm_rows[arm])
        if REHEARSAL:
            perm = M.rng(99, 2, M.POP_KEY[pop]).permutation(len(d['problems']))
            d['y_prob'] = d['y_prob'][perm]
            idx = {int(p): i for i, p in enumerate(d['problems'])}
            d['y_row'] = d['y_prob'][[idx[int(g)] for g in d['groups']]]
        mbpp = mbpp_sets(d['problems'], d['y_prob']) if pop == 'rust_strict' else None
        wants_p = pop in ('rust_strict', 'python_240')
        have_p = wants_p and (REHEARSAL or out['gates']['p_ok'])
        svd_info, Pm, Pc = {}, None, None
        if REHEARSAL:
            Cm, Pm, Pc = stand_ins(d)
            ext_sha = None
        else:
            Cm, X, Xc, ext_sha = c_and_posmean(arm, d['rows'], have_p, guard)
            if have_p:
                Pm, svd_info['P'] = centred_svd(X, 83, M.POP_KEY[pop])
                Pc, svd_info['P_code'] = centred_svd(Xc, 84, M.POP_KEY[pop])
                del X, Xc
        sig = {'G': d['G'], 'G_free': d['G'][:, list(M.G_FREE)], 'C': Cm, 'C_dist': Cm[:, C_DIST], 'C_len': Cm[:, C_LEN]}
        if have_p:
            sig.update({'P': Pm, 'P_code': Pc})
        A = Pop(pop, d, sig, pool)
        A.mbpp = mbpp
        log(f"== {pop}: {len(d['rows'])} rows, {len(d['problems'])} problems, {int(d['y_prob'].sum())} positive")
        res = {'rows': len(d['rows']), 'problems': len(d['problems']), 'positive': int(d['y_prob'].sum()),
               'extract_sha256': ext_sha, 'svd': svd_info, 'have_p': have_p}
        flagged = [] if REHEARSAL else out['gates'][f'{arm}_detail']['flagged_problems']
        if pop == 'rust_strict':
            res.update(analyse_rust_strict(A, have_p))
            if flagged:
                res['primary_without_flagged'] = without_flagged(A, flagged, res['primary'])
        elif pop == 'python_240':
            res['levels'] = {k: A.level(v) for k, v in (('G', ['G']), ('C', ['C']), ('G+C', ['G', 'C']))}
            res['primary'] = A.increment(['G', 'C'], ['C'], ('L',), S_SECONDARY, 0)
            res['S0_raw'] = A.diff(['G', 'C'], ['G_free', 'C'])
            if have_p:
                res['S1_raw'] = A.diff(['G', 'C', 'P'], ['G_free', 'C', 'P'])
                res['S2_raw'] = A.diff(['P', 'C'], ['C'])
                res['S2_code_raw'] = A.diff(['P_code', 'C'], ['C'])
                res['S3_raw'] = A.diff(['P', 'G_free'], ['G'])
                res['S3_desc_P_vs_G'] = A.diff(['P'], ['G'])
                res['S3_desc_P_code_vs_G'] = A.diff(['P_code'], ['G'])
            res.update(step0_block(A, have_p))
            if flagged:
                res['primary_without_flagged'] = without_flagged(A, flagged, res['primary'])
        else:
            res['levels'] = {k: A.level(v) for k, v in (('G', ['G']), ('C', ['C']), ('G+C', ['G', 'C']))}
            res['primary_raw'] = A.diff(['G', 'C'], ['C'])
        out[pop] = res
        out['seconds'] = round(time.time() - t_start)
        save(out)
    pool.close()
    log(f'wrote {save(out)} in {out["seconds"]} s')


if __name__ == '__main__':
    main()
