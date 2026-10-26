"""Phase 9, part ③: post hoc sensitivity of S1 (PREREGISTRATION_P9_M1.md §12).

S1 = the attribution-dependent columns beyond G_free + C + P (full G + C + P, base G_free + C + P), rerun with
the checks §6 gave the primary: the 10 alternative seed pairs (stream 85), the logistic learner, both MBPP+
label sets, and Python 240 with a band. Each variant gets its own design-L null centre (100 draws, stream 86,
key = population x 100 + 20 + variant), computed with the variant's seeds, learner and labels, and §6's reading.
Variant 0 (key 2, the registered settings) must reproduce §11 exactly.

  WORKERS=16  SMOKE=1 (labels permuted across problems, 2 null draws, variants 0, 1 and 11 only; writes to
  M1_WORK/s1_smoke.json, never to results/)
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_m1_common as M  # noqa: E402
import p9_m1_analysis as AN  # noqa: E402

WORKERS = int(os.environ.get('WORKERS', 16))
SMOKE = os.environ.get('SMOKE', '0') == '1'
S = 2 if SMOKE else 100
FULL, BASE = ['G', 'C', 'P'], ['G_free', 'C', 'P']
REGISTERED = {'plugin': 0.0174, 'c0': 0.0027, 'upper_bracket': 0.0317}


def log(*a):
    print(*a, flush=True)


def check_code(arm):
    """The extraction's sha256 (checked by load_extract) and the hashes of the code that made it."""
    meta = json.load(open(os.path.join(M.WORK, f'extract_{arm}.json')))
    bad = [k for k, v in (meta.get('guard') or {}).items()
           if k.endswith('.py') and M.sha256(os.path.join(M.REPO, k))[:16] != v]
    if bad:
        sys.exit(f'extraction {arm}: code changed since it was made: {bad}')
    return meta['sha256']


def band_task(args):
    """One design-L null draw with the variant's seeds and learner (M.band_task with oof's keywords)."""
    r, ctx, Xfull, Xbase, kw = args
    yp = M.null_labels('L', r, ctx)
    if yp.min() == yp.max():
        return None
    y_row = yp[ctx['row_problem_index']]
    return M.auc(yp, M.oof(Xfull, ctx['groups'], y_row, ctx['problems'], **kw)) - \
        M.auc(yp, M.oof(Xbase, ctx['groups'], y_row, ctx['problems'], **kw))


def variant(pool, A, key, y_prob=None, y_row=None, kw=None):
    kw = kw or {}
    t0 = time.time()
    d = A.d
    yp = d['y_prob'] if y_prob is None else y_prob
    yr = d['y_row'] if y_row is None else y_row
    Xf, Xb = A.X(FULL), A.X(BASE)
    s_full = M.oof(Xf, d['groups'], yr, d['problems'], n_jobs=8, **kw)
    s_base = M.oof(Xb, d['groups'], yr, d['problems'], n_jobs=8, **kw)
    diff = M.summarise_diff(yp, s_full, s_base, A.D)
    ctx = M.band_context(Xb, d['groups'], d['problems'], yp, s_base, key)
    res = pool.map(band_task, [(r, ctx, Xf, Xb, kw) for r in range(S)], chunksize=1)
    v = np.array([x for x in res if x is not None])
    band = {'design': 'L', 'S': S, 'used': int(len(v)), 'key': key, 'c0': round(float(v.mean()), 4),
            'mc_se': round(float(v.std(ddof=1) / np.sqrt(len(v))), 4) if len(v) > 1 else None}
    out = {'diff': diff, 'band': band, 'reading': M.reading(diff, [band['c0']], 'G', 'G_free+C+P'),
           'seconds': round(time.time() - t0)}
    log(f"  key {key} {kw}: {diff['plugin']:+.4f} {diff['ci95']} c0 {band['c0']:+.4f} -> "
        f"[{out['reading']['lower_bracket']:+.4f}, {out['reading']['upper_bracket']:+.4f}] ({out['seconds']} s)")
    return out


def population(pop, tok, arm_rows, pool):
    arm = M.POPS[pop][0]
    d = M.load_pop(pop, tok, arm_rows[arm])
    if SMOKE:                                       # no real association is read
        perm = M.rng(99, 3, M.POP_KEY[pop]).permutation(len(d['problems']))
        d['y_prob'] = d['y_prob'][perm]
        idx = {int(p): i for i, p in enumerate(d['problems'])}
        d['y_row'] = d['y_prob'][[idx[int(g)] for g in d['groups']]]
    sha = check_code(arm)
    Cm, X, Xc, ext_sha = AN.c_and_posmean(arm, d['rows'], True, None)
    assert ext_sha == sha
    Pm, _ = AN.centred_svd(X, 83, M.POP_KEY[pop])
    del X, Xc
    sig = {'G': d['G'], 'G_free': d['G'][:, list(M.G_FREE)], 'C': Cm, 'P': Pm}
    A = AN.Pop(pop, d, sig, pool)
    return A, sha


def main():
    t_start = time.time()
    out = {'post_hoc': True, 'smoke': SMOKE, 'S': S}
    if not SMOKE:
        out['guard'] = M.guard()
    tok = M.tokenizer()
    arm_rows = {a: M.load_arm(a) for a in M.ARMS}
    pool = Pool(WORKERS)
    A, out['extract_sha256_rust'] = population('rust_strict', tok, arm_rows, pool)
    log(f"== rust_strict: {len(A.d['rows'])} rows, {len(A.d['problems'])} problems")
    v = {}
    v[0] = variant(pool, A, 2)
    if not SMOKE:
        ok = (v[0]['diff']['plugin'] == REGISTERED['plugin'] and v[0]['band']['c0'] == REGISTERED['c0']
              and v[0]['reading']['upper_bracket'] == REGISTERED['upper_bracket'])
        out['reference_reproduces_section_11'] = ok
        if not ok:
            out['variants'] = {'0': v[0]}
            json.dump(out, open(os.path.join(M.OUT, 's1_sensitivity.json'), 'w'), indent=1)
            sys.exit('variant 0 does not reproduce §11: nothing else is read')
    seeds = M.rng(85).integers(1000, 10 ** 6, size=(10, 2))
    for i, (pb, rs) in enumerate(seeds, start=1):
        if SMOKE and i > 1:
            break
        v[i] = variant(pool, A, 20 + i, kw={'perm_base': int(pb), 'rf_seed': int(rs)})
        v[i]['seeds'] = [int(pb), int(rs)]
    v[11] = variant(pool, A, 31, kw={'learner': 'lr'})
    if not SMOKE:
        mb = AN.mbpp_sets(A.d['problems'], A.d['y_prob'])
        idx = {int(p): i for i, p in enumerate(A.d['problems'])}
        for j, name in ((12, 'rule_section7'), (13, 'no_exclusions')):
            y, flipped = mb[name]
            y_row = y[[idx[int(g)] for g in A.d['groups']]]
            v[j] = variant(pool, A, 20 + j, y_prob=y, y_row=y_row)
            v[j]['labels'] = {'name': name, 'positive': int(y.sum()), 'flipped': flipped}
        del A
        A14, out['extract_sha256_python'] = population('python_240', tok, arm_rows, pool)
        log(f"== python_240: {len(A14.d['rows'])} rows, {len(A14.d['problems'])} problems")
        v[14] = variant(pool, A14, M.POP_KEY['python_240'] * 100 + 20 + 14)
    pool.close()
    out['variants'] = {str(k): x for k, x in sorted(v.items())}
    ub = {k: x['reading']['upper_bracket'] for k, x in v.items()}
    out['summary'] = {'upper_brackets': {str(k): u for k, u in sorted(ub.items())},
                      'max_upper_bracket': max(ub.values()),
                      'variants_at_or_above_0.05': [k for k, u in sorted(ub.items()) if u >= 0.05],
                      'plugins': {str(k): x['diff']['plugin'] for k, x in sorted(v.items())},
                      'all_below_0.05': all(u < 0.05 for u in ub.values())}
    out['seconds'] = round(time.time() - t_start)
    dest = os.path.join(M.WORK, 's1_smoke.json') if SMOKE else os.path.join(M.OUT, 's1_sensitivity.json')
    with open(dest + '.tmp', 'w') as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    os.replace(dest + '.tmp', dest)
    log(json.dumps(out['summary'], indent=1))
    log(f'wrote {dest} in {out["seconds"]} s')


if __name__ == '__main__':
    main()
