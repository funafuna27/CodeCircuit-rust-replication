"""Phase 3, step 3: the pre-registered analysis. PREREGISTRATION_WITHIN.md §3, §5.

One graph per sample; each pair is a passing and a failing solution to the
same problem, matched on length. The classifier never sees a problem it is
scored on, and since every problem contributes exactly one pass and one fail,
problem difficulty cannot help it.

  A    within-pair accuracy: P(pass) of the passing member > the failing one's
  test K refits with the two labels of each pair swapped with probability 1/2
  S1   the same on features residualised on [tokens, tokens^2, chars, lines]
  S2   a length-only model -- did the matching work?

  FEATURES=gemma_within_features.jsonl  ROWS=within_rows.jsonl
  PAIRS=within_pairs.json  OUT=results/within/analysis.json  K=100
A pair missing either member's graph is dropped, and the count is printed.
"""
import json
import os

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

FEATURES = os.environ.get('FEATURES', 'gemma_within_features.jsonl')
ROWS = os.environ.get('ROWS', 'within_rows.jsonl')
PAIRS = os.environ.get('PAIRS', 'within_pairs.json')
OUT = os.environ.get('OUT', '')
K = int(os.environ.get('K', 100))
N_REPEATS, N_FOLDS, N_BOOT = 10, 5, 5000
SEED = int(os.environ.get('SEED', 0))
N_JOBS = int(os.environ.get('N_JOBS', 4))


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def oof(X, y, groups, problems):
    """Per-sample out-of-fold P(pass), averaged over repeats. Folds hold whole
    problems; the surrogate-permutation scheme is 5_problem_level.py's."""
    idx = {int(p): i for i, p in enumerate(problems)}
    acc = np.zeros(len(y))
    for rep in range(N_REPEATS):
        perm = np.random.default_rng(100 + rep).permutation(len(problems))
        sur = np.array([perm[idx[int(g)]] for g in groups])
        p = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups=sur):
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS)
            clf.fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        acc += p
    return acc / N_REPEATS


def within(s, ip, ifl):
    d = s[ip] - s[ifl]
    return float(np.mean((d > 0) + 0.5 * (d == 0)))


def boot(s, ip, ifl, g):
    n = len(ip)
    out = [within(s, ip[b], ifl[b]) for b in (g.integers(0, n, n) for _ in range(N_BOOT))]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def swap_null(X, y, groups, problems, ip, ifl, g):
    null = []
    for k in range(K):
        flip = g.random(len(ip)) < 0.5
        pp, pf = np.where(flip, ifl, ip), np.where(flip, ip, ifl)
        ys = y.copy()
        ys[pp], ys[pf] = 1, 0
        null.append(within(oof(X, ys, groups, problems), pp, pf))
        if (k + 1) % 25 == 0:
            print(f"    {k + 1}/{K} refits", flush=True)
    return np.array(null)


def test(name, X, y, groups, problems, ip, ifl, stream):
    s = oof(X, y, groups, problems)
    a = within(s, ip, ifl)
    lo, hi = boot(s, ip, ifl, np.random.default_rng([SEED, stream]))
    null = swap_null(X, y, groups, problems, ip, ifl, np.random.default_rng([SEED, stream + 1]))
    p = (1 + int((null >= a).sum())) / (1 + K)
    print(f"  {name:28} A = {a:.4f}  95% CI [{lo:.4f}, {hi:.4f}]   label-swap refits: "
          f"mean {null.mean():.4f}, sd {null.std(ddof=1):.4f};  p = {p:.4f}")
    return {'A': a, 'ci': [lo, hi], 'null_mean': float(null.mean()),
            'null_sd': float(null.std(ddof=1)), 'p': p}, s


def main():
    rows = {int(r['task_id']): r for r in read_jsonl(ROWS)}
    feats = {}
    for r in read_jsonl(FEATURES):
        if int(r['expr_id']) in feats:
            raise SystemExit(f"two feature rows for sample {r['expr_id']}")
        feats[int(r['expr_id'])] = r
    with open(PAIRS, encoding='utf-8') as f:
        pairs = json.load(f)
    full = [p for p in pairs if p['pass_id'] in feats and p['fail_id'] in feats]
    bs = {feats[i]['batch_size'] for p in full for i in (p['pass_id'], p['fail_id'])}
    if bs != {64}:
        raise SystemExit(f"batch sizes {bs}; the study uses 64")
    print(f"{len(pairs)} pairs; {len(full)} have both graphs "
          f"({len(pairs) - len(full)} dropped for a missing graph)")

    ids = [i for p in full for i in (p['pass_id'], p['fail_id'])]
    X = np.array([feats[i]['feature'] for i in ids], dtype=float)
    if not np.isfinite(X).all():
        raise SystemExit("non-finite features; fix the featuriser, do not impute")
    y = np.array([rows[i]['label'] for i in ids])
    groups = np.array([rows[i]['problem'] for i in ids])
    problems = np.array(sorted(set(groups.tolist())))
    ip, ifl = np.arange(0, len(ids), 2), np.arange(1, len(ids), 2)
    assert (y[ip] == 1).all() and (y[ifl] == 0).all()
    text = [rows[i]['text'] for i in ids]
    tokens = np.array([rows[i]['tokens'] for i in ids], dtype=float)
    chars = np.array([len(t) for t in text], dtype=float)
    lines = np.array([t.count('\n') + 1. for t in text])
    print(f"  tokens pass - fail: mean {np.mean(tokens[ip] - tokens[ifl]):+.2f}, "
          f"|d| median {np.median(np.abs(tokens[ip] - tokens[ifl])):.0f}")

    res = {'n_pairs': len(full), 'n_dropped': len(pairs) - len(full)}
    print("\n=== A: within-pair accuracy (primary) ===")
    res['A'], _ = test('graph features', X, y, groups, problems, ip, ifl, 10)
    print("\n=== S1: features residualised on length ===")
    cov = np.column_stack([tokens, tokens ** 2, chars, lines])
    Xr = X - LinearRegression().fit(cov, X).predict(cov)
    res['S1'], _ = test('residualised features', Xr, y, groups, problems, ip, ifl, 20)
    print("\n=== S2: length alone (did the matching work?) ===")
    sl = oof(np.column_stack([tokens, chars, lines]), y, groups, problems)
    a = within(sl, ip, ifl)
    lo, hi = boot(sl, ip, ifl, np.random.default_rng([SEED, 30]))
    res['S2'] = {'A': a, 'ci': [lo, hi]}
    print(f"  {'length only':28} A = {a:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    print("\n=== descriptive: per-feature paired differences (Wilcoxon, BH) ===")
    d = X[ip] - X[ifl]
    pv = np.array([stats.wilcoxon(d[:, j])[1] if np.any(d[:, j] != 0) else 1.0
                   for j in range(d.shape[1])])
    order = np.argsort(pv)
    q = np.empty_like(pv)
    q[order] = np.minimum.accumulate((pv[order] * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]
    hits = [(int(j), float(np.median(d[:, j])), float(q[j])) for j in order if q[j] < 0.1]
    print(f"  features with q < 0.1: {len(hits)} of {d.shape[1]}")
    for j, md, qq in hits:
        print(f"    dim {j:2}  median pass - fail {md:+.4g}  q {qq:.3g}")
    res['features_q_lt_0.1'] = hits

    A, S1 = res['A'], res['S1']
    if A['p'] < 0.05 and A['A'] > 0.5:
        reading = ("the graph tells a correct solution from an incorrect one to the same "
                   "problem, beyond length" if S1['p'] < 0.05 and S1['A'] > 0.5 else
                   "a within-problem signal that cannot be separated from the length "
                   "difference matching left")
    else:
        reading = "not detectable at this n -- not evidence of absence"
    se = np.sqrt(0.25 / len(full))
    print(f"\n=== reading (PREREGISTRATION_WITHIN.md §5) ===\n  {reading}")
    print(f"  resolvable at 80% power, one-sided 0.05: A ~ {0.5 + 2.486 * se:.3f} "
          f"(normal approximation, n = {len(full)} pairs)")
    res['reading'] = reading
    if OUT:
        os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
