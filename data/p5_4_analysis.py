"""Phase 5, step 4: the pre-registered analysis. PREREGISTRATION_MATCHED.md §3.

Every pair is a survivor and a killed one-token mutant of one passing solution,
with the same Gemma token count and code log-probabilities at most 1 nat apart.
Several pairs may share a problem: folds keep a problem's pairs together (as in
Phases 3 and 4) and the interval resamples problems.

  Ac    PRIMARY. The 49 graph features centred within each pair; Phase 3's
        forest, folds and out-of-fold scores (w3_within_analysis.oof); A = the
        share of pairs whose survivor scores higher, ties 1/2; one-sided test
        against K = 500 refits with each pair's labels swapped with
        probability 1/2; 95% interval from 5000 bootstrap draws of problems.
  S2c   matching check: the same on centred [code log-probability, chars], K = 200
  S2+c  the output distribution after matching: centred [code log-probability,
        edit-token log-probability, minimum token log-probability, mean entropy,
        chars], K = 200
  S1c   Phase 4's m5 statistic: centred features minus a within-pair
        fixed-effects fit on [code log-probability, chars] (no intercept, no
        label), K = 200; f reported
  G     feature 1, pruned feature nodes: paired Wilcoxon signed-rank, one-sided,
        H1 the killed member has more
  descriptive: the matched gap, per-feature paired Wilcoxon with BH, family,
        site and kill kinds, pairs per problem

  FEATURES=gemma_matched_features.jsonl  ROWS=matched_rows.jsonl
  PAIRS=matched_pairs.json  OUT=results/matched/analysis.json  N_JOBS (w3)
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w3_within_analysis as w3  # noqa: E402

FEATURES = os.environ.get('FEATURES', 'gemma_matched_features.jsonl')
ROWS = os.environ.get('ROWS', 'matched_rows.jsonl')
PAIRS = os.environ.get('PAIRS', 'matched_pairs.json')
OUT = os.environ.get('OUT', 'results/matched/analysis.json')
K_PRIMARY = int(os.environ.get('K_PRIMARY', 500))
K_SECONDARY = int(os.environ.get('K_SECONDARY', 200))
N_BOOT = 5000
NODES = 1                                           # pruned feature-node count


def kill_kind(err):
    err = err or ''
    if 'TIMEOUT' in err:
        return 'timeout'
    if 'PANIC' in err:
        return 'panic in a call'
    if err.startswith('FAIL'):
        return 'wrong value'
    return 'aborted'


def boot_problems(s, ip, ifl, pprob, g):
    """2.5 and 97.5 percentiles of A over bootstrap draws of whole problems."""
    probs = np.unique(pprob)
    members = {p: np.flatnonzero(pprob == p) for p in probs}
    out = []
    for _ in range(N_BOOT):
        sel = np.concatenate([members[p] for p in g.choice(probs, len(probs))])
        out.append(w3.within(s, ip[sel], ifl[sel]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    rows = {int(r['task_id']): r for r in w3.read_jsonl(ROWS)}
    feats = {}
    for r in w3.read_jsonl(FEATURES):
        if int(r['expr_id']) in feats:
            raise SystemExit(f"two feature rows for {r['expr_id']}")
        feats[int(r['expr_id'])] = r
    with open(PAIRS, encoding='utf-8') as f:
        pairs = json.load(f)
    full = [p for p in pairs if p['pass_id'] in feats and p['fail_id'] in feats]
    if {feats[i]['batch_size'] for p in full for i in (p['pass_id'], p['fail_id'])} != {64}:
        raise SystemExit("a batch size other than 64")
    print(f"{len(pairs)} pairs; {len(full)} have both graphs "
          f"({len(pairs) - len(full)} dropped for a missing graph)")

    ids = [i for p in full for i in (p['pass_id'], p['fail_id'])]
    X = np.array([feats[i]['feature'] for i in ids], dtype=float)
    if not np.isfinite(X).all():
        raise SystemExit("non-finite features; fix the featuriser, do not impute")
    if any(feats[i]['original_expression'] != rows[i]['text'] for i in ids):
        raise SystemExit("a feature row's text differs from its pair row")
    y = np.array([rows[i]['label'] for i in ids])
    groups = np.array([rows[i]['problem'] for i in ids])
    problems = np.array(sorted(set(groups.tolist())))
    ip, ifl = np.arange(0, len(ids), 2), np.arange(1, len(ids), 2)
    pprob = groups[ip]
    assert (y[ip] == 1).all() and (y[ifl] == 0).all() and (groups[ip] == groups[ifl]).all()
    tokens = np.array([rows[i]['tokens'] for i in ids])
    if not (tokens[ip] == tokens[ifl]).all():
        raise SystemExit("a pair differs in token count; the design forbids it")
    chars = np.array([len(rows[i]['text']) for i in ids], dtype=float)
    sur = {k: np.array([rows[i]['surprise'][k] for i in ids], dtype=float)
           for k in ('code_logprob', 'edit_logprob', 'min_logprob', 'mean_entropy')}
    lp = sur['code_logprob']
    if np.max(np.abs(lp[ip] - lp[ifl])) > 1.0 + 1e-9:
        raise SystemExit("a pair outside the 1-nat caliper")

    def centred(M):
        return M - np.repeat((M[ip] + M[ifl]) / 2, 2, axis=0)

    def test(name, M, stream, k):
        w3.K = k
        s = w3.oof(M, y, groups, problems)
        a = w3.within(s, ip, ifl)
        lo, hi = boot_problems(s, ip, ifl, pprob, np.random.default_rng([w3.SEED, stream]))
        null = w3.swap_null(M, y, groups, problems, ip, ifl,
                            np.random.default_rng([w3.SEED, stream + 1]))
        p = (1 + int((null >= a).sum())) / (1 + k)
        print(f"  {name:30} A = {a:.4f}  95% CI [{lo:.4f}, {hi:.4f}]   label-swap refits (K = {k}): "
              f"mean {null.mean():.4f}, sd {null.std(ddof=1):.4f};  p = {p:.4f}")
        return {'A': a, 'ci': [lo, hi], 'K': k, 'null_mean': float(null.mean()),
                'null_sd': float(null.std(ddof=1)), 'p': p}

    d = lp[ip] - lp[ifl]
    per = np.unique(pprob, return_counts=True)[1]
    kinds = {}
    for p in full:
        kinds[kill_kind(p['killed_error'])] = kinds.get(kill_kind(p['killed_error']), 0) + 1
    res = {'n_pairs': len(full), 'n_dropped': len(pairs) - len(full), 'n_problems': len(problems),
           'gap': {'median_abs': float(np.median(np.abs(d))), 'mean': float(d.mean()),
                   'killed_lower': float(np.mean(d > 0)),
                   'f': float(d.mean() ** 2 / (d.mean() ** 2 + d.var(ddof=1)))},
           'same_family': float(np.mean([p['same_family'] for p in full])),
           'same_site': float(np.mean([p['survivor_edit']['pos'] == p['killed_edit']['pos']
                                       for p in full])),
           'killed_by': kinds, 'pairs_per_problem_max': int(per.max())}
    print(f"  {len(problems)} problems, pairs per problem 1..{per.max()} (median {np.median(per):.0f})")
    print(f"  code log-probability survivor - killed: |d| median {res['gap']['median_abs']:.3f}, "
          f"mean {d.mean():+.3f}, killed lower in {res['gap']['killed_lower']:.3f}; f = {res['gap']['f']:.3f}")
    print(f"  chars survivor - killed: mean {np.mean(chars[ip] - chars[ifl]):+.3f}; same family "
          f"{res['same_family']:.3f}, same site {res['same_site']:.3f}; killed by: "
          + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))

    print("\n=== Ac: graph features centred within each pair (PRIMARY) ===")
    res['Ac'] = test('graph features, centred', centred(X), 10, K_PRIMARY)
    print("\n=== S2c: matching check, [code log-probability, chars] centred ===")
    res['S2c'] = test('surprise, centred', centred(np.column_stack([lp, chars])), 20, K_SECONDARY)
    print("\n=== S2+c: the output distribution after matching ===")
    res['S2pc'] = test('output distribution, centred', centred(np.column_stack(
        [lp, sur['edit_logprob'], sur['min_logprob'], sur['mean_entropy'], chars])), 30, K_SECONDARY)
    print("\n=== S1c: centred features, within-pair surprise removed ===")
    cw, xw = centred(np.column_stack([lp, chars])), centred(X)
    beta = np.linalg.lstsq(cw, xw, rcond=None)[0]
    res['S1c'] = test('centred, surprise removed', xw - cw @ beta, 40, K_SECONDARY)

    print("\n=== G: pruned feature nodes, killed > survivor ===")
    dn = X[ip, NODES] - X[ifl, NODES]
    w = stats.wilcoxon(dn, alternative='less')
    res['G'] = {'killed_more': float(np.mean(dn < 0)), 'equal': float(np.mean(dn == 0)),
                'median_survivor_minus_killed': float(np.median(dn)), 'p': float(w.pvalue),
                'spearman_with_gap': float(stats.spearmanr(dn, d).statistic)}
    print(f"  killed has more nodes in {res['G']['killed_more']:.3f} of pairs (equal in "
          f"{res['G']['equal']:.3f}); median survivor - killed {np.median(dn):+.1f}; "
          f"Wilcoxon one-sided p = {w.pvalue:.4f}; Spearman with the gap {res['G']['spearman_with_gap']:+.2f}")

    print("\n=== descriptive: per-feature paired differences (Wilcoxon, BH) ===")
    dx = X[ip] - X[ifl]
    pv = np.array([stats.wilcoxon(dx[:, j])[1] if np.any(dx[:, j] != 0) else 1.0
                   for j in range(dx.shape[1])])
    order = np.argsort(pv)
    q = np.empty_like(pv)
    q[order] = np.minimum.accumulate((pv[order] * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]
    hits = [(int(j), float(np.median(dx[:, j])), float(q[j])) for j in order if q[j] < 0.1]
    print(f"  features with q < 0.1: {len(hits)} of {dx.shape[1]}")
    for j, md, qq in hits:
        print(f"    dim {j:2}  median survivor - killed {md:+.4g}  q {qq:.3g}")
    res['features_q_lt_0.1'] = hits

    ac, s2 = res['Ac'], res['S2c']
    if s2['p'] < 0.05:
        reading = ("matching did not neutralise the code log-probability: Ac is not read as "
                   "beyond it; S1c is read in its place, with Phase 4's f rule")
    elif ac['p'] < 0.05:
        reading = ("the graph tells a passing one-token mutant from a failing one at identical "
                   "length and matched code log-probability")
    else:
        reading = "not detectable at this n -- not evidence of absence"
    print(f"\n=== reading (PREREGISTRATION_MATCHED.md §5) ===\n  {reading}")
    print(f"  resolvable at 80% power, one-sided 0.05: A ~ {0.5 + 2.486 * ac['null_sd']:.3f} "
          f"from the null's sd; {0.5 + 2.486 * np.sqrt(0.25 / len(full)):.3f} by sqrt(0.25/n)")
    res['reading'] = reading
    if OUT:
        os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
