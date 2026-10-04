"""Phase 4, step 3: the pre-registered analysis. PREREGISTRATION_MUTANT.md §3, §5.

Each pair is one passing solution carrying one of two one-token edits: one it
survives, one that kills it. The token count is identical within every pair, so
the members' length cannot separate them; what is left to rule out is how
surprising each edit is to the model.

  A    within-pair accuracy of the graph features: P(pass) of the survivor above
       the killed mutant's. Phase 3's classifier, folds, bootstrap and
       label-swap null (w3_within_analysis.test), unchanged.
  S1   the same on features with the within-pair differences in the model's
       surprise removed: each feature's fixed-effects slope on [code
       log-probability, chars], both centred within each pair, is subtracted.
       Within-pair, not pooled: Phase 3 found a pooled fit also strips the
       placement of whole pairs among problems (PREREGISTRATION_WITHIN.md §9b).
  S2   the model's surprise alone: [code log-probability, chars, tokens].
  descriptive: per-feature paired Wilcoxon with Benjamini-Hochberg; the gap in
       code log-probability, killed minus survivor.

  FEATURES=gemma_mutant_features.jsonl  ROWS=mutant_rows.jsonl
  PAIRS=mutant_pairs.json  LOGPROB=mutant_logprob.jsonl
  OUT=results/mutant/analysis.json  K=100
A pair missing either member's graph is dropped, and the count is printed.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w3_within_analysis as w3  # noqa: E402

FEATURES = os.environ.get('FEATURES', 'gemma_mutant_features.jsonl')
ROWS = os.environ.get('ROWS', 'mutant_rows.jsonl')
PAIRS = os.environ.get('PAIRS', 'mutant_pairs.json')
LOGPROB = os.environ.get('LOGPROB', 'mutant_logprob.jsonl')
OUT = os.environ.get('OUT', 'results/mutant/analysis.json')


def main():
    rows = {int(r['task_id']): r for r in w3.read_jsonl(ROWS)}
    lps = {int(r['task_id']): r for r in w3.read_jsonl(LOGPROB)}
    feats = {}
    for r in w3.read_jsonl(FEATURES):
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
    tokens = np.array([rows[i]['tokens'] for i in ids], dtype=float)
    if not (tokens[ip] == tokens[ifl]).all():
        raise SystemExit("a pair differs in token count; the design forbids it")
    chars = np.array([len(rows[i]['text']) for i in ids], dtype=float)
    lp = np.array([lps[i]['code_logprob'] for i in ids])
    dlp = lp[ifl] - lp[ip]
    print(f"  chars survivor - killed: mean {np.mean(chars[ip] - chars[ifl]):+.2f}")
    print(f"  code log-probability, killed - survivor: mean {dlp.mean():+.3f}, "
          f"median {np.median(dlp):+.3f}, killed lower in {np.mean(dlp < 0):.2f} of pairs, "
          f"Wilcoxon p {stats.wilcoxon(dlp)[1]:.4f}")

    def pair_mean(M):
        return np.repeat((M[ip] + M[ifl]) / 2, 2, axis=0)

    res = {'n_pairs': len(full), 'n_dropped': len(pairs) - len(full),
           'logprob_gap': {'mean': float(dlp.mean()), 'median': float(np.median(dlp)),
                           'killed_lower': float(np.mean(dlp < 0))}}
    print("\n=== A: within-pair accuracy (primary) ===")
    res['A'], _ = w3.test('graph features', X, y, groups, problems, ip, ifl, 10)

    print("\n=== S1: within-pair surprise removed ===")
    cov = np.column_stack([lp, chars])
    cw, xw = cov - pair_mean(cov), X - pair_mean(X)
    beta = np.linalg.lstsq(cw, xw, rcond=None)[0]
    r2 = 1 - ((xw - cw @ beta) ** 2).sum() / (xw ** 2).sum()
    print(f"  surprise and chars explain {r2:.3f} of the within-pair feature variance")
    res['S1'], _ = w3.test('surprise removed', X - cw @ beta, y, groups, problems, ip, ifl, 20)
    res['S1']['within_r2'] = float(r2)

    print("\n=== S2: the model's surprise alone ===")
    res['S2'], _ = w3.test('surprise only', np.column_stack([lp, chars, tokens]),
                           y, groups, problems, ip, ifl, 30)

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
        print(f"    dim {j:2}  median survivor - killed {md:+.4g}  q {qq:.3g}")
    res['features_q_lt_0.1'] = hits

    A, S1 = res['A'], res['S1']
    if A['p'] < 0.05 and A['A'] > 0.5:
        reading = ("the graph tells a passing solution from a broken copy of it, at "
                   "identical length and beyond the model's surprise at the edit"
                   if S1['p'] < 0.05 and S1['A'] > 0.5 else
                   "a signal at identical length that cannot be separated from the "
                   "model's surprise at the edit")
    else:
        reading = "not detectable at this n -- not evidence of absence"
    se = np.sqrt(0.25 / len(full))
    print(f"\n=== reading (PREREGISTRATION_MUTANT.md §5) ===\n  {reading}")
    print(f"  resolvable at 80% power, one-sided 0.05: A ~ {0.5 + 2.486 * se:.3f} "
          f"(normal approximation, n = {len(full)} pairs)")
    res['reading'] = reading
    if OUT:
        os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
