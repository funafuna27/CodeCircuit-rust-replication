"""EXPLORATORY, not pre-registered: what did S1 remove? PREREGISTRATION_WITHIN.md §9.

Phase 3's pre-registered result: A = 0.624 (p 0.020), but S1 -- every feature
residualised on [tokens, tokens^2, chars, lines] pooled over all 178 rows --
fell to 0.534 (p 0.267). That pooled fit removes two different things at once:

  within   how each member's length departs from its pair's mean. Within a pair
           the lengths are nearly matched and unsigned (tokens pass - fail mean
           +0.00, chars median +2, lines mean -0.15), so if the signal lives
           here it is length after all.
  between  how the pair's mean length places it among the other problems. This
           moves both members of a pair by the same amount, so it leaves every
           within-pair feature difference untouched; it can only change where
           the forest's thresholds fall.

Both corrections use the same four covariates as S1 and never see a label, so
the label-swap null stays valid:

  E1  remove the within part only: the fixed-effects slope of X on the
      covariates, both centred within each pair; X - (cov - pair mean) beta
  E2  remove the between part only: X - fit of X on the pair-mean covariates

Reading, written before the run:
  E1 near A and E2 near S1  -> S1's loss came from the between part; the
      within-pair signal is not explained by the members' residual length
      difference, but the forest reads the features against problem length
  E1 near S1                -> the within-pair length difference carries the
      signal: the graph signal is length
  neither                   -> no clean split; report the numbers
"near" is judged on the point estimates against A - S1 = 0.090, with the
confidence intervals reported alongside; nothing here re-reads §5's table.

  FEATURES, ROWS, PAIRS, K as in w3_within_analysis.py;
  OUT=results/exploratory/within_length_decomposition.json
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w3_within_analysis as w3  # noqa: E402

OUT = os.environ.get('OUT', 'results/exploratory/within_length_decomposition.json')


def main():
    rows = {int(r['task_id']): r for r in w3.read_jsonl(w3.ROWS)}
    feats = {int(r['expr_id']): r for r in w3.read_jsonl(w3.FEATURES)}
    with open(w3.PAIRS, encoding='utf-8') as f:
        pairs = [p for p in json.load(f) if p['pass_id'] in feats and p['fail_id'] in feats]
    ids = [i for p in pairs for i in (p['pass_id'], p['fail_id'])]
    X = np.array([feats[i]['feature'] for i in ids], dtype=float)
    y = np.array([rows[i]['label'] for i in ids])
    groups = np.array([rows[i]['problem'] for i in ids])
    problems = np.array(sorted(set(groups.tolist())))
    ip, ifl = np.arange(0, len(ids), 2), np.arange(1, len(ids), 2)
    assert (y[ip] == 1).all() and (y[ifl] == 0).all()
    text = [rows[i]['text'] for i in ids]
    tokens = np.array([rows[i]['tokens'] for i in ids], dtype=float)
    chars = np.array([len(t) for t in text], dtype=float)
    lines = np.array([t.count('\n') + 1. for t in text])
    cov = np.column_stack([tokens, tokens ** 2, chars, lines])
    print(f"{len(pairs)} pairs")

    def pair_mean(M):
        m = (M[ip] + M[ifl]) / 2
        return np.repeat(m, 2, axis=0)  # rows are ordered pass, fail per pair

    cw, xw = cov - pair_mean(cov), X - pair_mean(X)
    beta = np.linalg.lstsq(cw, xw, rcond=None)[0]
    X_e1 = X - cw @ beta
    cb = pair_mean(cov)
    X_e2 = X - LinearRegression().fit(cb, X).predict(cb)

    d_e2 = (X_e2[ip] - X_e2[ifl]) - (X[ip] - X[ifl])
    print(f"  check: E2 changes within-pair differences by at most {np.abs(d_e2).max():.2e}")
    r2 = 1 - ((xw - cw @ beta) ** 2).sum() / (xw ** 2).sum()
    print(f"  within-pair length explains {r2:.3f} of the within-pair feature variance")

    res = {'exploratory': True, 'n_pairs': len(pairs), 'within_r2': float(r2)}
    print("\n=== E1: within-pair length removed ===")
    res['E1'], _ = w3.test('within-pair residualised', X_e1, y, groups, problems, ip, ifl, 40)
    print("\n=== E2: between-pair length removed ===")
    res['E2'], _ = w3.test('between-pair residualised', X_e2, y, groups, problems, ip, ifl, 50)
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
