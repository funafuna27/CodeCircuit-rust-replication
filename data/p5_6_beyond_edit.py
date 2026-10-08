"""Phase 5, EXPLORATORY, not pre-registered: do the graph features add to the
edits' surface identity? PREREGISTRATION_MATCHED.md §8, written and committed
after E's values were known and before any Phase 5 graph was featurised.

E (p5_5_edit_baseline.py) -- which token was replaced, by which, in which
family, and where -- ranks the survivor higher in 0.676 of Phase 5's pairs and
0.667 of Phase 4's, about as often as Phase 4's centred graph features did. So
Ac alone cannot show that the graph knows more than the kind of edit.

Three forests, Phase 3's procedure (w3_within_analysis.oof), on values centred
within each pair:
  G    the 49 graph features (Ac's input; its out-of-fold scores equal Ac's)
  E    the edit identity, encoded as in p5_5
  GE   both together
Per pair, 1 if the survivor scores higher, 1/2 for a tie, else 0. Reported:
A for each, and the differences G - E and GE - E with 95% intervals from 5000
bootstrap draws of problems (the same draws for both). The intervals carry
which pairs were drawn, not the forests' own refitting noise.

Reading, written before it first ran:
  GE - E, interval above 0     the graph features add to the edits' surface
                               identity at matched surprise (exploratory: a
                               lead for a pre-registered test, not a finding)
  GE - E, interval covering 0  not shown to add to it
  G - E                        Ac against E, descriptive

  FEATURES=gemma_matched_features.jsonl  ROWS=matched_rows.jsonl
  PAIRS=matched_pairs.json  OUT=results/matched/beyond_edit.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w3_within_analysis as w3  # noqa: E402
from p5_5_edit_baseline import encode  # noqa: E402

FEATURES = os.environ.get('FEATURES', 'gemma_matched_features.jsonl')
ROWS = os.environ.get('ROWS', 'matched_rows.jsonl')
PAIRS = os.environ.get('PAIRS', 'matched_pairs.json')
OUT = os.environ.get('OUT', 'results/matched/beyond_edit.json')
N_BOOT = 5000


def main():
    rows = {int(r['task_id']): r for r in w3.read_jsonl(ROWS)}
    feats = {int(r['expr_id']): r for r in w3.read_jsonl(FEATURES)}
    with open(PAIRS, encoding='utf-8') as f:
        pairs = json.load(f)
    full = [p for p in pairs if p['pass_id'] in feats and p['fail_id'] in feats]
    ids = [i for p in full for i in (p['pass_id'], p['fail_id'])]
    y = np.array([rows[i]['label'] for i in ids])
    groups = np.array([rows[i]['problem'] for i in ids])
    problems = np.array(sorted(set(groups.tolist())))
    ip, ifl = np.arange(0, len(ids), 2), np.arange(1, len(ids), 2)
    assert (y[ip] == 1).all() and (y[ifl] == 0).all()

    def centred(M):
        return M - np.repeat((M[ip] + M[ifl]) / 2, 2, axis=0)

    G = centred(np.array([feats[i]['feature'] for i in ids], dtype=float))
    E = centred(np.array([encode(rows[i]) for i in ids], dtype=float))
    print(f"{len(full)} pairs over {len(problems)} problems (exploratory)")
    correct = {}
    for name, M in (('G', G), ('E', E), ('GE', np.column_stack([G, E]))):
        s = w3.oof(M, y, groups, problems)
        d = s[ip] - s[ifl]
        correct[name] = (d > 0) + 0.5 * (d == 0)
        print(f"  {name:3} A = {correct[name].mean():.4f}")

    pprob = groups[ip]
    members = {p: np.flatnonzero(pprob == p) for p in problems}
    g = np.random.default_rng([w3.SEED, 60])
    diffs = {'G-E': [], 'GE-E': []}
    for _ in range(N_BOOT):
        sel = np.concatenate([members[p] for p in g.choice(problems, len(problems))])
        diffs['G-E'].append(correct['G'][sel].mean() - correct['E'][sel].mean())
        diffs['GE-E'].append(correct['GE'][sel].mean() - correct['E'][sel].mean())
    res = {'exploratory': True, 'n_pairs': len(full),
           'A': {k: float(v.mean()) for k, v in correct.items()}}
    for k, v in diffs.items():
        point = res['A'][k.split('-')[0]] - res['A']['E']
        lo, hi = np.percentile(v, [2.5, 97.5])
        res[k] = {'diff': float(point), 'ci': [float(lo), float(hi)]}
        print(f"  {k:5} {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    if OUT:
        os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
