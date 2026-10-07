"""Phase 4, POST HOC: written on 2026-10-08 after m3 and m4 had run and their
results had been seen. Not pre-registered; PREREGISTRATION_MUTANT.md §9.

m4 found that the graph features, centred within each pair, rank the survivor
higher (Ac = 0.674, p 0.0099), and so does the model's surprise alone (S2c =
0.694, p 0.0099). The descriptive tests point the same way: the killed member's
graph is larger, and the more so the less probable it is. So Ac may be the
surprise read back through the graph. S1c asks whether it is more than that:
A on the within-pair-centred features after S1's within-pair fixed-effects
removal of [code log-probability, chars], the same fit as m3's S1. The centring
and the fit use no label, so the label-swap null holds.

Reading, written before this script first ran:
  S1c p < 0.05   the graph separates the members beyond a linear within-pair
                 effect of the model's surprise; post hoc, so a lead for a
                 fresh pre-registered test, not a finding
  S1c p >= 0.05  the centred graph signal cannot be separated from the model's
                 surprise at this n. Not evidence that the graph reads only
                 surprise: the fit also removes about f = 0.17 of any
                 surprise-independent difference (§8), and 98 pairs resolve
                 only a large effect
  Either way the removal is linear; a forest can still use a nonlinear
  function of surprise that the fit leaves in the features.

  same inputs as m3; OUT=results/mutant/posthoc.json  K=100
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w3_within_analysis as w3  # noqa: E402

FEATURES = os.environ.get('FEATURES', 'gemma_mutant_features.jsonl')
ROWS = os.environ.get('ROWS', 'mutant_rows.jsonl')
PAIRS = os.environ.get('PAIRS', 'mutant_pairs.json')
LOGPROB = os.environ.get('LOGPROB', 'mutant_logprob.jsonl')
OUT = os.environ.get('OUT', 'results/mutant/posthoc.json')


def main():
    rows = {int(r['task_id']): r for r in w3.read_jsonl(ROWS)}
    lps = {int(r['task_id']): r for r in w3.read_jsonl(LOGPROB)}
    feats = {int(r['expr_id']): r for r in w3.read_jsonl(FEATURES)}
    with open(PAIRS, encoding='utf-8') as f:
        pairs = json.load(f)
    full = [p for p in pairs if p['pass_id'] in feats and p['fail_id'] in feats]
    ids = [i for p in full for i in (p['pass_id'], p['fail_id'])]
    X = np.array([feats[i]['feature'] for i in ids], dtype=float)
    y = np.array([rows[i]['label'] for i in ids])
    groups = np.array([rows[i]['problem'] for i in ids])
    problems = np.array(sorted(set(groups.tolist())))
    ip, ifl = np.arange(0, len(ids), 2), np.arange(1, len(ids), 2)
    assert (y[ip] == 1).all() and (y[ifl] == 0).all()
    chars = np.array([len(rows[i]['text']) for i in ids], dtype=float)
    lp = np.array([lps[i]['code_logprob'] for i in ids])

    def centred(M):
        return M - np.repeat((M[ip] + M[ifl]) / 2, 2, axis=0)

    print(f"{len(full)} pairs analysed ({len(pairs) - len(full)} dropped for a missing graph)")
    cw, xw = centred(np.column_stack([lp, chars])), centred(X)
    beta = np.linalg.lstsq(cw, xw, rcond=None)[0]
    print("\n=== S1c: centred graph features, within-pair surprise removed (post hoc) ===")
    res = {'post_hoc': True, 'n_pairs': len(full)}
    res['S1c'], _ = w3.test('centred, surprise removed', xw - cw @ beta,
                            y, groups, problems, ip, ifl, 60)
    if OUT:
        os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
