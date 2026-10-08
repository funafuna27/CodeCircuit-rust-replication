"""Phase 5, EXPLORATORY, not pre-registered: what the edits' surface identity
says about the members. PREREGISTRATION_MATCHED.md §8, written and committed
after the pairs and before any graph.

The two members of a pair carry different edits: which token was replaced,
by which, in which family, and where. That surface identity may itself tell
a surviving edit from a killing one (a boundary change such as > -> >= is
often harmless), without any model. A graph feature that responds to the
edit's identity would separate the members for the same reason.

  E    Phase 3's forest and test (as Ac, K = 200) on the centred edit identity
       of each member: one-hot of the replaced token, one-hot of the new token,
       one-hot of the family, and the edit's position as a share of the code
  uses no graph and no model; the interval resamples problems

Reading, written before it first ran:
  E p < 0.05   the edits' surface identity alone tells the members apart; an
               Ac that does not clearly exceed E does not show the graph adds
               to it (Ac against E is descriptive; the difference is not tested)
  E p >= 0.05  the surface identity, as encoded here, does not separate them

  ROWS=matched_rows.jsonl  PAIRS=matched_pairs.json  OUT=results/matched/edit_baseline.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w3_within_analysis as w3  # noqa: E402
from m1_make_mutants import FAMILIES  # noqa: E402
from p5_4_analysis import boot_problems  # noqa: E402

ROWS = os.environ.get('ROWS', 'matched_rows.jsonl')
PAIRS = os.environ.get('PAIRS', 'matched_pairs.json')
OUT = os.environ.get('OUT', 'results/matched/edit_baseline.json')
K = int(os.environ.get('K', 200))
TOKENS = sorted({t for ts in FAMILIES.values() for t in ts})
FAMS = sorted(FAMILIES)


def encode(r):
    e = r['edit']
    v = [float(e['from'] == t) for t in TOKENS] + [float(e['to'] == t) for t in TOKENS]
    v += [float(e['family'] == f) for f in FAMS]
    return v + [e['pos'] / max(1, len(r['generated_code']))]


def main():
    rows = {int(r['task_id']): r for r in w3.read_jsonl(ROWS)}
    with open(PAIRS, encoding='utf-8') as f:
        pairs = json.load(f)
    ids = [i for p in pairs for i in (p['pass_id'], p['fail_id'])]
    X = np.array([encode(rows[i]) for i in ids])
    y = np.array([rows[i]['label'] for i in ids])
    groups = np.array([rows[i]['problem'] for i in ids])
    problems = np.array(sorted(set(groups.tolist())))
    ip, ifl = np.arange(0, len(ids), 2), np.arange(1, len(ids), 2)
    assert (y[ip] == 1).all() and (y[ifl] == 0).all()
    Xc = X - np.repeat((X[ip] + X[ifl]) / 2, 2, axis=0)

    print(f"{len(pairs)} pairs over {len(problems)} problems (exploratory, no graph)")
    w3.K = K
    s = w3.oof(Xc, y, groups, problems)
    a = w3.within(s, ip, ifl)
    lo, hi = boot_problems(s, ip, ifl, groups[ip], np.random.default_rng([w3.SEED, 50]))
    null = w3.swap_null(Xc, y, groups, problems, ip, ifl, np.random.default_rng([w3.SEED, 51]))
    p = (1 + int((null >= a).sum())) / (1 + K)
    print(f"  E, edit identity, centred    A = {a:.4f}  95% CI [{lo:.4f}, {hi:.4f}]   label-swap "
          f"refits (K = {K}): mean {null.mean():.4f}, sd {null.std(ddof=1):.4f};  p = {p:.4f}")
    if OUT:
        os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump({'exploratory': True, 'n_pairs': len(pairs), 'E': {
                'A': a, 'ci': [lo, hi], 'K': K, 'null_mean': float(null.mean()),
                'null_sd': float(null.std(ddof=1)), 'p': p}}, f, indent=1)


if __name__ == '__main__':
    main()
