"""Phase 4, EXPLORATORY, not pre-registered: the within-pair-centred analyses.
PREREGISTRATION_MUTANT.md §8 (2026-10-07). Written and committed before
m3_mutant_analysis.py first ran.

An audit before the analysis found that a forest fed levels can hardly use a
difference inside a pair when the levels vary far more between problems than
inside one: on synthetic pairs whose killed member is the less probable in 93%
of pairs, the forest scored A = 0.52 on log-probability levels and 0.91 on the
same values centred within each pair. The graph features have the same shape --
their spread between problems dwarfs what a one-token edit can change -- so the
pre-registered A and S2 may miss a difference that is there. Centring each pair
on its own mean removes the between-pair variation and uses no label, so
Phase 3's label-swap null stays valid.

  Ac   A on within-pair-centred graph features
  S2c  the same on within-pair-centred [code log-probability, chars]
  f    mean^2 / (mean^2 + var) of the survivor - killed code log-probability:
       about the share of a surprise-independent feature difference that S1's
       within-pair fit absorbs (§8)
  descriptive, over the analysed pairs: same-family and same-site shares; how
       the killed members fail; the per-feature within-pair R^2 of [code
       log-probability, chars] beside m3's pooled figure, which one raw-scale
       feature dominates

Reading, written before any result (exploratory; not a row of §5):
  Ac significant and A not   the graph carries a within-pair signal at
                             identical length that the pre-registered forest,
                             fed levels, could not use
  S2c above 0.5              the model's surprise separates the members
  Ac against S2c             which separates the members better; descriptive,
                             the difference is not tested

  same inputs as m3; OUT=results/mutant/centred.json  K=100
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
OUT = os.environ.get('OUT', 'results/mutant/centred.json')


def kill_kind(err):
    err = err or ''
    if 'TIMEOUT' in err:
        return 'timeout'
    if 'PANIC' in err:
        return 'panic in a call'
    if err.startswith('FAIL'):
        return 'wrong value'
    return 'aborted'


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
    d = lp[ip] - lp[ifl]
    f_abs = d.mean() ** 2 / (d.mean() ** 2 + d.var(ddof=1))
    res = {'exploratory': True, 'n_pairs': len(full),
           'gap': {'mean_survivor_minus_killed': float(d.mean()), 'sd': float(d.std(ddof=1)),
                   'killed_lower': float(np.mean(d > 0)), 'f_absorbed_by_S1': float(f_abs)}}
    print(f"  code log-probability, survivor - killed: mean {d.mean():+.3f}, sd {d.std(ddof=1):.3f}, "
          f"killed lower in {np.mean(d > 0):.3f}; S1 absorbs about f = {f_abs:.3f}")

    same_family = np.mean([p['same_family'] for p in full])
    same_site = np.mean([rows[p['pass_id']]['edit']['pos'] == rows[p['fail_id']]['edit']['pos']
                         for p in full])
    kinds = {}
    for p in full:
        k = kill_kind(p['killed_error'])
        kinds[k] = kinds.get(k, 0) + 1
    print(f"  same family {same_family:.3f}, same site {same_site:.3f}; killed by: "
          + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    res['descriptive'] = {'same_family': float(same_family), 'same_site': float(same_site),
                          'killed_by': kinds}

    cw, xw = centred(np.column_stack([lp, chars])), centred(X)
    beta = np.linalg.lstsq(cw, xw, rcond=None)[0]
    ss_res, ss_tot = ((xw - cw @ beta) ** 2).sum(0), (xw ** 2).sum(0)
    live = ss_tot > 0
    r2_cols = 1 - ss_res[live] / ss_tot[live]
    r2_pooled = 1 - ss_res.sum() / ss_tot.sum()
    top = int(np.argmax(ss_tot))
    print(f"  within-pair R^2 of [log-prob, chars]: pooled {r2_pooled:.3f} (feature {top} holds "
          f"{ss_tot[top] / ss_tot.sum():.3f} of the pooled sum of squares); per feature median "
          f"{np.median(r2_cols):.3f}, IQR {np.percentile(r2_cols, 25):.3f}-{np.percentile(r2_cols, 75):.3f}")
    res['within_r2'] = {'pooled': float(r2_pooled), 'per_feature_median': float(np.median(r2_cols)),
                        'dominant_feature': top, 'dominant_share': float(ss_tot[top] / ss_tot.sum())}

    print("\n=== Ac: graph features centred within each pair ===")
    res['Ac'], _ = w3.test('graph features, centred', centred(X), y, groups, problems, ip, ifl, 40)
    print("\n=== S2c: [code log-probability, chars] centred within each pair ===")
    res['S2c'], _ = w3.test('surprise, centred', cw, y, groups, problems, ip, ifl, 50)

    if OUT:
        os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
