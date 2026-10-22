"""Phase 9, part ③ (M1): an independent recomputation of the primary point estimates.

Written apart from p9_m1_analysis.py: C is rebuilt from the extraction with its own code-token rule (offsets,
not the analysis's loop), the out-of-fold scores come from 5_problem_level.oof_problem_scores itself (not
p9_m1_common.oof), and AUROC is a Mann-Whitney count. Compares AUROC(C), AUROC(G + C) and the primary Δ
(Rust strict), and the Python 240 primary, with results/p9_m1/analysis.json.
"""
import importlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('N_JOBS', '4')
pl = importlib.import_module('5_problem_level')


def mw_auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def main():
    out = {}
    for pop, arm, feats, labs, sub, rows_f in (
            ('rust_strict', 'rust', 'gemma_mbpp_rust_features.jsonl', 'gemma_mbpp_rust_correctness.jsonl',
             'subset_rust_strict.txt', 'gemma_mbpp_rust_incremental_capped.jsonl'),
            ('python_240', 'python', 'gemma_mbpp_features.jsonl', 'gemma_mbpp_python_correctness.jsonl',
             'subset_python_parses.txt', 'gemma_mbpp_incremental.jsonl')):
        F = [json.loads(l) for l in open(os.path.join(HERE, feats))]
        F.sort(key=lambda r: r['row_index'])
        src = [json.loads(l) for l in open(os.path.join(HERE, rows_f))]
        lab = {int(r['task_id']): int(bool(r['passes'])) for r in map(json.loads, open(os.path.join(HERE, labs)))}
        keep = {int(l) for l in open(os.path.join(HERE, sub)) if l.strip() and not l.startswith('#')}
        z = np.load(os.path.join(os.path.expanduser('~/p9_work/m1'), f'extract_{arm}.npz'))
        keys, tptr, off, logp, ent, final, ids = (z[k] for k in ('keys', 'tptr', 'off', 'logp', 'ent', 'final', 'ids'))
        at = {int(k): i for i, k in enumerate(keys)}
        X, Cm, g, y = [], [], [], []
        for r in F:
            if int(r['expr_id']) not in keep:
                continue
            i = at[r['row_index']]
            a, b = tptr[i], tptr[i + 1]
            if b - a > 140:                       # the 140-token cap, BOS counted
                continue
            s = src[r['row_index']]
            boundary = len(s['text']) - len(s['generated_code']) - 1      # the space before the code
            o = off[a:b]
            code = np.flatnonzero((o[:, 0] >= boundary) & (np.arange(b - a) > 0))
            lp, en = logp[a:b][code], ent[a:b][code]
            t1, t2, ef = final[i]
            n_prose = (b - a) - 1 - len(code)
            Cm.append([t1, ef, t1 - t2, lp.sum(), lp.mean(), lp.min(), en.mean(), len(code), n_prose])
            X.append(r['feature'])
            g.append(int(r['expr_id']))
            y.append(lab[int(r['expr_id'])])
        Cm, X, g, y = np.array(Cm), np.array(X, float), np.array(g), np.array(y)
        problems = np.array(sorted(set(g.tolist())))
        yp = np.array([lab[int(p)] for p in problems])
        sC = pl.oof_problem_scores(Cm, g, y, problems)
        sGC = pl.oof_problem_scores(np.column_stack([X, Cm]), g, y, problems)
        out[pop] = {'rows': len(g), 'problems': len(problems), 'auroc_C': round(mw_auc(yp, sC), 4),
                    'auroc_G+C': round(mw_auc(yp, sGC), 4), 'primary': round(mw_auc(yp, sGC) - mw_auc(yp, sC), 4)}
    ref = json.load(open(os.path.join(HERE, 'results', 'p9_m1', 'analysis.json')))
    out['reference'] = {'rust_strict': {'auroc_C': ref['rust_strict']['levels']['C']['auroc'],
                                        'auroc_G+C': ref['rust_strict']['levels']['G+C']['auroc'],
                                        'primary': ref['rust_strict']['primary']['diff']['plugin']},
                        'python_240': {'auroc_C': ref['python_240']['levels']['C']['auroc'],
                                       'auroc_G+C': ref['python_240']['levels']['G+C']['auroc'],
                                       'primary': ref['python_240']['primary']['diff']['plugin']}}
    out['agree'] = all(out[p][k] == out['reference'][p][k] for p in ('rust_strict', 'python_240')
                       for k in ('auroc_C', 'auroc_G+C', 'primary'))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
