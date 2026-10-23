"""Phase 9, part ②: independent recomputation of the analysis's point estimates (not imported by it).

From the raw files, without p9_plus_common or p9_plus_analysis: the population counts (a3_relabel_v2 and the
case table), each task's AUROC of the saved out-of-fold scores (sklearn's roc_auc_score), and the code's mean
log p per program from the extraction with its within-task AUROC. Compared with analysis.json to 4 decimals.

  PLUS_OUT=results/p9_plus  PLUS_WORK=~/p9_work/plus
"""
import gzip
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('PLUS_OUT', os.path.join(HERE, 'results', 'p9_plus'))
WORK = os.environ.get('PLUS_WORK', '/home/user1/p9_work/plus')
STATE = os.path.join(HERE, 'results', 'p9_state')
ARTEFACTS = {593, 806, 790, 771, 559, 109, 633, 564, 119, 142, 753, 409, 274, 435}


def rows(path):
    with gzip.open(path, 'rt') as fh:
        return [json.loads(l) for l in fh]


def main():
    an = json.load(open(os.path.join(OUT, 'analysis.json')))
    res = {}
    cases = json.load(open(os.path.join(STATE, 'a3_cases.json')))['tasks']
    small = {int(t) for t, v in cases.items() if v.get('n_cases') is not None and v['n_cases'] < 20}
    per = defaultdict(lambda: [0, 0])
    for r in rows(os.path.join(STATE, 'a3_relabel_v2.jsonl.gz')):
        o = r['O']
        if r['task_id'] in small | ARTEFACTS or o['stage'] != 'ok':
            continue
        per[r['task_id']][0 if o['n_fail'] + o['n_panic'] == 0 else 1] += 1
    mixed = {t for t, (h, f) in per.items() if h and f}
    res['population'] = [len(mixed), sum(sum(per[t]) for t in mixed), sum(per[t][0] for t in mixed),
                         sum(per[t][1] for t in mixed)]
    sc = rows(os.path.join(OUT, 'scores.jsonl.gz'))
    by = defaultdict(list)
    for r in sc:
        by[r['task']].append(r)
    for key, name in (('s_CP', 'Q1'), ('s_C', 'S1_C'), ('s_P', 'P_code_alone')):
        a = [roc_auc_score([r['y'] for r in rs], [r[key] for r in rs]) for _, rs in sorted(by.items())]
        res[name] = round(float(np.mean(a)) - 0.5, 4)
    with np.load(os.path.join(WORK, 'extract_plus.npz')) as z:
        keys, tptr, off, lp = z['keys'].tolist(), z['tptr'], z['off'], z['logp']
    prose_len = {}
    for r in rows(os.path.join(STATE, 'pool_samples.jsonl.gz')):
        prose_len[int(r['task_id'])] = len(r['text'])
    mean_lp = {}
    for i, k in enumerate(keys):
        t = int(k.split('_')[0])
        o, l = off[tptr[i]:tptr[i + 1]], lp[tptr[i]:tptr[i + 1]]
        sel = (o[:, 0] >= prose_len[t]) & (np.arange(len(o)) > 0)
        mean_lp[k] = float(l[sel].mean())
    a = [roc_auc_score([r['y'] for r in rs], [mean_lp[r['id']] for r in rs]) for _, rs in sorted(by.items())]
    res['code_logp_mean'] = round(float(np.mean(a)) - 0.5, 4)
    ref = {'population': [an['population'][k] for k in ('tasks', 'programs', 'holds', 'flips')],
           'Q1': an['Q1']['mean'], 'S1_C': an['S1_C']['mean'],
           'P_code_alone': an['descriptive']['P_code_alone']['mean'],
           'code_logp_mean': an['descriptive']['single_columns']['code_logp_mean']['mean']}
    agree = {k: res[k] == ref[k] for k in ref}
    print(json.dumps({'recomputed': res, 'analysis': ref, 'agree': agree}, indent=1))
    with open(os.path.join(OUT, 'check.json'), 'w') as fh:
        json.dump({'recomputed': res, 'analysis': ref, 'agree': agree}, fh, indent=1)


if __name__ == '__main__':
    main()
