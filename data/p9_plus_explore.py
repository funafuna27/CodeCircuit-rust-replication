"""Phase 9, part ②: post hoc checks, written after the registered results (PREREGISTRATION_P9_PLUS.md §11).
Nothing here changes a registered decision.

  1. Monte Carlo error of Q1's permutation p: 1,800 more permutations of the same family (streams
     [20261023, 31, k], k = 200..1999), with the registered 200 kept: p at K = 2,000.
  2. Length: holds were shorter than flips within a task (§11). The registered forest on the centred
     code-token count alone (L); P_code + L against L; P_code's out-of-fold score with a pooled linear
     length slope removed (centred within task), then its within-task AUROC; C without the length column.
  3. Per task: counts and d_t for C + P_code, C and P_code.
The signals are rebuilt exactly as the analysis built them; the C + P_code and P_code scores must equal the
saved ones.

  WORKERS=16  K_TOTAL=2000
"""
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.sparse import csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_plus_common as PC  # noqa: E402
import p9_plus_analysis as AN  # noqa: E402

WORKERS = int(os.environ.get('WORKERS', 16))
K_TOTAL = int(os.environ.get('K_TOTAL', 2000))


def main():
    t0 = time.time()
    guard = PC.guard()
    pop = PC.population()
    groups = np.array([r['task'] for r in pop])
    y = np.array([r['y'] for r in pop])
    # the pre-registration has gained its results since the extraction; the scripts must be the same
    z, meta = AN.load_extract(pop, {k: v for k, v in guard.items() if k.endswith('.py') or k == 'HEAD'})
    C = AN.c_matrix(pop, z)
    Xs = csr_matrix((z['pm_code_data'].astype(np.float64), z['pm_code_indices'], z['pm_code_indptr']),
                    shape=(len(pop), int(z['shape'][1])))
    Cc = AN.centre(C, groups)
    P, _ = AN.within_svd(Xs, groups, (33,))
    CP = np.hstack([Cc, P])
    with gzip.open(os.path.join(PC.OUT, 'scores.jsonl.gz'), 'rt') as fh:
        saved = [json.loads(l) for l in fh]
    if [r['id'] for r in saved] != [r['id'] for r in pop]:
        sys.exit('saved scores are not in population order')
    s_cp = PC.oof_rows(CP, groups, y, AN.PERM_BASE)
    s_p = PC.oof_rows(P, groups, y, AN.PERM_BASE)
    dev = max(np.abs(s_cp - [r['s_CP'] for r in saved]).max(), np.abs(s_p - [r['s_P'] for r in saved]).max())
    if dev > 1e-12:
        sys.exit(f'rebuilt scores differ from the saved ones by {dev}')
    an = json.load(open(os.path.join(PC.OUT, 'analysis.json')))
    out = {'post_hoc': True, 'guard': guard, 'rebuilt_scores_max_abs_diff': float(dev)}
    T = float((PC.task_aucs(s_cp, groups, y) - 0.5).mean())

    # 1. more permutations of Q1
    null200 = np.load(os.path.join(PC.OUT, 'permutation_null.npy'))[:, 0]
    with Pool(WORKERS, initializer=AN._init, initargs=([CP], y, groups)) as pool:
        more = np.array(pool.map(AN._perm, range(200, K_TOTAL), chunksize=1))[:, 0]
    null = np.concatenate([null200, more])
    out['Q1_more_permutations'] = {
        'T': round(T, 4), 'K': int(len(null)), 'exceed': int((null >= T).sum()),
        'p': round(float((1 + (null >= T).sum()) / (len(null) + 1)), 4),
        'p_first_200_registered': an['Q1']['p_permutation'],
        'p_permutations_200_to_end': round(float((1 + (more >= T).sum()) / (len(more) + 1)), 4),
        'null_mean': round(float(null.mean()), 4), 'null_q95': round(float(np.percentile(null, 95)), 4)}
    print('permutations', out['Q1_more_permutations'], f'{time.time() - t0:.0f} s', flush=True)

    # 2. length
    draws = PC.rng(32).integers(0, 44, size=(PC.N_BOOT, 44))
    L = Cc[:, 7:8]
    s_l = PC.oof_rows(L, groups, y, AN.PERM_BASE)
    s_pl = PC.oof_rows(np.hstack([P, L]), groups, y, AN.PERM_BASE)
    s_cdist = PC.oof_rows(Cc[:, :7], groups, y, AN.PERM_BASE)
    d = {k: PC.task_aucs(v, groups, y) - 0.5 for k, v in (('L', s_l), ('P_plus_L', s_pl), ('C_without_length', s_cdist),
                                                          ('P', s_p), ('CP', s_cp))}
    sp_c = AN.centre(s_p[:, None], groups)[:, 0]
    slope = float(np.dot(sp_c, L[:, 0]) / np.dot(L[:, 0], L[:, 0]))
    d['P_minus_linear_length'] = PC.task_aucs(sp_c - slope * L[:, 0], groups, y) - 0.5
    out['length'] = {k: PC.summarise(v, draws) for k, v in d.items()}
    inc = d['P_plus_L'] - d['L']
    out['length']['P_plus_L_minus_L'] = dict(PC.summarise(inc, draws), p_sign_flip=PC.sign_flip_greater(inc, (40,)))
    out['length']['P_minus_linear_length']['p_sign_flip'] = PC.sign_flip_greater(d['P_minus_linear_length'], (41,))
    out['length']['slope_of_P_score_on_centred_tokens'] = slope
    out['length']['corr_P_score_centred_tokens'] = round(float(np.corrcoef(sp_c, L[:, 0])[0, 1]), 4)
    tasks = np.unique(groups)
    out['length']['holds_minus_flips_tokens_median_over_tasks'] = float(np.median(
        [L[(groups == t) & (y == 1), 0].mean() - L[(groups == t) & (y == 0), 0].mean() for t in tasks]))

    # 3. per task
    dc = PC.task_aucs(PC.oof_rows(Cc, groups, y, AN.PERM_BASE), groups, y) - 0.5
    out['per_task'] = [{'task': int(t), 'holds': int(((groups == t) & (y == 1)).sum()),
                        'flips': int(((groups == t) & (y == 0)).sum()), 'd_CP': round(float(d['CP'][i]), 4),
                        'd_C': round(float(dc[i]), 4), 'd_P': round(float(d['P'][i]), 4)} for i, t in enumerate(tasks)]
    out['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(PC.OUT, 'explore.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ('guard', 'per_task')}, indent=1))


if __name__ == '__main__':
    main()
