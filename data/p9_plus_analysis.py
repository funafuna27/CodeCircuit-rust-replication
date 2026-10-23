"""Phase 9, part ②: the registered analysis (PREREGISTRATION_P9_PLUS.md §4-§6).

Signals per program, centred on their task's mean (label-free): C (8 columns of part ③'s C; the prose
length is constant within a task) and P_code (45 directions of a label-free truncated SVD of the code-position
posmean, centred within task). Learner: part ③'s forest on rows, out of fold over tasks (10 repeats x 5 folds,
fold maps from default_rng(100 + repeat)). Statistic: the mean over tasks of AUC_t - 0.5, AUC_t the within-task
AUROC of holds above flips.

  Q1 (primary)  C + P_code; one-sided permutation test: labels permuted within task (counts kept), the whole
                pipeline refitted, K = 200 (streams [20261023, 31, k]); interval: 5,000 task draws (stream 32)
  S1, S2        only if Q1 rejects; Holm over the two at 0.05. S1: C alone, the same permutations. S2: the
                increment AUC_t(C + P_code) - AUC_t(C), one-sided task-level sign flip (stream 35)
  descriptive   P_code alone; the logistic learner; label-free single columns; without the 8 tasks whose
                audited flip was an artefact; tasks with 2+ holds and 2+ flips; pooled pairs; the permutation null

  REHEARSAL=1: no guard; stand-in signals and labels permuted within task (no real association is read);
  writes to PLUS_WORK/rehearsal.   K_PERM=200  WORKERS=16
"""
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, svds

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_plus_common as PC  # noqa: E402

REHEARSAL = os.environ.get('REHEARSAL', '0') == '1'
K_PERM = int(os.environ.get('K_PERM', 200))
WORKERS = int(os.environ.get('WORKERS', 16))
K_SVD = 45
PERM_BASE = 100
C_NAMES = ('final_top1', 'final_entropy', 'final_top1_minus_top2', 'code_logp_sum', 'code_logp_mean',
           'code_logp_min', 'code_entropy_mean', 'n_code')
_G = {}


def log(*a):
    print(*a, flush=True)


def load_extract(pop, guard):
    path = os.path.join(PC.WORK, 'extract_plus.npz')
    meta = json.load(open(path.replace('.npz', '.json')))
    if PC.sha256(path) != meta['sha256']:
        sys.exit(f'{path}: sha256 differs from the one recorded at extraction')
    changed = [k for k in guard if k != 'HEAD' and (meta.get('guard') or {}).get(k) != guard[k]]
    if changed:
        sys.exit(f'{path}: made by other versions of {changed}')
    with np.load(path) as z:
        z = {k: z[k] for k in z.files}
    if z['keys'].tolist() != [r['id'] for r in pop]:
        sys.exit('extraction keys differ from the population')
    return z, meta


def c_matrix(pop, z):
    tptr, off_all, lp_all, en_all = z['tptr'], z['off'], z['logp'], z['ent']
    rows = []
    for i, r in enumerate(pop):
        a, b = tptr[i], tptr[i + 1]
        off, lp, en = off_all[a:b], lp_all[a:b], en_all[a:b]
        code = [t for t in range(1, b - a) if off[t][0] >= len(r['prose'])]
        if not code or len(code) != int(z['n_code'][i]):
            sys.exit(f"{r['id']}: {len(code)} code tokens, extraction counted {z['n_code'][i]}")
        top1, top2, ent_f = z['final'][i]
        rows.append([top1, ent_f, top1 - top2, lp[code].sum(), lp[code].mean(), lp[code].min(), en[code].mean(),
                     len(code)])
    C = np.array(rows, dtype=float)
    if not np.isfinite(C).all():
        sys.exit('C has non-finite entries')
    return C


def centre(X, groups):
    out = np.array(X, dtype=float)
    for t in np.unique(groups):
        m = groups == t
        out[m] -= out[m].mean(axis=0)
    return out


def within_svd(Xs, groups, stream):
    """45 leading right singular vectors of Xs with each row centred on its task's mean; rows projected."""
    _, inv = np.unique(groups, return_inverse=True)
    n = Xs.shape[0]
    T = csr_matrix((np.ones(n), (np.arange(n), inv)))
    Mu = (diags(1.0 / np.asarray(T.sum(axis=0)).ravel()) @ (T.T @ Xs)).tocsr()
    A = LinearOperator(Xs.shape, dtype=np.float64,
                       matvec=lambda v: Xs @ v - T @ (Mu @ v),
                       rmatvec=lambda u: Xs.T @ u - Mu.T @ (T.T @ u),
                       matmat=lambda V: Xs @ V - T @ (Mu @ V),
                       rmatmat=lambda U: Xs.T @ U - Mu.T @ (T.T @ U))
    v0 = PC.rng(*stream).standard_normal(min(Xs.shape))
    _, s, Vt = svds(A, k=K_SVD, v0=v0, solver='arpack')
    o = np.argsort(-s)
    s, Vt = s[o], Vt[o]
    Vt = Vt * np.sign(Vt[np.arange(K_SVD), np.argmax(np.abs(Vt), axis=1)])[:, None]
    P = np.asarray(Xs @ Vt.T) - np.asarray(T @ (Mu @ Vt.T))
    tot = float(Xs.multiply(Xs).sum() - (np.asarray(T.sum(axis=0)).ravel() * np.asarray(Mu.multiply(Mu).sum(axis=1)).ravel()).sum())
    return P, {'share_of_within_task_ss': round(float((s ** 2).sum() / tot), 4), 'singular_values': s[:5].round(4).tolist()}


def stand_ins(pop):
    """REHEARSAL: label-free stand-ins with the real shapes (C 8 columns, a sparse posmean_code)."""
    r = PC.rng(99, 1)
    n = len(pop)
    C = r.normal(size=(n, 8))
    C[:, 7] = [len(p['code']) for p in pop]
    nnz = 400
    ind = np.sort(r.choice(26 * 16384, size=(n, nnz)), axis=1)
    Xs = csr_matrix((r.exponential(size=n * nnz), ind.ravel(), np.arange(0, n * nnz + 1, nnz)), shape=(n, 26 * 16384))
    Xs.sum_duplicates()
    return C, Xs


def _init(Xs_list, y, groups):
    _G['X'], _G['y'], _G['g'] = Xs_list, y, groups


def _perm(k):
    yp = PC.permute_within(_G['y'], _G['g'], PC.rng(31, k))
    return [float((PC.task_aucs(PC.oof_rows(X, _G['g'], yp, PERM_BASE), _G['g'], yp) - 0.5).mean()) for X in _G['X']]


def main():
    t_start = time.time()
    guard = None if REHEARSAL else PC.guard()
    pop = PC.population()
    groups = np.array([r['task'] for r in pop])
    y = np.array([r['y'] for r in pop])
    if REHEARSAL:
        y = PC.permute_within(y, groups, PC.rng(99, 2))        # no real association is read
        C, Xs = stand_ins(pop)
        meta, sha = {'rehearsal': True}, None
    else:
        z, meta = load_extract(pop, guard)
        C = c_matrix(pop, z)
        Xs = csr_matrix((z['pm_code_data'].astype(np.float64), z['pm_code_indices'], z['pm_code_indptr']),
                        shape=(len(pop), int(z['shape'][1])))
        sha = meta['sha256']
    tasks = np.unique(groups)
    out = {'part': '2', 'rehearsal': REHEARSAL, 'guard': guard, 'extract_sha256': sha,
           'population': {'tasks': len(tasks), 'programs': len(pop), 'holds': int(y.sum()), 'flips': int((1 - y).sum())}}
    log('population', out['population'])

    Cc = centre(C, groups)
    P, svd_info = within_svd(Xs, groups, (33,))
    out['svd'] = svd_info
    CP = np.hstack([Cc, P])
    log('signals built', svd_info, f'{time.time() - t_start:.0f} s')

    draws = PC.rng(32).integers(0, len(tasks), size=(PC.N_BOOT, len(tasks)))
    s = {name: PC.oof_rows(X, groups, y, PERM_BASE) for name, X in (('CP', CP), ('C', Cc), ('P', P))}
    d = {name: PC.task_aucs(v, groups, y) - 0.5 for name, v in s.items()}
    log('observed', {k: round(float(v.mean()), 4) for k, v in d.items()}, f'{time.time() - t_start:.0f} s')

    # permutation null for Q1 (C + P_code) and S1 (C), the same permuted label sets
    with Pool(WORKERS, initializer=_init, initargs=([CP, Cc], y, groups)) as pool:
        null = np.array(pool.map(_perm, range(K_PERM), chunksize=1))
    log('permutations done', f'{time.time() - t_start:.0f} s')

    def perm_p(obs, col):
        return float((1 + (null[:, col] >= obs).sum()) / (K_PERM + 1))

    q1 = PC.summarise(d['CP'], draws)
    q1.update({'p_permutation': perm_p(d['CP'].mean(), 0), 'K': K_PERM,
               'p_sign_flip_reference': PC.sign_flip_greater(d['CP'], (36,))})
    q1['rejected'] = q1['p_permutation'] < 0.05
    out['Q1'] = q1
    s1 = PC.summarise(d['C'], draws)
    s1['p_permutation'] = perm_p(d['C'].mean(), 1)
    inc = d['CP'] - d['C']
    s2 = PC.summarise(inc, draws)
    s2['p_sign_flip'] = PC.sign_flip_greater(inc, (35,))
    if q1['rejected']:
        ps = {'S1': s1['p_permutation'], 'S2': s2['p_sign_flip']}
        order = sorted(ps, key=ps.get)
        rej = {k: False for k in ps}
        if ps[order[0]] <= 0.025:
            rej[order[0]] = True
            rej[order[1]] = ps[order[1]] <= 0.05
        s1['rejected'], s2['rejected'] = rej['S1'], rej['S2']
    s1['tested'] = s2['tested'] = bool(q1['rejected'])
    out['S1_C'], out['S2_increment_P_over_C'] = s1, s2
    out['permutation_null'] = {name: {'mean': round(float(null[:, i].mean()), 4),
                                      'q95': round(float(np.percentile(null[:, i], 95)), 4)}
                               for i, name in enumerate(('CP', 'C'))}

    desc = {'P_code_alone': PC.summarise(d['P'], draws)}
    for name, X in (('CP', CP), ('C', Cc)):
        sl = PC.oof_rows(X, groups, y, PERM_BASE, learner='logit')
        desc[f'{name}_logistic'] = PC.summarise(PC.task_aucs(sl, groups, y) - 0.5, draws)
    raw = {n: C[:, j] for j, n in enumerate(C_NAMES)}
    if not REHEARSAL:
        for j, n in enumerate(('f0_active_features', 'f3_top_salient_prob', 'f4_salient_entropy')):
            raw[n] = z['f'][:, j]
    desc['single_columns'] = {n: PC.summarise(PC.task_aucs(v, groups, y) - 0.5, draws) for n, v in raw.items()}
    keep = ~np.isin(tasks, PC.AUDIT_ARTEFACT_TASKS)
    dk = d['CP'][keep]
    desc['Q1_without_audit_artefact_tasks'] = dict(
        PC.summarise(dk, PC.rng(37).integers(0, len(dk), size=(PC.N_BOOT, len(dk)))),
        p_sign_flip=PC.sign_flip_greater(dk, (38,)))
    h = np.array([(y[groups == t] == 1).sum() for t in tasks])
    f = np.array([(y[groups == t] == 0).sum() for t in tasks])
    big = (h >= 2) & (f >= 2)
    desc['Q1_tasks_2plus_each'] = PC.summarise(d['CP'][big], PC.rng(39).integers(0, big.sum(), size=(PC.N_BOOT, big.sum())))
    pairs = h * f
    desc['pooled_pairs_auc'] = {name: round(float(((d[name] + 0.5) * pairs).sum() / pairs.sum()), 4) for name in d}
    out['descriptive'] = desc
    out['seconds'] = round(time.time() - t_start, 1)

    dest = os.path.join(PC.WORK, 'rehearsal') if REHEARSAL else PC.OUT
    os.makedirs(dest, exist_ok=True)
    with gzip.open(os.path.join(dest, 'scores.jsonl.gz'), 'wt') as fh:
        for i, r in enumerate(pop):
            fh.write(json.dumps({'id': r['id'], 'task': r['task'], 'y': int(y[i]), 's_CP': s['CP'][i],
                                 's_C': s['C'][i], 's_P': s['P'][i]}) + '\n')
    np.save(os.path.join(dest, 'permutation_null.npy'), null)
    with open(os.path.join(dest, 'analysis.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    log(json.dumps({k: v for k, v in out.items() if k != 'guard'}, indent=1))


if __name__ == '__main__':
    main()
