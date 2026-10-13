"""Phase 6's primary statistic, as fixed in PREREGISTRATION_OWNFIX.md §3.

A conditional logit for matched pairs: one row per pair, the outcome "R is the
passing member" (always 1 as observed), no intercept, the covariates being
R - C differences. Fitted with Firth's penalty on the ridge-penalised
information,
    l(b) - (lam/2) sum_{j in P} b_j^2 + 1/2 log det(X'WX + lam P),
by the modified-score Newton iteration. The test of the graph term is a wild
score bootstrap over problems with leverage-adjusted (CR2, or CR3) scores.
"""
import numpy as np

LAM = 1.0
MEMBER = ('code_logprob', 'edit_logprob', 'min_logprob', 'mean_entropy', 'site_entropy',
          'orig_logprob', 'tokens_to_end')
TOKENS = ('<', '<=', '>', '>=', '==', '!=', '+', '-', '*', '/', '%', '+=', '-=', '*=', '/=', '%=',
          '&&', '||', 'true', 'false', '0', '1', '2', '..', '..=')
CONTEXTS = ('if', 'while', 'for', 'match', 'return', 'let', 'other')


def baseline(rows_r, rows_c, member=MEMBER, extra=None):
    """b of §3, R minus C, one row per pair. rows_*: dicts with 'surprise',
    'edit' and 'context'. member: the member-level quantities available (Phase 5
    lacks the site terms). extra: optional (n, m) array of further R - C columns,
    unpenalised (features 0, 3, 4 in the secondary). Returns (B, pen, names,
    dropped)."""
    val = lambda rows, k: np.array([float(r['surprise'][k]) for r in rows])
    cols, names, pen = [], [], []
    d = val(rows_r, 'code_logprob') - val(rows_c, 'code_logprob')
    cont = [('d_code_logprob', d), ('sign_code_logprob', np.sign(d)), ('d_abs_code_logprob', d * np.abs(d))]
    for k in member[1:]:
        cont.append((f'd_{k}', val(rows_r, k) - val(rows_c, k)))
    if extra is not None:
        cont += [(f'extra_{j}', extra[:, j]) for j in range(extra.shape[1])]
    # drop zero columns, then exact linear combinations among the continuous ones, last first
    dropped = [n for n, c in cont if not np.any(c != 0)]
    cont = [(n, c) for n, c in cont if np.any(c != 0)]
    while cont:
        M = np.column_stack([c for _, c in cont])
        if np.linalg.matrix_rank(M) == M.shape[1]:
            break
        for i in range(len(cont) - 1, -1, -1):
            rest = [c for j, (_, c) in enumerate(cont) if j != i]
            if rest and np.linalg.matrix_rank(np.column_stack(rest)) == np.linalg.matrix_rank(M):
                dropped.append(cont[i][0])
                cont.pop(i)
                break
    for n, c in cont:
        cols.append(c); names.append(n); pen.append(False)
    # squares of the member-level quantities, centred and scaled over all members
    for k in member:
        both = np.r_[val(rows_r, k), val(rows_c, k)]
        sd = both.std()
        if sd == 0:
            dropped.append(f'sq_{k}')
            continue
        zr, zc = (val(rows_r, k) - both.mean()) / sd, (val(rows_c, k) - both.mean()) / sd
        c = zr ** 2 - zc ** 2
        if np.any(c != 0):
            cols.append(c); names.append(f'sq_{k}'); pen.append(True)
        else:
            dropped.append(f'sq_{k}')
    # one-hot differences, all levels kept
    for block, levels, get in (('from', TOKENS, lambda r: r['edit']['from']),
                               ('to', TOKENS, lambda r: r['edit']['to']),
                               ('context', CONTEXTS, lambda r: r['context'])):
        for lv in levels:
            c = np.array([float(get(a) == lv) - float(get(b) == lv) for a, b in zip(rows_r, rows_c)])
            if np.any(c != 0):
                cols.append(c); names.append(f'{block}={lv}'); pen.append(True)
            else:
                dropped.append(f'{block}={lv}')
    return np.column_stack(cols), np.array(pen), names, dropped


def graph_z(nodes_r, nodes_c, c):
    """z of §3: the difference in log(1 + pruned feature nodes), divided by c."""
    return (np.log1p(np.asarray(nodes_r, float)) - np.log1p(np.asarray(nodes_c, float))) / c


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit(X, pen, lam=LAM, maxit=200, tol=1e-9):
    """Penalised (Firth + ridge on columns where pen is True) conditional logit
    with all outcomes 1. Returns beta, p, W-weights and the information matrix."""
    n, k = X.shape
    P = np.diag(pen.astype(float)) * lam
    beta = np.zeros(k)

    def objective(b):
        eta = X @ b
        p = _sig(eta)
        w = p * (1 - p)
        info = X.T @ (X * w[:, None]) + P
        sign, logdet = np.linalg.slogdet(info)
        if sign <= 0:
            return -np.inf
        return np.sum(-np.logaddexp(0, -eta)) - 0.5 * lam * np.sum(b[pen] ** 2) + 0.5 * logdet

    obj = objective(beta)
    for _ in range(maxit):
        p = _sig(X @ beta)
        w = p * (1 - p)
        info = X.T @ (X * w[:, None]) + P
        inv = np.linalg.inv(info)
        h = np.einsum('ij,jk,ik->i', X * np.sqrt(w)[:, None], inv, X * np.sqrt(w)[:, None])
        score = X.T @ (1 - p + h * (0.5 - p)) - P @ beta
        step = inv @ score
        t = 1.0
        while t > 1e-6:
            new = beta + t * step
            o = objective(new)
            if o >= obj - 1e-12:
                break
            t /= 2
        beta, delta = new, abs(o - obj)
        obj = o
        if delta < tol and np.max(np.abs(t * step)) < 1e-7:
            break
    p = _sig(X @ beta)
    w = p * (1 - p)
    return beta, p, w, X.T @ (X * w[:, None]) + P


def _inv_sqrt(M, power=-0.5):
    vals, vecs = np.linalg.eigh((M + M.T) / 2)
    vals = np.maximum(vals, 1e-8)
    return (vecs * vals ** power) @ vecs.T


def score_test(z, B, pen, groups, rng, n_boot=9999, adjust='CR2', lam=LAM):
    """Wild score bootstrap over problems for the coefficient of z, given the
    baseline columns B (pen marks their ridge-penalised columns). Returns
    (S_U, p): S_U = sum of the problem scores (its sign is the direction)."""
    gamma, _, _, _ = fit(B, pen, lam)
    eta = B @ gamma
    p, q = _sig(eta), _sig(-eta)                              # q = 1 - p without cancellation
    w = p * q
    P = np.diag(pen.astype(float)) * lam
    BtWB = B.T @ (B * w[:, None]) + P
    inv = np.linalg.inv(BtWB)
    zt = z - B @ (inv @ (B.T @ (w * z)))                    # z residualised on b
    if np.linalg.norm(np.sqrt(w) * zt) < 1e-8 * max(np.linalg.norm(np.sqrt(w) * z), 1e-300):
        return float(np.sum(zt * q)), float('nan')            # z lies in the span of b: not testable
    sw = np.sqrt(w)
    e = np.exp(-eta / 2)                                      # (1 - p) / sqrt(w), stably
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    U, Ua = np.zeros(len(uniq)), np.zeros(len(uniq))
    for gi, g in enumerate(uniq):
        idx = np.flatnonzero(groups == g)
        U[gi] = np.sum(zt[idx] * q[idx])
        Wb = B[idx] * sw[idx, None]
        Hgg = Wb @ inv @ Wb.T
        A = _inv_sqrt(np.eye(len(idx)) - Hgg, -0.5 if adjust == 'CR2' else -1.0)
        Ua[gi] = (zt[idx] * sw[idx]) @ A @ e[idx]
    S = U.sum()
    eps = rng.choice([-1.0, 1.0], size=(n_boot, len(uniq)))
    boot = eps @ Ua
    if not (np.isfinite(S) and np.isfinite(boot).all()):
        raise FloatingPointError("non-finite score statistic")
    pval = (1 + np.sum(np.abs(boot) >= abs(S))) / (1 + n_boot)
    return float(S), float(pval)


def cr2_interval(z, B, pen, groups, lam=LAM, level=0.95):
    """beta_G from the full penalised fit, with a CR2 sandwich interval
    (t with G - 1 degrees of freedom). Descriptive."""
    from scipy import stats
    X = np.column_stack([z, B])
    penx = np.r_[False, pen]
    beta, _, _, _ = fit(X, penx, lam)
    eta = X @ beta
    p, q = _sig(eta), _sig(-eta)
    w = p * q
    inv = np.linalg.inv(X.T @ (X * w[:, None]) + np.diag(penx.astype(float)) * lam)
    sw = np.sqrt(w)
    xmx = np.einsum('ij,jk,ik->i', X, inv, X)
    r_sw = np.exp(-eta / 2) + sw * xmx * (0.5 - p)        # (1 - p + h(1/2 - p)) / sqrt(w), stably
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in uniq:
        idx = np.flatnonzero(groups == g)
        Wx = X[idx] * sw[idx, None]
        A = _inv_sqrt(np.eye(len(idx)) - Wx @ inv @ Wx.T, -0.5)
        s = Wx.T @ (A @ r_sw[idx])
        meat += np.outer(s, s)
    G = len(uniq)
    V = inv @ meat @ inv                                    # CR2: no G/(G-1) factor
    if not np.isfinite(V).all():
        raise FloatingPointError("non-finite sandwich")
    se = float(np.sqrt(V[0, 0]))
    t = stats.t.ppf(0.5 + level / 2, G - 1)
    return float(beta[0]), (float(beta[0] - t * se), float(beta[0] + t * se))
