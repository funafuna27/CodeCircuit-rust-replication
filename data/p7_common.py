"""Phase 7, shared library: the closing comparison on the retained pairs.
PREREGISTRATION_CLOSING.md §2-§3. Importing it prints nothing.

  loaders      Phase 5's 259 pairs and Phase 4's 98, their rows and stored features
  orientation  label-free work (p7_0, p7_2, p7_3) orders each pair's members by a
               rule that uses no label: the member whose generated code sorts first
               comes first ('canonical'). Only orientation() reads which member
               passed MBPP's asserts. Every pair keeps its place in the pair file,
               so the survivor-first order is Phases 4/5's own row order.
  blocks       E, S2+, O, C and the graph blocks; normal scores for the tests
  forest       a copy of w3_within_analysis.oof with its two seeds as parameters
  GCM          the clustered generalised covariance measure, its nuisances (p-hat,
               m-hat), the Rademacher max-|T| test and Romano-Wolf
  multiplicity weighted Holm for the primary pair, Holm for the secondary three
  streams      every random draw is rng(s, ...) = default_rng([20261014, s, ...])

data/1_code_correct.py is never imported, opened or run. p5_5_edit_baseline
imports m1_make_mutants, which loads 1_rust_correct.py, as p5_5 and p5_6 did.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_expit, logit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w3_within_analysis as w3  # noqa: E402

ROOT = 20261014
STREAMS = {1: 'problem bootstrap, Phase 5', 2: 'Rademacher signs, real labels',
           3: 'synthetic labels and their signs', 4: 'synthetic designs', 5: 'reserved',
           6: 'power', 7: 'null bands', 8: 'alternative folds and forest seeds',
           9: 'verification intervals, Phase 5', 10: 'verification intervals, Phase 4',
           11: 'N45 noise and the shadow block', 12: 'SVD start, Phase 5',
           13: 'SVD start, Phase 4', 14: 'bootstrap, Phase 4',
           15: 'fresh size draws for a version-2 fallback', 16: "S_U reproduction (its p is discarded)"}
NEVER = {0, 20261012, 20261005, 20261008, 20261001, 77}
assert ROOT not in NEVER

KEEP = {5: '/home/user1/matched_graph_keep', 4: '/home/user1/mutant_graph_keep'}
WORK = os.environ.get('WORK', '/home/user1/closing_work')
OUT = os.environ.get('OUT_DIR', os.path.join(HERE, 'results', 'closing'))
FILES = {5: {'rows': 'matched_rows.jsonl', 'feats': 'gemma_matched_features.jsonl',
             'pairs': 'matched_pairs.json'},
         4: {'rows': 'mutant_rows.jsonl', 'feats': 'gemma_mutant_features.jsonl',
             'pairs': 'mutant_pairs.json', 'logprob': 'mutant_logprob.jsonl'}}
C_REF = 0.004917333931110322                      # results/ownfix/reference.json 'c'
S2P_KEYS = ('code_logprob', 'edit_logprob', 'min_logprob', 'mean_entropy')

# graph columns (graph_dataset._extract_advanced_features)
G_CONST = (42, 43, 47, 48)                        # constant, or 2 x column 41 (43)
G_OUT = (3, 4)                                    # the final position's top probability, entropy
G_FREE = (0, 3, 4)                                # need no attribution
G43 = tuple(j for j in range(49) if j not in G_CONST + G_OUT)
GATTR = tuple(j for j in G43 if j != 0)
G45 = tuple(j for j in range(49) if j not in G_CONST)
GATTR46 = tuple(j for j in range(49) if j not in G_FREE)
NODES = 1                                         # pruned feature nodes


def rng(s, *k):
    if s not in STREAMS:
        raise ValueError(f"stream {s} is not registered")
    return np.random.default_rng([ROOT, s, *k])


def path(name):
    return os.path.join(HERE, name)


def read_jsonl(name):
    with open(path(name) if not os.path.isabs(name) else name, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(p, obj):
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, default=_jsonable)
    os.replace(tmp, p)


def _jsonable(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


def drop_cache(p):
    """Tell the kernel to drop p from the page cache (the graphs total ~320 GiB)."""
    fd = os.open(p, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def tokenizer():
    """Gemma's tokenizer from the local snapshot; by model name, HF_HUB_OFFLINE
    crashes in transformers' _patch_mistral_regex."""
    from transformers import AutoTokenizer
    snaps = glob.glob(os.path.expanduser(
        '~/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/*/'))
    if len(snaps) != 1:
        raise SystemExit(f"expected one tokenizer snapshot, found {snaps}")
    return AutoTokenizer.from_pretrained(snaps[0])


def edit_at(row):
    return len(row['prose']) + 1 + row['edit']['pos']


def edit_token(row, offs):
    """p5_2's rule: the token whose characters cover the edit's first character."""
    at = edit_at(row)
    return next(t for t in range(1, len(offs)) if offs[t][0] <= at < offs[t][1])


def edit_span(row, offs):
    """Tokens overlapping the edit's characters [at, at + max(len(from), len(to)))."""
    at = edit_at(row)
    w = max(len(row['edit']['from']), len(row['edit']['to']))
    return [t for t in range(1, len(offs)) if offs[t][0] < at + w and offs[t][1] > at]


# ---------------------------------------------------------------- loaders

def load_pairs(phase):
    """The analysed pairs, members in canonical order (no label is used)."""
    f = FILES[phase]
    rows_list = read_jsonl(f['rows'])
    rows = {int(r['task_id']): r for r in rows_list}
    row_index = {int(r['task_id']): k for k, r in enumerate(rows_list)}
    feats = {}
    for r in read_jsonl(f['feats']):
        e = int(r['expr_id'])
        if e in feats:
            raise SystemExit(f"two feature rows for {e}")
        feats[e] = r
    with open(path(f['pairs']), encoding='utf-8') as fh:
        pairs = json.load(fh)
    full = [p for p in pairs if p['pass_id'] in feats and p['fail_id'] in feats]
    dropped = [p['problem'] for p in pairs if not (p['pass_id'] in feats and p['fail_id'] in feats)]
    if phase == 5 and not (len(pairs) == 259 and not dropped):
        raise SystemExit(f"Phase 5: {len(pairs)} pairs, dropped {dropped}")
    if phase == 4 and not (len(full) == 98 and dropped == [633]):
        raise SystemExit(f"Phase 4: {len(full)} pairs, dropped {dropped}")
    ids = []
    for p in full:
        a, b = int(p['pass_id']), int(p['fail_id'])
        for i in (a, b):
            r = feats[i]
            if r['batch_size'] != 64 or r['original_expression'] != rows[i]['text'] \
                    or not np.isfinite(r['feature']).all() or len(r['feature']) != 49:
                raise SystemExit(f"feature row {i} fails a check")
        ca, cb = rows[a]['generated_code'], rows[b]['generated_code']
        if ca == cb:
            raise SystemExit(f"pair {a}/{b}: identical code")
        ids += [a, b] if ca < cb else [b, a]
    ids = np.array(ids)
    groups = np.array([rows[i]['problem'] for i in ids])
    ia, ib = np.arange(0, len(ids), 2), np.arange(1, len(ids), 2)
    if not (groups[ia] == groups[ib]).all():
        raise SystemExit("a pair spans two problems")
    tokens = np.array([rows[i]['tokens'] for i in ids])
    if not (tokens[ia] == tokens[ib]).all():
        raise SystemExit("a pair differs in token count")
    return {'phase': phase, 'ids': ids, 'rows': rows, 'feats': feats, 'row_index': row_index,
            'pairs': full, 'groups': groups, 'pprob': groups[ia],
            'problems': np.array(sorted(set(groups.tolist()))), 'n': len(full)}


def orientation(P):
    """+1 where the canonical first member is the survivor (passes MBPP's asserts),
    -1 where it is the killed one. The only place a label is read."""
    o = []
    for k, p in enumerate(P['pairs']):
        if P['rows'][p['pass_id']]['label'] != 1 or P['rows'][p['fail_id']]['label'] != 0:
            raise SystemExit(f"pair {k}: row labels disagree with the pair file")
        o.append(1 if P['ids'][2 * k] == p['pass_id'] else -1)
    return np.array(o)


def member_perm(o):
    """Row order that puts the member with o-orientation 'first' first: rows of
    pairs with o = -1 are swapped."""
    idx = np.arange(2 * len(o)).reshape(-1, 2)
    idx[o < 0] = idx[o < 0][:, ::-1]
    return idx.ravel()


def dpair(M):
    return M[0::2] - M[1::2]


def mpair(M):
    return (M[0::2] + M[1::2]) / 2


def centred(M):
    return M - np.repeat(mpair(M), 2, axis=0)


def live_levels(Z):
    """Pair means of Z's columns that vary over pairs (levels for the nuisances)."""
    lev = mpair(Z)
    keep = lev.std(axis=0) > 1e-12
    return lev[:, keep]


def normal_scores(M):
    """Each column replaced by the normal scores of its ranks over all members
    (ties averaged): label-free, orientation-free, tames heavy tails."""
    M = np.asarray(M, float)
    out = np.empty_like(M)
    n = len(M)
    for j in range(M.shape[1]):
        out[:, j] = stats.norm.ppf((stats.rankdata(M[:, j]) - 0.5) / n)
    return out


# ---------------------------------------------------------------- blocks

def base_blocks(P):
    """Member matrices in canonical order: E, S2P, O, C, C63, G49, f0 and the
    normal-scored test blocks of the graph."""
    from p5_5_edit_baseline import encode
    ids, rows, feats = P['ids'], P['rows'], P['feats']
    E = np.array([encode(rows[i]) for i in ids], float)
    G49 = np.array([feats[i]['feature'] for i in ids], float)
    chars = np.array([len(rows[i]['text']) for i in ids], float)
    if P['phase'] == 5:
        S2P = np.column_stack([np.array([[rows[i]['surprise'][k] for k in S2P_KEYS] for i in ids], float),
                               chars])
    else:
        lp = {int(r['task_id']): r for r in read_jsonl(FILES[4]['logprob'])}
        S2P = np.column_stack([[lp[i]['code_logprob'] for i in ids],
                               [lp[i]['edit_logprob'] for i in ids], chars]).astype(float)
    O = np.column_stack([S2P, G49[:, 3], G49[:, 4]])
    B = {'E': E, 'S2P': S2P, 'O': O, 'C': np.column_stack([E, O]), 'C63': np.column_stack([E, S2P]),
         'G49': G49, 'GFREE': G49[:, list(G_FREE)], 'GATTR46': G49[:, list(GATTR46)],
         'f0': np.log1p(G49[:, [0]]),
         'G43n': normal_scores(G49[:, list(G43)]), 'GATTRn': normal_scores(G49[:, list(GATTR)]),
         'G45n': normal_scores(G49[:, list(G45)])}
    f1 = np.log1p(G49[:, NODES])
    B['z'] = (f1[0::2] - f1[1::2]) / C_REF                    # pair level, canonical
    noise = rng(11, P['phase']).standard_normal((P['n'], 45))
    N = np.empty((2 * P['n'], 45))
    N[0::2], N[1::2] = noise / 2, -noise / 2
    B['N45'] = N
    return B


def cplus_columns(P, tok):
    """tokens_to_end and the 7-level syntactic context per member, as
    p6_3b_reference.prep computes them (copied: prep is nested in main there)."""
    os.environ.setdefault('ROUND', '0')               # p6_3_match reads it at import
    from p6_3_match import context
    from p6_stats import CONTEXTS
    tte, ctx = [], []
    for i in P['ids']:
        r = P['rows'][i]
        offs = tok(r['text'], return_offsets_mapping=True)['offset_mapping']
        code = [t for t in range(1, len(offs)) if offs[t][0] >= len(r['prose'])]
        tm = edit_token(r, offs)
        tte.append(len(code) - 1 - code.index(tm))
        c = context(r['generated_code'], r['edit']['pos'])
        ctx.append([float(c == lv) for lv in CONTEXTS])
    return np.array(tte, float)[:, None], np.array(ctx, float)


# ---------------------------------------------------------------- forest

def folds(groups, problems, rep, perm_seed=None):
    """w3.oof's folds for repeat rep (member rows)."""
    idx = {int(p): i for i, p in enumerate(problems)}
    seed = 100 + rep if perm_seed is None else perm_seed(rep)
    perm = np.random.default_rng(seed).permutation(len(problems))
    sur = np.array([perm[idx[int(g)]] for g in groups])
    return list(GroupKFold(n_splits=w3.N_FOLDS).split(np.zeros(len(groups)), groups=sur))


def pair_folds(groups, problems, rep, perm_seed=None):
    """The same folds at the pair level (members 2k, 2k+1 form pair k)."""
    out = []
    for tr, te in folds(groups, problems, rep, perm_seed):
        tep = te[te % 2 == 0] // 2
        trp = tr[tr % 2 == 0] // 2
        if not (len(tep) * 2 == len(te) and len(trp) * 2 == len(tr)):
            raise RuntimeError("a fold splits a pair")
        out.append((trp, tep))
    return out


def oof_seeded(X, y, groups, problems, perm_seed=None, rf_seed=42, n_jobs=None):
    """w3_within_analysis.oof with its fold and forest seeds as parameters; the
    defaults (100 + rep, 42) reproduce it bit for bit (self_test)."""
    n_jobs = w3.N_JOBS if n_jobs is None else n_jobs
    acc = np.zeros(len(y))
    for rep in range(w3.N_REPEATS):
        p = np.zeros(len(y))
        for tr, te in folds(groups, problems, rep, perm_seed):
            clf = RandomForestClassifier(n_estimators=100, random_state=rf_seed, n_jobs=n_jobs)
            clf.fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        acc += p
    return acc / w3.N_REPEATS


def self_test(X, y, groups, problems):
    return bool(np.array_equal(oof_seeded(X, y, groups, problems), w3.oof(X, y, groups, problems)))


def correct(s):
    """Per pair, members first-then-second: 1 if the first scores higher, 1/2 if tied."""
    d = s[0::2] - s[1::2]
    return (d > 0) + 0.5 * (d == 0)


def boot_draws(pprob, g, n=5000):
    """p5_4.boot_problems' draws: whole problems with replacement."""
    probs = np.unique(pprob)
    members = {p: np.flatnonzero(pprob == p) for p in probs}
    return [np.concatenate([members[p] for p in g.choice(probs, len(probs))]) for _ in range(n)]


def boot_ci(v, draws):
    m = np.array([v[d].mean() for d in draws])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# ---------------------------------------------------------------- nuisances

def _pi_symmetric(s):
    return np.clip((s[0::2] + 1 - s[1::2]) / 2, 0.01, 0.99)


def lin_pi(dZ, lev, groups, problems):
    """Version 2's linear part: no-intercept logistic regression (C = 1) on
    standardised mirrored pairs, symmetrised, out of fold, averaged over repeats."""
    X = np.column_stack([dZ, lev]) if lev.shape[1] else dZ
    Xm = np.column_stack([-dZ, lev]) if lev.shape[1] else -dZ
    out = np.zeros(len(dZ))
    for rep in range(w3.N_REPEATS):
        for tr, te in pair_folds(groups, problems, rep):
            Xt = np.vstack([X[tr], Xm[tr]])
            sc = StandardScaler().fit(Xt)
            yt = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
            lr = LogisticRegression(C=1.0, fit_intercept=False, max_iter=5000).fit(sc.transform(Xt), yt)
            out[te] += (lr.predict_proba(sc.transform(X[te]))[:, 1] + 1
                        - lr.predict_proba(sc.transform(Xm[te]))[:, 1]) / 2
    return np.clip(out / w3.N_REPEATS, 0.01, 0.99)


def platt(pi, groups, problems):
    """Slope-only recalibration p = sigmoid(lambda * logit(pi)), lambda in
    [0.1, 10] fitted on the other folds of repeat 0 (the first member passes)."""
    li = logit(pi)
    ph = np.empty(len(pi))
    lams = []
    for tr, te in pair_folds(groups, problems, 0):
        r = minimize_scalar(lambda lam: -np.sum(log_expit(lam * li[tr])), bounds=(0.1, 10.0),
                            method='bounded')
        lams.append(float(r.x))
        ph[te] = expit(r.x * li[te])
    return ph, lams


def phat(Zc, lev, groups, problems, version=1, n_jobs=None, perm_seed=None, rf_seed=42):
    """P(the first member passes | Z), per pair. Zc: centred member rows of Z,
    passer first; lev: the pairs' levels. Phase 3's forest on [Zc, lev], out of
    fold, symmetrised, (version 2: averaged with lin_pi), slope-recalibrated."""
    X = np.column_stack([Zc, np.repeat(lev, 2, axis=0)]) if lev.shape[1] else Zc
    y = np.tile([1, 0], len(Zc) // 2)
    pi = _pi_symmetric(oof_seeded(X, y, groups, problems, perm_seed, rf_seed, n_jobs))
    if version == 2:
        pi = np.clip((pi + lin_pi(dpair(Zc), lev, groups, problems)) / 2, 0.01, 0.99)
    ph, lams = platt(pi, groups, problems)
    return ph, pi, lams


def mhat_column(dZ, lev, db, groups, problems):
    """E[dB_j | Z], label-free, per pair: a forest on mirrored training pairs
    {(x, db), (x-, -db)}, x = [dZ, lev], x- = [-dZ, lev], predicting
    (f(x) - f(x-)) / 2 out of fold, averaged over the 10 repeats. Also returns the
    odd part of a no-intercept ridge (lambda 1) on standardised mirrored data,
    which version 2 averages in."""
    X = np.column_stack([dZ, lev]) if lev.shape[1] else dZ
    Xm = np.column_stack([-dZ, lev]) if lev.shape[1] else -dZ
    rf_pred, lin_pred = np.zeros(len(dZ)), np.zeros(len(dZ))
    for rep in range(w3.N_REPEATS):
        for tr, te in pair_folds(groups, problems, rep):
            Xt, yt = np.vstack([X[tr], Xm[tr]]), np.r_[db[tr], -db[tr]]
            rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=1 / 3,
                                       random_state=42, n_jobs=1).fit(Xt, yt)
            rf_pred[te] += (rf.predict(X[te]) - rf.predict(Xm[te])) / 2
            sc = StandardScaler().fit(Xt)
            rg = Ridge(alpha=1.0, fit_intercept=False).fit(sc.transform(Xt), yt)
            lin_pred[te] += (rg.predict(sc.transform(X[te])) - rg.predict(sc.transform(Xm[te]))) / 2
    return rf_pred / w3.N_REPEATS, lin_pred / w3.N_REPEATS


# ---------------------------------------------------------------- the test

def problem_sums(R, pprob, problems):
    return np.stack([R[pprob == g].sum(axis=0) for g in problems])


def gcm(ph, dB, mh, pprob, problems, eps=None, rw=False):
    """Clustered GCM of H0: Y independent of B given Z. Pairs passer first, so
    Y = 1 and its residual is 1 - p-hat. Returns T per live column, S = max |T|,
    rho, and with eps (B x problems Rademacher signs) the max-|T| p-value and,
    with rw, Romano-Wolf step-down p-values."""
    R = (1 - ph)[:, None] * (dB - mh)
    U = problem_sums(R, pprob, problems)
    den = np.sqrt((U ** 2).sum(axis=0))
    ok = den > 0
    T = U[:, ok].sum(axis=0) / den[ok]
    res = {'cols': np.flatnonzero(ok).tolist(), 'dropped': np.flatnonzero(~ok).tolist(),
           'T': T.tolist(), 'S': float(np.max(np.abs(T)))}
    ry = 1 - ph
    rb = dB[:, ok] - mh[:, ok]
    res['rho'] = (R[:, ok].sum(axis=0) / np.sqrt(np.sum(ry ** 2) * np.sum(rb ** 2, axis=0))).tolist()
    if eps is not None:
        Ts = np.abs(eps @ U[:, ok]) / den[ok]
        res['p'] = float((1 + np.sum(Ts.max(axis=1) >= res['S'])) / (1 + len(eps)))
        if rw:
            res['rw_p'] = romano_wolf(T, Ts).tolist()
    return res


def gcm_fast(ph, dB, mh, pprob_idx, n_prob, eps):
    """gcm's S and p for the simulations (pprob_idx: problem index per pair)."""
    R = (1 - ph)[:, None] * (dB - mh)
    U = np.zeros((n_prob, R.shape[1]))
    np.add.at(U, pprob_idx, R)
    den = np.sqrt((U ** 2).sum(axis=0))
    ok = den > 0
    T = U[:, ok].sum(axis=0) / den[ok]
    S = float(np.max(np.abs(T)))
    Ts = np.abs(eps @ U[:, ok]) / den[ok]
    return T, S, float((1 + np.sum(Ts.max(axis=1) >= S)) / (1 + len(eps)))


def romano_wolf(T, Ts):
    """Step-down max-|T| adjusted p per column; Ts = |T*| (B x J)."""
    order = np.argsort(-np.abs(T))
    out, prev = np.empty(len(T)), 0.0
    for k, j in enumerate(order):
        m = Ts[:, order[k:]].max(axis=1)
        prev = max(prev, (1 + np.sum(m >= abs(T[j]))) / (1 + len(Ts)))
        out[j] = prev
    return out


def weighted_holm2(p1, p2, a1=0.035, a2=0.015, alpha=0.05):
    """Primary family: H1 at a1, H2 at a2; a rejection passes its level to the
    other, tested at alpha. A withdrawn test (p None) is never rejected."""
    r1 = p1 is not None and p1 <= a1
    r2 = p2 is not None and p2 <= a2
    if r1 and not r2:
        r2 = p2 is not None and p2 <= alpha
    elif r2 and not r1:
        r1 = p1 is not None and p1 <= alpha
    return bool(r1), bool(r2)


def holm(ps, alpha=0.05):
    """Holm over a list of p (None = withdrawn, not rejected, weight kept)."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: (ps[i] is None, ps[i] if ps[i] is not None else 2))
    rej = [False] * m
    for k, i in enumerate(order):
        if ps[i] is None or ps[i] > alpha / (m - k):
            break
        rej[i] = True
    return rej
