"""Phase 9, part ③ (M1): shared pieces. PREREGISTRATION_P9_M1.md fixes what they compute.

Populations, labels and stored G (as Phase 2), the out-of-fold forest (5_problem_level's, with its seeds
as arguments), the problem draws, the synthetic-label null bands (designs L and F) and the reading table.
"""
import glob
import hashlib
import importlib
import json
import os
import subprocess
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
WORK = os.environ.get('M1_WORK', '/home/user1/p9_work/m1')
OUT = os.environ.get('M1_OUT', os.path.join(HERE, 'results', 'p9_m1'))
ROOT = 20261022
TOKEN_CAP = 140
N_BOOT = 5000
SCRIPTS = ('p9_m1_common.py', 'p9_m1_extract.py', 'p9_m1_gates.py', 'p9_m1_analysis.py', 'p9_common.py',
           '5_problem_level.py')
# §2: rows, problems, positive per population (checked whenever a population is loaded)
POP_COUNTS = {'rust_strict': (1654, 206, 95), 'rust_loose': (1734, 214, 99),
              'python_240': (1137, 240, 114), 'python_256': (1337, 256, 114)}
G_FREE = (0, 3, 4)

ARMS = {
    'rust': {'features': 'gemma_mbpp_rust_features.jsonl', 'labels': 'gemma_mbpp_rust_correctness.jsonl',
             'rows': 'gemma_mbpp_rust_incremental_capped.jsonl'},
    'python': {'features': 'gemma_mbpp_features.jsonl', 'labels': 'gemma_mbpp_python_correctness.jsonl',
               'rows': 'gemma_mbpp_incremental.jsonl'},
}
POPS = {'rust_strict': ('rust', 'subset_rust_strict.txt'), 'rust_loose': ('rust', None),
        'python_240': ('python', 'subset_python_parses.txt'), 'python_256': ('python', None)}
POP_KEY = {'rust_strict': 0, 'rust_loose': 1, 'python_256': 2, 'python_240': 3}
# Phase 2's AUROCs of stored G (all rows, step 0): gate R
GATE_R = {'rust_strict': (0.7887, 0.6540), 'rust_loose': (0.7953, 0.6732),
          'python_240': (0.7091, 0.6928), 'python_256': (0.7269, 0.6931)}


def rng(*k):
    return np.random.default_rng([ROOT, *k])


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def guard():
    """Refuse unless every PREREGISTRATION_P9*.md and the M1 scripts are committed and unchanged."""
    files = sorted(glob.glob(os.path.join(REPO, 'PREREGISTRATION_P9*.md'))) + [os.path.join(HERE, s) for s in SCRIPTS]
    out = {}
    for f in files:
        rel = os.path.relpath(f, REPO)
        logged = subprocess.run(['git', '-C', REPO, 'log', '-1', '--format=%H', '--', rel],
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', REPO, 'status', '--porcelain', '--', rel],
                               capture_output=True, text=True).stdout.strip()
        if not logged or dirty:
            sys.exit(f'refused: {rel} is not committed or has uncommitted changes')
        out[rel] = sha256(f)[:16]
    out['HEAD'] = subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    return out


def load_extract(role, current_guard=None):
    """An extraction, read once into memory; its sha256 and the hashes of the code that made it are checked
    against the record written at extraction time (and against the current code, if given)."""
    path = os.path.join(WORK, f'extract_{role}.npz')
    meta = json.load(open(path.replace('.npz', '.json')))
    if sha256(path) != meta['sha256']:
        sys.exit(f'{path}: sha256 differs from the one recorded at extraction')
    if current_guard is not None:
        changed = [k for k in current_guard if k != 'HEAD' and (meta.get('guard') or {}).get(k) != current_guard[k]]
        if changed:
            sys.exit(f'{path}: made by other versions of {changed}')
    with np.load(path) as z:
        return {k: z[k] for k in z.files}, meta


def tokenizer():
    import p9_common
    return p9_common.tokenizer()


def load_arm(arm):
    """Feature rows of an arm in row_index order, with the prose split from the code."""
    a = ARMS[arm]
    feats = read_jsonl(os.path.join(HERE, a['features']))
    feats.sort(key=lambda r: r.get('row_index', 0))
    src = read_jsonl(os.path.join(HERE, a['rows']))
    out = []
    for r in feats:
        s = src[r['row_index']]
        text = r['original_expression']
        if s['text'] != text or int(s['task_id']) != int(r['expr_id']):
            raise SystemExit(f"{arm} row {r['row_index']}: feature row and source row disagree")
        code = s['generated_code']
        prose = text[:len(text) - len(code) - 1]
        if prose + ' ' + code != text:
            raise SystemExit(f"{arm} row {r['row_index']}: text is not prose + ' ' + code")
        out.append({'row_index': r['row_index'], 'task_id': int(r['expr_id']), 'step': int(r['step_number']),
                    'text': text, 'prose': prose, 'G': r['feature']})
    return out


def load_pop(pop, tok=None, arm_rows=None):
    """A population as Phase 2 analysed it: subset, then the 140-token cap (BOS counted), row_index order."""
    arm, sub = POPS[pop]
    rows = arm_rows if arm_rows is not None else load_arm(arm)
    keep = subset(sub)
    tok = tok or tokenizer()
    out = [r for r in rows if (keep is None or r['task_id'] in keep)
           and len(tok(r['text'])['input_ids']) <= TOKEN_CAP]
    lab = labels(arm)
    g = np.array([r['task_id'] for r in out])
    problems = np.array(sorted(set(g.tolist())))
    got = (len(out), len(problems), int(sum(lab[int(p)] for p in problems)))
    if got != POP_COUNTS[pop]:
        raise SystemExit(f'{pop}: rows/problems/positive {got}, not {POP_COUNTS[pop]}')
    return {'pop': pop, 'arm': arm, 'rows': out, 'groups': g, 'step': np.array([r['step'] for r in out]),
            'G': np.array([r['G'] for r in out], dtype=float), 'problems': problems,
            'y_prob': np.array([lab[int(p)] for p in problems]), 'y_row': np.array([lab[int(t)] for t in g]),
            'row_index': np.array([r['row_index'] for r in out])}


def labels(arm):
    return {int(r['task_id']): int(bool(r['passes'])) for r in read_jsonl(os.path.join(HERE, ARMS[arm]['labels']))}


def subset(name):
    if name is None:
        return None
    return {int(l) for l in open(os.path.join(HERE, name), encoding='utf-8') if l.strip() and not l.startswith('#')}


def oof(X, groups, y, problems, perm_base=100, rf_seed=42, n_jobs=1, n_repeats=10, n_folds=5, learner='rf'):
    """5_problem_level.oof_problem_scores with its seeds as arguments (equal to it at the defaults)."""
    idx = {int(p): i for i, p in enumerate(problems)}
    acc = np.zeros(len(problems))
    for rep in range(n_repeats):
        perm = np.random.default_rng(perm_base + rep).permutation(len(problems))
        surrogate = np.array([perm[idx[int(g)]] for g in groups])
        p = np.zeros(len(y), dtype=float)
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups=surrogate):
            if learner == 'rf':
                clf = RandomForestClassifier(n_estimators=100, random_state=rf_seed, n_jobs=n_jobs)
                clf.fit(X[tr], y[tr])
                p[te] = clf.predict_proba(X[te])[:, 1]
            else:
                Xtr, Xte = normal_scores_fit(X[tr], X[te])
                clf = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr, y[tr])
                p[te] = clf.predict_proba(Xte)[:, 1]
        acc += np.array([p[groups == q].mean() for q in problems])
    return acc / n_repeats


def normal_scores_fit(Xtr, Xte):
    """Rank normal scores fitted on the training rows, interpolated for the test rows; constant columns dropped."""
    from scipy.stats import norm, rankdata
    keep = Xtr.std(axis=0) > 0
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    n = len(Xtr)
    out_tr, out_te = np.empty_like(Xtr, dtype=float), np.empty_like(Xte, dtype=float)
    for j in range(Xtr.shape[1]):
        out_tr[:, j] = norm.ppf((rankdata(Xtr[:, j], method='average') - 0.5) / n)   # ties share a score
        u, inv = np.unique(Xtr[:, j], return_inverse=True)
        qu = np.bincount(inv, weights=out_tr[:, j]) / np.bincount(inv)
        out_te[:, j] = np.interp(Xte[:, j], u, qu)
    return out_tr, out_te


def auc(y, s):
    return float(roc_auc_score(y, s))


def draws(pop, n):
    return rng(81, POP_KEY[pop]).integers(0, n, size=(N_BOOT, n))


def boot(y, scores, D):
    """scores: list of problem-score vectors. Per draw, the AUROC of each (single-class draws dropped)."""
    out, dropped = [], 0
    for b in D:
        if y[b].min() == y[b].max():
            dropped += 1
            continue
        out.append([roc_auc_score(y[b], s[b]) for s in scores])
    return np.array(out), dropped


def summarise_level(y, s, D):
    a, dropped = boot(y, [s], D)
    return {'auroc': round(auc(y, s), 4), 'ci95': [round(float(np.percentile(a[:, 0], 2.5)), 4),
                                                   round(float(np.percentile(a[:, 0], 97.5)), 4)],
            'draws_dropped': dropped}


def summarise_diff(y, s1, s2, D):
    a, dropped = boot(y, [s1, s2], D)
    d = a[:, 0] - a[:, 1]
    return {'plugin': round(auc(y, s1) - auc(y, s2), 4), 'boot_mean': round(float(d.mean()), 4),
            'ci95': [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            'draws_dropped': dropped}


# ---------- synthetic-label null bands (PREREGISTRATION_P9_M1.md §6) ----------

def problem_summary(B, groups, problems):
    """Standardised per-problem means and maxima of B's columns (constant ones dropped)."""
    Z = np.array([np.r_[B[groups == p].mean(axis=0), B[groups == p].max(axis=0)] for p in problems])
    sd = Z.std(axis=0)
    Z = Z[:, sd > 0]
    return (Z - Z.mean(axis=0)) / Z.std(axis=0)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def expected_auc(eta, p):
    """eta's expected AUROC when y_i ~ Bernoulli(p_i) independently, as the ratio of the expected number of
    concordant positive-negative pairs (ties 1/2) to the expected number of such pairs (§6)."""
    u, inv = np.unique(eta, return_inverse=True)
    neg = np.bincount(inv, weights=1 - p, minlength=len(u))
    below = np.cumsum(neg) - neg                          # (1 - p_j) summed over strictly lower values
    s_below = below[inv] + 0.5 * (neg[inv] - (1 - p))     # ties (other than i itself) count 1/2
    num = np.sum(p * s_below)
    den = p.sum() * (1 - p).sum() - np.sum(p * (1 - p))
    return num / den


def calibrate_L(eta, target_auc, base_rate):
    """k, b with mean sigmoid(k eta + b) = base rate and expected AUROC = target."""
    def b_for(k):
        lo, hi = -50.0, 50.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if _sig(k * eta + mid).mean() < base_rate:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lo, hi = 0.0, 50.0
    for _ in range(60):
        k = (lo + hi) / 2
        if expected_auc(eta, _sig(k * eta + b_for(k))) < target_auc:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    return k, b_for(k)


def null_labels(design, r, ctx):
    """One synthetic problem-label vector drawn from the base block only."""
    g = rng(86 if design == 'L' else 87, ctx['key'], r)
    if design == 'L':
        w = g.standard_normal(ctx['Z'].shape[1])
        eta = ctx['Z'] @ w
        eta = (eta - eta.mean()) / eta.std()
        k, b = calibrate_L(eta, ctx['target_auc'], ctx['base_rate'])
        p = _sig(k * eta + b)
    else:
        p = ctx['p_hat']
    return (g.random(len(p)) < p).astype(int)


def band_task(args):
    """One null draw: refit [X, B] and B on synthetic labels; return the increment in AUROC."""
    design, r, ctx, Xfull, Xbase = args
    yp = null_labels(design, r, ctx)
    if yp.min() == yp.max():
        return None
    y_row = yp[ctx['row_problem_index']]
    s_full = oof(Xfull, ctx['groups'], y_row, ctx['problems'])
    s_base = oof(Xbase, ctx['groups'], y_row, ctx['problems'])
    return auc(yp, s_full) - auc(yp, s_base)


def band_context(B, groups, problems, y_prob, s_base_real, key):
    """Everything a null draw needs: designs L (random weights on B's problem summaries, calibrated to the
    real AUROC(B) and base rate) and F (B's out-of-fold score on the real labels, slope-recalibrated)."""
    idx = {int(p): i for i, p in enumerate(problems)}
    s = np.clip(s_base_real, 1e-4, 1 - 1e-4)
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(np.log(s / (1 - s))[:, None], y_prob)
    return {'key': key, 'Z': problem_summary(B, groups, problems), 'target_auc': auc(y_prob, s_base_real),
            'base_rate': float(y_prob.mean()), 'p_hat': lr.predict_proba(np.log(s / (1 - s))[:, None])[:, 1],
            'groups': groups, 'problems': problems,
            'row_problem_index': np.array([idx[int(g)] for g in groups])}


def run_band(pool, design, S, ctx, Xfull, Xbase):
    res = pool.map(band_task, [(design, r, ctx, Xfull, Xbase) for r in range(S)], chunksize=1)
    v = np.array([x for x in res if x is not None])
    return {'design': design, 'S': S, 'used': int(len(v)), 'c0': round(float(v.mean()), 4),
            'mc_se': round(float(v.std(ddof=1) / np.sqrt(len(v))), 4),
            'band95': [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}


# ---------- the reading table (PREREGISTRATION_P9_M1.md §6) ----------

def reading(diff, c0s, x='X', b='B', widen=1.28, sesoi=0.05):
    """diff: summarise_diff output; c0s: the null centres; x, b: the tested and base blocks' names. The
    bracket uses the widened interval, the smaller c0 for the upper bound and the larger for the lower."""
    d, (lo, hi) = diff['plugin'], diff['ci95']
    lo_w, hi_w = d - widen * (d - lo), d + widen * (hi - d)
    U = hi_w - min(c0s)
    Lb = lo_w - max(c0s)
    pt = round(d - float(np.mean(c0s)), 4)
    if U < 0:
        row = f'{x} + {b} ranks below {b} beyond the null centre: {x} costs this forest'
    elif Lb > 0 and U < sesoi:
        row = f'{x} adds a little ({pt:+.4f}), less than {sesoi}'
    elif Lb > 0:
        row = f'{x} adds ({pt:+.4f}); its size is not bounded below {sesoi}'
    elif U < sesoi:
        row = f'{x} adds less than {sesoi} (upper bound {U:+.4f}); not distinguishable from 0'
    else:
        row = f'not determined at this n: [{Lb:+.4f}, {U:+.4f}]'
    return {'point_corrected': pt, 'lower_bracket': round(Lb, 4), 'upper_bracket': round(U, 4),
            'widened_ci95': [round(lo_w, 4), round(hi_w, 4)], 'row': row}
