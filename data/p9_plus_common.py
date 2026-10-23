"""Phase 9, part ②: shared pieces (PREREGISTRATION_P9_PLUS.md fixes what they compute).

The population (distinct passing programs of the mixed tasks, with their MBPP+ verdicts), the texts the
forward pass scores, the guard, the out-of-fold learner on rows, the within-task AUROC and the tests.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import p9_common as C  # noqa: E402

OUT = os.environ.get('PLUS_OUT', os.path.join(HERE, 'results', 'p9_plus'))
WORK = os.environ.get('PLUS_WORK', '/home/user1/p9_work/plus')
ROOT = 20261023
N_BOOT = 5000
SCRIPTS = ('p9_plus_common.py', 'p9_plus_extract.py', 'p9_plus_analysis.py', 'p9_c_audit.py', 'p9_common.py',
           'p9_m1_extract.py', 'p9_m1_common.py')
# §2: mixed tasks, programs, holds, flips (asserted whenever the population is built)
COUNTS = (44, 908, 474, 434)
AUDIT_ARTEFACT_TASKS = (113, 269, 451, 459, 472, 598, 794, 800)     # §8: the sampled flip was an artefact


def rng(*k):
    return np.random.default_rng([ROOT, *k])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def guard():
    """Refuse unless every PREREGISTRATION_P9*.md and the scripts of this part are committed and clean."""
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


def population():
    """The mixed tasks' distinct programs (registered label set, mode O, timeouts left out), ordered by task and
    then by first member (phase, sample). Each: id, task, members, code (as first sampled), prose (the task's
    pool text), text = prose + ' ' + code, and y (1 holds, 0 flips). The extraction does not read y."""
    import p9_c_audit as A
    cases = json.load(open(os.path.join(C.OUT_DIR, 'a3_cases.json')))['tasks']
    small = {int(t) for t, v in cases.items() if v.get('n_cases') is not None and v['n_cases'] < A.MIN_CASES}
    excl = small | set(A.ARTEFACTS)
    per = defaultdict(list)
    for r in C.read_jsonl(os.path.join(C.OUT_DIR, 'a3_relabel_v2.jsonl.gz')):
        v = A.verdict(r['O'])
        if r['task_id'] not in excl and v in ('holds', 'flips'):
            per[r['task_id']].append((r, v))
    mixed = sorted(t for t, rs in per.items() if {v for _, v in rs} == {'holds', 'flips'})
    pool = C.read_jsonl(os.path.join(C.OUT_DIR, 'pool_samples.jsonl.gz'))
    code_of = {(int(r['task_id']), int(r['sample'])): r['generated_code'] for r in pool}
    code_of.update({(int(r['task_id']), -1): r['generated_code']
                    for r in C.read_jsonl(os.path.join(HERE, 'gemma_mbpp_rust_responses.jsonl'))})
    prose_of = defaultdict(set)
    for r in pool:
        prose_of[int(r['task_id'])].add(r['text'])
    out = []
    for t in mixed:
        if len(prose_of[t]) != 1:
            sys.exit(f'task {t}: {len(prose_of[t])} prose variants')
        prose = next(iter(prose_of[t]))
        rs = sorted(per[t], key=lambda rv: (rv[0]['members'][0][1], rv[0]['members'][0][0]))
        for n, (r, v) in enumerate(rs):
            s, _ = r['members'][0]
            code = code_of[(t, s)]
            if len({C.ws_key(code_of[(t, m)]) for m, _ in r['members']}) != 1:
                sys.exit(f'task {t}: the members of one program differ beyond whitespace')
            out.append({'id': f'{t}_{n:03d}', 'task': t, 'members': r['members'], 'code': code, 'prose': prose,
                        'text': prose + ' ' + code, 'y': 1 if v == 'holds' else 0})
    y = Counter(r['y'] for r in out)
    got = (len(mixed), len(out), y[1], y[0])
    if got != COUNTS:
        sys.exit(f'population {got}, not the registered {COUNTS}')
    return out


def oof_rows(X, groups, y, perm_base, rf_seed=42, n_repeats=10, n_folds=5, learner='rf'):
    """Out-of-fold probability of y = 1 per row, averaged over repeats: GroupKFold over tasks, the task-to-fold
    map permuted per repeat with default_rng(perm_base + repeat) (5_problem_level's scheme, on rows)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    import p9_m1_common as M
    tasks = np.unique(groups)
    idx = {int(t): i for i, t in enumerate(tasks)}
    acc = np.zeros(len(y))
    for rep in range(n_repeats):
        perm = np.random.default_rng(perm_base + rep).permutation(len(tasks))
        surrogate = np.array([perm[idx[int(g)]] for g in groups])
        p = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups=surrogate):
            if learner == 'rf':
                clf = RandomForestClassifier(n_estimators=100, random_state=rf_seed, n_jobs=1).fit(X[tr], y[tr])
                p[te] = clf.predict_proba(X[te])[:, 1]
            else:
                Xtr, Xte = M.normal_scores_fit(X[tr], X[te])
                clf = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr, y[tr])
                p[te] = clf.predict_proba(Xte)[:, 1]
        acc += p
    return acc / n_repeats


def task_aucs(s, groups, y):
    """Per task (sorted), the AUROC of s for holds (y = 1) above flips (y = 0); ties count one half."""
    out = []
    for t in np.unique(groups):
        m = groups == t
        pos, neg = s[m][y[m] == 1], s[m][y[m] == 0]
        d = pos[:, None] - neg[None, :]
        out.append(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)
    return np.array(out)


def permute_within(y, groups, r):
    """y permuted within each task (counts kept)."""
    out = y.copy()
    for t in np.unique(groups):
        m = np.where(groups == t)[0]
        out[m] = y[m][r.permutation(len(m))]
    return out


def sign_flip_greater(d, stream, n_flips=99999):
    """One-sided problem-level sign flip of mean(d) > 0: 99,999 flips plus the identity."""
    r = rng(*stream)
    obs = d.mean()
    s = r.choice([-1.0, 1.0], size=(n_flips, len(d)))
    return float((1 + ((s * d).mean(1) >= obs).sum()) / (n_flips + 1))


def summarise(d, draws):
    """Mean of per-task values with its bootstrap interval over tasks, the SD and the one-sided 80% MDE."""
    b = d[draws].mean(1)
    sd = float(d.std(ddof=1))
    return {'tasks': int(len(d)), 'mean': round(float(d.mean()), 4),
            'ci95': [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
            'sd': round(sd, 4), 'mde80_one_sided': round((1.6449 + 0.8416) * sd / np.sqrt(len(d)), 4)}
