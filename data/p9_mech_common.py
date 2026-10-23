"""Phase 9, part ① at gate 3: shared pieces for PREREGISTRATION_P9_MECH.md (patching and graphs)."""
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUT = os.environ.get('MECH_OUT', os.path.join(HERE, 'results', 'p9_mech'))
WORK = os.environ.get('MECH_WORK', '/home/user1/p9_work/mech')
ROOT = 20261023
SPANS = os.path.join(OUT, 'spans.jsonl.gz')
SCRIPTS = ('p9_mech_common.py', 'p9_mech_spans.py', 'p9_mech_patch.py', 'p9_mech_graphs.py', 'p9_mech_analysis.py',
           'p9_b_analysis.py', 'p9_common.py', 'p9_m1_extract.py', 'p9_m1_common.py', 'graph_dataset.py',
           'igraph_features.py')
N_LAYERS = 26
READ_MIN_DIST = 20


def rng(*k):
    return np.random.default_rng([ROOT, *k])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def guard():
    """Refuse unless every PREREGISTRATION_P9*.md and the scripts this part relies on are committed and clean."""
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


def g_items(items):
    """G's items: the first primary item of each problem, in b_items.jsonl.gz order."""
    first = {}
    for it in items:
        if it['role'] == 'primary':
            first.setdefault(it['key'][0], it)
    return [first[p] for p in sorted(first)]


def item_id(it):
    return f"{it['key'][0]}_{it['key'][1]}_{it['x']['name']}_{it['y']['name']}_{it['q_place']}"


def hier(rows, val):
    """items -> (source, x) -> source -> problem, equal weights (part ①'s aggregation)."""
    from collections import defaultdict
    sx = defaultdict(list)
    for r in rows:
        v = val(r)
        if v is not None:
            sx[(r['problem'], r['source'], r['x'])].append(v)
    src = defaultdict(list)
    for (p, s, _), v in sx.items():
        src[(p, s)].append(np.mean(v))
    prob = defaultdict(list)
    for (p, _), v in src.items():
        prob[p].append(np.mean(v))
    return {p: float(np.mean(v)) for p, v in prob.items()}


def test_greater(per_problem, stream, n_flip=99999, n_boot=5000):
    """One-sided problem-level sign flip, H1: mean > 0; percentile bootstrap interval over problems."""
    v = np.array([per_problem[k] for k in sorted(per_problem)])
    if len(v) < 2:
        return {'problems': len(v)}
    flips = rng(stream).choice([-1.0, 1.0], size=(n_flip, len(v)))
    obs = v.mean()
    p = (1 + np.sum((flips * v).mean(1) >= obs)) / (n_flip + 1)
    rb = rng(stream + 1)
    boots = np.array([v[rb.integers(len(v), size=len(v))].mean() for _ in range(n_boot)])
    return {'problems': int(len(v)), 'mean': round(float(obs), 4),
            'ci95': [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            'p_one_sided_greater': round(float(p), 5), 'sd': round(float(v.std(ddof=1)), 4),
            'mde80_one_sided': round(float((1.645 + 0.842) * v.std(ddof=1) / np.sqrt(len(v))), 4)}
