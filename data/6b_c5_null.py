"""C5's null distribution, from refits on shuffled labels. PREREGISTRATION.md §4, §8 v4.

C5 correlates the two arms' per-problem scores. Version 2 tested it against a
null that permutes problems, which assumes a Rust score vector carries no
structure the Python one shares unless both learned the same labels. The blind
rehearsal refuted that before any real Rust label was used: refitting the Rust
model on shuffled labels, six times, gave rho against the real Python scores
of -0.425 to +0.388 (sd ~0.30) where the permutation null has sd ~0.12. A
label-free Rust score still ranks problems along whatever axis dominates its
features -- length, above all -- and the Python score shares that axis
(+0.50 with -max prefix chars).

So the decision p-value comes from here: refit the Rust population K times with
its labels permuted across its problems, correlate each out-of-fold score vector
with the fixed Python scores on the cross set, and ask how often a model that
learned nothing real about Rust correctness matches the observed rho.

  RUST_FEATURES  RUST_LABELS  SUBSET_FILE   the Rust population, exactly as in 5
  RUST_SCORES    the population's scores from 5_problem_level.py (checked here)
  PY_SCORES      CROSS_SET                  as in 6_cross_arm.py
  K              refits (100)               ALLOW_PARTIAL  rehearsal only

Before refitting it refits once on the real labels and demands the result equal
RUST_SCORES bit for bit, which proves the rows here are the rows there.
"""
import importlib.util
import json
import os
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('protocol', os.path.join(HERE, '5_problem_level.py'))
protocol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(protocol)

RUST_FEATURES = os.environ.get('RUST_FEATURES', 'gemma_mbpp_rust_features.jsonl')
RUST_LABELS = os.environ.get('RUST_LABELS', 'gemma_mbpp_rust_correctness.jsonl')
SUBSET_FILE = os.environ.get('SUBSET_FILE', 'subset_rust_strict.txt')
RUST_SCORES = os.environ.get('RUST_SCORES', 'results/rust/scores_rust_strict.json')
PY_SCORES = os.environ.get('PY_SCORES', 'results/python/scores_python_240.json')
CROSS_SET = os.environ.get('CROSS_SET', 'subset_cross_primary.txt')
K = int(os.environ.get('K', 100))
SEED = int(os.environ.get('SEED', 0))
ALLOW_PARTIAL = os.environ.get('ALLOW_PARTIAL', '0') == '1'


def read_ids(path):
    with open(path, encoding='utf-8') as f:
        return {int(l) for l in f if l.strip() and not l.startswith('#')}


def population():
    """The rows 5_problem_level.py fits on, prepared the same way."""
    rows = protocol.read_jsonl(RUST_FEATURES)
    rows.sort(key=lambda r: r.get('row_index', 0))
    labels = {int(r['task_id']): int(bool(r['passes']))
              for r in protocol.read_jsonl(RUST_LABELS)}
    if SUBSET_FILE:
        wanted = read_ids(SUBSET_FILE)
        rows = [r for r in rows if int(r['expr_id']) in wanted]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(protocol.MODEL_NAME)
    keep = [len(tok(r['original_expression'])['input_ids']) <= protocol.TOKEN_CAP
            for r in rows]
    rows = [r for r, k in zip(rows, keep) if k]
    X = np.array([r['feature'] for r in rows], dtype=float)
    g = np.array([int(r['expr_id']) for r in rows])
    problems = np.array(sorted(set(g.tolist())))
    return X, g, problems, labels


def main():
    t0 = time.time()
    X, g, problems, labels = population()
    y_prob = np.array([labels[int(p)] for p in problems])
    print(f"population: {len(problems)} problems, {len(g)} rows "
          f"({SUBSET_FILE or 'whole file'})")

    with open(RUST_SCORES, encoding='utf-8') as f:
        saved = {int(p['task_id']): p['score'] for p in json.load(f)['problems']}
    sc = protocol.oof_problem_scores(X, g, np.array([labels[int(t)] for t in g]), problems)
    same = (set(saved) == set(problems.tolist()) and
            all(saved[int(p)] == float(s) for p, s in zip(problems, sc)))
    if not same:
        raise SystemExit(f"refit on the real labels does not reproduce {RUST_SCORES}; "
                         f"the rows here are not the rows there")
    print(f"  refit on the real labels reproduces {RUST_SCORES} bit for bit")

    with open(PY_SCORES, encoding='utf-8') as f:
        py = {int(p['task_id']): p['score'] for p in json.load(f)['problems']}
    cross = sorted(read_ids(CROSS_SET))
    have = set(problems.tolist()) & set(py)
    if not set(cross) <= have:
        if not ALLOW_PARTIAL:
            raise SystemExit(f"{len(set(cross) - have)} cross-set problems lack a score")
        cross = [t for t in cross if t in have]
        print(f"  !!! ALLOW_PARTIAL: cross set cut to {len(cross)} -- rehearsal only")
    pos = {int(p): i for i, p in enumerate(problems)}
    idx = np.array([pos[t] for t in cross])
    s_py = np.array([py[t] for t in cross])
    rho = float(stats.spearmanr(s_py, sc[idx])[0])

    rng = np.random.default_rng([SEED, 100])
    null = []
    for k in range(K):
        shuffled = dict(zip(problems.tolist(), rng.permutation(y_prob)))
        y_row = np.array([shuffled[int(t)] for t in g])
        null.append(float(stats.spearmanr(s_py, protocol.oof_problem_scores(
            X, g, y_row, problems)[idx])[0]))
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{K} refits, {time.time() - t0:.0f}s")
    null = np.array(null)
    p = (1 + int((null >= rho).sum())) / (1 + K)

    print(f"\n=== C5 against label-free refits ({CROSS_SET}, {len(cross)} problems) ===")
    print(f"  observed rho          {rho:+.4f}")
    print(f"  label-free refits     mean {null.mean():+.4f}  sd {null.std(ddof=1):.4f}  "
          f"5-95% [{np.percentile(null, 5):+.4f}, {np.percentile(null, 95):+.4f}]  (K={K})")
    print(f"  one-sided p = {p:.4f}  (H1: rho exceeds what a model that learned nothing "
          f"real about Rust correctness produces; floor 1/{K + 1})")
    print(f"  this p is C5's decision statistic; 6_cross_arm.py's permutation p assumes "
          f"exchangeability, which the rehearsal refuted")


if __name__ == '__main__':
    main()
