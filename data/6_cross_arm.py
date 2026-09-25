"""Cross-arm statistics and the evaluation-only subsets. Written before the Rust
scores exist; PREREGISTRATION.md §4 (C5) and §8 (version 4) fix what it computes.

Nothing here fits a model. Every number is an arm's own out-of-fold score
vector -- written by 5_problem_level.py with SCORES_OUT -- RESTRICTED to a set
of problems. That is how the Python comparator 0.7816 on the 129 shared
problems was computed, and refitting on 80 or 120 problems would confound the
subset with a smaller training set (see 5a_make_subsets.py).

  PY_SCORES      results/python/scores_python_240.json   primary: like-for-like fit
  RUST_SCORES    results/rust/scores_rust_strict.json    primary: strict fit;
                 unset -> print the Python side only, which is how it was locked
  CROSS_SET      subset_cross_primary.txt                 120 problems
  RUST_ONLY_SET  subset_rust_only_strict.txt              80 problems
  ALLOW_PARTIAL  1 only for the blind rehearsal on an incomplete feature file

Every statistic draws from its own seeded generator, so adding the Rust file
cannot move a Python number that was locked without it. Paired contrasts use
the same estimator as 5_problem_level.py's paired_boot (mean of the bootstrap
differences, percentile CI, P(<=0)), so a C1 restricted to a subset here is
the same quantity as the population C1 there.

C5's rho and its bootstrap interval are computed here, but not its test: the
permutation p printed beside it assumes the two score vectors share no
structure, and refits on shuffled labels showed they do (6b_c5_null.py, which
supplies the decision p-value).
"""
import json
import os

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

PY_SCORES = os.environ.get('PY_SCORES', 'results/python/scores_python_240.json')
RUST_SCORES = os.environ.get('RUST_SCORES', '')
CROSS_SET = os.environ.get('CROSS_SET', 'subset_cross_primary.txt')
RUST_ONLY_SET = os.environ.get('RUST_ONLY_SET', 'subset_rust_only_strict.txt')
ALLOW_PARTIAL = os.environ.get('ALLOW_PARTIAL', '0') == '1'
N_BOOT = int(os.environ.get('N_BOOT', 5000))
N_PERM = int(os.environ.get('N_PERM', 10000))
SEED = int(os.environ.get('SEED', 0))


def rng(stream):
    return np.random.default_rng([SEED, stream])


def read_ids(path):
    with open(path, encoding='utf-8') as f:
        return sorted({int(l) for l in f if l.strip() and not l.startswith('#')})


def load_scores(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    by_id = {int(p['task_id']): p for p in d['problems']}
    print(f"  {path}: {len(by_id)} problems, fit on "
          f"{d['subset_file'] or 'the whole file'} ({d['features_file']}, "
          f"labels {d['labels_file']})")
    return by_id


def restrict(scores, ids, what):
    missing = [t for t in ids if t not in scores]
    if missing:
        if not ALLOW_PARTIAL:
            raise SystemExit(f"{what}: {len(missing)} of {len(ids)} problems have no "
                             f"score (first {missing[:5]}). The population fit must "
                             f"cover the whole set; do not shrink it.")
        ids = [t for t in ids if t in scores]
        print(f"  !!! ALLOW_PARTIAL: {what} cut to {len(ids)} problems -- rehearsal only")
    return ids


def vec(scores, ids, key):
    return np.array([scores[t][key] for t in ids], dtype=float)


def boot_auc(y, s, g):
    obs = roc_auc_score(y, s)
    out = []
    for _ in range(N_BOOT):
        b = g.integers(0, len(y), len(y))
        if len(set(y[b].tolist())) > 1:
            out.append(roc_auc_score(y[b], s[b]))
    return obs, np.percentile(out, 2.5), np.percentile(out, 97.5)


def paired_boot(y1, s1, y2, s2, g):
    """AUROC(y1, s1) - AUROC(y2, s2) over the same resampled problems."""
    d = []
    for _ in range(N_BOOT):
        b = g.integers(0, len(y1), len(y1))
        if len(set(y1[b].tolist())) > 1 and len(set(y2[b].tolist())) > 1:
            d.append(roc_auc_score(y1[b], s1[b]) - roc_auc_score(y2[b], s2[b]))
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), float((d <= 0).mean())


def spearman(a, b):
    return float(stats.spearmanr(a, b)[0])


def fmt(t):
    return f"{t[0]:.4f}  95% CI [{t[1]:.4f}, {t[2]:.4f}]"


def main():
    print("=== inputs ===")
    py = load_scores(PY_SCORES)
    rust = load_scores(RUST_SCORES) if RUST_SCORES else None
    cross = read_ids(CROSS_SET)
    print(f"  cross set {CROSS_SET}: {len(cross)} problems")

    cross_py = restrict(py, cross, 'cross set in the Python scores')
    y_p = vec(py, cross_py, 'label')
    s_p = vec(py, cross_py, 'score')

    print(f"\n=== Python on the cross set (no refit) ===")
    a_p = boot_auc(y_p, s_p, rng(1))
    print(f"  {len(cross_py)} problems, {int(y_p.sum())} passing in Python")
    print(f"  AUROC {fmt(a_p)}")
    if rust is None:
        print("\n  RUST_SCORES unset: Python side only.")
        return

    cross = restrict(rust, cross_py, 'cross set in the Rust scores')
    if len(cross) < len(cross_py):
        # rehearsal only: re-cut the Python side to the same problems
        y_p, s_p = vec(py, cross, 'label'), vec(py, cross, 'score')
        a_p = boot_auc(y_p, s_p, rng(1))
        print(f"  !!! Python re-cut to the {len(cross)} problems Rust also has: {fmt(a_p)}")
    y_r = vec(rust, cross, 'label')
    s_r = vec(rust, cross, 'score')

    print(f"\n=== Rust on the cross set (no refit) ===")
    a_r = boot_auc(y_r, s_r, rng(2))
    print(f"  {len(cross)} problems, {int(y_r.sum())} passing in Rust")
    print(f"  AUROC {fmt(a_r)}")
    agree = float((y_r == y_p).mean())
    print(f"  labels agree across arms on {agree*100:.1f}% of these problems")

    m, lo, hi, pv = paired_boot(y_r, s_r, y_p, s_p, rng(3))
    print(f"\n  Rust - Python, paired over problems: {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"  P(<=0)={pv:.3f}")
    print("  descriptive only: section 6 shows a gap below ~0.11 cannot be resolved here,"
          " and no difference is attributed to language")

    print(f"\n=== C5: do the arms rank the same problems as hard? ===")
    rho = spearman(s_p, s_r)
    g = rng(4)
    boots = []
    for _ in range(N_BOOT):
        b = g.integers(0, len(cross), len(cross))
        if len(set(s_p[b].tolist())) > 1:
            boots.append(spearman(s_p[b], s_r[b]))
    g = rng(5)
    null = np.array([spearman(s_p, g.permutation(s_r)) for _ in range(N_PERM)])
    p_perm = (1 + int((null >= rho).sum())) / (1 + N_PERM)
    print(f"  Spearman rho {rho:+.4f}  95% CI [{np.percentile(boots, 2.5):+.4f}, "
          f"{np.percentile(boots, 97.5):+.4f}]  (bootstrap over {len(cross)} problems)")
    print(f"  one-sided permutation p = {p_perm:.4f}  (H1 rho > 0; null "
          f"{null.mean():+.4f} +- {null.std():.4f}, floor 1/{N_PERM + 1})")
    print("  NOT the decision statistic: permuting problems assumes the score vectors share"
          " no structure,\n  which label-free refits refute; C5 is decided by 6b_c5_null.py")
    print("  a positive rho says both arms read language-invariant difficulty; it is NOT"
          " evidence that the effect generalises across languages")

    print(f"\n=== Rust headline on problems Python never saw (no refit) ===")
    only = restrict(rust, read_ids(RUST_ONLY_SET), f'{RUST_ONLY_SET} in the Rust scores')
    y_o = vec(rust, only, 'label')
    s_o = vec(rust, only, 'score')
    print(f"  {RUST_ONLY_SET}: {len(only)} problems, {int(y_o.sum())} passing")
    lev = boot_auc(y_o, s_o, rng(6))
    print(f"  level (all rows)                  {fmt(lev)}")
    only0 = [t for t in only if rust[t]['score_step0'] is not None]
    if len(only0) < len(only):
        print(f"  WARNING: {len(only) - len(only0)} problems have no step-0 score")
    y0 = vec(rust, only0, 'label')
    s_a = vec(rust, only0, 'score')
    s_0 = vec(rust, only0, 'score_step0')
    st0 = boot_auc(y0, s_0, rng(7))
    print(f"  step-0 only (signature, no body)  {fmt(st0)}")
    c1 = paired_boot(y0, s_a, y0, s_0, rng(8))
    print(f"  C1 contrast  {c1[0]:+.4f}  95% CI [{c1[1]:+.4f}, {c1[2]:+.4f}]  P(<=0)={c1[3]:.3f}")

    print(f"\n=== summary ===")
    print(f"  cross set ({len(cross)})   Python {a_p[0]:.4f}   Rust {a_r[0]:.4f}   "
          f"Rust-Python {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"  C5 rho {rho:+.4f}  (decided by 6b_c5_null.py; permutation p {p_perm:.4f} "
          f"is not the test)")
    print(f"  Rust-only ({len(only)})  level {lev[0]:.4f} [{lev[1]:.4f}, {lev[2]:.4f}]   "
          f"C1 {c1[0]:+.4f} [{c1[1]:+.4f}, {c1[2]:+.4f}]")


if __name__ == '__main__':
    main()
