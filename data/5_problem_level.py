"""The evaluation protocol for both arms. Fixed before the Rust graphs exist.

Upstream's 5_classification.py scores 1420 ROWS with a positional 70/30 split.
Neither is right for this target, and the Python arm was measured three times
before that was clear:

  - execution labels are constant within a problem, so a row-level AUROC weights
    each problem by its row count, which is its solution length, which
    correlates with the label. That single choice made the graph features look
    like they had lost to surface heuristics; at equal weight they win;
  - a seed-to-seed sd over GroupShuffleSplit reshuffles the SAME problems, so it
    measures split noise, not sampling error, and is 4-10x too narrow. A paired
    t or Wilcoxon over seeds is not a hypothesis test either: its p-value is a
    function of how many seeds you draw (one comparison ran p=0.2305 at 20
    seeds, 0.0494 at 50, 0.0010 at 100);
  - the effective sample size is the number of problems, not the number of rows.
    ICC of a task-constant label is 1 by construction, so n_eff = n_problems.

So: one aggregated prediction per held-out problem, and every interval a
bootstrap over problems.

  FEATURES_FILE  gemma_mbpp_features.jsonl | gemma_mbpp_rust_features.jsonl
  LABELS_FILE    gemma_mbpp_python_correctness.jsonl | gemma_mbpp_rust_correctness.jsonl
                 (both carry task_id + passes; the schemas already match)
  TOKEN_CAP      140 -- applied to BOTH arms or they differ in preprocessing
"""
import json
import os

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

FEATURES_FILE = os.environ.get('FEATURES_FILE', 'gemma_mbpp_features.jsonl')
LABELS_FILE = os.environ.get('LABELS_FILE', 'gemma_mbpp_python_correctness.jsonl')
TOKEN_CAP = int(os.environ.get('TOKEN_CAP', 140))
MODEL_NAME = os.environ.get('MODEL_NAME', 'google/gemma-2-2b-it')
N_REPEATS = int(os.environ.get('N_REPEATS', 10))
N_FOLDS = int(os.environ.get('N_FOLDS', 5))
N_BOOT = int(os.environ.get('N_BOOT', 5000))
N_PERM = int(os.environ.get('N_PERM', 1000))
SEED = int(os.environ.get('SEED', 0))

RNG = np.random.default_rng(SEED)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def oof_problem_scores(X, groups, y, problems):
    """Every problem scored out of fold, averaged over repeats.

    Each repeat permutes the problem -> surrogate-group map so fold membership
    differs between repeats; no problem ever straddles a fold.
    """
    idx = {int(p): i for i, p in enumerate(problems)}
    acc = np.zeros(len(problems))
    for rep in range(N_REPEATS):
        perm = np.random.default_rng(100 + rep).permutation(len(problems))
        surrogate = np.array([perm[idx[int(g)]] for g in groups])
        p = np.zeros(len(y), dtype=float)
        for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups=surrogate):
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        acc += np.array([p[groups == q].mean() for q in problems])
    return acc / N_REPEATS


def boot_ci(score, y_problem, n=None):
    n = n or N_BOOT
    obs = roc_auc_score(y_problem, score)
    out = []
    for _ in range(n):
        b = RNG.integers(0, len(y_problem), len(y_problem))
        if len(set(y_problem[b].tolist())) > 1:
            out.append(roc_auc_score(y_problem[b], score[b]))
    out = np.array(out)
    return obs, np.percentile(out, 2.5), np.percentile(out, 97.5)


def paired_boot(s1, s2, y_problem, n=None):
    n = n or N_BOOT
    d = []
    for _ in range(n):
        b = RNG.integers(0, len(y_problem), len(y_problem))
        if len(set(y_problem[b].tolist())) > 1:
            d.append(roc_auc_score(y_problem[b], s1[b])
                     - roc_auc_score(y_problem[b], s2[b]))
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), float((d <= 0).mean())


def main():
    rows = read_jsonl(FEATURES_FILE)
    rows.sort(key=lambda r: r.get('row_index', 0))
    labels = {int(r['task_id']): int(bool(r['passes'])) for r in read_jsonl(LABELS_FILE)}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    n_tok = np.array([len(tok(r['original_expression'])['input_ids']) for r in rows])
    keep = n_tok <= TOKEN_CAP
    dropped = int((~keep).sum())
    rows = [r for r, k in zip(rows, keep) if k]
    n_tok = n_tok[keep]
    print(f"{FEATURES_FILE}: {len(rows)} rows after a {TOKEN_CAP}-token cap "
          f"({dropped} dropped)")

    missing = {int(r['expr_id']) for r in rows} - set(labels)
    if missing:
        raise SystemExit(f"{len(missing)} problems have features but no label: "
                         f"{sorted(missing)[:10]}")

    X = np.array([r['feature'] for r in rows], dtype=float)
    g = np.array([int(r['expr_id']) for r in rows])
    step = np.array([r['step_number'] for r in rows], dtype=float)
    chars = np.array([len(r['original_expression']) for r in rows], dtype=float)
    lines = np.array([r['original_expression'].count('\n') + 1. for r in rows])

    problems = np.array(sorted(set(g.tolist())))
    y_prob = np.array([labels[int(p)] for p in problems])
    y_row = np.array([labels[int(t)] for t in g])
    n_steps_of = {int(p): int((g == p).sum()) for p in problems}
    n_steps = np.array([n_steps_of[int(t)] for t in g], dtype=float)

    print(f"  {len(problems)} problems, {int(y_prob.sum())} positive "
          f"({y_prob.mean()*100:.1f}%), {len(rows)/len(problems):.2f} rows each")
    print(f"  tokens: mean {n_tok.mean():.1f}, max {n_tok.max()}")
    const = int((X.var(axis=0) < 1e-12).sum())
    print(f"  features: {X.shape[1]} dims, {const} constant "
          f"-> {X.shape[1]-const} effective\n")

    print("=== headline: graph topology, problem level ===")
    sc = oof_problem_scores(X, g, y_row, problems)
    obs, lo, hi = boot_ci(sc, y_prob)
    print(f"  AUROC {obs:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
          f"(bootstrap over {len(problems)} problems)")

    print("\n=== surface baselines, same problems ===")
    surf_rows = np.column_stack([chars, lines, step, n_steps])
    sc_surf = oof_problem_scores(surf_rows, g, y_row, problems)
    singles = {
        '-n_steps': -np.array([n_steps_of[int(p)] for p in problems], dtype=float),
        '-max prefix chars': -np.array([chars[g == p].max() for p in problems]),
        '-max prefix lines': -np.array([lines[g == p].max() for p in problems]),
    }
    named = {'trained RF on 4 surface dims': sc_surf, **singles}
    for nm, s in named.items():
        o, l, h = boot_ci(s, y_prob)
        print(f"  {nm:30} {o:.4f}  95% CI [{l:.4f}, {h:.4f}]")

    print("\n=== paired: graph minus each baseline (same problems resampled) ===")
    for nm, s in named.items():
        m, l, h, p = paired_boot(sc, s, y_prob)
        mark = '  CI excludes 0' if l > 0 or h < 0 else ''
        print(f"  vs {nm:30} {m:+.4f}  95% CI [{l:+.4f}, {h:+.4f}]  "
              f"P(<=0)={p:.3f}{mark}")

    print("\n=== is it above chance? permutation over problems ===")
    null = np.array([roc_auc_score(RNG.permutation(y_prob), sc) for _ in range(N_PERM)])
    p_perm = float((null >= obs).mean())
    print(f"  observed {obs:.4f} | null {null.mean():.4f} +- {null.std():.4f} "
          f"| p = {p_perm:.4f}  ({N_PERM} permutations)")

    print("\n=== learning curve over problems ===")
    for frac in (0.25, 0.5, 0.75, 1.0):
        aucs = []
        for s in range(10):
            rg = np.random.default_rng(1000 + s)
            perm = rg.permutation(problems)
            n_te = max(2, int(len(problems) * 0.2))
            te_p, pool = set(perm[:n_te].tolist()), perm[n_te:]
            tr_p = set(pool[:max(2, int(len(pool) * frac))].tolist())
            tr, te = np.isin(g, list(tr_p)), np.isin(g, list(te_p))
            if len(set(y_row[tr].tolist())) < 2:
                continue
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X[tr], y_row[tr])
            pr = clf.predict_proba(X[te])[:, 1]
            tt = np.array(sorted(te_p))
            s_te = np.array([pr[g[te] == q].mean() for q in tt])
            y_te = np.array([labels[int(q)] for q in tt])
            if len(set(y_te.tolist())) > 1:
                aucs.append(roc_auc_score(y_te, s_te))
        a = np.array(aucs)
        print(f"  train on {int(len(problems)*0.8*frac):4} problems: "
              f"{a.mean():.4f} +- {a.std(ddof=1):.4f} (n={len(a)})")

    print("\n=== length-proxy check ===")
    r = np.array([abs(stats.spearmanr(X[:, j], chars).statistic)
                  if X[:, j].var() > 1e-12 else 0.0 for j in range(X.shape[1])])
    proxies = np.flatnonzero(r > 0.7)
    print(f"  {len(proxies)} of {X.shape[1]} dims correlate |r|>0.7 with prefix chars "
          f"(max {r.max():.3f})")
    if len(proxies) and len(proxies) < X.shape[1]:
        rest = [j for j in range(X.shape[1]) if j not in set(proxies.tolist())]
        s2 = oof_problem_scores(X[:, rest], g, y_row, problems)
        o2, l2, h2 = boot_ci(s2, y_prob)
        print(f"  dropping them: {o2:.4f}  95% CI [{l2:.4f}, {h2:.4f}] "
              f"(vs {obs:.4f} with all dims)")

    print(f"\n=== summary line ===")
    print(f"  {FEATURES_FILE} | {len(problems)} problems | "
          f"AUROC {obs:.4f} [{lo:.4f}, {hi:.4f}] | perm p={p_perm:.4f}")


if __name__ == '__main__':
    main()
