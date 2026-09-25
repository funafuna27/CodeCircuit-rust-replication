"""The evaluation protocol for both arms. Fixed before the Rust graphs exist.

See PREREGISTRATION.md. The short version of why it looks like this:

Upstream's 5_classification.py scores 1420 ROWS with a positional 70/30 split.
Neither is right for this target, and the Python arm was measured three times
before that was clear. Execution labels are constant within a problem, so a
row-level AUROC weights each problem by its solution length, which correlates
with the label; a seed-to-seed sd measures split noise rather than sampling
error; and the effective sample size is the number of problems, not rows.

The primary output is NOT the AUROC. It is the C1 contrast:

    AUROC(all rows) - AUROC(step-0 rows only)

Step 0 is the prompt plus the function signature and no body -- no logic that
could be right or wrong. On the Python arm that contrast is +0.0340
[-0.0160, +0.0846] (PREREGISTRATION.md §3a): a graph taken before the model
writes any logic predicts correctness as well as one taken over the whole
generation. Reporting the level alone would let a problem-difficulty detector
pass as "predicts whether generated code is correct", which is the paper's claim.

  FEATURES_FILE  gemma_mbpp_features.jsonl | gemma_mbpp_rust_features.jsonl
  LABELS_FILE    gemma_mbpp_python_correctness.jsonl | gemma_mbpp_rust_correctness.jsonl
                 (both carry task_id + passes; the schemas already match)
  TOKEN_CAP      140 -- applied to BOTH arms or they differ in preprocessing
  SCORES_OUT     optional path; writes the per-problem out-of-fold scores that
                 6_cross_arm.py needs for C5 and the evaluation-only subsets
"""
import json
import os

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
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
N_JOBS = int(os.environ.get('N_JOBS', -1))
SEED = int(os.environ.get('SEED', 0))
# Restrict to a task-id POPULATION, one per line, and refit on it: the 240
# Python problems that parse, the 206 strict Rust problems. The cross-arm and
# Rust-only sets are not populations -- 6_cross_arm.py restricts an existing
# score vector to them and never refits (5a_make_subsets.py explains why).
SUBSET_FILE = os.environ.get('SUBSET_FILE', '')
# Writing the scores consumes no randomness, so every printed number is the
# same with or without it.
SCORES_OUT = os.environ.get('SCORES_OUT', '')

RNG = np.random.default_rng(SEED)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def oof_problem_scores(X, groups, y, problems):
    """Every problem scored out of fold, averaged over repeats.

    Each repeat permutes the problem -> surrogate-group map so fold membership
    differs between repeats; no problem ever straddles a fold. (Checked: the
    reshuffling is real, adjusted Rand index between repeat partitions ~0.001,
    mean 4.3 of 5 distinct folds visited per problem.)
    """
    idx = {int(p): i for i, p in enumerate(problems)}
    acc = np.zeros(len(problems))
    for rep in range(N_REPEATS):
        perm = np.random.default_rng(100 + rep).permutation(len(problems))
        surrogate = np.array([perm[idx[int(g)]] for g in groups])
        p = np.zeros(len(y), dtype=float)
        for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups=surrogate):
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS)
            clf.fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        acc += np.array([p[groups == q].mean() for q in problems])
    return acc / N_REPEATS


def boot_ci(score, y_problem, n=None):
    """Bootstrap over PROBLEMS.

    Conditional on the fitted scoring rule: the score vector is computed once,
    so model-fitting variance is excluded and this is about 1.28x too narrow.
    Stated rather than corrected, because widening it would make the locked
    baseline non-reproducible.
    """
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


def residualise(X, covars):
    """Strip everything linearly predictable from length and position.

    Replaces the drop-|r|>0.7 filter of version 1, which a matrix built to be
    100% length by construction passed at every noise level -- a test nothing
    can fail. It was also computed at the row unit while the claim is at the
    problem unit (23 of 49 dims exceed the threshold when recomputed correctly,
    against 10 at the row unit).
    """
    return X - LinearRegression().fit(covars, X).predict(covars)


def main():
    rows = read_jsonl(FEATURES_FILE)
    rows.sort(key=lambda r: r.get('row_index', 0))
    labels = {int(r['task_id']): int(bool(r['passes'])) for r in read_jsonl(LABELS_FILE)}

    if SUBSET_FILE:
        wanted = {int(l) for l in open(SUBSET_FILE, encoding='utf-8')
                  if l.strip() and not l.startswith('#')}
        before = len({int(r['expr_id']) for r in rows})
        rows = [r for r in rows if int(r['expr_id']) in wanted]
        print(f"SUBSET_FILE {SUBSET_FILE}: {before} problems -> "
              f"{len({int(r['expr_id']) for r in rows})}")

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
    if not np.isfinite(X).all():
        # A single NaN used to pass silently: it escapes the constant-dim count
        # (nan < 1e-12 is False), escapes any correlation filter (abs(nan) > 0.7
        # is False), and sklearn fits it without a word. Reachable from
        # graph_dataset.py's std() over one node and mean() over an empty mask.
        bad = np.argwhere(~np.isfinite(X))
        raise SystemExit(f"feature matrix has {len(bad)} non-finite entries, first "
                         f"at row {bad[0][0]} dim {bad[0][1]}. Fix the featuriser; "
                         f"do not impute.")

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

    print("=== level: graph topology over all rows ===")
    sc = oof_problem_scores(X, g, y_row, problems)
    obs, lo, hi = boot_ci(sc, y_prob)
    print(f"  AUROC {obs:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
          f"(bootstrap over {len(problems)} problems, conditional on the fitted rule)")

    print("\n=== C1 (PRIMARY): does the generated body add anything? ===")
    s0 = step == 0
    p0 = np.array([p for p in problems if s0[g == p].any()])
    if len(p0) < len(problems):
        print(f"  WARNING: {len(problems)-len(p0)} problems have no step-0 row")
    sc0_sub = oof_problem_scores(X[s0], g[s0], y_row[s0], p0)
    y0 = np.array([labels[int(p)] for p in p0])
    o0, l0, h0 = boot_ci(sc0_sub, y0)
    print(f"  step-0 only (signature, no body)  {o0:.4f}  95% CI [{l0:.4f}, {h0:.4f}]")
    pos = {int(p): i for i, p in enumerate(problems)}
    pos0 = {int(p): i for i, p in enumerate(p0)}
    common = np.array([p for p in problems if int(p) in pos0])
    a = np.array([sc[pos[int(p)]] for p in common])
    b = np.array([sc0_sub[pos0[int(p)]] for p in common])
    yc = np.array([labels[int(p)] for p in common])
    m, cl, ch, pv = paired_boot(a, b, yc)
    verdict = ('EXCLUDES ZERO -- the body contributes' if cl > 0
               else 'includes zero -- the body adds nothing measurable')
    print(f"  C1 contrast  {m:+.4f}  95% CI [{cl:+.4f}, {ch:+.4f}]  "
          f"P(<=0)={pv:.3f}")
    print(f"  -> {verdict}")

    print("\n=== C2: surface baselines and the paired contrast ===")
    surf_rows = np.column_stack([chars, lines, step, n_steps])
    sc_surf = oof_problem_scores(surf_rows, g, y_row, problems)
    per_problem = lambda v: np.array([v[g == p].max() for p in problems])
    singles = {
        '-n_steps': -np.array([n_steps_of[int(p)] for p in problems], dtype=float),
        '-max prefix chars': -per_problem(chars),
        '-max prefix lines': -per_problem(lines),
    }
    if np.allclose(singles['-n_steps'], singles['-max prefix lines']):
        print("  note: -n_steps and -max prefix lines are the same vector "
              "(rows are line prefixes), so there are 3 distinct baselines, not 4")
    named = {'trained RF on 4 surface dims': sc_surf, **singles}
    best_nm, best = None, -1.0
    for nm, s in named.items():
        o, l, h = boot_ci(s, y_prob)
        print(f"  {nm:30} {o:.4f}  95% CI [{l:.4f}, {h:.4f}]")
        if o > best:
            best_nm, best = nm, o
    for nm, s in named.items():
        m2, l2, h2, p2 = paired_boot(sc, s, y_prob)
        mark = '  CI excludes 0' if l2 > 0 or h2 < 0 else '  unresolved'
        star = '  <- C2' if nm == best_nm else ''
        print(f"  vs {nm:30} {m2:+.4f}  95% CI [{l2:+.4f}, {h2:+.4f}]  "
              f"P(<=0)={p2:.3f}{mark}{star}")
    print(f"  best surface is {best_nm} at {best:.4f}; minimum resolvable paired "
          f"gap at n={len(problems)} is about 0.06")

    print("\n=== C3: above chance? ===")
    null = np.array([roc_auc_score(RNG.permutation(y_prob), sc) for _ in range(N_PERM)])
    p_perm = (1 + int((null >= obs).sum())) / (1 + N_PERM)
    print(f"  CI excludes 0.50: {'yes' if lo > 0.5 else 'NO'}")
    print(f"  permutation p = {p_perm:.4f} (null {null.mean():.4f} +- {null.std():.4f}, "
          f"floor 1/{N_PERM+1})")
    print("  note: the permutation holds the score vector fixed, so this is the same "
          "test as the CI, not a second one")

    print("\n=== C4: is it length? ===")
    cov = np.column_stack([chars, np.log1p(chars), chars ** 2, step, n_steps,
                           lines, n_tok, n_tok.astype(float) ** 2])
    Xr = residualise(X, cov)
    sc_r = oof_problem_scores(Xr, g, y_row, problems)
    o4, l4, h4 = boot_ci(sc_r, y_prob)
    print(f"  residualised on 8 length/position terms: {o4:.4f} "
          f"95% CI [{l4:.4f}, {h4:.4f}]  (vs {obs:.4f})")
    print(f"  loss {obs - o4:+.4f}  -> {'passes' if obs - o4 < 0.05 else 'FAILS'} "
          f"the 0.05 criterion")

    print("\n=== descriptive: learning curve (not a prediction) ===")
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
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS)
            clf.fit(X[tr], y_row[tr])
            pr = clf.predict_proba(X[te])[:, 1]
            tt = np.array(sorted(te_p))
            s_te = np.array([pr[g[te] == q].mean() for q in tt])
            y_te = np.array([labels[int(q)] for q in tt])
            if len(set(y_te.tolist())) > 1:
                aucs.append(roc_auc_score(y_te, s_te))
        a = np.array(aucs)
        # spread over splits, NOT an interval -- all four fractions reuse the
        # same test problems within a split
        print(f"  train on {int(len(problems)*0.8*frac):4} problems: "
              f"{a.mean():.4f} (split spread {a.std(ddof=1):.4f}, n={len(a)})")

    print(f"\n=== summary ===")
    print(f"  {FEATURES_FILE} | {len(problems)} problems")
    print(f"  level  {obs:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"  C1     {m:+.4f} [{cl:+.4f}, {ch:+.4f}]   <- the primary result")
    print(f"  C4     {obs - o4:+.4f} loss after residualising on length")

    if SCORES_OUT:
        out = {
            'features_file': FEATURES_FILE, 'labels_file': LABELS_FILE,
            'subset_file': SUBSET_FILE or None, 'token_cap': TOKEN_CAP,
            'n_repeats': N_REPEATS, 'n_folds': N_FOLDS,
            'level': [obs, lo, hi], 'step0': [o0, l0, h0], 'c1': [m, cl, ch],
            'problems': [{'task_id': int(p), 'label': int(labels[int(p)]),
                          'score': float(sc[pos[int(p)]]),
                          'score_step0': (float(sc0_sub[pos0[int(p)]])
                                          if int(p) in pos0 else None)}
                         for p in problems],
        }
        with open(SCORES_OUT, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1)
        print(f"\n  per-problem scores for {len(problems)} problems -> {SCORES_OUT}")


if __name__ == '__main__':
    main()
