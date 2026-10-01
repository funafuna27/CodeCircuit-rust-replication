"""Exploratory control for C1 -- NOT pre-registered.

Written 2026-10-01, after the pre-registered Rust result (commit 08a6663) had
been seen, and committed before this script was first run.

C1 compares a model fitted and scored on every row of a problem (~8 rows in
Rust, ~5 in Python) with one fitted and scored on a single row, step 0. The two
sides differ in more than the generated body: the multi-row model has several
times the training rows, and averaging its probabilities over a problem's rows
reduces noise at test time. Either could raise AUROC(all rows) with no
information from the code at all. Python's C1 was null despite that advantage;
Rust's was +0.135, and the advantage could be part of it.

This control removes both differences -- ONE row per problem on each side:

    step 0     the prompt and the signature, no body (exactly C1's step-0 side)
    last row   the prompt and the most code the 140-token cap keeps

THE question, fixed before the first run:

    AUROC(last row) - AUROC(step 0), paired bootstrap over problems.

    CI excludes zero, positive -> the generated code carries information beyond
                                  the prompt even with one row on each side.
    CI includes zero           -> at this n, C1 cannot be separated from the
                                  multi-row advantage.

Printed for context only, not as further tests: the all-rows level (it must
reproduce the pre-registered number), the last row residualised on length as
in C4, the last row against -max prefix chars (the surface baseline it most
resembles), and all rows against the last row.

    FEATURES_FILE  LABELS_FILE  SUBSET_FILE   one population, as in 5_problem_level.py
    OUT            optional json: every number printed, and per-problem scores
"""
import importlib.util
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('protocol', os.path.join(HERE, '5_problem_level.py'))
protocol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(protocol)

OUT = os.environ.get('OUT', '')


def rng(k):
    return np.random.default_rng([protocol.SEED, 200 + k])


def boot_auc(y, s, g):
    obs = roc_auc_score(y, s)
    out = []
    for _ in range(protocol.N_BOOT):
        b = g.integers(0, len(y), len(y))
        if len(set(y[b].tolist())) > 1:
            out.append(roc_auc_score(y[b], s[b]))
    return [obs, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def paired(y, s1, s2, g):
    """AUROC(s1) - AUROC(s2) on the same resampled problems; 5_problem_level's estimator."""
    d = []
    for _ in range(protocol.N_BOOT):
        b = g.integers(0, len(y), len(y))
        if len(set(y[b].tolist())) > 1:
            d.append(roc_auc_score(y[b], s1[b]) - roc_auc_score(y[b], s2[b]))
    d = np.array(d)
    return [float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float((d <= 0).mean())]


def main():
    rows = protocol.read_jsonl(protocol.FEATURES_FILE)
    rows.sort(key=lambda r: r.get('row_index', 0))
    labels = {int(r['task_id']): int(bool(r['passes']))
              for r in protocol.read_jsonl(protocol.LABELS_FILE)}
    if protocol.SUBSET_FILE:
        with open(protocol.SUBSET_FILE, encoding='utf-8') as f:
            wanted = {int(l) for l in f if l.strip() and not l.startswith('#')}
        rows = [r for r in rows if int(r['expr_id']) in wanted]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(protocol.MODEL_NAME)
    n_tok = np.array([len(tok(r['original_expression'])['input_ids']) for r in rows])
    keep = n_tok <= protocol.TOKEN_CAP
    rows = [r for r, k in zip(rows, keep) if k]
    n_tok = n_tok[keep].astype(float)

    X = np.array([r['feature'] for r in rows], dtype=float)
    g = np.array([int(r['expr_id']) for r in rows])
    step = np.array([r['step_number'] for r in rows], dtype=float)
    chars = np.array([len(r['original_expression']) for r in rows], dtype=float)
    lines = np.array([r['original_expression'].count('\n') + 1. for r in rows])
    problems = np.array(sorted(set(g.tolist())))
    y_row = np.array([labels[int(t)] for t in g])
    y = np.array([labels[int(p)] for p in problems])
    max_step = {int(p): step[g == p].max() for p in problems}
    n_steps = np.array([float((g == t).sum()) for t in g])

    s0 = step == 0
    last = np.array([s == max_step[int(t)] for s, t in zip(step, g)])
    for name, m in (('step-0', s0), ('last', last)):
        per = {int(p): int(m[g == p].sum()) for p in problems}
        if set(per.values()) != {1}:
            raise SystemExit(f"{name}: not exactly one row per problem")
    same = sum(max_step[int(p)] == 0 for p in problems)
    print(f"{protocol.FEATURES_FILE} | {protocol.SUBSET_FILE or 'whole file'}")
    print(f"  {len(problems)} problems, {int(y.sum())} positive, {len(rows)} rows "
          f"({len(rows) / len(problems):.2f} per problem)")
    print(f"  problems whose last row IS step 0 (nothing past the signature): {same}")

    def at(mask, v):  # one value per problem, in `problems` order
        order = np.argsort(g[mask], kind='stable')
        assert np.array_equal(g[mask][order], problems)
        return v[mask][order]

    oof = protocol.oof_problem_scores
    sc_all = oof(X, g, y_row, problems)
    sc0 = oof(X[s0], g[s0], y_row[s0], problems)
    scL = oof(X[last], g[last], y_row[last], problems)
    cov = np.column_stack([chars, np.log1p(chars), chars ** 2, step, n_steps,
                           lines, n_tok, n_tok ** 2])
    XrL = protocol.residualise(X[last], cov[last])
    scLr = oof(XrL, g[last], y_row[last], problems)
    neg_chars = -at(last, chars)

    res = {'features_file': protocol.FEATURES_FILE, 'labels_file': protocol.LABELS_FILE,
           'subset_file': protocol.SUBSET_FILE or None, 'n_problems': int(len(problems)),
           'n_positive': int(y.sum()), 'rows_per_problem': len(rows) / len(problems),
           'last_is_step0': int(same)}
    print("\n  one-problem AUROCs (bootstrap over problems)")
    for k, (name, s) in enumerate((('all rows (pre-registered level)', sc_all),
                                   ('step 0 only', sc0), ('last row only', scL),
                                   ('last row, residualised on length', scLr),
                                   ('-max prefix chars (surface)', neg_chars))):
        a = boot_auc(y, s, rng(k))
        res[name] = a
        print(f"    {name:34} {a[0]:.4f}  95% CI [{a[1]:.4f}, {a[2]:.4f}]")

    q = paired(y, scL, sc0, rng(10))
    res['THE CONTRAST last - step0'] = q
    verdict = ('EXCLUDES ZERO -- the code adds information with one row per side'
               if q[1] > 0 else 'includes zero -- not separable from the multi-row advantage')
    print(f"\n  THE CONTRAST  last row - step 0   {q[0]:+.4f}  95% CI [{q[1]:+.4f}, {q[2]:+.4f}]"
          f"  P(<=0)={q[3]:.3f}")
    print(f"  -> {verdict}")

    print("\n  context only")
    for k, (name, a, b) in enumerate((('last residualised - step 0', scLr, sc0),
                                      ('last row - (-max prefix chars)', scL, neg_chars),
                                      ('all rows - last row', sc_all, scL)), start=11):
        q = paired(y, a, b, rng(k))
        res[name] = q
        print(f"    {name:32} {q[0]:+.4f}  95% CI [{q[1]:+.4f}, {q[2]:+.4f}]  P(<=0)={q[3]:.3f}")

    if OUT:
        res['problems'] = [{'task_id': int(p), 'label': int(l), 'all': float(a),
                            'step0': float(b), 'last': float(c), 'last_resid': float(d)}
                           for p, l, a, b, c, d in zip(problems, y, sc_all, sc0, scL, scLr)]
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1)
        print(f"\n  -> {OUT}")


if __name__ == '__main__':
    main()
