"""Phase 9, A3 summary: how many of Gemma's passing programs MBPP+ overturns, and what a
within-problem study of the overturned ones could detect. Descriptive and planning only.

Verdict of a distinct program in a mode:
  holds        compiles, finishes, every MBPP+ case right
  flips        a case wrong or panicking
  timeout      the 30 s run limit (or the compile limit) hit; not counted as a flip
  crash        no PASS line (an abort, e.g. stack overflow); counted as a flip in the sensitivity only
  harness      the harness did not compile (a literal the signature cannot take); excluded

Tables
  F0  gate: programs failing a base case in tasks whose MBPP+ base inputs equal the stored asserts
      (must be 0 in mode O: a stored pass passes the same inputs again)
  F1  verdicts per mode, programs and samples; flips only under overflow checks
  F2  flips by cause: a base case (MBPP+ corrected the task's expected values) or plus cases only
  F3  by phase (2 = greedy, 3, 5, 6)
  F4  per task: tasks with a flip; mixed tasks (some programs hold, some flip)
  F5  Phase 2's greedy generations (the between-problem labels of ③): label changes, strict 206
  V   variance components of the flip label among distinct programs (one-way ANOVA ICC over tasks)
  P   power of a within-problem AUROC (holds ranked above flips) on the real mixed-task structure:
      binormal scores with a common within-task shift, problem-level sign-flip test of the
      task-mean centred AUROC, alpha 0.05 two-sided, 2000 simulations per cell (stream [20261022, 3])

  RELABEL=results/p9_state/a3_relabel.jsonl.gz  OUT=results/p9_state/a3_summary.json
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_common as C  # noqa: E402

RELABEL = os.environ.get('RELABEL', os.path.join(C.OUT_DIR, 'a3_relabel.jsonl.gz'))
CASES = os.environ.get('CASES', os.path.join(C.OUT_DIR, 'a3_cases.json'))
OUT = os.environ.get("OUT", os.path.join(C.OUT_DIR, "a3_summary.json"))
# EXCLUDE=comma-separated task ids left out (the audit of 2026-10-22: tasks below MIN_CASES, artefact tasks)
N_SIM = int(os.environ.get('N_SIM', 2000))


def verdict(r):
    st = r['stage']
    if st == 'ok':
        return 'holds' if r['n_fail'] + r['n_panic'] == 0 else 'flips'
    if st in ('run_timeout', 'compile_timeout'):
        return 'timeout'
    if st == 'no_pass_line':
        return 'crash'
    return 'harness'


def within_auc(scores, groups, labels):
    """Mean over tasks of the within-task AUROC (holds above flips), tasks with both kinds only."""
    vals = []
    for g in np.unique(groups):
        m = groups == g
        s, y = scores[m], labels[m]
        pos, neg = s[y == 1], s[y == 0]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            vals.append(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)
    return np.array(vals)


def power(struct, aucs, rng, n_sim):
    """struct: list of (n_holds, n_flips) per mixed task. Test: sign-flip over tasks of (AUC_g - 0.5)."""
    out = {}
    groups = np.concatenate([[g] * (a + b) for g, (a, b) in enumerate(struct)])
    labels = np.concatenate([[1] * a + [0] * b for a, b in struct])
    for auc in aucs:
        d = np.sqrt(2) * norm.ppf(auc)
        rej = 0
        for _ in range(n_sim):
            s = rng.normal(size=len(labels)) + d * labels + rng.normal(size=len(struct))[groups] * 1.0
            v = within_auc(s, groups, labels) - 0.5
            t = abs(v.mean())
            flips = rng.choice([-1, 1], size=(999, len(v)))
            p = (1 + (np.abs((flips * v).mean(1)) >= t).sum()) / 1000
            rej += p < 0.05
        out[f'{auc:.2f}'] = round(float(rej / n_sim), 3)
    return out


def main():
    rows = C.read_jsonl(RELABEL)
    excl = {int(t) for t in os.environ.get("EXCLUDE", "").split(",") if t}
    rows = [r for r in rows if r["task_id"] not in excl]
    with open(CASES, encoding='utf-8') as f:
        cases = json.load(f)['tasks']
    out, log = {}, []
    pr = lambda *a: log.append(' '.join(str(x) for x in a))
    modes = ('O', 'O_ovf')
    for r in rows:
        t = cases[str(r['task_id'])]
        base = {i for i, c in enumerate(t['cases']) if c['base']}
        for m in modes:
            r[m]['v'] = verdict(r[m])
            bad = set(r[m].get('bad', []))
            r[m]['base_bad'] = bool(bad & base)
        r['base_agrees'] = t['base_agrees_with_stored']
        r['n_members'] = len(r['members'])

    # F0
    f0 = [r for r in rows if r['base_agrees'] and r['O']['base_bad']]
    out['F0'] = {'base_fail_where_base_agrees': len(f0), 'tasks': sorted({r['task_id'] for r in f0})}
    pr('F0 gate: programs failing a base case where MBPP+ base = stored asserts (mode O):', out['F0'])

    # F1
    f1 = {}
    for m in modes:
        f1[m] = {'programs': dict(Counter(r[m]['v'] for r in rows)),
                 'samples': dict(sum((Counter({r[m]['v']: r['n_members']}) for r in rows), Counter()))}
    f1['flip_only_with_overflow_checks'] = sum(r['O']['v'] == 'holds' and r['O_ovf']['v'] == 'flips' for r in rows)
    f1['flip_O_holds_ovf'] = sum(r['O']['v'] == 'flips' and r['O_ovf']['v'] == 'holds' for r in rows)
    out['F1'] = f1
    pr('F1 verdicts:', json.dumps(f1))

    # F2
    fl = [r for r in rows if r['O']['v'] == 'flips']
    out['F2'] = {'flips': len(fl), 'base_case_fails': sum(r['O']['base_bad'] for r in fl),
                 'base_case_fails_in_corrected_tasks': sum(r['O']['base_bad'] and r['base_agrees'] is False for r in fl),
                 'plus_only': sum(not r['O']['base_bad'] for r in fl),
                 'flips_in_float_result_tasks': sum(cases[str(r['task_id'])]['float_result'] for r in fl),
                 'flips_in_custom_assertion_tasks': sum(cases[str(r['task_id'])]['custom_assertion'] for r in fl)}
    pr('F2 causes (mode O):', out['F2'])

    # F3
    f3 = defaultdict(Counter)
    for r in rows:
        for s, ph in r['members']:
            f3[ph][r['O']['v']] += 1
    out['F3'] = {ph: dict(c) for ph, c in sorted(f3.items())}
    pr('F3 samples by phase (mode O):', out['F3'])

    # F4
    per = defaultdict(Counter)
    for r in rows:
        if r['O']['v'] in ('holds', 'flips'):
            per[r['task_id']][r['O']['v']] += 1
    mixed = {t: c for t, c in per.items() if c['holds'] and c['flips']}
    out['F4'] = {'tasks': len(per), 'tasks_with_flip': sum(bool(c['flips']) for c in per.values()),
                 'tasks_all_flip': sum(bool(c['flips']) and not c['holds'] for c in per.values()),
                 'mixed_tasks': len(mixed),
                 'mixed_programs': sum(c['holds'] + c['flips'] for c in mixed.values()),
                 'mixed_flips': sum(c['flips'] for c in mixed.values()),
                 'mixed_pairs': sum(c['holds'] * c['flips'] for c in mixed.values()),
                 'mixed_tasks_corrected_base': sum(cases[str(t)]['base_agrees_with_stored'] is False for t in mixed)}
    pr('F4 tasks (mode O, distinct programs):', out['F4'])

    # F5: Phase 2 greedy
    g = [r for r in rows if any(ph == '2' for _, ph in r['members'])]
    strict = {int(t) for t, v in cases.items() if v['graded_all_tests']}
    out['F5'] = {'greedy_passing_in_usable_tasks': len(g),
                 'flips_O': sum(r['O']['v'] == 'flips' for r in g),
                 'flips_O_ovf': sum(r['O_ovf']['v'] == 'flips' for r in g),
                 'timeouts_O': sum(r['O']['v'] == 'timeout' for r in g),
                 'strict_flips_O': sum(r['O']['v'] == 'flips' and r['task_id'] in strict for r in g),
                 'strict_passing': sum(r['task_id'] in strict for r in g),
                 'flipped_tasks': sorted(r['task_id'] for r in g if r['O']['v'] == 'flips')}
    pr('F5 Phase 2 greedy generations:', out['F5'])

    # V: variance components of the flip label over tasks (distinct programs, mode O)
    ys, gs = [], []
    for r in rows:
        if r['O']['v'] in ('holds', 'flips'):
            ys.append(1.0 if r['O']['v'] == 'flips' else 0.0)
            gs.append(r['task_id'])
    ys, gs = np.array(ys), np.array(gs)
    tasks = np.unique(gs)
    n_i = np.array([np.sum(gs == t) for t in tasks])
    means = np.array([ys[gs == t].mean() for t in tasks])
    N, k = len(ys), len(tasks)
    msb = np.sum(n_i * (means - ys.mean()) ** 2) / (k - 1)
    msw = np.sum([np.sum((ys[gs == t] - m) ** 2) for t, m in zip(tasks, means)]) / (N - k)
    n0 = (N - np.sum(n_i ** 2) / N) / (k - 1)
    s2b = max(0.0, (msb - msw) / n0)
    out['V'] = {'programs': int(N), 'tasks': int(k), 'flip_rate': round(float(ys.mean()), 3),
                'between_task_var': round(float(s2b), 4), 'within_task_var': round(float(msw), 4),
                'icc': round(float(s2b / (s2b + msw)), 3), 'programs_per_task_median': float(np.median(n_i))}
    pr('V variance components (flip label, distinct programs):', out['V'])

    # P: power on the mixed tasks' real structure
    rng = np.random.default_rng([20261022, 3])
    struct = [(c['holds'], c['flips']) for c in mixed.values()]
    out['P'] = {'mixed_tasks': len(struct), 'n_sim': N_SIM,
                'power_at_auc': power(struct, [0.55, 0.60, 0.65, 0.70, 0.75], rng, N_SIM) if struct else None}
    pr('P power of a within-task AUROC test (alpha 0.05):', out['P'])

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, default=str)
    with open(OUT.replace('.json', '.log'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log) + '\n')
    print('\n'.join(log))


if __name__ == '__main__':
    main()
