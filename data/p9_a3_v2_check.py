"""Phase 9, A3 v2: the type-directed harness, checked against version 1, and the flips audited.

Gate R (regression): every program that version 1 judged to the end ('ok') in a mode must get the same
  verdict and the same set of failing cases in version 2.
Gate F0: no program fails a base case in a task whose MBPP+ base inputs equal the stored asserts
  (now including the 27 tasks version 1 could not compile).
Rules fixed at gate 1 (2026-10-22) before any analysis read a flip label:
  - a run that times out is rerun once at 120 s; still timing out, the program is 'timeout' and is
    left out of the flip labels (reported);
  - a task enters a within-task analysis only with at least MIN_CASES usable MBPP+ cases.
Audit:
  - all-flip tasks: tasks with 2+ programs judged where every program flips; per task the share of
    cases failed, whether its base agrees, a custom MBPP+ assertion, float results;
  - semantic classes of a flip's failing cases: any argument or expected value with a negative
    integer, an empty string or container, a non-ASCII character; panics only. A flip all of whose
    failing cases fall in these classes is 'trap-only' (Python and Rust can differ there: floor
    division and remainder, byte vs character length, empty-input conventions).

  V1=results/p9_state/a3_relabel.jsonl.gz  V2=results/p9_state/a3_relabel_v2.jsonl.gz
"""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_common as C  # noqa: E402

V1 = os.environ.get('V1', os.path.join(C.OUT_DIR, 'a3_relabel.jsonl.gz'))
V2 = os.environ.get('V2', os.path.join(C.OUT_DIR, 'a3_relabel_v2.jsonl.gz'))
CASES = os.path.join(C.OUT_DIR, 'a3_cases.json')
OUT = os.path.join(C.OUT_DIR, 'a3_v2_check.json')
MIN_CASES = 20


def verdict(r):
    if r['stage'] == 'ok':
        return 'holds' if r['n_fail'] + r['n_panic'] == 0 else 'flips'
    return {'run_timeout': 'timeout', 'compile_timeout': 'timeout', 'no_pass_line': 'crash'}.get(r['stage'], 'harness')


def classes(v):
    out = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, int) and x < 0:
            out.add('negative')
        elif isinstance(x, float) and x < 0:
            out.add('negative')
        elif isinstance(x, str):
            if not x:
                out.add('empty')
            if any(ord(ch) > 127 for ch in x):
                out.add('non_ascii')
        elif isinstance(x, list):
            if not x:
                out.add('empty')
            for y in x:
                walk(y)
    walk(v)
    return out


def main():
    cases = json.load(open(CASES))['tasks']
    key = lambda r: (r['task_id'], tuple(map(tuple, r['members']))[0])
    v1 = {key(r): r for r in C.read_jsonl(V1)}
    v2 = {key(r): r for r in C.read_jsonl(V2)}
    assert set(v1) == set(v2), 'program sets differ'
    out = {}
    # gate R
    bad = []
    for k, a in v1.items():
        b = v2[k]
        for m in ('O', 'O_ovf'):
            if a[m]['stage'] == 'ok':
                if verdict(a[m]) != verdict(b[m]) or sorted(a[m]['bad'])[:40] != sorted(b[m].get('bad', []))[:40]:
                    bad.append((k[0], m, verdict(a[m]), verdict(b[m])))
    out['gate_R'] = {'compared': sum(v1[k][m]['stage'] == 'ok' for k in v1 for m in ('O', 'O_ovf')),
                     'changed': len(bad), 'examples': bad[:10]}
    # gate F0 and verdict counts
    f0, cnt = [], Counter()
    for k, r in v2.items():
        t = cases[str(r['task_id'])]
        base = {i for i, c in enumerate(t['cases']) if c['base']}
        cnt[(verdict(r['O']), verdict(r['O_ovf']))] += 1
        if t['base_agrees_with_stored'] and set(r['O'].get('bad', [])) & base:
            f0.append(r['task_id'])
    out['gate_F0'] = {'fails': len(f0), 'tasks': sorted(set(f0))}
    out['verdicts_O_by_O_ovf'] = {f'{a}|{b}': n for (a, b), n in cnt.most_common()}
    out['was_harness_v1'] = dict(Counter(verdict(v2[k]['O']) for k in v1 if v1[k]['O']['stage'] == 'no_compile'))
    out['timeouts_after_rerun'] = sorted({r['task_id'] for r in v2.values() if verdict(r['O']) == 'timeout'})
    out['retried'] = sum(bool(r['O'].get('retried')) for r in v2.values())
    small = sorted(int(t) for t, v in cases.items() if v.get('n_cases') is not None and v['n_cases'] < MIN_CASES)
    out['tasks_below_min_cases'] = {'min_cases': MIN_CASES, 'tasks': small}
    # all-flip tasks
    per = defaultdict(list)
    for r in v2.values():
        per[r['task_id']].append(r)
    allflip = []
    for t, rs in per.items():
        vs = [verdict(r['O']) for r in rs]
        judged = [r for r, v in zip(rs, vs) if v in ('holds', 'flips')]
        if len(judged) >= 2 and all(verdict(r['O']) == 'flips' for r in judged):
            c = cases[str(t)]
            share = sorted(len(r['O']['bad']) / c['n_cases'] for r in judged)
            allflip.append({'task': t, 'programs': len(judged), 'median_share_failed': round(share[len(share) // 2], 3),
                            'base_agrees': c['base_agrees_with_stored'], 'custom_assertion': c['custom_assertion'],
                            'float_result': c['float_result'], 'n_cases': c['n_cases']})
    out['all_flip_tasks'] = sorted(allflip, key=lambda x: -x['programs'])
    # semantic classes of the flips
    trap = Counter()
    for r in v2.values():
        if verdict(r['O']) != 'flips':
            continue
        c = cases[str(r['task_id'])]['cases']
        badset = r['O']['bad']
        cls = [classes(c[i]['args']) | classes(c[i]['exp']) for i in badset]
        only_panic = set(badset) == set(r['O'].get('panics', [])) and badset
        tags = set()
        for name in ('negative', 'empty', 'non_ascii'):
            if all(name in s for s in cls):
                tags.add(name)
        if all(s & {'negative', 'empty', 'non_ascii'} for s in cls):
            tags.add('trap_only')
        if only_panic:
            tags.add('panic_only')
        for tg in tags or {'none'}:
            trap[tg] += 1
    out['flip_classes'] = dict(trap)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
