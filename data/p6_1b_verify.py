"""Phase 6, step 1b: is a repair R correct beyond MBPP's few asserts?
PREREGISTRATION_OWNFIX.md §2.

A repair passes MBPP's asserts, usually three. Some do so by luck (Phase 5's
task 404). Each problem with a stable repair gets extra inputs:

  inputs     each assert's arguments, one argument changed at a time and kept in
             its type and sign: integers +-1, +-2, 0, 1; floats +-0.5, x2, 0;
             strings reversed, without the first character, with the first
             character appended; lists reversed, sorted, sorted descending,
             without the last element, with the first duplicated, with the first
             element changed as an integer or float would be; tuples one element
             at a time; booleans flipped. Non-negative values stay non-negative,
             nothing becomes empty. Originals and duplicates are dropped; at most
             MAX_INPUTS = 40, in the order generated.
  expected   MBPP's Python reference solution, run in a subprocess with a time
             limit; an input it raises on, or whose value is not of the
             signature's Rust type, is dropped.
  calibrated an input is kept only if every one of the problem's passing Gemma
             samples (whitespace-distinct, the first CALIB = 10 in sample order)
             returns the reference's value on it in the Rust judge. The
             reference and Gemma's passing samples disagree on some boundary
             inputs (task 66's reference counts 0 as positive); those inputs say
             nothing about R and are dropped.
  verified   R passes every kept input: True; fails one: False; fewer than
             MIN_INPUTS = 10 kept, or no passing sample to calibrate with: None.

Recomputed every round for every problem with a stable repair, using the
passing samples drawn so far; a problem whose calibration samples are unchanged
reuses its inputs. Descriptive: whether each control also returns S's values on
the kept inputs (the differential harness of p6_1). Writes VERIFY (json: per
repair, per control and per problem).

  SOURCES=ownfix_sources.jsonl  MUTANTS=ownfix_mutants.jsonl  VERIFY=ownfix_verify.json
"""
import ast
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m1_make_mutants as m1  # noqa: E402
from mbpp_rust import rust_type  # noqa: E402
from p6_1_repairs import _diff, load_samples  # noqa: E402

SOURCES = os.environ.get('SOURCES', 'ownfix_sources.jsonl')
MUTANTS = os.environ.get('MUTANTS', 'ownfix_mutants.jsonl')
VERIFY = os.environ.get('VERIFY', 'ownfix_verify.json')
WORKERS = int(os.environ.get('WORKERS', 6))
MAX_INPUTS, MIN_INPUTS, CALIB = 40, 10, 10


def vary(v):
    """Values near v of the same kind; never negative if v is not, never empty."""
    if isinstance(v, bool):
        return [not v]
    if isinstance(v, int):
        out = [v + 1, v - 1, v + 2, v - 2, 0, 1]
        return [x for x in out if x != v and (v < 0 or x >= 0)]
    if isinstance(v, float):
        out = [v + 0.5, v - 0.5, v * 2, 0.0]
        return [x for x in out if x != v and (v < 0 or x >= 0)]
    if isinstance(v, str):
        out = [v[::-1], v[1:], v + v[:1]]
        return [x for x in out if x != v and x]
    if isinstance(v, list):
        out = []
        if len(v) > 1:
            out += [v[::-1], v[:-1]]
            try:
                out += [sorted(v), sorted(v, reverse=True)]
            except TypeError:
                pass
        if v:
            out.append([v[0]] + v)
            out += [[x] + v[1:] for x in vary(v[0])]
        return [x for x in out if x != v and x]
    if isinstance(v, tuple):
        return [v[:i] + (x,) + v[i + 1:] for i in range(len(v)) for x in vary(v[i])]
    return []


def inputs_for(sig):
    seen = {json.dumps(list(a), sort_keys=True, default=str) for a, _, _ in sig['cases']}
    out = []
    for args, _, _ in sig['cases']:
        for j, a in enumerate(args):
            for x in vary(a):
                if rust_type(x, owned=False) != sig['params'][j]:
                    continue
                new = list(args[:j]) + [x] + list(args[j + 1:])
                key = json.dumps(new, sort_keys=True, default=str)
                if key not in seen:
                    seen.add(key)
                    out.append(new)
    return out[:MAX_INPUTS]


RUNNER = r'''
import ast, json, sys
ns = {}
exec(sys.argv[1], ns)
f = ns[sys.argv[2]]
out = []
for args in ast.literal_eval(sys.stdin.read()):
    try:
        out.append([True, repr(f(*args))])
    except Exception as e:
        out.append([False, type(e).__name__])
print(json.dumps(out))
'''


def reference_values(task, sig, inputs):
    """[(args, value)] for the inputs the reference solution answers."""
    code = '\n'.join(task['test_imports']) + '\n' + task['code']
    with tempfile.TemporaryDirectory(prefix='p6ref_') as td:
        runner = Path(td) / 'run.py'
        runner.write_text(RUNNER, encoding='utf-8')
        try:
            r = subprocess.run([sys.executable, '-I', str(runner), code, sig['name']],
                               input=repr([tuple(a) for a in inputs]), capture_output=True,
                               text=True, timeout=20, cwd=td)
            res = json.loads(r.stdout)
        except Exception:
            return []
    import ast
    out = []
    for args, (ok, val) in zip(inputs, res):
        if not ok:
            continue
        try:
            v = ast.literal_eval(val)
        except Exception:
            continue
        if isinstance(v, float) and not math.isfinite(v):
            continue
        if rust_type(v, owned=True) == sig['ret']:
            out.append((args, v))
    return out


def _judge_cases(args):
    """(tid, code, cases) -> the set of failing case indices (from the judge's own
    program, read in full), or None if it does not compile or does not finish."""
    tid, code, cases = args
    sig = dict(m1._typeable[tid], cases=cases)
    with tempfile.TemporaryDirectory(prefix='p6ver_') as td:
        rs, binp = Path(td) / 'v.rs', Path(td) / 'v'
        rs.write_text(m1.judge_mod.build_source(code, sig), encoding='utf-8')
        try:
            c = subprocess.run([m1.judge_mod.RUSTC, '-O', '--edition', '2021', str(rs), '-o', str(binp)],
                               capture_output=True, text=True, timeout=m1.judge_mod.COMPILE_TIMEOUT)
            if c.returncode != 0:
                return None
            r = subprocess.run([str(binp)], capture_output=True, text=True,
                               timeout=m1.judge_mod.RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            return None
    m = re.search(r'PASS (\d+) of (\d+)', r.stdout)
    if not m:
        return None
    bad = {int(i) for i in re.findall(r'(?:FAIL|PANIC) (\d+)', r.stdout)}
    return bad if int(m.group(1)) + len(bad) == len(cases) else None


def main():
    from datasets import load_dataset
    ds = load_dataset('mbpp', 'sanitized')
    tasks = {int(t['task_id']): t for s in ('train', 'test', 'validation', 'prompt') for t in ds[s]}
    typeable, _ = m1.load_typeable()
    sources = {(s['task_id'], s['sample']): s for s in m1.read_jsonl(SOURCES) if s['stable']}
    reps = [m for m in m1.read_jsonl(MUTANTS) if m.get('is_r') and (m['task_id'], m['sample']) in sources]
    problems = sorted({m['task_id'] for m in reps})
    samples = load_samples()
    passing = {}
    for (t, smp), (s, lab) in sorted(samples.items()):
        if t in problems and lab['passes'] and not s['truncated']:
            passing.setdefault(t, {}).setdefault(re.sub(r'\s+', '', s['generated_code']), s['generated_code'])
    old = json.load(open(VERIFY, encoding='utf-8')) if os.path.exists(VERIFY) else {'problems': {}, 'mutants': {}}
    res = {'problems': {}, 'mutants': {}}
    live_cases = {}
    with Pool(WORKERS, initializer=m1._init) as pool:
        for t in problems:
            calib = list(passing.get(t, {}).values())[:CALIB]
            ckey = [re.sub(r'\s+', '', c) for c in calib]
            prev = old['problems'].get(str(t))
            if prev and prev['calib'] == ckey:
                cases = ast.literal_eval(prev['cases'])          # repr keeps tuples tuples
            elif not calib:
                cases = []
            else:
                sig = typeable[t]
                ref = reference_values(tasks[t], sig, inputs_for(sig))
                is_close = any(c[2] for c in sig['cases'])
                cand = [(a, v, is_close) for a, v in ref]
                fails = [f for f in pool.map(_judge_cases, [(t, c, cand) for c in calib]) if f is not None]
                # a passing sample that does not compile or finish on the inputs is left out
                bad = set().union(*fails) if fails else set()
                cases = [c for i, c in enumerate(cand) if i not in bad] if fails else []
            res['problems'][str(t)] = {'calib': ckey, 'n_calib': len(calib), 'cases': repr(cases),
                                       'n_inputs': len(cases)}
            live_cases[t] = cases
        todo = [m for m in reps]
        verdicts = pool.map(_judge_cases, [(m['task_id'], m['code'], live_cases[m['task_id']])
                                           for m in todo if len(live_cases[m['task_id']]) >= MIN_INPUTS])
        # descriptive: does a control also return S's values on the kept extra inputs?
        ctrl = [m for m in m1.read_jsonl(MUTANTS) if m.get('is_c') and (m['task_id'], m['sample']) in sources
                and len(live_cases.get(m['task_id'], [])) >= MIN_INPUTS]
        cres = pool.map(_diff, [(m['task_id'], sources[(m['task_id'], m['sample'])]['code'], m['code'],
                                 [c[0] for c in live_cases[m['task_id']]]) for m in ctrl], chunksize=2)
    res['controls'] = {f"{m['task_id']}|{m['sample']}|{m['pos']}|{m['to']}": r for m, r in zip(ctrl, cres)}
    it = iter(verdicts)
    for m in todo:
        p = res['problems'][str(m['task_id'])]
        key = f"{m['task_id']}|{m['sample']}|{m['pos']}|{m['to']}"
        if p['n_calib'] == 0 or p['n_inputs'] < MIN_INPUTS:
            res['mutants'][key] = None
        else:
            f = next(it)
            res['mutants'][key] = (f is not None and not f)
    with open(VERIFY, 'w', encoding='utf-8') as fh:
        json.dump(res, fh)
    vals = list(res['mutants'].values())
    print(f"{len(problems)} problems with a stable repair: calibrated inputs median "
          f"{sorted(p['n_inputs'] for p in res['problems'].values())[len(problems) // 2] if problems else 0}; "
          f"repairs verified {sum(v is True for v in vals)}, failed {sum(v is False for v in vals)}, "
          f"unverifiable {sum(v is None for v in vals)} (of {len(vals)})")


if __name__ == '__main__':
    main()
