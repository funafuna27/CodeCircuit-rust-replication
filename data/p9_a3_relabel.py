"""Phase 9, A3 step 2: judge Gemma's passing programs again on MBPP+'s inputs.

Programs: every sample a phase labelled passing (the pool of Phases 3, 5 and 6) and Phase 2's
greedy generations that pass, in the tasks MBPP+ covers (p9_a3_mbppplus.py). Each distinct program
(whitespace removed, within a task) is judged once per mode; every sample keeps its program's
verdict.

The harness is the finite-aware judge's (p8_judge.finite_case / build_source), unchanged in kind:
one catch_unwind per case, exact equality, and the finite-aware isclose branch where MBPP wrote
math.isclose (f64 results only). Only the cases differ: MBPP+'s inputs that fit the signature,
with MBPP+'s expected values. Literals are written by a stricter emitter than mbpp_rust's (escapes
for control characters, i64::MIN, one-element tuples), since MBPP+'s inputs contain them.
No input is calibrated on Gemma's samples: the expected value is MBPP+'s.

Modes: `-O` (the study's) and `-O -C overflow-checks=on`.

Safety: the programs are Gemma's code. Each runs in a Pool worker under an 8 GiB address-space
limit (inherited by rustc and the binary), in its own temporary directory, with timeouts:
compile 120 s, run 30 s (MBPP+ has about 100 cases per task, against MBPP's 3).

  OUT=results/p9_state/a3_relabel.jsonl.gz  WORKERS=6  [LIMIT=n]
"""
import gzip
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
import p9_common as C  # noqa: E402
import p8_judge  # noqa: E402
from mbpp_rust import I64_MIN  # noqa: E402

CASES = os.environ.get('CASES', os.path.join(C.OUT_DIR, 'a3_cases.json'))
OUT = os.environ.get('OUT', os.path.join(C.OUT_DIR, 'a3_relabel.jsonl.gz'))
WORKERS = int(os.environ.get('WORKERS', 6))
LIMIT = int(os.environ.get('LIMIT', 0))
COMPILE_TIMEOUT, RUN_TIMEOUT = 120, 30
MODES = {'O': ['-O'], 'O_ovf': ['-O', '-C', 'overflow-checks=on']}
MAIN_RE = p8_judge._NEW.MAIN_RE


def esc(s):
    out = []
    for ch in s:
        o = ord(ch)
        if ch == '\\':
            out.append('\\\\')
        elif ch == '"':
            out.append('\\"')
        elif ch == '\n':
            out.append('\\n')
        elif ch == '\r':
            out.append('\\r')
        elif ch == '\t':
            out.append('\\t')
        elif ch == '\0':
            out.append('\\0')
        elif o < 0x20 or o == 0x7f or 0xd800 <= o <= 0xdfff:
            out.append('\\u{%x}' % o)
        else:
            out.append(ch)
    return ''.join(out)


def lit(v, owned=True):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, int):
        return 'i64::MIN' if v == I64_MIN else f'{v}i64'
    if isinstance(v, float):
        return f'{v!r}f64'
    if isinstance(v, str):
        return f'String::from("{esc(v)}")' if owned else f'"{esc(v)}"'
    if isinstance(v, list):
        return 'vec![' + ', '.join(lit(x, True) for x in v) + ']'
    if isinstance(v, tuple):
        inner = ', '.join(lit(x, True) for x in v)
        return f'({inner},)' if len(v) == 1 else f'({inner})'
    raise ValueError(f'no literal for {v!r}')


def build_case(i, sig, case):
    call = f"{sig['name']}({', '.join(lit(a, owned=(sig['params'][j] != '&str')) for j, a in enumerate(case['args']))})"
    exp = lit(case['exp'], owned=True)
    if case['is_close'] and sig['ret'] == 'f64':
        check = (f"match r {{ Ok(v) => {{ let e = {exp}; "
                 f"if v == e || (v.is_finite() && e.is_finite() && (v - e).abs() <= 1e-3 * v.abs().max(e.abs())) "
                 f"{{ p += 1; }} else {{ println!(\"FAIL {i}\"); }} }}, "
                 f"Err(_) => println!(\"PANIC {i}\") }}")
    else:
        check = (f"match r {{ Ok(v) => {{ if v == {exp} {{ p += 1; }} "
                 f"else {{ println!(\"FAIL {i}\"); }} }}, "
                 f"Err(_) => println!(\"PANIC {i}\") }}")
    return f"    let r = std::panic::catch_unwind(|| {call});\n    {check};\n"


def build_source(code, sig, cases):
    body = MAIN_RE.sub('fn __generated_main(', code)
    return ("#![allow(warnings)]\n" + body + "\n\nfn main() {\n    std::panic::set_hook(Box::new(|_| {}));\n"
            "    let mut p = 0;\n" + ''.join(build_case(i, sig, c) for i, c in enumerate(cases))
            + f"    println!(\"PASS {{}} of {len(cases)}\", p);\n}}\n")


def judge(code, sig, cases, flags):
    with tempfile.TemporaryDirectory(prefix='p9a3_') as td:
        rs, binp = Path(td) / 'p.rs', Path(td) / 'p'
        rs.write_text(build_source(code, sig, cases), encoding='utf-8')
        t0 = time.time()
        try:
            c = subprocess.run([p8_judge.RUSTC, *flags, '--edition', '2021', str(rs), '-o', str(binp)],
                               capture_output=True, text=True, timeout=COMPILE_TIMEOUT, cwd=td)
        except subprocess.TimeoutExpired:
            return {'stage': 'compile_timeout'}
        if c.returncode != 0:
            m = re.search(r'^error(\[[^\]]+\])?: .+$', c.stderr, re.MULTILINE)
            return {'stage': 'no_compile', 'error': (m.group(0) if m else c.stderr[:160])[:160]}
        ct = time.time() - t0
        try:
            r = subprocess.run([str(binp)], capture_output=True, text=True, timeout=RUN_TIMEOUT, cwd=td)
        except subprocess.TimeoutExpired:
            return {'stage': 'run_timeout', 'compile_sec': round(ct, 1)}
        m = re.search(r'PASS (\d+) of (\d+)', r.stdout)
        fails = [int(x) for x in re.findall(r'FAIL (\d+)', r.stdout)]
        panics = [int(x) for x in re.findall(r'PANIC (\d+)', r.stdout)]
        if not m:
            return {'stage': 'no_pass_line', 'n_fail': len(fails), 'n_panic': len(panics),
                    'error': (r.stderr.strip().splitlines() or [''])[-1][:160]}
        return {'stage': 'ok', 'n_pass': int(m.group(1)), 'n_cases': int(m.group(2)), 'n_fail': len(fails),
                'n_panic': len(panics), 'bad': sorted(fails + panics)[:40], 'compile_sec': round(ct, 1)}


_CASES = None


def _init():
    global _CASES
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    p8_judge.typeable()
    with open(CASES, encoding='utf-8') as f:
        _CASES = json.load(f)['tasks']


def work(item):
    task, code = item
    sig = p8_judge.typeable()[task]
    cases = _CASES[str(task)]['cases']
    out = {'task_id': task}
    for mode, flags in MODES.items():
        try:
            out[mode] = judge(code, sig, cases, flags)
        except Exception as e:
            out[mode] = {'stage': 'judge_error', 'error': type(e).__name__}
    return out


def main():
    with open(CASES, encoding='utf-8') as f:
        cases = json.load(f)['tasks']
    usable = {int(t) for t, v in cases.items() if v.get('n_cases')}
    S = C.OUT_DIR
    samples = C.read_jsonl(os.path.join(S, 'pool_samples.jsonl.gz'))
    labels = {(int(r['task_id']), int(r['sample'])): r for r in C.read_jsonl(os.path.join(S, 'pool_labels.jsonl.gz'))}
    progs = []                                   # (task, sample, phase, code)
    for r in samples:
        k = (int(r['task_id']), int(r['sample']))
        if C.truthy(labels[k]['passes']) and k[0] in usable:
            progs.append((k[0], k[1], r['phase'], r['generated_code']))
    greedy = C.read_jsonl(os.path.join(HERE, 'gemma_mbpp_rust_responses.jsonl'))
    gl = {int(r['task_id']): r for r in C.read_jsonl(os.path.join(HERE, 'gemma_mbpp_rust_correctness.jsonl'))}
    for r in greedy:
        t = int(r['task_id'])
        if C.truthy(gl[t]['passes']) and t in usable:
            progs.append((t, -1, '2', r['generated_code']))
    uniq, members = {}, {}
    for t, s, ph, code in progs:
        key = (t, C.ws_key(code))
        if key not in uniq:
            uniq[key] = (t, code)
        members.setdefault(key, []).append([s, ph])
    items = list(uniq.items())
    if LIMIT:
        items = items[:LIMIT]
    print(f'{len(progs)} passing programs in {len(usable)} usable tasks; {len(uniq)} distinct; judging {len(items)} '
          f'x {len(MODES)} modes, {WORKERS} workers', flush=True)
    t0, n = time.time(), 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with Pool(WORKERS, initializer=_init) as pool, gzip.open(OUT + '.part', 'wt', encoding='utf-8') as f:
        for key, res in zip((k for k, _ in items), pool.imap(work, [v for _, v in items], chunksize=1)):
            res['members'] = members[key]
            f.write(json.dumps(res) + '\n')
            n += 1
            if n % 250 == 0 or n == len(items):
                print(f'  {n}/{len(items)}  {time.time() - t0:.0f} s', flush=True)
    os.replace(OUT + '.part', OUT)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
