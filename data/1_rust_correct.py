"""Label the Rust generations by compiling and running them.

This replaces 1_code_correct.py, which asks gpt-4o-mini to score each line. That
judge is not usable as ground truth here: its prompt mandates the step-function
shape of its own labels, it is told to ignore function names, and it disagrees
with execution on 69 of 256 Python problems in the direction of passing broken
code. See the readme's replication log.

Each generation is compiled once with rustc and run once. Every assert is
evaluated inside catch_unwind so an index-out-of-bounds in case 2 does not hide
the fact that case 1 passed, and the whole binary runs under a wall-clock
timeout so a non-terminating solution is a failure rather than a hang.

Output columns per task:
  compiles     rustc exited 0
  passes       compiles AND every assert held
  n_pass       asserts that held (0 if it does not compile)
  n_assert     asserts attempted
  stage        'compile' | 'run' | 'ok' -- where it stopped
  error        first rustc error code or panic line, truncated
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mbpp_rust import load_typeable, rust_literal

RESPONSES = os.environ.get('RESPONSES', 'gemma_mbpp_rust_responses.jsonl')
OUTPUT = os.environ.get('OUTPUT', 'gemma_mbpp_rust_correctness.jsonl')
RUSTC = os.environ.get('RUSTC', str(Path.home() / '.cargo/bin/rustc'))
COMPILE_TIMEOUT = int(os.environ.get('COMPILE_TIMEOUT', 60))
RUN_TIMEOUT = int(os.environ.get('RUN_TIMEOUT', 10))

# The generated code may define its own main; ours has to be the entry point.
MAIN_RE = re.compile(r'\bfn\s+main\s*\(')


def build_case(i, sig, args, expected):
    """One assert, evaluated without letting a panic kill the rest."""
    call_args = ', '.join(rust_literal(a, owned=(sig['params'][j] != '&str'))
                          for j, a in enumerate(args))
    call = f"{sig['name']}({call_args})"
    exp = rust_literal(expected, owned=True)
    if sig['ret'] == 'f64':
        # MBPP's float asserts use math.isclose(rel_tol=0.001).
        check = (f"match r {{ Ok(v) => {{ let e = {exp}; "
                 f"if (v - e).abs() <= 1e-3 * e.abs().max(1.0) {{ p += 1; }} "
                 f"else {{ println!(\"FAIL {i}\"); }} }}, "
                 f"Err(_) => println!(\"PANIC {i}\") }}")
    else:
        check = (f"match r {{ Ok(v) => {{ if v == {exp} {{ p += 1; }} "
                 f"else {{ println!(\"FAIL {i}\"); }} }}, "
                 f"Err(_) => println!(\"PANIC {i}\") }}")
    return (f"    let r = std::panic::catch_unwind(|| {call});\n"
            f"    {check};\n")


def build_source(code, sig):
    body = MAIN_RE.sub('fn __generated_main(', code)
    cases = ''.join(build_case(i, sig, a, e) for i, (a, e) in enumerate(sig['cases']))
    return (
        "#![allow(warnings)]\n"
        f"{body}\n\n"
        "fn main() {\n"
        "    std::panic::set_hook(Box::new(|_| {}));\n"
        "    let mut p = 0;\n"
        f"{cases}"
        f"    println!(\"PASS {{}} of {len(sig['cases'])}\", p);\n"
        "}\n"
    )


def judge(code, sig, workdir):
    src = build_source(code, sig)
    rs = workdir / f"t{sig['task_id']}.rs"
    binp = workdir / f"t{sig['task_id']}"
    rs.write_text(src, encoding='utf-8')

    try:
        c = subprocess.run([RUSTC, '-O', '--edition', '2021',
                            str(rs), '-o', str(binp)],
                           capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return dict(compiles=False, passes=False, n_pass=0, stage='compile',
                    error='rustc TIMEOUT')
    if c.returncode != 0:
        m = re.search(r'^error(\[[^\]]+\])?: .+$', c.stderr, re.MULTILINE)
        if m:
            first = m.group(0)
        else:
            lines = c.stderr.strip().splitlines()
            first = lines[0] if lines else 'rustc failed with no message'
        return dict(compiles=False, passes=False, n_pass=0, stage='compile',
                    error=first[:160])

    try:
        r = subprocess.run([str(binp)], capture_output=True, text=True,
                           timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return dict(compiles=True, passes=False, n_pass=0, stage='run',
                    error='binary TIMEOUT')
    finally:
        binp.unlink(missing_ok=True)

    m = re.search(r'PASS (\d+) of (\d+)', r.stdout)
    if not m:
        return dict(compiles=True, passes=False, n_pass=0, stage='run',
                    error=(r.stderr.strip().splitlines() or ['no PASS line'])[-1][:160])
    n_pass, n_total = int(m.group(1)), int(m.group(2))
    return dict(compiles=True, passes=n_pass == n_total, n_pass=n_pass,
                stage='ok' if n_pass == n_total else 'run',
                error='' if n_pass == n_total else
                      ' '.join(re.findall(r'(?:FAIL|PANIC) \d+', r.stdout))[:160])


def main():
    if not Path(RUSTC).exists():
        sys.exit(f"rustc not found at {RUSTC}; set RUSTC= or install rustup")
    typeable, _ = load_typeable()

    rows = []
    with open(RESPONSES, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"{len(rows)} generations to judge with {RUSTC}")

    results = []
    with tempfile.TemporaryDirectory(prefix='rustjudge_') as td:
        wd = Path(td)
        for n, row in enumerate(rows, 1):
            sig = typeable[int(row['task_id'])]
            v = judge(row['generated_code'], sig, wd)
            results.append({'task_id': int(row['task_id']),
                            'n_assert': len(sig['cases']), **v})
            if n % 25 == 0 or n == len(rows):
                print(f"  {n}/{len(rows)}")

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(results)
    comp = sum(r['compiles'] for r in results)
    pas = sum(r['passes'] for r in results)
    print(f"\nwrote {n} -> {OUTPUT}")
    print(f"  compiles: {comp}/{n} ({comp/n*100:.1f}%)")
    print(f"  passes:   {pas}/{n} ({pas/n*100:.1f}%)")
    print(f"  of those that compile, {pas}/{comp} ({pas/comp*100:.1f}%) pass"
          if comp else "  nothing compiled")
    from collections import Counter
    print("\n  where they stop:")
    for st, k in Counter(r['stage'] for r in results).most_common():
        print(f"    {k:4}  {st}")


if __name__ == '__main__':
    main()
