"""Label the Python generations by running MBPP's own tests.

The counterpart to 1_rust_correct.py, and the replacement for
1_code_correct.py, which asks gpt-4o-mini to score each line and must not be
used: its prompt mandates the step-function shape of its own labels, it is told
to ignore function names, and it disagrees with execution on 69 of 256 problems
in the direction of passing broken code. (It also carries a hardcoded OpenAI key
inherited from upstream.)

Two bugs in an earlier version of this labelling, both found by independent
re-derivation and both corrected here:

  1. test_imports was never executed, so the 13 problems whose asserts call
     math.isclose failed spuriously. Pass rate 105/256 -> 114/256.
  2. The function name was pulled out of the assert with a regex that returns
     nothing on those same asserts.

MBPP never states the name the tests call, so 210 of 256 generations fail on
NameError before any assert runs. The aliasing rule that forgives this is stated
in full below because an earlier version left its fallback undocumented, and
that fallback is the entire difference between two independent reimplementations
of the same described method (113 vs 114).
"""
import ast
import builtins
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RESPONSES = os.environ.get('RESPONSES', 'gemma_mbpp_responses.jsonl')
OUTPUT = os.environ.get('OUTPUT', 'gemma_mbpp_python_correctness.jsonl')
TIMEOUT = int(os.environ.get('TIMEOUT', 10))
BUILTINS = set(dir(builtins))


def target_name(test_src):
    """The name the assert calls.

    The first Call in the assert whose func is a bare Name, in source order,
    preferring a name not in builtins. If every candidate is a builtin, take the
    first one anyway -- MBPP task 126's test is literally `assert sum(10,15)==6`,
    and that fallback fires exactly once over the whole dataset.
    """
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return None, False
    names = [n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    if not names:
        return None, False
    for n in names:
        if n not in BUILTINS:
            return n, False
    return names[0], True


def single_def(code):
    """The one top-level function, if there is exactly one."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    defs = [n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return defs[0] if len(defs) == 1 else None


def run(code, imports, asserts, alias=None):
    """Each assert in its own process, so one crash does not hide the rest."""
    n_pass, err = 0, ''
    for a in asserts:
        parts = list(imports) + [code]
        if alias:
            parts.append(f'{alias[0]} = {alias[1]}')
        parts.append(a)
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False,
                                         encoding='utf-8') as f:
            f.write('\n'.join(parts))
            path = f.name
        try:
            p = subprocess.run([sys.executable, path], capture_output=True,
                               timeout=TIMEOUT, text=True)
            if p.returncode == 0:
                n_pass += 1
            elif not err:
                tail = [l for l in p.stderr.strip().splitlines() if l.strip()]
                err = tail[-1][:160] if tail else 'nonzero exit'
        except subprocess.TimeoutExpired:
            err = err or 'TIMEOUT'
        finally:
            Path(path).unlink(missing_ok=True)
    return n_pass, err


def main():
    from datasets import load_dataset
    mbpp = {}
    ds = load_dataset('mbpp', 'sanitized')
    for split in ('train', 'test', 'validation', 'prompt'):
        for ex in ds[split]:
            mbpp[int(ex['task_id'])] = ex

    gens = {}
    with open(RESPONSES, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                gens[int(r['task_id'])] = r['generated_code']
    print(f"{len(gens)} generations, {len(mbpp)} MBPP tasks")

    out, n_alias, n_fallback = [], 0, 0
    for n, (tid, code) in enumerate(sorted(gens.items()), 1):
        ex = mbpp[tid]
        imports, asserts = list(ex['test_imports']), list(ex['test_list'])
        s_pass, s_err = run(code, imports, asserts)

        want, fb = target_name(asserts[0])
        have = single_def(code)
        n_fallback += fb
        if want and have and want != have:
            n_alias += 1
            a_pass, a_err = run(code, imports, asserts, alias=(want, have))
        else:
            a_pass, a_err = s_pass, s_err

        out.append({'task_id': tid,
                    'strict': s_pass == len(asserts),
                    'passes': a_pass == len(asserts),
                    'n_pass': a_pass, 'n_assert': len(asserts),
                    'strict_n_pass': s_pass,
                    'want': want, 'have': have, 'builtin_fallback': fb,
                    'error': a_err})
        if n % 64 == 0 or n == len(gens):
            print(f"  {n}/{len(gens)}")

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    s = sum(r['strict'] for r in out)
    a = sum(r['passes'] for r in out)
    print(f"\nwrote {len(out)} -> {OUTPUT}")
    print(f"  strict  {s}/{len(out)} ({s/len(out)*100:.1f}%)")
    print(f"  aliased {a}/{len(out)} ({a/len(out)*100:.1f}%)")
    print(f"  an alias was applied to {n_alias} generations; "
          f"the builtin fallback fired {n_fallback} time(s)")


if __name__ == '__main__':
    main()
