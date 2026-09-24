"""Shared MBPP -> Rust helpers: signature derivation and literal emission.

Used by 0_generate_responses_rust.py (to put the required signature in the
prompt) and by 1_rust_correct.py (to build the compilable test harness).

Why the signature goes in the prompt: MBPP never states the function name the
tests call. On the Python arm that cost 210 of 256 generations a NameError
before any assert ran, and the LLM judge used upstream is explicitly told to
ignore function names, so it never saw the failure. In Rust a name or arity
mismatch is a compile error, so ported naively the label would measure naming
rather than reasoning.

Types are read off the literals in test_list and must agree across every assert
for a task. A task whose types cannot be pinned down is rejected, not guessed
at -- 350 of MBPP sanitized's 427 tasks survive that rule.
"""
import ast

__all__ = ['rust_type', 'rust_literal', 'derive_signature', 'load_typeable']


I64_MIN, I64_MAX = -(2 ** 63), 2 ** 63 - 1


def rust_type(v, owned=True):
    """Python literal -> Rust type, or None if not simply representable."""
    if isinstance(v, bool):          # before int: bool is a subclass of int
        return 'bool'
    if isinstance(v, int):
        # Range-checked: an earlier version mapped every Python int to i64, so
        # MBPP task 67's 55-digit assert produced a literal rustc refuses
        # ("integer literal is too large") and the task was recorded as a model
        # compile failure. Exactly one task in all 427 is affected, but an
        # unchecked cast is a trap for any regenerated dataset.
        return 'i64' if I64_MIN <= v <= I64_MAX else None
    if isinstance(v, float):
        return 'f64'
    if isinstance(v, str):
        return 'String' if owned else '&str'
    if isinstance(v, list):
        if not v:
            return None              # empty literal: element type unknown
        inner = {rust_type(x, owned=True) for x in v}
        if len(inner) != 1 or None in inner:
            return None              # heterogeneous or unrepresentable
        return f'Vec<{inner.pop()}>'
    if isinstance(v, tuple):
        if not v:
            return None
        parts = [rust_type(x, owned=True) for x in v]
        if any(p is None for p in parts):
            return None
        return '(' + ', '.join(parts) + ')'
    return None                      # dict, set, None, everything else


def rust_literal(v, owned=True):
    """Python literal -> Rust literal source text."""
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, int):
        return f'{v}i64'
    if isinstance(v, float):
        return f'{v}f64'
    if isinstance(v, str):
        esc = v.replace('\\', '\\\\').replace('"', '\\"')
        return f'String::from("{esc}")' if owned else f'"{esc}"'
    if isinstance(v, list):
        return 'vec![' + ', '.join(rust_literal(x, True) for x in v) + ']'
    if isinstance(v, tuple):
        return '(' + ', '.join(rust_literal(x, True) for x in v) + ')'
    raise ValueError(f'no Rust literal for {v!r}')


def _unwrap_call(node):
    """Find (call, expected, isclose) inside one assert's test expression.

    `isclose` says whether MBPP wrote this assert with math.isclose. An earlier
    version applied a 1e-3 tolerance to every f64-returning task; only 11 of the
    21 use isclose, and the other 10 demand exact equality. It changed no label
    on this dataset (the closest approach is 34x inside the threshold) but it is
    the wrong test and would bite on any regenerated set.
    """
    if isinstance(node, ast.Compare) and len(node.comparators) == 1:
        # Only == is handled. MBPP sanitized contains no non-Eq comparison
        # assert, but silently grading `!=` as `==` would be invisible.
        if not isinstance(node.ops[0], ast.Eq):
            return None, None, False
        left, right = node.left, node.comparators[0]
        for cand, other in ((left, right), (right, left)):
            if isinstance(cand, ast.Call) and isinstance(cand.func, ast.Name):
                return cand, other, False
        return None, None, False
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node, ast.Constant(value=True), False     # assert f(...)
        if isinstance(node.func, ast.Attribute) and node.args:
            inner = node.args[0]                      # assert math.isclose(f(...), v)
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                is_close = node.func.attr == 'isclose'
                return (inner, node.args[1] if len(node.args) > 1 else None,
                        is_close)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        call, _, _ = _unwrap_call(node.operand)
        if call is not None:
            return call, ast.Constant(value=False), False    # assert not f(...)
    return None, None, False


def derive_signature(task):
    """(signature dict, None) if the task is typeable, else (None, reason)."""
    names, arg_sigs, ret_types, cases, reasons = set(), [], set(), [], []
    for src in task['test_list']:
        try:
            stmt = ast.parse(src).body[0]
        except (SyntaxError, IndexError):
            reasons.append('assert does not parse')
            continue
        if not isinstance(stmt, ast.Assert):
            reasons.append('not an assert')
            continue
        call, expected, is_close = _unwrap_call(stmt.test)
        if call is None:
            reasons.append('no bare function call found')
            continue
        names.add(call.func.id)
        if call.keywords:
            reasons.append('keyword argument')
            continue
        try:
            args = [ast.literal_eval(a) for a in call.args]
        except (ValueError, SyntaxError):
            reasons.append('non-literal argument')
            continue
        ts = [rust_type(a, owned=False) for a in args]
        if any(t is None for t in ts):
            reasons.append('argument type not representable')
            continue
        arg_sigs.append(tuple(ts))
        if expected is None:
            reasons.append('no expected value')
            continue
        try:
            exp = ast.literal_eval(expected)
        except (ValueError, SyntaxError):
            reasons.append('non-literal expected value')
            continue
        rt = rust_type(exp, owned=True)
        if rt is None:
            reasons.append('return type not representable')
            continue
        ret_types.add(rt)
        cases.append((args, exp, is_close))

    if len(names) != 1:
        return None, f'{len(names)} distinct function names'
    if not arg_sigs:
        return None, reasons[0] if reasons else 'no usable assert'
    if len(set(arg_sigs)) != 1:
        return None, 'argument types disagree across asserts'
    if len(ret_types) != 1:
        return None, ('return types disagree across asserts' if ret_types
                      else (reasons[0] if reasons else 'no return type'))

    name, ret, params = names.pop(), ret_types.pop(), list(arg_sigs[0])
    plist = ', '.join(f'a{i}: {t}' for i, t in enumerate(params))
    # graded_all_tests is the field that matters for the headline. Skipping an
    # assert and accepting the task anyway leaves 15 of 350 graded on a strict
    # subset, and 4 of those carry passes=True -- task 744's two None-returning
    # asserts are both dropped, so it is graded on one case that a stub passes.
    # Nothing in the output used to say so.
    return {'task_id': int(task['task_id']), 'name': name, 'params': params,
            'ret': ret, 'signature': f'fn {name}({plist}) -> {ret}',
            'cases': cases, 'prompt': task['prompt'],
            'n_mbpp_asserts': len(task['test_list']),
            'graded_all_tests': len(cases) == len(task['test_list']),
            'skipped_reasons': reasons}, None


def load_typeable():
    """Every MBPP sanitized task that can be given a Rust signature."""
    from datasets import load_dataset
    ds = load_dataset('mbpp', 'sanitized')
    out, rejected = {}, {}
    for split in ('train', 'test', 'validation', 'prompt'):
        for task in ds[split]:
            sig, why = derive_signature(task)
            if sig:
                out[sig['task_id']] = sig
            else:
                rejected[int(task['task_id'])] = why
    return out, rejected
