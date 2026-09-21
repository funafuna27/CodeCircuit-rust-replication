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


def rust_type(v, owned=True):
    """Python literal -> Rust type, or None if not simply representable."""
    if isinstance(v, bool):          # before int: bool is a subclass of int
        return 'bool'
    if isinstance(v, int):
        return 'i64'
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
    """Find the (call, expected) pair inside one assert's test expression."""
    if isinstance(node, ast.Compare) and len(node.comparators) == 1:
        left, right = node.left, node.comparators[0]
        for cand, other in ((left, right), (right, left)):
            if isinstance(cand, ast.Call) and isinstance(cand.func, ast.Name):
                return cand, other
        return None, None
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node, ast.Constant(value=True)     # assert f(...)
        if isinstance(node.func, ast.Attribute) and node.args:
            inner = node.args[0]                      # assert math.isclose(f(...), v)
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                return inner, node.args[1] if len(node.args) > 1 else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        call, _ = _unwrap_call(node.operand)
        if call is not None:
            return call, ast.Constant(value=False)    # assert not f(...)
    return None, None


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
        call, expected = _unwrap_call(stmt.test)
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
        cases.append((args, exp))

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
    return {'task_id': int(task['task_id']), 'name': name, 'params': params,
            'ret': ret, 'signature': f'fn {name}({plist}) -> {ret}',
            'cases': cases, 'prompt': task['prompt']}, None


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
