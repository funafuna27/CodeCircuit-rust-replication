"""Phase 9, A3 step 1: MBPP+'s inputs and expected outputs as Rust test cases. Nothing is executed.

MBPP+ (evalplus/mbppplus, fetched by p9_a3_fetch.py into a directory outside the repository)
stores each task's tests as Python source: `inputs = [...]`, `results = [...]`, a per-task
`assertion(out, exp, atol)` and a loop calling it. That source is untrusted data. It is parsed
with ast.parse; the two lists and the loop's `atol` are read with ast.literal_eval. Where a list
holds the names `inf` / `nan` (MBPP+ imports `from math import inf`), literal_eval refuses it,
and a restricted evaluator accepts exactly the node types literal_eval accepts plus those two
names; the tasks that needed it are listed. A task without a literal `results` list (its
expected values come from running code) is not used.

For each of the 350 typeable tasks (mbpp_rust.load_typeable) that MBPP+ covers, every input is
checked against the task's Rust signature: each argument must fit its parameter type and the
expected value the return type (empty containers fit; an int fits f64 and is written as a float).
Inputs that do not fit are dropped and counted. The first len(test_list) inputs are MBPP's own
(base) inputs; their expected values are compared with the stored assert values.

The comparison is the study's judge, unchanged in kind: exact equality, or the finite-aware
isclose branch where MBPP wrote math.isclose. Where MBPP+'s own assertion would compare
differently (a float result compared with np.allclose, a custom assertion), the task is flagged.

  SRC=/home/user1/p9data/mbppplus  OUT=results/p9_state/a3_cases.json
"""
import ast
import glob
import hashlib
import json
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
from mbpp_rust import I64_MAX, I64_MIN, load_typeable  # noqa: E402

SRC = os.environ.get('SRC', '/home/user1/p9data/mbppplus')
OUT = os.environ.get('OUT', os.path.join(HERE, 'results', 'p9_state', 'a3_cases.json'))


def literal(node):
    """ast.literal_eval, or the same with the names inf and nan allowed. Returns (value, extended)."""
    try:
        return ast.literal_eval(node), False
    except ValueError:
        pass

    def ev(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name) and n.id in ('inf', 'nan'):
            return math.inf if n.id == 'inf' else math.nan
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError('unary on a non-number')
            return -v if isinstance(n.op, ast.USub) else +v
        if isinstance(n, ast.List):
            return [ev(e) for e in n.elts]
        if isinstance(n, ast.Tuple):
            return tuple(ev(e) for e in n.elts)
        if isinstance(n, ast.Set):
            return {ev(e) for e in n.elts}
        if isinstance(n, ast.Dict):
            return {ev(k): ev(v) for k, v in zip(n.keys, n.values)}
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)) and isinstance(n.right, ast.Constant) \
                and isinstance(n.right.value, complex):
            return ev(n.left) + n.right.value if isinstance(n.op, ast.Add) else ev(n.left) - n.right.value
        raise ValueError(f'not a literal: {type(n).__name__}')
    return ev(node), True


def parse_task(src):
    """{'inputs', 'results', 'atol', 'assertion_hash', 'extended'} or {'error': ...}."""
    mod = ast.parse(src)
    out, ext = {}, False
    for n in mod.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id in ('inputs', 'results'):
            try:
                out[n.targets[0].id], e = literal(n.value)
                ext |= e
            except (ValueError, SyntaxError, TypeError) as e:
                return {'error': f'{n.targets[0].id}: {e}'[:120]}
        elif isinstance(n, ast.FunctionDef) and n.name == 'assertion':
            out['assertion_src'] = ast.unparse(n)
        elif isinstance(n, ast.For):
            for c in ast.walk(n):
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == 'assertion' \
                        and len(c.args) == 3:
                    try:
                        out['atol'] = ast.literal_eval(c.args[2])
                    except ValueError:
                        out['atol'] = 'non-literal'
    if 'inputs' not in out or 'results' not in out:
        return {'error': 'no literal inputs/results'}
    if len(out['inputs']) != len(out['results']):
        return {'error': 'inputs and results differ in length'}
    out['extended'] = ext
    out['assertion_hash'] = hashlib.sha256(out.pop('assertion_src', '').encode()).hexdigest()[:12]
    return out


def split_tuple(t):
    """'(i64, Vec<(i64, i64)>)' -> ['i64', 'Vec<(i64, i64)>']"""
    inner, parts, depth, cur = t[1:-1], [], 0, ''
    for ch in inner:
        if ch in '<(':
            depth += 1
        elif ch in '>)':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def fits(v, t):
    if t == 'bool':
        return isinstance(v, bool)
    if t == 'i64':
        return isinstance(v, int) and not isinstance(v, bool) and I64_MIN <= v <= I64_MAX
    if t == 'f64':
        return isinstance(v, (int, float)) and not isinstance(v, bool) and \
            (isinstance(v, float) or I64_MIN <= v <= I64_MAX)
    if t in ('String', '&str'):
        return isinstance(v, str)
    if t.startswith('Vec<'):
        return isinstance(v, list) and all(fits(x, t[4:-1]) for x in v)
    if t.startswith('('):
        ps = split_tuple(t)
        return isinstance(v, tuple) and len(v) == len(ps) and all(fits(x, p) for x, p in zip(v, ps))
    return False


def has_nonfinite(v):
    if isinstance(v, float):
        return not math.isfinite(v)
    if isinstance(v, (list, tuple)):
        return any(has_nonfinite(x) for x in v)
    return False


def as_typed(v, t):
    """The value with ints turned into floats where the type says f64 (so it is written 2.0f64)."""
    if t == 'f64':
        return float(v)
    if t.startswith('Vec<'):
        return [as_typed(x, t[4:-1]) for x in v]
    if t.startswith('('):
        return tuple(as_typed(x, p) for x, p in zip(v, split_tuple(t)))
    return v


def main():
    paths = glob.glob(os.path.join(SRC, 'data', 'test-*.parquet'))
    assert len(paths) == 1, paths
    import pyarrow.parquet as pq
    rows = pq.read_table(paths[0]).to_pylist()
    typeable, _ = load_typeable()
    common_hash = Counter()
    parsed = {}
    for r in rows:
        try:
            p = parse_task(r['test'])
        except SyntaxError as e:
            p = {'error': f'syntax: {e}'[:120]}
        p['n_test_list'] = len(r['test_list'])
        parsed[int(r['task_id'])] = p
        if 'assertion_hash' in p:
            common_hash[p['assertion_hash']] += 1
    default_hash = common_hash.most_common(1)[0][0]

    tasks, reasons = {}, Counter()
    for tid in sorted(set(parsed) & set(typeable)):
        p, sig = parsed[tid], typeable[tid]
        rec = {'n_mbpp_asserts': sig['n_mbpp_asserts'], 'graded_all_tests': sig['graded_all_tests'],
               'ret': sig['ret'], 'params': sig['params']}
        if 'error' in p:
            rec['error'] = p['error']
            reasons[p['error'].split(':')[0]] += 1
            tasks[tid] = rec
            continue
        is_close = any(c[2] for c in sig['cases'])
        ptypes = [('&str' if t == '&str' else t) for t in sig['params']]
        cases, drop = [], Counter()
        for i, (inp, exp) in enumerate(zip(p['inputs'], p['results'])):
            if not isinstance(inp, list) or len(inp) != len(ptypes):
                drop['arity'] += 1
                continue
            if not all(fits(a, t) for a, t in zip(inp, ptypes)):
                drop['argument type'] += 1
                continue
            if any(has_nonfinite(a) for a in inp):
                drop['non-finite argument'] += 1
                continue
            if not fits(exp, sig['ret']):
                drop['result type'] += 1
                continue
            if has_nonfinite(exp):
                drop['non-finite expected'] += 1
                continue
            cases.append({'i': i, 'base': i < p['n_test_list'],
                          'args': [as_typed(a, t) for a, t in zip(inp, ptypes)],
                          'exp': as_typed(exp, sig['ret']), 'is_close': is_close})
        base_stored = [c for c in sig['cases']]
        base_mbppp = [c for c in cases if c['base']]
        agree = None
        if len(base_mbppp) == len(base_stored) == p['n_test_list']:
            agree = all(list(a) == c['args'] and (b == c['exp'] or (isinstance(b, float) and math.isclose(b, c['exp'], rel_tol=1e-9)))
                        for (a, b, _), c in zip(base_stored, base_mbppp))
        rec.update({'n_inputs': len(p['inputs']), 'n_cases': len(cases), 'dropped': dict(drop),
                    'base_kept': len(base_mbppp), 'base_agrees_with_stored': agree,
                    'atol': p.get('atol'), 'custom_assertion': p['assertion_hash'] != default_hash,
                    'extended_literals': p['extended'], 'is_close': is_close,
                    'float_result': 'f64' in sig['ret'], 'cases': cases})
        tasks[tid] = rec

    summary = {
        'mbppplus_tasks': len(rows), 'parsed_ok': sum('error' not in p for p in parsed.values()),
        'parse_errors': dict(Counter(p['error'].split(':')[0] for p in parsed.values() if 'error' in p)),
        'typeable': len(typeable), 'overlap': len(tasks),
        'overlap_strict': sum(t['graded_all_tests'] for t in tasks.values()),
        'overlap_usable': sum(t.get('n_cases', 0) > 0 for t in tasks.values()),
        'overlap_errors': dict(reasons),
        'cases_total': sum(t.get('n_cases', 0) for t in tasks.values()),
        'inputs_total': sum(t.get('n_inputs', 0) for t in tasks.values()),
        'cases_per_task_quartiles': None,
        'dropped': dict(sum((Counter(t.get('dropped', {})) for t in tasks.values()), Counter())),
        'base_agrees_with_stored': dict(Counter(str(t.get('base_agrees_with_stored')) for t in tasks.values())),
        'custom_assertion': sum(t.get('custom_assertion', False) for t in tasks.values()),
        'float_result_tasks': sum(t.get('float_result', False) for t in tasks.values()),
        'extended_literal_tasks': [k for k, t in tasks.items() if t.get('extended_literals')],
        'atol_values': dict(Counter(str(t.get('atol')) for t in tasks.values())),
    }
    import statistics
    nc = [t['n_cases'] for t in tasks.values() if t.get('n_cases')]
    summary['cases_per_task_quartiles'] = statistics.quantiles(nc, n=4) + [min(nc), max(nc)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'tasks': {str(k): v for k, v in tasks.items()}}, f)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
