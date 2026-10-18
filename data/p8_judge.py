"""Phase 8: the finite-aware Rust judge. Design §2 (PREREGISTRATION_INTERVENTION.md); SPEC decision 1.

The study's judge (1_rust_correct.py, unchanged) checks an assert MBPP wrote with math.isclose as
    v == e || (v - e).abs() <= 1e-3 * v.abs().max(e.abs())
which accepts v = +-inf for any finite e (inf <= inf is true). Phase 8 judges with the same program
except for that one branch:
    v == e || (v.is_finite() && e.is_finite() && (v - e).abs() <= 1e-3 * v.abs().max(e.abs()))
`v == e` is unchanged (equal infinities still match; MBPP's expected values are all finite), NaN never
matches (Python's isclose rule). Every other assert is built by the study's build_case, unchanged, so
the two judges write byte-identical programs for every task that is not f64 + isclose.

This module holds two private copies of 1_rust_correct: _OLD (never patched; used only to show that
the old path reproduces the stored labels) and _NEW (build_case = finite_case). init_worker() also
patches build_case in every other loaded copy of 1_rust_correct (m1_make_mutants and p6_1b_verify
each load their own), so code that judges through m1._judge inside a worker is finite-aware too.

  judge(code, sig, workdir)         the finite-aware judge; the same return dict as 1_rust_correct.judge
  judge_old(code, sig, workdir)     the study's judge, unchanged (agreement checks only)
  judge3(task, code, finite=True)   Phase 6's stability protocol (p6_1_repairs: one judging, two more
                                    if it passes): {'verdicts': [...], 'stable_pass': bool}
  judge_cases(task, code, cases, finite=True)   p6_1b_verify._judge_cases: the set of failing case
                                    indices of the judge's own program on `cases`, or None if it does
                                    not compile or finish
  verify_extra(task, code, finite=True, cases=None)   p6_1b_verify's rule on the calibrated extra
                                    inputs of ownfix_verify.json['problems'][str(task)] (or on `cases`):
                                    True (passes every input) / False / None (< 10 inputs or no
                                    calibrating sample)
  label(task, code, finite=True)    {'label': 'verified' | 'lucky' | 'unverifiable' | 'fail' |
                                    'no_compile' | 'judge_error', 'verdicts', 'stable_pass', 'extra',
                                    'extra_bad'}; 'judge_error' = the judge itself raised (e.g. a
                                    MemoryError under the rlimit); none occurs at the Phase 6 sites
  probe_values(task, code, cases)   the Debug form of each case's return value ('PANIC' if it
                                    panics), or None (diagnostics: which outputs are non-finite)
  init_worker()                     Pool initializer: 8 GiB RLIMIT_AS (inherited by rustc and the
                                    binary), the typeable signatures, and the patch above
  canaries()                        INFINITY / -INFINITY / NaN -> fail, exact -> pass, 1.0005*e
                                    (isclose) -> pass, 1.002*e -> fail; the old judge passes INFINITY

Usage (cwd data/):  python p8_judge.py --test      (canaries; a few seconds of rustc)
Env: RUSTC, COMPILE_TIMEOUT, RUN_TIMEOUT as 1_rust_correct.py; HF_DATASETS_OFFLINE=1 is set if unset.
data/1_code_correct.py is never opened, imported or run.
"""
import ast
import gc
import importlib.util
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import types
import warnings
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
from mbpp_rust import load_typeable, rust_literal  # noqa: E402

JUDGE_FILE = os.path.join(HERE, '1_rust_correct.py')
VERIFY_FILE = os.path.join(HERE, 'ownfix_verify.json')
MEM_LIMIT = 8 << 30                       # as m1_make_mutants.MEM_LIMIT
MIN_INPUTS, CALIB = 10, 10                # p6_1b_verify.MIN_INPUTS, CALIB (checked in --test)


def _load_copy(name):
    spec = importlib.util.spec_from_file_location(name, JUDGE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_OLD = _load_copy('p8_judge_old')
_OLD.__p8_role__ = 'old'
OLD_CASE = _OLD.build_case


def finite_case(i, sig, args, expected, is_close):
    """1_rust_correct.build_case with the tolerant f64 branch made finite-aware; every other case
    is the study's own text."""
    if not (is_close and sig['ret'] == 'f64'):
        return OLD_CASE(i, sig, args, expected, is_close)
    call_args = ', '.join(rust_literal(a, owned=(sig['params'][j] != '&str'))
                          for j, a in enumerate(args))
    call = f"{sig['name']}({call_args})"
    exp = rust_literal(expected, owned=True)
    check = (f"match r {{ Ok(v) => {{ let e = {exp}; "
             f"if v == e || (v.is_finite() && e.is_finite() && (v - e).abs() <= 1e-3 * v.abs().max(e.abs())) "
             f"{{ p += 1; }} else {{ println!(\"FAIL {i}\"); }} }}, "
             f"Err(_) => println!(\"PANIC {i}\") }}")
    return (f"    let r = std::panic::catch_unwind(|| {call});\n"
            f"    {check};\n")


finite_case.__p8_finite__ = True
_NEW = _load_copy('p8_judge_new')
_NEW.build_case = finite_case
_NEW.__p8_role__ = 'new'
RUSTC = _NEW.RUSTC

TYPEABLE = None
_VERIFY = None


# ---------------------------------------------------------------- setup
def patch_all_copies():
    """Set build_case = finite_case in every loaded module object of 1_rust_correct.py except _OLD.
    Returns (n patched now, n copies that are finite-aware in total)."""
    target = os.path.realpath(JUDGE_FILE)
    n_new, n_all = 0, 0
    with warnings.catch_warnings():              # touching deprecated module objects (torch) warns
        warnings.simplefilter('ignore')
        mods = [o for o in gc.get_objects() if isinstance(o, types.ModuleType) and o is not _OLD
                and isinstance(getattr(o, '__file__', None), str)
                and os.path.realpath(o.__file__) == target]
    for o in mods:
        if o.build_case is not finite_case:
            o.build_case = finite_case
            n_new += 1
        n_all += 1
    return n_new, n_all


def typeable():
    global TYPEABLE
    if TYPEABLE is None:
        TYPEABLE, _ = load_typeable()
    return TYPEABLE


def verify_data():
    global _VERIFY
    if _VERIFY is None:
        with open(VERIFY_FILE, encoding='utf-8') as f:
            _VERIFY = json.load(f)
    return _VERIFY


def init_worker():
    """Pool initializer: the 8 GiB address-space limit (inherited by rustc and the binary), the
    typeable signatures, and every other copy of 1_rust_correct made finite-aware."""
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT, MEM_LIMIT))
    typeable()
    patch_all_copies()


def _mod(finite):
    return _NEW if finite else _OLD


# ---------------------------------------------------------------- judging
def judge(code, sig, workdir):
    return _NEW.judge(code, sig, Path(workdir))


def judge_old(code, sig, workdir):
    return _OLD.judge(code, sig, Path(workdir))


def judge_once(task, code, finite=True):
    """m1_make_mutants._judge with the chosen judge: a judge exception becomes compiles=None."""
    try:
        with tempfile.TemporaryDirectory(prefix='p8judge_') as td:
            return _mod(finite).judge(code, typeable()[task], Path(td))
    except Exception as e:                       # e.g. MemoryError reading output
        return {'compiles': None, 'passes': None, 'n_pass': 0, 'stage': 'judge',
                'error': f'{type(e).__name__}'[:160]}


def judge3(task, code, finite=True):
    """Phase 6's protocol (p6_1_repairs): one judging; if it passes, two more. A program passes
    stably iff all three pass."""
    vs = [judge_once(task, code, finite)]
    if vs[0]['compiles'] and vs[0]['passes']:
        vs += [judge_once(task, code, finite) for _ in range(2)]
    return {'verdicts': vs, 'stable_pass': len(vs) == 3 and all(v['compiles'] and v['passes'] for v in vs)}


def judge_cases(task, code, cases, finite=True):
    """p6_1b_verify._judge_cases with the chosen judge: the set of failing case indices, or None if
    the program does not compile or does not finish."""
    sig = dict(typeable()[task], cases=cases)
    mod = _mod(finite)
    with tempfile.TemporaryDirectory(prefix='p8ver_') as td:
        rs, binp = Path(td) / 'v.rs', Path(td) / 'v'
        rs.write_text(mod.build_source(code, sig), encoding='utf-8')
        try:
            c = subprocess.run([mod.RUSTC, '-O', '--edition', '2021', str(rs), '-o', str(binp)],
                               capture_output=True, text=True, timeout=mod.COMPILE_TIMEOUT)
            if c.returncode != 0:
                return None
            r = subprocess.run([str(binp)], capture_output=True, text=True, timeout=mod.RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            return None
    m = re.search(r'PASS (\d+) of (\d+)', r.stdout)
    if not m:
        return None
    bad = {int(i) for i in re.findall(r'(?:FAIL|PANIC) (\d+)', r.stdout)}
    return bad if int(m.group(1)) + len(bad) == len(cases) else None


def stored_cases(task):
    """The calibrated extra inputs p6_1b_verify kept for this problem, or None if it has none."""
    p = verify_data()['problems'].get(str(task))
    if p is None or p['n_calib'] == 0 or p['n_inputs'] < MIN_INPUTS:
        return None
    return ast.literal_eval(p['cases'])          # repr keeps tuples tuples


def verify_extra(task, code, finite=True, cases=None, detail=False):
    """p6_1b_verify's verdict: True iff the program compiles, finishes and passes every calibrated
    input; None if the problem has < MIN_INPUTS inputs (or no calibrating sample)."""
    if cases is None:
        cases = stored_cases(task)
    elif len(cases) < MIN_INPUTS:
        cases = None
    if cases is None:
        return (None, None) if detail else None
    f = judge_cases(task, code, cases, finite)
    v = f is not None and not f
    return (v, None if f is None else sorted(f)) if detail else v


def label(task, code, finite=True):
    """The Phase 8 label of a program: judge3, then the extra inputs for a stable pass."""
    j = judge3(task, code, finite)
    v1 = j['verdicts'][0]
    out = {'verdicts': j['verdicts'], 'stable_pass': j['stable_pass'], 'extra': None, 'extra_bad': None}
    if v1['compiles'] is None:
        out['label'] = 'judge_error'
    elif v1['compiles'] is False:
        out['label'] = 'no_compile'
    elif not j['stable_pass']:
        out['label'] = 'fail'
    else:
        out['extra'], out['extra_bad'] = verify_extra(task, code, finite, detail=True)
        out['label'] = {True: 'verified', False: 'lucky', None: 'unverifiable'}[out['extra']]
    return out


def probe_values(task, code, cases=None):
    """Debug form of each case's value (as final/g1_inf.py): ['inf', '0.5', 'PANIC', ...], or None
    if the program does not compile or finish."""
    sig = typeable()[task]
    cases = sig['cases'] if cases is None else cases
    body = _OLD.MAIN_RE.sub('fn __generated_main(', code)
    lines = []
    for i, (a, _, _) in enumerate(cases):
        ca = ', '.join(rust_literal(x, owned=(sig['params'][j] != '&str')) for j, x in enumerate(a))
        lines.append(f'    let r = std::panic::catch_unwind(|| {sig["name"]}({ca}));\n'
                     f'    match r {{ Ok(v) => println!("VAL {i} {{:?}}", v), Err(_) => println!("VAL {i} PANIC") }};\n')
    src = ('#![allow(warnings)]\n' + body + '\nfn main() {\n    std::panic::set_hook(Box::new(|_| {}));\n'
           + ''.join(lines) + '}\n')
    with tempfile.TemporaryDirectory(prefix='p8val_') as td:
        rs, binp = Path(td) / 'v.rs', Path(td) / 'v'
        rs.write_text(src, encoding='utf-8')
        try:
            c = subprocess.run([RUSTC, '-O', '--edition', '2021', str(rs), '-o', str(binp)],
                               capture_output=True, text=True, timeout=_OLD.COMPILE_TIMEOUT)
            if c.returncode != 0:
                return None
            r = subprocess.run([str(binp)], capture_output=True, text=True, timeout=_OLD.RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            return None
    vals = {}
    for ln in r.stdout.split('\n'):
        if ln.startswith('VAL '):
            _, i, v = ln.split(' ', 2)
            vals[int(i)] = v
    return [vals.get(i) for i in range(len(cases))]


def nonfinite(value):
    """A probe_values string that is a non-finite f64."""
    return value in ('inf', '-inf', 'NaN')


def rustc_version():
    return subprocess.run([RUSTC, '--version'], capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- canaries
CANARY_SIG = {'task_id': 990001, 'name': 'canary', 'params': ['f64'], 'ret': 'f64',
              'cases': [([1.0], 2.5, True), ([2.0], 2.5, True)]}
CANARY_EXACT_SIG = dict(CANARY_SIG, task_id=990002, cases=[([1.0], 2.5, False), ([2.0], 2.5, False)])
CANARIES = [  # (name, sig, body, expected passes: finite judge, old judge)
    ('INFINITY', CANARY_SIG, 'f64::INFINITY', False, True),
    ('NEG_INFINITY', CANARY_SIG, 'f64::NEG_INFINITY', False, True),
    ('NaN', CANARY_SIG, 'f64::NAN', False, False),
    ('exact', CANARY_SIG, '2.5', True, True),
    ('1.0005*e isclose', CANARY_SIG, '2.5 * 1.0005', True, True),
    ('1.002*e isclose', CANARY_SIG, '2.5 * 1.002', False, False),
    ('exact assert, 1.0005*e', CANARY_EXACT_SIG, '2.5 * 1.0005', False, False),
    ('exact assert, INFINITY', CANARY_EXACT_SIG, 'f64::INFINITY', False, False),
    ('exact assert, exact', CANARY_EXACT_SIG, '2.5', True, True),
]


def canaries():
    """Returns (ok, rows). Each canary is judged by both judges; also checks that the two judges
    write identical programs for the non-tolerant case."""
    rows, ok = [], True
    with tempfile.TemporaryDirectory(prefix='p8can_') as td:
        for name, sig, body, exp_new, exp_old in CANARIES:
            code = f'fn canary(a0: f64) -> f64 {{ let _ = a0; {body} }}'
            vn = judge(code, sig, Path(td))
            vo = judge_old(code, sig, Path(td))
            good = vn['compiles'] and vn['passes'] == exp_new and vo['passes'] == exp_old
            ok &= bool(good)
            rows.append({'canary': name, 'finite': vn['passes'], 'old': vo['passes'],
                         'expected_finite': exp_new, 'expected_old': exp_old, 'ok': bool(good)})
    same = _OLD.build_source('fn canary(a0: f64) -> f64 { a0 }', CANARY_EXACT_SIG) == \
        _NEW.build_source('fn canary(a0: f64) -> f64 { a0 }', CANARY_EXACT_SIG)
    rows.append({'canary': 'non-isclose program identical in both judges', 'ok': same})
    ok &= same
    return ok, rows


def _selftest():
    ok, rows = canaries()
    for r in rows:
        print(('PASS ' if r['ok'] else 'FAIL ') + json.dumps(r))
    import p6_1b_verify as v                     # constants as Phase 6 used them
    c_ok = (v.MIN_INPUTS, v.CALIB) == (MIN_INPUTS, CALIB)
    print(('PASS' if c_ok else 'FAIL') + f' p6_1b_verify MIN_INPUTS, CALIB = {v.MIN_INPUTS}, {v.CALIB}')
    n_new, n_all = patch_all_copies()
    m1_ok = v.m1.judge_mod.build_case is finite_case and _OLD.build_case is OLD_CASE
    print(('PASS' if m1_ok else 'FAIL') + f' patch: {n_new} copies patched now, {n_all} finite-aware '
          f'(m1 copy finite: {v.m1.judge_mod.build_case is finite_case}; _OLD untouched: {_OLD.build_case is OLD_CASE})')
    print(rustc_version())
    return ok and c_ok and m1_ok


if __name__ == '__main__':
    if '--test' in sys.argv:
        sys.exit(0 if _selftest() else 1)
    print(__doc__)
