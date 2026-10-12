"""Phase 6, step 1: Gemma's own failing solutions (S), their one-token repairs (R)
and output-identical controls (C). PREREGISTRATION_OWNFIX.md §2.

  S   a distinct sample Gemma wrote: compiles, fails at least one assert through
      a wrong value or a panic (not a timeout), not truncated, prose + ' ' + code
      <= 140 Gemma tokens; within a problem, a sample whose code with all
      whitespace removed equals an earlier kept one (sample order) is dropped.
      Its verdict must be the same in 3 judgings (the draw's label and two more):
      compiles, passes, n_pass and the FAIL/PANIC line all equal.
  mutants  Phase 4's (m1_make_mutants: sites, families, the exact token-count
      filter, the judge under an 8 GiB limit), imported unchanged, on every
      stable S.
  R   a mutant that passes every assert, in 3 judgings.
  C   a mutant that compiles, fails with S's FAIL/PANIC line, and returns S's
      value on every assert: S and the mutant are compiled into one binary, in
      separate modules, and each assert's call is compared by its Debug form
      ({:?}; both panicking counts as equal); 3 runs, all equal. Its site must be
      live: another replacement at the same position compiles and does not
      return S's values (it passes, fails differently, or differs in the harness).
  The differential harness runs only for sources that have a stable R. As a
  guard it first compares S with itself; where that is not 'same' (e.g. S
  returns NaN), the source's harness verdicts are unknown: no control, no
  liveness witness.

Incremental: sources already in SOURCES are not judged again. Appends to
SOURCES (one row per kept S) and MUTANTS (one row per judged mutant).

  SOURCES=ownfix_sources.jsonl  MUTANTS=ownfix_mutants.jsonl  WORKERS=6
  TEST_P4=1 uses Phase 4's killed pair members as S instead (pipeline tests
  only: each has a known repair, its own passing source)
"""
import glob
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m1_make_mutants as m1  # noqa: E402
from mbpp_rust import rust_literal  # noqa: E402

SOURCES = os.environ.get('SOURCES', 'ownfix_sources.jsonl')
MUTANTS = os.environ.get('MUTANTS', 'ownfix_mutants.jsonl')
WORKERS = int(os.environ.get('WORKERS', 6))
TEST_P4 = os.environ.get('TEST_P4', '0') == '1'
LIMIT = int(os.environ.get('LIMIT', 0))
SIG_RE = re.compile(r'(?:FAIL|PANIC) \d+')
MAIN_RE = re.compile(r'\bfn\s+main\s*\(')


def norm(code):
    return re.sub(r'\s+', '', code)


def signature(v):
    """What a failing verdict says, for comparing judgings."""
    return (v['compiles'], v['passes'], v['n_pass'], v.get('error', ''))


def failure_ok(v):
    """Fails through a wrong value or a panic: not a timeout, not a crash."""
    return (v['compiles'] and v['passes'] is False and v.get('stage') == 'run'
            and bool(v.get('error')) and ' '.join(SIG_RE.findall(v['error'])) == v['error'])


def load_samples():
    files = [('within_samples.jsonl', 'within_labels.jsonl')]
    for pat in ('matched_samples_r*.jsonl', 'ownfix_samples_r*.jsonl'):
        for s in sorted(glob.glob(pat)):
            files.append((s, s.replace('_samples_', '_labels_')))
    out = {}
    for sf, lf in files:
        if not os.path.exists(lf):
            raise SystemExit(f"{sf} has no labels ({lf})")
        labels = {(r['task_id'], r['sample']): r for r in m1.read_jsonl(lf)}
        for r in m1.read_jsonl(sf):
            key = (r['task_id'], r['sample'])
            if key in out:
                raise SystemExit(f"sample {key} drawn twice")
            out[key] = (r, labels[key])
    return out


def test_sources():
    """Phase 4's killed pair members: failing programs with a known repair."""
    rows = {r['task_id']: r for r in m1.read_jsonl('mutant_rows.jsonl')}
    out = {}
    for p in json.load(open('mutant_pairs.json', encoding='utf-8')):
        k = rows[p['fail_id']]
        out[(p['problem'], 900 + p['source_sample'])] = (
            {'task_id': p['problem'], 'sample': 900 + p['source_sample'], 'text': k['prose'],
             'generated_code': k['generated_code'], 'truncated': False},
            {'compiles': True, 'passes': False, 'n_pass': -1, 'stage': 'run',
             'error': p['killed_error']})
    return out


def diff_source(code_s, code_c, sig, arg_lists=None):
    """One binary: S and C in their own modules, each call's value compared by its
    Debug form ({:?}), so NaN equals NaN and -0.0 differs from 0.0. The calls are
    the asserts' arguments, or arg_lists if given."""
    name = sig['name']
    pub = re.compile(r'(?<![\w])(?:pub\s+)?fn\s+' + re.escape(name) + r'\b')

    def module(tag, code):
        body = MAIN_RE.sub('fn __generated_main(', code)
        body = pub.sub(f'pub fn {name}', body, count=1)
        return f"mod {tag} {{\n{body}\n}}\n"

    cases = []
    calls = arg_lists if arg_lists is not None else [c[0] for c in sig['cases']]
    for i, args in enumerate(calls):
        a = ', '.join(rust_literal(x, owned=(sig['params'][j] != '&str')) for j, x in enumerate(args))
        cases.append(
            f"    let a = std::panic::catch_unwind(|| s::{name}({a}));\n"
            f"    let b = std::panic::catch_unwind(|| c::{name}({a}));\n"
            f"    match (a, b) {{ (Ok(x), Ok(y)) => {{ if format!(\"{{:?}}\", x) == format!(\"{{:?}}\", y) "
            f"{{ println!(\"SAME {i}\"); }} "
            f"else {{ println!(\"DIFF {i}\"); }} }}, (Err(_), Err(_)) => println!(\"SAME {i}\"), "
            f"_ => println!(\"DIFF {i}\") }}\n")
    return ("#![allow(warnings)]\n" + module('s', code_s) + module('c', code_c)
            + "fn main() {\n    std::panic::set_hook(Box::new(|_| {}));\n"
            + ''.join(cases) + "    println!(\"END\");\n}\n")


def _judge3(args):
    """args: (tid, code, n): the judge's verdict n times."""
    tid, code, n = args
    return [m1._judge((tid, code)) for _ in range(n)]


def _diff(args):
    """args: (tid, code_s, code_c[, arg_lists]): 'same' if every call agrees in 3
    runs, 'diff' if any run shows a difference, else 'error: ...'."""
    import subprocess
    tid, code_s, code_c = args[:3]
    arg_lists = args[3] if len(args) > 3 else None
    sig = m1._typeable[tid]
    with tempfile.TemporaryDirectory(prefix='p6diff_') as td:
        rs, binp = Path(td) / 'd.rs', Path(td) / 'd'
        rs.write_text(diff_source(code_s, code_c, sig, arg_lists), encoding='utf-8')
        try:
            c = subprocess.run([m1.judge_mod.RUSTC, '-O', '--edition', '2021', str(rs), '-o', str(binp)],
                               capture_output=True, text=True, timeout=m1.judge_mod.COMPILE_TIMEOUT)
        except subprocess.TimeoutExpired:
            return 'error: rustc timeout'
        if c.returncode != 0:
            return 'error: does not compile together'
        n = len(arg_lists) if arg_lists is not None else len(sig['cases'])
        for _ in range(3):
            try:
                r = subprocess.run([str(binp)], capture_output=True, text=True,
                                   timeout=m1.judge_mod.RUN_TIMEOUT)
            except subprocess.TimeoutExpired:
                return 'error: run timeout'
            same = re.findall(r'^SAME (\d+)$', r.stdout, re.M)
            diff = re.findall(r'^DIFF (\d+)$', r.stdout, re.M)
            if diff:
                return 'diff'
            if len(same) != n or 'END' not in r.stdout:
                return 'error: incomplete run'
        return 'same'


def main():
    if not Path(m1.judge_mod.RUSTC).exists():
        sys.exit(f"rustc not found at {m1.judge_mod.RUSTC}")
    samples = test_sources() if TEST_P4 else load_samples()
    tok = m1.AutoTokenizer.from_pretrained(m1.MODEL_NAME)
    ntok = lambda text: len(tok(text)['input_ids'])

    kept, seen = [], set()
    for key in sorted(samples):
        s, lab = samples[key]
        if s['truncated'] or not failure_ok(lab):
            continue
        n = ntok(s['text'] + ' ' + s['generated_code'])
        if n > m1.TOKEN_CAP:
            continue
        dkey = (key[0], norm(s['generated_code']))
        if dkey in seen:
            continue
        seen.add(dkey)
        kept.append((key, n))
    if LIMIT:
        allowed = sorted({k[0] for k, _ in kept})[:LIMIT]
        kept = [(k, n) for k, n in kept if k[0] in allowed]
    done = set()
    if os.path.exists(SOURCES):
        done = {(r['task_id'], r['sample']) for r in m1.read_jsonl(SOURCES)}
    new = [(k, n) for k, n in kept if k not in done]
    print(f"{len(kept)} failing sources over {len({k[0] for k, _ in kept})} problems; "
          f"{len(done)} done before, {len(new)} new", flush=True)

    with Pool(WORKERS, initializer=m1._init) as pool:
        # S: two more judgings; stable if all agree with the draw's label
        reps = pool.map(_judge3, [(k[0], samples[k][0]['generated_code'], 2) for k, _ in new], chunksize=2)
        stable = {}
        for (k, n), vs in zip(new, reps):
            lab = samples[k][1]
            ok = all(failure_ok(v) and v['error'] == lab['error'] and v['n_pass'] == lab['n_pass']
                     for v in vs)
            if TEST_P4:
                ok = all(failure_ok(v) for v in vs) and vs[0]['error'] == vs[1]['error']
            stable[k] = (ok, vs[0])
        good = [(k, n) for k, n in new if stable[k][0]]
        print(f"  S stable in 3 judgings: {len(good)} of {len(new)}", flush=True)

        batch = []
        for (tid, smp), n in good:
            s = samples[(tid, smp)][0]
            for m in m1.mutants(s['generated_code']):
                if ntok(s['text'] + ' ' + m['code']) == n:
                    batch.append({'task_id': tid, 'sample': smp, 'tokens': n, **m})
        print(f"  {len(batch)} token-preserving mutants to judge", flush=True)
        verdicts = pool.map(m1._judge, [(b['task_id'], b['code']) for b in batch], chunksize=4)
        for b, v in zip(batch, verdicts):
            b.update(v)

        cand_r = [b for b in batch if b['compiles'] and b['passes']]
        reps = pool.map(_judge3, [(b['task_id'], b['code'], 2) for b in cand_r], chunksize=2)
        for b, vs in zip(cand_r, reps):
            b['r_stable'] = all(v['compiles'] and v['passes'] for v in vs)
        with_r = {(b['task_id'], b['sample']) for b in cand_r if b['r_stable']}
        print(f"  repairs: {len(cand_r)} passing mutants, stable {sum(b['r_stable'] for b in cand_r)}; "
              f"sources with a stable repair: {len(with_r)}", flush=True)

        s_err = {k: stable[k][1]['error'] for k, _ in good}
        cand_c = [b for b in batch if (b['task_id'], b['sample']) in with_r and b['compiles']
                  and b['passes'] is False and b.get('error') == s_err[(b['task_id'], b['sample'])]]
        # guard: the harness must call S equal to itself, or its verdicts there mean nothing
        selfd = pool.map(_diff, [(k[0], samples[k][0]['generated_code'], samples[k][0]['generated_code'])
                                 for k in sorted(with_r)], chunksize=2)
        unstable = {k for k, r in zip(sorted(with_r), selfd) if r != 'same'}
        res = pool.map(_diff, [(b['task_id'], samples[(b['task_id'], b['sample'])][0]['generated_code'],
                                b['code']) for b in cand_c], chunksize=2)
        for b, r in zip(cand_c, res):
            b['diff'] = 'unknown: S differs from itself' if (b['task_id'], b['sample']) in unstable else r
        if unstable:
            print(f"  harness-unstable sources (S not equal to itself): {sorted(unstable)}", flush=True)

    # liveness: another replacement at the same position changes some assert's value
    by_site = defaultdict(list)
    for b in batch:
        by_site[(b['task_id'], b['sample'], b['pos'])].append(b)

    def changes_output(m, k):
        if not m['compiles']:
            return False
        if m['passes'] or m.get('error') != s_err[k]:
            return True
        return m.get('diff') == 'diff'

    for b in batch:
        k = (b['task_id'], b['sample'])
        b['output_identical'] = b.get('diff') == 'same'
        b['live'] = any(changes_output(o, k) for o in by_site[(b['task_id'], b['sample'], b['pos'])]
                        if o is not b)
        b['is_r'] = bool(b.get('r_stable'))
        b['is_c'] = b['output_identical'] and b['live']

    with open(MUTANTS, 'a', encoding='utf-8') as f:
        for b in batch:
            f.write(json.dumps(b, ensure_ascii=False) + '\n')
    n_c = defaultdict(int)
    n_r = defaultdict(int)
    for b in batch:
        n_c[(b['task_id'], b['sample'])] += b['is_c']
        n_r[(b['task_id'], b['sample'])] += b['is_r']
    with open(SOURCES, 'a', encoding='utf-8') as f:
        for (tid, smp), n in new:
            s, lab = samples[(tid, smp)]
            f.write(json.dumps({'task_id': tid, 'sample': smp, 'tokens': n, 'code': s['generated_code'],
                                'prose': s['text'], 'stable': stable[(tid, smp)][0],
                                'harness_unstable': (tid, smp) in unstable,
                                'error': lab['error'], 'n_pass': lab['n_pass'],
                                'n_repairs': n_r[(tid, smp)], 'n_controls': n_c[(tid, smp)]},
                               ensure_ascii=False) + '\n')
    usable = sum(1 for k, _ in good if n_r[k] and n_c[k])
    print(f"  controls: {len(cand_c)} same-failure candidates; output-identical "
          f"{sum(b['output_identical'] for b in batch)}, and live {sum(b['is_c'] for b in batch)}; "
          f"harness errors {sum(1 for b in cand_c if str(b.get('diff')).startswith('error'))}")
    print(f"  usable sources (a stable repair and a control): {usable} of {len(new)}")


if __name__ == '__main__':
    main()
