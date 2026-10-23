"""Phase 9, part ②: audit of flips inside mixed tasks (PREREGISTRATION_P9_MUTSTATE.md §7, "Part ②'s labels").

Before part ②'s addendum, a random sample of the flips in mixed tasks is read by eye, to estimate how many
are artefacts of MBPP+ (the classes found in the all-flip audit of 2026-10-22) rather than Gemma's errors.
The estimate is reported; no case and no program is removed because of it (inputs are never calibrated on
Gemma's samples). No signal of any program has been computed when this runs.

Population: mode-O flips (distinct programs) in the mixed tasks of the registered label set (tasks with
fewer than 20 usable cases and the 14 artefact tasks left out): 44 tasks.
Sample: one flip per mixed task, uniform among the task's flips (programs ordered by their first member),
stream [20261023, 21]. Per program: the MBPP prompt, the signature, the stored asserts, the program, MBPP+'s
reference code, and up to SHOW failing cases (random order, stream [20261023, 22]) with the program's value.
Gemma's program runs only in a subprocess (8 GiB address space, compile 120 s, run 30 s), as the judge runs.

Classes, fixed before any program was read:
  genuine        at least one failing case has an input the prompt admits (its wording, the signature's
                 meaning and the stored asserts' domain) and an expected value the prompt determines, and
                 the program's value is wrong there
  convention     every failing case's expected value reflects a choice the prompt leaves open (tie-breaking,
                 order, rounding, the form of the result)
  precondition   every failing case's input lies outside what the prompt states or implies (the answer is
                 undefined or the task is ill-posed there: a negative length, n = 0 for "the n-th", ...)
  numeric        every failing case needs values beyond i64/f64 where Python is unbounded or exact
  semantics      every failing case turns on a Python-Rust difference the prompt does not settle (floor
                 division and remainder of negatives, bytes vs characters, regex dialect, set order)
  mixed_artefact every failing case is an artefact, of more than one class
  unclear        none of the above can be decided from the prompt
A program is an artefact only if every failing case shown is one; when all shown cases are artefacts and the
program fails more, all its failing cases are shown (SHOW_ALL=1 for that task).

  OUT=results/p9_state/c_audit_sample.txt  SHOW=6  [TASKS=comma-separated, SHOW_ALL=1]
"""
import glob
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
import p9_common as C  # noqa: E402
import p9_a3_relabel as R  # noqa: E402
import p8_judge  # noqa: E402

ARTEFACTS = [593, 806, 790, 771, 559, 109, 633, 564, 119, 142, 753, 409, 274, 435]
MIN_CASES = 20
SHOW = int(os.environ.get('SHOW', 6))
SHOW_ALL = os.environ.get('SHOW_ALL', '0') == '1'
ONLY = {int(t) for t in os.environ.get('TASKS', '').split(',') if t}
OUT = os.environ.get('OUT', os.path.join(C.OUT_DIR, 'c_audit_sample.txt'))
PLUS = '/home/user1/p9data/mbppplus/data/test-*.parquet'


def verdict(r):
    if r['stage'] == 'ok':
        return 'holds' if r['n_fail'] + r['n_panic'] == 0 else 'flips'
    return {'run_timeout': 'timeout', 'compile_timeout': 'timeout', 'no_pass_line': 'crash'}.get(r['stage'], 'harness')


def sample():
    cases = json.load(open(R.CASES))['tasks']
    small = {int(t) for t, v in cases.items() if v.get('n_cases') is not None and v['n_cases'] < MIN_CASES}
    excl = small | set(ARTEFACTS)
    per = defaultdict(lambda: {'holds': [], 'flips': []})
    for r in C.read_jsonl(os.path.join(C.OUT_DIR, 'a3_relabel_v2.jsonl.gz')):
        v = verdict(r['O'])
        if r['task_id'] not in excl and v in ('holds', 'flips'):
            per[r['task_id']][v].append(r)
    mixed = sorted(t for t, d in per.items() if d['holds'] and d['flips'])
    rng = np.random.default_rng([20261023, 21])
    picks = []
    for t in mixed:
        fl = sorted(per[t]['flips'], key=lambda r: (r['members'][0][1], r['members'][0][0]))
        picks.append((t, fl[int(rng.integers(len(fl)))], len(per[t]['holds']), len(fl)))
    return cases, picks


def run_cases(code, sig, cs):
    lines = []
    for i, c in enumerate(cs):
        call = f"{sig['name']}({', '.join(R.lit(a, sig['params'][j]) for j, a in enumerate(c['args']))})"
        lines.append(f'    let r = std::panic::catch_unwind(|| {call});\n'
                     f'    match r {{ Ok(v) => println!("VAL {i} {{:?}}", v), Err(_) => println!("VAL {i} PANIC") }};\n')
    src = ('#![allow(warnings)]\n' + R.MAIN_RE.sub('fn __generated_main(', code) +
           '\nfn main() {\n    std::panic::set_hook(Box::new(|_| {}));\n' + ''.join(lines) + '}\n')
    with tempfile.TemporaryDirectory() as td:
        Path(td, 'a.rs').write_text(src)
        cp = subprocess.run([p8_judge.RUSTC, '-O', '--edition', '2021', 'a.rs', '-o', 'a'], cwd=td,
                            capture_output=True, timeout=120)
        if cp.returncode != 0:
            return None
        try:
            out = subprocess.run(['./a'], cwd=td, capture_output=True, text=True, timeout=30).stdout
        except subprocess.TimeoutExpired:
            return {}
    return {int(m.group(1)): m.group(2) for m in re.finditer(r'VAL (\d+) (.*)', out)}


def main():
    import pyarrow.parquet as pq
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    sigs = p8_judge.typeable()
    plus = {int(r['task_id']): r for r in pq.read_table(glob.glob(PLUS)[0]).to_pylist()}
    code_of = {(int(r['task_id']), int(r['sample'])): r['generated_code']
               for r in C.read_jsonl(os.path.join(C.OUT_DIR, 'pool_samples.jsonl.gz'))}
    code_of.update({(int(r['task_id']), -1): r['generated_code']                 # Phase 2's greedy programs
                    for r in C.read_jsonl(os.path.join(HERE, 'gemma_mbpp_rust_responses.jsonl'))})
    cases, picks = sample()
    rng = np.random.default_rng([20261023, 22])
    log = [f'{len(picks)} mixed tasks; one flip each (stream [20261023, 21]); failing cases shown: '
           f'{"all" if SHOW_ALL else SHOW} (stream [20261023, 22])']
    for t, r, n_h, n_f in picks:
        order = rng.permutation(len(r['O']['bad']))            # drawn for every task, so the stream is fixed
        if ONLY and t not in ONLY:
            continue
        sig = sigs[t]
        s, ph = r['members'][0]
        code = code_of[(t, s)]
        bad = [r['O']['bad'][i] for i in order] if SHOW_ALL else [r['O']['bad'][i] for i in order[:SHOW]]
        tc = cases[str(t)]
        vals = run_cases(code, sig, [tc['cases'][i] for i in bad])
        log.append('=' * 100)
        log.append(f'task {t}: {n_h} hold / {n_f} flip; sampled program: members {r["members"][:3]}'
                   f'{"..." if len(r["members"]) > 3 else ""}; fails {len(r["O"]["bad"])} of {tc["n_cases"]} cases'
                   f' ({r["O"]["n_panic"]} panics); base agrees with stored asserts: {tc["base_agrees_with_stored"]}')
        log.append(f'  MBPP prompt: {sig["prompt"]}')
        log.append(f'  signature: {sig["signature"]}')
        log.append(f'  stored asserts: {[(a, e) for a, e, _ in sig["cases"]]}'[:600])
        log.append('  program:\n    ' + code.replace('\n', '\n    '))
        log.append('  MBPP+ reference:\n    ' + plus[t]['code'].strip().replace('\n', '\n    '))
        for k, i in enumerate(bad):
            c = tc['cases'][i]
            got = None if vals is None else vals.get(k, 'NO OUTPUT')
            log.append(f'  case {i}{" (base)" if c["base"] else ""}: args {json.dumps(c["args"])[:300]}\n'
                       f'      expected {json.dumps(c["exp"])[:200]}\n      program  {str(got)[:200]}')
    text = '\n'.join(log) + '\n'
    if not ONLY:
        with open(OUT, 'w', encoding='utf-8') as fh:
            fh.write(text)
    print(text)


if __name__ == '__main__':
    main()
