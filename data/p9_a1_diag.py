"""Phase 9, A1: rustc's JSON diagnostics as labels.

Every sample is compiled twice, never run:
  lib      the generated code alone, `--crate-type lib --emit=metadata`, warnings on. This runs
           name resolution, type checking and borrow checking, and keeps lints such as unused_mut.
  harness  the program the finite-aware judge builds (p8_judge: the code with any `fn main(`
           renamed, plus the assert harness and `#![allow(warnings)]`), compiled exactly as the
           judge compiles it (`-O --edition 2021`, full build). The binary is deleted unrun.
           Its exit status must reproduce the stored `compiles` label.

Each diagnostic keeps its level, its code (an error code such as E0384, or a lint name), its
message, its spans and its children's spans (help and note, with suggested replacements). A span
is mapped back to the generated code as character offsets; spans in the harness part are marked.

rustc skips borrow checking in a function whose body has a type or name-resolution error (checked
by hand on rustc 1.98.1, ~/p9work/a1_cases.py); other functions are still checked. Counts of
borrow-check errors are therefore lower bounds wherever such errors coexist.

Safety: nothing generated is executed. Each worker runs rustc under an 8 GiB address-space limit
with a 60 s timeout, in its own temporary directory. data/1_code_correct.py is never opened.

  SAMPLES=within_samples.jsonl  OUT=results/p9_state/a1_diag.jsonl.gz  WORKERS=6  [LIMIT=n]
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
from multiprocessing import Pool
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
import p8_judge  # noqa: E402  (finite-aware judge; its build_source is the harness)

SAMPLES = os.environ.get('SAMPLES', 'within_samples.jsonl')
OUT = os.environ.get('OUT', os.path.join('results', 'p9_state', 'a1_diag.jsonl.gz'))
WORKERS = int(os.environ.get('WORKERS', 6))
LIMIT = int(os.environ.get('LIMIT', 0))
RUSTC = p8_judge.RUSTC
TIMEOUT = 60
MEM_LIMIT = 8 << 30
MAIN_RE = p8_judge._NEW.MAIN_RE
HARNESS_HEAD = '#![allow(warnings)]\n'


def byte_to_char(text):
    """Map every UTF-8 byte offset of `text` (and its end) to a character offset."""
    out = []
    for i, ch in enumerate(text):
        out.extend([i] * len(ch.encode('utf-8')))
    out.append(len(text))
    return out


def body_map(code):
    """For the judge's renamed body: a list mapping each body character to a code character."""
    m, last = [], 0
    for mt in MAIN_RE.finditer(code):
        m.extend(range(last, mt.start()))
        m.extend([mt.start()] * len('fn __generated_main('))
        last = mt.end()
    m.extend(range(last, len(code)))
    m.append(len(code))
    return m


def parse(stderr, b2c, to_code):
    """Diagnostics from rustc's JSON stream. to_code(char offset in the source) -> (where, offset)."""
    def span(s):
        a, b = b2c[min(s['byte_start'], len(b2c) - 1)], b2c[min(s['byte_end'], len(b2c) - 1)]
        wa, ca = to_code(a)
        wb, cb = to_code(b)
        out = {'p': bool(s.get('is_primary')), 'w': wa, 's': ca, 'e': cb if wb == wa else ca}
        if s.get('label'):
            out['label'] = s['label'][:160]
        if s.get('suggested_replacement') is not None:
            out['repl'] = s['suggested_replacement'][:80]
        return out

    diags = []
    for line in stderr.splitlines():
        if not line.startswith('{'):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get('$message_type', 'diagnostic') != 'diagnostic' or d.get('level') not in ('error', 'warning'):
            continue
        code = (d.get('code') or {}).get('code')
        if not d.get('spans') and code is None:       # "aborting due to", "N warnings emitted"
            continue
        diags.append({
            'level': d['level'], 'code': code, 'msg': d['message'][:200],
            'spans': [span(s) for s in d['spans']],
            'children': [{'level': c['level'], 'msg': c['message'][:160], 'spans': [span(s) for s in c['spans']]}
                         for c in d.get('children', [])],
        })
    return diags


def compile_lib(code, td):
    src = Path(td) / 'lib.rs'
    src.write_text(code, encoding='utf-8')
    b2c = byte_to_char(code)
    t0 = time.time()
    try:
        p = subprocess.run([RUSTC, '--edition', '2021', '--crate-type', 'lib', '--emit=metadata',
                            '--error-format=json', '--crate-name', 'gen', str(src), '--out-dir', td],
                           capture_output=True, text=True, timeout=TIMEOUT, cwd=td)
    except subprocess.TimeoutExpired:
        return {'rc': None, 'timeout': True, 'sec': round(time.time() - t0, 2), 'diags': []}
    diags = parse(p.stderr, b2c, lambda c: ('code', c))
    return {'rc': p.returncode, 'timeout': False, 'sec': round(time.time() - t0, 2), 'diags': diags}


def compile_harness(code, sig, td):
    src_text = p8_judge._NEW.build_source(code, sig)
    body = MAIN_RE.sub('fn __generated_main(', code)
    assert src_text.startswith(HARNESS_HEAD + body + '\n'), 'harness layout changed'
    head, bmap = len(HARNESS_HEAD), body_map(code)

    def to_code(c):
        if c < head:
            return ('head', 0)
        if c - head < len(body):
            return ('code', bmap[c - head])
        if c - head == len(body):
            return ('code', len(code))
        return ('harness', c - head - len(body))

    src = Path(td) / 'h.rs'
    binp = Path(td) / 'h'
    src.write_text(src_text, encoding='utf-8')
    b2c = byte_to_char(src_text)
    t0 = time.time()
    try:
        p = subprocess.run([RUSTC, '-O', '--edition', '2021', '--error-format=json', str(src), '-o', str(binp)],
                           capture_output=True, text=True, timeout=TIMEOUT, cwd=td)
    except subprocess.TimeoutExpired:
        return {'rc': None, 'timeout': True, 'sec': round(time.time() - t0, 2), 'diags': []}
    finally:
        binp.unlink(missing_ok=True)
    return {'rc': p.returncode, 'timeout': False, 'sec': round(time.time() - t0, 2),
            'diags': parse(p.stderr, b2c, to_code)}


def _init():
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT, MEM_LIMIT))
    p8_judge.typeable()


def work(row):
    task, code = int(row['task_id']), row['generated_code']
    sig = p8_judge.typeable()[task]
    out = {'task_id': task, 'sample': int(row['sample'])}
    for mode in ('lib', 'harness'):
        try:
            with tempfile.TemporaryDirectory(prefix='p9a1_') as td:
                out[mode] = compile_lib(code, td) if mode == 'lib' else compile_harness(code, sig, td)
        except Exception as e:                     # e.g. MemoryError reading rustc's output
            out[mode] = {'rc': None, 'timeout': False, 'error': type(e).__name__, 'diags': []}
    return out


def main():
    if not Path(RUSTC).exists():
        sys.exit(f'rustc not found at {RUSTC}')
    import p9_common
    rows = p9_common.read_jsonl(SAMPLES)
    if LIMIT:
        rows = rows[:LIMIT]
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    print(f'{len(rows)} samples, {WORKERS} workers, {p8_judge.rustc_version()}', flush=True)
    t0, n = time.time(), 0
    with Pool(WORKERS, initializer=_init) as pool, gzip.open(OUT + '.part', 'wt', encoding='utf-8') as f:
        for r in pool.imap(work, rows, chunksize=4):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            n += 1
            if n % 500 == 0 or n == len(rows):
                print(f'  {n}/{len(rows)}  {time.time() - t0:.0f} s', flush=True)
    os.replace(OUT + '.part', OUT)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
