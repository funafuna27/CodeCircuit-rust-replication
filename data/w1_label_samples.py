"""Phase 3, step 1: execution labels for every sample. PREREGISTRATION_WITHIN.md §2.

The judge is 1_rust_correct.py's, imported and unchanged -- the one checked
twice, independently, in Phase 2 (350/350). Each call gets its own temporary
directory, so samples of one problem can be judged in parallel without
colliding on the t{task_id}.rs file name the judge writes.

WORKERS is kept modest: the judge runs each binary under a 10 s timeout, and an
overloaded machine must not turn a slow correct solution into a failure.

  SAMPLES=within_samples.jsonl  OUTPUT=within_labels.jsonl  WORKERS=6
"""
import importlib.util
import json
import os
import sys
import tempfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mbpp_rust import load_typeable  # noqa: E402

_spec = importlib.util.spec_from_file_location('judge_mod', os.path.join(HERE, '1_rust_correct.py'))
judge_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judge_mod)

SAMPLES = os.environ.get('SAMPLES', 'within_samples.jsonl')
OUTPUT = os.environ.get('OUTPUT', 'within_labels.jsonl')
WORKERS = int(os.environ.get('WORKERS', 6))

_typeable = None


def _init():
    global _typeable
    _typeable, _ = load_typeable()


def work(row):
    sig = _typeable[int(row['task_id'])]
    with tempfile.TemporaryDirectory(prefix='wjudge_') as td:
        v = judge_mod.judge(row['generated_code'], sig, Path(td))
    return {'task_id': int(row['task_id']), 'sample': int(row['sample']), **v}


def main():
    if not Path(judge_mod.RUSTC).exists():
        sys.exit(f"rustc not found at {judge_mod.RUSTC}")
    with open(SAMPLES, encoding='utf-8') as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f"{len(rows)} samples to judge with {WORKERS} workers")
    with Pool(WORKERS, initializer=_init) as pool:
        out = []
        for i, r in enumerate(pool.imap(work, rows, chunksize=4), 1):
            out.append(r)
            if i % 250 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}", flush=True)
    out.sort(key=lambda r: (r['task_id'], r['sample']))
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    comp = sum(r['compiles'] for r in out)
    pas = sum(r['passes'] for r in out)
    print(f"\nwrote {len(out)} -> {OUTPUT}")
    print(f"  compile {comp}/{len(out)} ({comp / len(out) * 100:.1f}%), "
          f"pass {pas}/{len(out)} ({pas / len(out) * 100:.1f}%)")
    print("  where they stop: " + ", ".join(f"{k} {v}" for k, v in
                                           Counter(r['stage'] for r in out).most_common()))
    timeouts = sum('TIMEOUT' in (r.get('error') or '') for r in out)
    print(f"  timeouts: {timeouts}")


if __name__ == '__main__':
    main()
