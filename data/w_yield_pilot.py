"""Phase 3 yield pilot: how many pairs will a design produce? No graph is built.

The smoke test (20 problems x 4 samples) gave no pair at all: within a problem
the compiling samples nearly always agree on pass/fail, and half the samples
failed to compile, many for wrapping rather than code. Before the real draw,
compare on the first 40 strict problems, K = 20:

  temperature 0.8 vs 1.0     x     Phase 2 cleaning vs + within_clean.tidy

Only yield is measured -- compile rate, mixed problems, pairs -- never anything
a graph would see, so choosing the design from it cannot steer the result.

  SAMPLES_T08 / SAMPLES_T10   w0 outputs (TIDY=0; code_phase2 and raw stored)
  OUT_DIR                     labels and tidy variants are written here
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mbpp_rust import load_typeable  # noqa: E402
from within_clean import tidy  # noqa: E402

_spec = importlib.util.spec_from_file_location('judge_mod', os.path.join(HERE, '1_rust_correct.py'))
judge_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judge_mod)

OUT_DIR = Path(os.environ.get('OUT_DIR', 'within_yield'))
WORKERS = int(os.environ.get('WORKERS', 6))
_typeable = None


def _init():
    global _typeable
    _typeable, _ = load_typeable()


def _judge(args):
    tid, code = args
    with tempfile.TemporaryDirectory(prefix='wyield_') as td:
        return judge_mod.judge(code, _typeable[tid], Path(td))


def label(pool, rows):
    return pool.map(_judge, [(int(r['task_id']), r['generated_code']) for r in rows], chunksize=4)


def write(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def pairs(samples, labels, tag):
    env = dict(os.environ, SAMPLES=str(samples), LABELS=str(labels),
               OUT_ROWS=str(OUT_DIR / f'rows_{tag}.jsonl'),
               OUT_PAIRS=str(OUT_DIR / f'pairs_{tag}.json'))
    out = subprocess.run([sys.executable, os.path.join(HERE, 'w2_make_pairs.py')],
                         env=env, capture_output=True, text=True).stdout
    get = lambda pat: int(re.search(pat, out).group(1))
    return {'eligible': get(r'eligible: (\d+) samples'),
            'mixed': get(r'mixed problems[^:]*: (\d+)')}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    typeable, _ = load_typeable()
    with Pool(WORKERS, initializer=_init) as pool:
        for temp in ('08', '10'):
            src = os.environ[f'SAMPLES_T{temp}']
            rows = [json.loads(l) for l in open(src, encoding='utf-8') if l.strip()]
            n_prob = len({r['task_id'] for r in rows})
            labs = label(pool, rows)
            write(OUT_DIR / f'labels_T{temp}_phase2.jsonl',
                  [{'task_id': r['task_id'], 'sample': r['sample'], **v} for r, v in zip(rows, labs)])

            tidy_rows, changed = [], []
            for i, r in enumerate(rows):
                name = re.search(r'fn\s+(\w+)', r['signature']).group(1)
                t = tidy(r['generated_code'], name)
                tidy_rows.append(dict(r, generated_code=t))
                if t != r['generated_code']:
                    changed.append(i)
            tlabs = list(labs)
            for i, v in zip(changed, label(pool, [tidy_rows[i] for i in changed])):
                tlabs[i] = v
            write(OUT_DIR / f'samples_T{temp}_tidy.jsonl', tidy_rows)
            write(OUT_DIR / f'labels_T{temp}_tidy.jsonl',
                  [{'task_id': r['task_id'], 'sample': r['sample'], **v} for r, v in zip(rows, tlabs)])

            for tag, s, labs_, lab_path in (
                    ('phase2', src, labs, OUT_DIR / f'labels_T{temp}_phase2.jsonl'),
                    ('tidy', OUT_DIR / f'samples_T{temp}_tidy.jsonl', tlabs,
                     OUT_DIR / f'labels_T{temp}_tidy.jsonl')):
                comp = sum(v['compiles'] for v in labs_)
                pas = sum(v['passes'] for v in labs_)
                y = pairs(s, lab_path, f'T{temp}_{tag}')
                print(f"T={'0.8' if temp == '08' else '1.0'}  {tag:6}  compile {comp}/{len(rows)} "
                      f"({comp / len(rows) * 100:4.1f}%)  pass {pas:3}  eligible {y['eligible']:3}  "
                      f"mixed {y['mixed']:2}/{n_prob}  -> ~{y['mixed'] / n_prob * 334:.0f} pairs "
                      f"over 334 problems", flush=True)
            print(f"  tidy changed {len(changed)} of {len(rows)} samples")


if __name__ == '__main__':
    main()
