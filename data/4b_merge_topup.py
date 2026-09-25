"""Merge the rebuilt graphs of the 16 repaired Python problems into a COPY of the
Python features. Run after run_python_topup.sh has built them; the locked
features file is only ever read.

  ORIG_FEATURES   gemma_mbpp_features.jsonl              locked, never written
  TOPUP_FEATURES  gemma_mbpp_features_repaired.jsonl     from the top-up run
  TOPUP_DATA      gemma_mbpp_incremental_repaired.jsonl  the 16 problems and their rows
  OUTPUT          gemma_mbpp_features_merged.jsonl

The other 240 problems' lines are copied byte for byte in their original order,
and the output is re-read at the end to prove it. The repaired problems' rows
follow them, with row_index offset by 1,000,000 so they sort last in
5_problem_level.py and stay traceable to their line in TOPUP_DATA.

The top-up runs with MAX_PROMPT_CHARS=99999 and relies on the analysis-time
140-token cap. That is not a looser filter: on this data the cap is strictly
tighter than the original 550-character one (the longest row inside the cap is
486 characters, and no repaired row is over 550 characters yet inside the cap).

Writes nothing unless the top-up is complete, unduplicated, finite and built at
BATCH_SIZE=64 -- a short merge would quietly change the Python population.
"""
import json
import os
import sys

import numpy as np

ORIG = os.environ.get('ORIG_FEATURES', 'gemma_mbpp_features.jsonl')
TOPUP = os.environ.get('TOPUP_FEATURES', 'gemma_mbpp_features_repaired.jsonl')
TOPUP_DATA = os.environ.get('TOPUP_DATA', 'gemma_mbpp_incremental_repaired.jsonl')
PARSES = os.environ.get('PARSES', 'subset_python_parses.txt')
OUTPUT = os.environ.get('OUTPUT', 'gemma_mbpp_features_merged.jsonl')
OFFSET = 1_000_000
BATCH_SIZE = 64


def fail(msg):
    sys.exit(f"!!! {msg}\n!!! nothing written")


def read_lines(path):
    with open(path, encoding='utf-8') as f:
        return [l if l.endswith('\n') else l + '\n' for l in f if l.strip()]


def main():
    if os.path.abspath(OUTPUT) in (os.path.abspath(ORIG), os.path.abspath(TOPUP)):
        fail("OUTPUT would overwrite an input")

    orig_lines = read_lines(ORIG)
    orig = [json.loads(l) for l in orig_lines]
    orig_problems = {int(r['expr_id']) for r in orig}
    dims = len(orig[0]['feature'])

    data = [json.loads(l) for l in read_lines(TOPUP_DATA)]
    want = {}
    for r in data:
        want[int(r['task_id'])] = want.get(int(r['task_id']), 0) + 1
    repaired = set(want)

    with open(PARSES, encoding='utf-8') as f:
        parses = {int(l) for l in f if l.strip() and not l.startswith('#')}
    if repaired != orig_problems - parses:
        fail(f"{TOPUP_DATA} covers {sorted(repaired)}, but the problems whose "
             f"generation does not parse are {sorted(orig_problems - parses)}")

    top = [json.loads(l) for l in read_lines(TOPUP)]
    stray = sorted({int(r['expr_id']) for r in top} - repaired)
    if stray:
        fail(f"{TOPUP} has rows for problems that were not repaired: {stray}")
    have = {}
    for r in top:
        have[int(r['expr_id'])] = have.get(int(r['expr_id']), 0) + 1
    short = {t: (have.get(t, 0), n) for t, n in want.items() if have.get(t, 0) != n}
    if short:
        fail(f"top-up incomplete, (have, want) rows per problem: {short}")
    keys = [(int(r['expr_id']), int(r['step_number'])) for r in top]
    if len(set(keys)) != len(keys):
        fail(f"{len(keys) - len(set(keys))} duplicated (expr_id, step_number) rows")
    bad_bs = {r.get('batch_size') for r in top} - {BATCH_SIZE}
    if bad_bs:
        fail(f"top-up built at batch_size {sorted(bad_bs)}, the study uses {BATCH_SIZE}")
    X = np.array([r['feature'] for r in top], dtype=float)
    if X.shape[1] != dims or not np.isfinite(X).all():
        fail(f"top-up features are {X.shape[1]}-dim (want {dims}) "
             f"with {int((~np.isfinite(X)).sum())} non-finite entries")

    kept = [l for r, l in zip(orig, orig_lines) if int(r['expr_id']) not in repaired]
    top.sort(key=lambda r: r['row_index'])
    new = []
    for r in top:
        r = dict(r)
        r['row_index'] = OFFSET + int(r['row_index'])
        r['source'] = 'python_topup'
        new.append(json.dumps(r, ensure_ascii=False) + '\n')

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        f.writelines(kept)
        f.writelines(new)

    # prove it: the 240 are untouched, the 16 are replaced, nothing else moved
    out_lines = read_lines(OUTPUT)
    out = [json.loads(l) for l in out_lines]
    assert out_lines[:len(kept)] == kept, "kept rows changed on the way through"
    assert {int(r['expr_id']) for r in out} == orig_problems, "problem set changed"
    ri = [r['row_index'] for r in out]
    assert ri == sorted(ri) and len(set(ri)) == len(ri), "row_index not unique and ordered"

    n_old = len(orig) - len(kept)
    print(f"{ORIG}: {len(orig)} rows, {len(orig_problems)} problems (read only)")
    print(f"  replaced {len(repaired)} problems: {n_old} old rows -> {len(new)} rebuilt rows")
    print(f"  kept {len(kept)} rows of the other {len(orig_problems) - len(repaired)} "
          f"problems, byte for byte")
    print(f"wrote {len(out)} rows -> {OUTPUT}")


if __name__ == '__main__':
    main()
