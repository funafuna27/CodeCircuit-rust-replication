"""Write the task-id sets the pre-registration names, from the label files alone.

There are two kinds of set, and they are used in two different ways
(PREREGISTRATION.md §2, §4 C5, §8 version 4):

  POPULATIONS -- given to 5_problem_level.py as SUBSET_FILE, which REFITS on them.
  A population says which problems the arm consists of at all.
    subset_rust_strict.txt        compiles, and every MBPP assert gradeable (206)
    subset_python_parses.txt      the Python generations that parse (240; written
                                  earlier, only checked here)

  EVALUATION SETS -- given to 6_cross_arm.py, which RESTRICTS an arm's existing
  out-of-fold score vector to them and never refits. Refitting on 80 or 120
  problems would confound the subset with a smaller training set while the
  learning curve is still climbing (§6). The locked Python comparator 0.7816 on
  the 129 shared problems was computed this way.
    subset_cross_primary.txt      shared, strict in Rust, parses in Python (120)
    subset_cross_loose.txt        shared, the version-2 definition (129)
    subset_cross_strict.txt       shared and strict in Rust (126): the cross set
                                  once the Python top-up repairs the 16 (§7)
    subset_rust_only_strict.txt   strict Rust problems Python never saw (80)
    subset_rust_only_loose.txt    compiling Rust problems Python never saw (85)

Every count is asserted, so a changed label file fails here instead of quietly
moving a pre-registered population.
"""
import json
import os

RUST_LABELS = os.environ.get('RUST_LABELS', 'gemma_mbpp_rust_correctness.jsonl')
PY_LABELS = os.environ.get('PY_LABELS', 'gemma_mbpp_python_correctness.jsonl')
PY_PARSES = 'subset_python_parses.txt'


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def read_ids(path):
    with open(path, encoding='utf-8') as f:
        return {int(l) for l in f if l.strip() and not l.startswith('#')}


def write_ids(path, ids, header, rust, expect):
    ids = sorted(ids)
    n_pos = sum(bool(rust[t]['passes']) for t in ids) if rust else None
    if len(ids) != expect:
        raise SystemExit(f"{path}: expected {expect} problems, got {len(ids)}. "
                         f"A label file changed; do not edit the expectation.")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# {header}\n")
        f.write(f"# {len(ids)} problems" +
                (f", {n_pos} passing in Rust" if n_pos is not None else "") +
                ". Written by 5a_make_subsets.py.\n")
        for t in ids:
            f.write(f"{t}\n")
    print(f"  {path:30} {len(ids):4}" + (f"  Rust pos {n_pos}" if n_pos is not None else ""))


def main():
    rust = {int(r['task_id']): r for r in read_jsonl(RUST_LABELS)}
    py = {int(r['task_id']) for r in read_jsonl(PY_LABELS)}
    parses = read_ids(PY_PARSES)

    if len(py) != 256 or len(parses) != 240 or not parses <= py:
        raise SystemExit("the Python label file or subset_python_parses.txt changed")

    loose = {t for t, r in rust.items() if r['compiles']}
    strict = {t for t in loose if rust[t]['graded_all_tests']}

    print("populations (5_problem_level.py SUBSET_FILE, refits):")
    write_ids('subset_rust_strict.txt', strict,
              'Rust population, strict: compiles and every MBPP assert gradeable',
              rust, 206)
    print(f"  {PY_PARSES:30} {len(parses):4}  (checked, not rewritten)")

    print("evaluation sets (6_cross_arm.py, restrict a score vector, never refit):")
    write_ids('subset_cross_primary.txt', strict & parses,
              'cross-arm primary: shared, strict in Rust, parses in Python',
              rust, 120)
    write_ids('subset_cross_loose.txt', loose & py,
              'cross-arm as defined in version 2: shared, compiles in Rust',
              rust, 129)
    write_ids('subset_cross_strict.txt', strict & py,
              'cross-arm after the Python top-up: shared, strict in Rust',
              rust, 126)
    write_ids('subset_rust_only_strict.txt', strict - py,
              'Rust headline where Python never saw the problem, strict',
              rust, 80)
    write_ids('subset_rust_only_loose.txt', loose - py,
              'Rust headline where Python never saw the problem, loose',
              rust, 85)


if __name__ == '__main__':
    main()
