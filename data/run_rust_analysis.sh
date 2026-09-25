#!/usr/bin/env bash
# The pre-registered Rust analysis, end to end (PREREGISTRATION.md §2-§5, §8 v4).
#
# Committed before the Rust features were complete, so the day the run finishes
# is mechanical: no command, population, set or threshold is chosen after a
# Rust number has been seen.
#
#   PY=/path/to/python ./run_rust_analysis.sh
#
# It refuses to analyse an incomplete file: the integrity step demands all 1734
# rows and all 214 compiling problems, which also makes it safe to start while
# another graph run (the Python top-up) is using the GPU.
#
# Rehearsal -- the identical path on a snapshot, labels shuffled across
# problems so nothing about the real feature-label relation is visible:
#   RUST_FEATURES=snap.jsonl RUST_LABELS=shuffled.jsonl OUT_DIR=rehearsal \
#   EXPECT_ROWS=600 ALLOW_PARTIAL=1 ./run_rust_analysis.sh

set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
RUST_FEATURES="${RUST_FEATURES:-gemma_mbpp_rust_features.jsonl}"
RUST_LABELS="${RUST_LABELS:-gemma_mbpp_rust_correctness.jsonl}"
OUT_DIR="${OUT_DIR:-results/rust}"
EXPECT_ROWS="${EXPECT_ROWS:-1734}"
export N_JOBS="${N_JOBS:-4}"
export ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
mkdir -p "$OUT_DIR"

if [ "$ALLOW_PARTIAL" = 1 ]; then
    echo "=== REHEARSAL: $RUST_FEATURES with labels $RUST_LABELS -- no number here is a result ==="
fi

echo "=== $(date '+%F %T') integrity of $RUST_FEATURES ==="
$PY - "$RUST_FEATURES" "$RUST_LABELS" "$EXPECT_ROWS" <<'PY' 2>&1 | tee "$OUT_DIR/integrity.log"
import json, sys
import numpy as np
path, labels_path, expect = sys.argv[1], sys.argv[2], int(sys.argv[3])
rd = lambda p: [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
rows = rd(path)
compiling = {int(r['task_id']) for r in rd(labels_path) if r['compiles']}
keys = [(int(r['expr_id']), int(r['step_number'])) for r in rows]
probs = {p for p, _ in keys}
X = np.array([r['feature'] for r in rows], dtype=float)
checks = [
    (f'{len(rows)} rows, expected {expect}', len(rows) == expect),
    ('batch_size is 64 on every row', {r['batch_size'] for r in rows} == {64}),
    ('no duplicated (problem, step)', len(set(keys)) == len(keys)),
    ('49 dims, all finite', X.shape[1] == 49 and bool(np.isfinite(X).all())),
    ('every problem has a step-0 row', probs == {p for p, s in keys if s == 0}),
    ('every problem is a compiling task', probs <= compiling),
]
if expect == 1734:
    checks.append((f'all {len(compiling)} compiling tasks present', probs == compiling))
print(f"  {len(rows)} rows, {len(probs)} problems")
for name, ok in checks:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
if not all(ok for _, ok in checks):
    sys.exit("!!! integrity failed; nothing analysed")
PY

level() {  # name, then env assignments for 5_problem_level.py
    local name=$1; shift
    echo "=== $(date '+%F %T') 5_problem_level.py, population: $name ==="
    env "$@" FEATURES_FILE="$RUST_FEATURES" LABELS_FILE="$RUST_LABELS" \
        SCORES_OUT="$OUT_DIR/scores_rust_$name.json" \
        $PY 5_problem_level.py 2>&1 | tee "$OUT_DIR/level_$name.log"
}
level strict SUBSET_FILE=subset_rust_strict.txt    # the headline, §2
level loose SUBSET_FILE=                           # reported alongside, never instead

cross() {  # name, Python scores, Rust population, cross set, Rust-only set
    echo "=== $(date '+%F %T') 6_cross_arm.py: $1 ==="
    PY_SCORES="$2" RUST_SCORES="$OUT_DIR/scores_rust_$3.json" \
    CROSS_SET="$4" RUST_ONLY_SET="$5" \
        $PY 6_cross_arm.py 2>&1 | tee "$OUT_DIR/cross_$1.log"
}
# primary: like-for-like Python fit (240) x strict Rust fit, on the 120 in both
cross primary results/python/scores_python_240.json strict \
      subset_cross_primary.txt subset_rust_only_strict.txt
# alongside: the version-2 definition, locked Python fit (256) x loose Rust fit
cross as_v2 results/python/scores_python_256.json loose \
      subset_cross_loose.txt subset_rust_only_loose.txt

c5null() {  # name, Rust population subset, Rust population, Python scores, cross set
    echo "=== $(date '+%F %T') 6b_c5_null.py: $1 (K=${K:-100} refits on shuffled labels) ==="
    RUST_FEATURES="$RUST_FEATURES" RUST_LABELS="$RUST_LABELS" SUBSET_FILE="$2" \
    RUST_SCORES="$OUT_DIR/scores_rust_$3.json" PY_SCORES="$4" CROSS_SET="$5" \
        $PY 6b_c5_null.py 2>&1 | tee "$OUT_DIR/c5null_$1.log"
}
c5null primary subset_rust_strict.txt strict results/python/scores_python_240.json \
       subset_cross_primary.txt
c5null as_v2 "" loose results/python/scores_python_256.json subset_cross_loose.txt

echo
echo "=== $(date '+%F %T') done. Read it against the table in PREREGISTRATION.md §5 ==="
for f in level_strict level_loose cross_primary cross_as_v2; do
    echo "--- $f ---"
    sed -n '/=== summary ===/,$p' "$OUT_DIR/$f.log"
done
for f in c5null_primary c5null_as_v2; do
    echo "--- $f ---"
    sed -n '/=== C5 against/,$p' "$OUT_DIR/$f.log"
done
