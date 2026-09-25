#!/usr/bin/env bash
# Rebuild the graphs for the 16 fence-corrupted Python problems.
#
# Run this only AFTER the Rust arm finishes -- it needs the same GPU. Measured
# cost: 124 rows within the 140-token cap, about 8.9 hours.
#
# Why it is needed: the fence-extraction fix landed in 0_generate_responses.py
# but the shipped generations were never repaired, so 16 of 256 Python problems
# have graphs built over code containing literal ``` lines. Their Rust
# counterparts are deleted by 2_incremental_data_rust.py, so the arms are
# asymmetric until this runs.
#
#   Usage:  nohup ./run_python_topup.sh > ~/python_topup.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")"

if pgrep -f 3_generate_graph.py > /dev/null; then
    echo "!!! a graph run is already using the GPU. Wait for it to finish."
    exit 1
fi

PY="${PY:-python}"

echo "=== $(date '+%F %T') repairing generations ==="
$PY 0b_repair_responses.py || exit 1

echo "=== $(date '+%F %T') re-labelling ==="
RESPONSES=gemma_mbpp_responses_repaired.jsonl \
OUTPUT=gemma_mbpp_python_correctness_repaired.jsonl \
$PY 1_python_correct.py || exit 1

echo "=== $(date '+%F %T') building incremental rows ==="
$PY 2c_incremental_repaired.py || exit 1

# PREREGISTRATION §7 cites the committed versions of these three files (117/256
# passing, 132 rows). The labeller runs every assert under a timeout, so a
# machine under load could flip a label; stop rather than build on a moved one.
REGEN="gemma_mbpp_responses_repaired.jsonl gemma_mbpp_python_correctness_repaired.jsonl gemma_mbpp_incremental_repaired.jsonl"
if ! git diff --quiet -- $REGEN; then
    git diff --stat -- $REGEN
    echo "!!! the regenerated inputs differ from the committed ones; nothing built"
    exit 1
fi
echo "regenerated inputs match the committed ones"

export DATA_FILE=gemma_mbpp_incremental_repaired.jsonl
export GRAPH_DIR=python_graph_repaired
export FEATURES_FILE=gemma_mbpp_features_repaired.jsonl
export CHUNK_SIZE=200
export BATCH_SIZE=64              # must match the original Python run
export MAX_PROMPT_CHARS=99999     # the 140-token cap is applied at analysis time
export EMPTY_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
mkdir -p "$GRAPH_DIR"

echo "=== $(date '+%F %T') generating graphs ==="
./run_chunked.sh || exit 1

echo "=== $(date '+%F %T') merging into a copy of the Python features ==="
$PY 4b_merge_topup.py || exit 1

mkdir -p results/python_topup
echo "=== $(date '+%F %T') scoring the repaired Python arm ==="
FEATURES_FILE=gemma_mbpp_features_merged.jsonl \
LABELS_FILE=gemma_mbpp_python_correctness_repaired.jsonl \
SCORES_OUT=results/python_topup/scores_python_topup.json N_JOBS=4 \
    $PY 5_problem_level.py 2>&1 | tee results/python_topup/level_topup.log
[ "${PIPESTATUS[0]}" -eq 0 ] || exit 1

# With all 256 Python generations clean, the cross set is every shared problem
# that is strict in Rust (126). Needs the Rust analysis to have run first.
if [ -f results/rust/scores_rust_strict.json ]; then
    echo "=== $(date '+%F %T') cross-arm against the repaired Python arm ==="
    PY_SCORES=results/python_topup/scores_python_topup.json \
    RUST_SCORES=results/rust/scores_rust_strict.json \
    CROSS_SET=subset_cross_strict.txt RUST_ONLY_SET=subset_rust_only_strict.txt \
        $PY 6_cross_arm.py 2>&1 | tee results/python_topup/cross_topup.log
    SUBSET_FILE=subset_rust_strict.txt RUST_SCORES=results/rust/scores_rust_strict.json \
    PY_SCORES=results/python_topup/scores_python_topup.json CROSS_SET=subset_cross_strict.txt \
        $PY 6b_c5_null.py 2>&1 | tee results/python_topup/c5null_topup.log
else
    echo "results/rust/scores_rust_strict.json not there yet; run 6_cross_arm.py later"
fi

echo
echo "=== done. Record the result in PREREGISTRATION.md section 8 as a deviation,"
echo "    alongside the locked 240-problem number it does not replace. ==="
