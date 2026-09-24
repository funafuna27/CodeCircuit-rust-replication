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

echo
echo "=== done. To use the repaired arm: ==="
echo "  1. merge gemma_mbpp_features_repaired.jsonl into a copy of"
echo "     gemma_mbpp_features.jsonl, replacing the 16 problems' rows"
echo "  2. re-run 5_problem_level.py with LABELS_FILE=gemma_mbpp_python_correctness_repaired.jsonl"
echo "  3. record the result in PREREGISTRATION.md section 8 as a deviation,"
echo "     alongside the locked 240-problem number it replaces"
