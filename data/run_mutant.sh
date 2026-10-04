#!/usr/bin/env bash
# Phase 4 end to end (PREREGISTRATION_MUTANT.md): mutate -> judge -> pair and
# gate -> surprise -> graphs -> analysis.
#
#   PHASE=pairs   mutate, judge, pair; stop at the gate              (CPU ~30-60 min)
#   PHASE=graphs  surprise, graphs and the analysis on the pairs written (GPU ~2 days)
#   PHASE=all     both (default)
#
#   nohup ./run_mutant.sh > ~/mutant_run.log 2>&1 &     with the env's python on PATH

set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
PHASE="${PHASE:-all}"
stage() { echo "=== $(date '+%F %T') $* ==="; }

if pgrep -f 3_generate_graph.py > /dev/null; then
    echo "!!! a graph run is using the GPU"; exit 1
fi

if [ "$PHASE" = pairs ] || [ "$PHASE" = all ]; then
    stage "mutating, judging and pairing"
    $PY m1_make_mutants.py; rc=$?
    if [ $rc -eq 2 ]; then
        stage "GATE NOT MET: stopping, as pre-registered"
        exit 2
    fi
    [ $rc -eq 0 ] || exit 1
    [ "$PHASE" = pairs ] && { stage "pairs done"; exit 0; }
fi

if [ "$PHASE" = graphs ] || [ "$PHASE" = all ]; then
    [ -f mutant_rows.jsonl ] || { echo "!!! no mutant_rows.jsonl: run PHASE=pairs first"; exit 1; }
    stage "the model's surprise at each member"
    $PY m2_logprob.py || exit 1

    stage "graphs"
    export DATA_FILE=mutant_rows.jsonl GRAPH_DIR=mutant_graph \
           FEATURES_FILE=gemma_mutant_features.jsonl
    export CHUNK_SIZE=200 BATCH_SIZE=64 MAX_PROMPT_CHARS=99999 EMPTY_CACHE=1
    export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
    mkdir -p "$GRAPH_DIR"
    ./run_chunked.sh || exit 1

    stage "analysis"
    mkdir -p results/mutant
    $PY m3_mutant_analysis.py 2>&1 | tee results/mutant/analysis.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
    stage "done"
fi
