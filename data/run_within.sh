#!/usr/bin/env bash
# Phase 3 end to end (PREREGISTRATION_WITHIN.md): sample -> label -> pair and
# gate -> graphs -> analysis.
#
#   PHASE=pilot   sample, label, pair; stop at the gate      (GPU ~1-2 h, CPU ~30 min)
#   PHASE=graphs  graphs and the analysis, on the pairs the pilot wrote (GPU ~2-3 days)
#   PHASE=all     both (default)
#
#   nohup ./run_within.sh > ~/within_run.log 2>&1 &     with the env's python on PATH

set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
PHASE="${PHASE:-all}"
stage() { echo "=== $(date '+%F %T') $* ==="; }

if pgrep -f 3_generate_graph.py > /dev/null; then
    echo "!!! a graph run is using the GPU"; exit 1
fi

if [ "$PHASE" = pilot ] || [ "$PHASE" = all ]; then
    stage "sampling, first draw (samples 0-9)"
    SAMPLE_OFFSET=0 $PY w0_sample_rust.py || exit 1
    stage "labelling"
    $PY w1_label_samples.py || exit 1
    stage "pairing and the gate"
    $PY w2_make_pairs.py; rc=$?
    if [ $rc -eq 2 ]; then
        stage "gate not met: the one pre-registered second draw (samples 10-19)"
        SAMPLE_OFFSET=10 $PY w0_sample_rust.py || exit 1
        $PY w1_label_samples.py || exit 1
        $PY w2_make_pairs.py; rc=$?
        if [ $rc -eq 2 ]; then
            stage "GATE NOT MET after the second draw: stopping, as pre-registered"
            exit 2
        fi
    fi
    [ $rc -eq 0 ] || exit 1
    [ "$PHASE" = pilot ] && { stage "pilot done"; exit 0; }
fi

if [ "$PHASE" = graphs ] || [ "$PHASE" = all ]; then
    [ -f within_rows.jsonl ] || { echo "!!! no within_rows.jsonl: run the pilot first"; exit 1; }
    stage "graphs"
    export DATA_FILE=within_rows.jsonl GRAPH_DIR=within_graph \
           FEATURES_FILE=gemma_within_features.jsonl
    export CHUNK_SIZE=200 BATCH_SIZE=64 MAX_PROMPT_CHARS=99999 EMPTY_CACHE=1
    export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
    mkdir -p "$GRAPH_DIR"
    ./run_chunked.sh || exit 1

    stage "analysis"
    mkdir -p results/within
    FEATURES=gemma_within_features.jsonl OUT=results/within/analysis.json \
        $PY w3_within_analysis.py 2>&1 | tee results/within/analysis.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
    stage "done"
fi
