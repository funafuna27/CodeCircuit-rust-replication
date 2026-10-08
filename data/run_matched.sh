#!/usr/bin/env bash
# Phase 5 end to end (PREREGISTRATION_MATCHED.md): sources -> mutants -> surprise
# -> matched pairs, in rounds until the target; then graphs -> analysis.
#
#   PHASE=pairs   round 0 on Phase 3's samples, then rounds 1-4 of 20 new samples
#                 per problem until 250 pairs (CPU + GPU, ~1-6 h); stops after
#                 the pairs are written so they can be committed before any graph
#   PHASE=graphs  graphs and the analysis on the pairs written (GPU ~3.5 days)
#
#   setsid nohup ./run_matched.sh > ~/matched_run.log 2>&1 &   with the env's python on PATH
#
# Exit codes: 0 done, 2 gate not met, 1 anything else.

set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
PHASE="${PHASE:-pairs}"
KEEP=~/matched_graph_keep
stage() { echo "=== $(date '+%F %T') $* ==="; }

if pgrep -f 3_generate_graph.py > /dev/null; then
    echo "!!! a graph run is using the GPU"; exit 1
fi

if [ "$PHASE" = pairs ]; then
    for r in 0 1 2 3 4; do
        if [ "$r" -gt 0 ]; then
            stage "round $r: 20 more samples per problem"
            ROUND=$r $PY p5_0_sample.py || exit 1
            stage "round $r: labels"
            SAMPLES=matched_samples_r$r.jsonl OUTPUT=matched_labels_r$r.jsonl \
                $PY w1_label_samples.py || exit 1
        fi
        stage "round $r: sources and mutants"
        $PY p5_1_mutants.py || exit 1
        stage "round $r: surprise"
        $PY p5_2_logprob.py || exit 1
        stage "round $r: matching"
        ROUND=$r $PY p5_3_match.py; rc=$?
        [ $rc -eq 0 ] && { stage "pairs done"; exit 0; }
        [ $rc -eq 2 ] && { stage "GATE NOT MET: stopping, as pre-registered"; exit 2; }
        [ $rc -eq 3 ] || exit 1
    done
    exit 1
fi

if [ "$PHASE" = graphs ]; then
    [ -f matched_rows.jsonl ] || { echo "!!! no matched_rows.jsonl: run PHASE=pairs first"; exit 1; }
    export DATA_FILE=matched_rows.jsonl GRAPH_DIR=matched_graph \
           FEATURES_FILE=gemma_matched_features.jsonl
    export CHUNK_SIZE=200 BATCH_SIZE=64 MAX_PROMPT_CHARS=99999 EMPTY_CACHE=1 MAX_GRAPH_ATTEMPTS=2
    export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
    mkdir -p "$GRAPH_DIR" "$KEEP"
    # chunk cleanup deletes most graphs once featurised: hard-link each finished
    # one outside the repository first (same filesystem, no extra space), while
    # at least 150 GB stay free
    keep() {
        local free; free=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)
        [ "$free" -ge 150 ] || { echo "!!! keeper paused: ${free} GB free"; return; }
        for f in "$GRAPH_DIR"/graph_*.pt; do
            [ -e "$f" ] || continue
            [ -e "$KEEP/$(basename "$f")" ] || ln "$f" "$KEEP/$(basename "$f")"
        done
    }
    ( while true; do keep; sleep 60; done ) &
    keeper=$!
    stage "graphs"
    ./run_chunked.sh; rc=$?
    kill $keeper 2>/dev/null; keep
    echo "graphs kept outside the repository: $(ls "$KEEP" | wc -l)"
    # 4: a graph was given up after repeated failures; §2 drops its pair
    [ $rc -eq 0 ] || [ $rc -eq 4 ] || exit 1

    stage "analysis"
    mkdir -p results/matched
    $PY p5_4_analysis.py 2>&1 | tee results/matched/analysis.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
    stage "done"
fi
