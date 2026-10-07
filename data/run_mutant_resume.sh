#!/bin/bash
# Phase 4 resume after the 2026-10-05 crash (PREREGISTRATION_MUTANT.md §8, item 1).
# A verbatim copy of /tmp/resume_mutant_inner.sh, which has been running since
# 2026-10-07 07:08: graphs with MAX_GRAPH_ATTEMPTS=2 (graph 147 is given up), then
# the pre-registered analysis. m2_logprob.py had already run. Started with the
# codecircuit env on PATH, from setsid nohup.
cd ~/CodeCircuit-rust/data || exit 1
stage() { echo "=== $(date '+%F %T') $* ==="; }
stage "graphs (resumed after the 2026-10-05 crash; MAX_GRAPH_ATTEMPTS=2)"
export DATA_FILE=mutant_rows.jsonl GRAPH_DIR=mutant_graph FEATURES_FILE=gemma_mutant_features.jsonl
export MAX_GRAPH_ATTEMPTS=2 CHUNK_SIZE=200 BATCH_SIZE=64 MAX_PROMPT_CHARS=99999 EMPTY_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
./run_chunked.sh; rc=$?; [ $rc -eq 0 ] || [ $rc -eq 4 ] || exit 1
stage "analysis"
mkdir -p results/mutant
python m3_mutant_analysis.py 2>&1 | tee results/mutant/analysis.log
[ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
stage "done"
