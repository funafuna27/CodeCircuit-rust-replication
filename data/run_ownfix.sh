#!/usr/bin/env bash
# Phase 6 end to end (PREREGISTRATION_OWNFIX.md): Gemma's failing samples S ->
# repairs R and output-identical controls C -> surprise -> pairs, in rounds until
# the target; then graphs -> analysis; then S's own graphs (descriptive tail).
#
#   PHASE=pairs   round 0 on the samples drawn so far, then rounds 1-8 of 20 new
#                 samples per problem until 250 pairs; stops once the pairs are
#                 written so they can be committed before any graph
#   PHASE=graphs  graphs of R and C, the analysis, then S's graphs and the tail
#
#   setsid nohup ./run_ownfix.sh > ~/ownfix_run.log 2>&1 &   with the env's python on PATH
# Exit codes: 0 done, 2 gate not met, 1 anything else.

set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
PHASE="${PHASE:-pairs}"
stage() { echo "=== $(date '+%F %T') $* ==="; }

if pgrep -f 3_generate_graph.py > /dev/null; then
    echo "!!! a graph run is using the GPU"; exit 1
fi

if [ "$PHASE" = pairs ]; then
    for r in 0 1 2 3 4 5 6 7 8; do
        if [ "$r" -gt 0 ]; then
            stage "round $r: 20 more samples per problem"
            ROUND=$r $PY p6_0_sample.py || exit 1
            if [ -s ownfix_samples_r$r.jsonl ]; then
                stage "round $r: labels"
                SAMPLES=ownfix_samples_r$r.jsonl OUTPUT=ownfix_labels_r$r.jsonl \
                    $PY w1_label_samples.py || exit 1
            fi
        fi
        stage "round $r: sources, repairs and controls"
        $PY p6_1_repairs.py || exit 1
        stage "round $r: repairs checked on extra inputs"
        $PY p6_1b_verify.py || exit 1
        stage "round $r: surprise"
        $PY p6_2_logprob.py || exit 1
        stage "round $r: pairs"
        ROUND=$r $PY p6_3_match.py; rc=$?
        if [ $rc -eq 0 ] || [ $rc -eq 2 ]; then
            stage "natural pairs (descriptive)"
            $PY p6_natural.py || true
            [ $rc -eq 0 ] && { stage "pairs done"; exit 0; }
            stage "GATE NOT MET: stopping, as pre-registered"; exit 2
        fi
        [ $rc -eq 3 ] || exit 1
    done
    exit 1
fi

graphs() {   # $1 rows, $2 graph dir, $3 features file
    export DATA_FILE=$1 GRAPH_DIR=$2 FEATURES_FILE=$3
    export CHUNK_SIZE=200 BATCH_SIZE=64 MAX_PROMPT_CHARS=99999 EMPTY_CACHE=1 MAX_GRAPH_ATTEMPTS=2
    export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
    local keep=~/${2}_keep
    mkdir -p "$GRAPH_DIR" "$keep"
    keeper() {
        local free; free=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)
        [ "$free" -ge 150 ] || { echo "!!! keeper paused: ${free} GB free"; return; }
        for f in "$GRAPH_DIR"/graph_*.pt; do
            [ -e "$f" ] || continue
            [ -e "$keep/$(basename "$f")" ] || ln "$f" "$keep/$(basename "$f")"
        done
    }
    ( while true; do keeper; sleep 60; done ) &
    local kp=$!
    ./run_chunked.sh; local rc=$?
    kill $kp 2>/dev/null; keeper
    echo "graphs kept outside the repository: $(ls "$keep" | wc -l)"
    # 4: a graph was given up after repeated failures; its pair is dropped
    [ $rc -eq 0 ] || [ $rc -eq 4 ]
}

if [ "$PHASE" = graphs ]; then
    [ -f ownfix_rows.jsonl ] || { echo "!!! no ownfix_rows.jsonl: run PHASE=pairs first"; exit 1; }
    [ -f p6_4_analysis.py ] || { echo "!!! p6_4_analysis.py is not written yet"; exit 1; }
    # graphs only from final pairs, unchanged since p6_3 wrote them, and committed
    $PY - <<'CHECK' || { echo "!!! pairs not final, changed since written, or not committed"; exit 1; }
import hashlib, json, subprocess, sys
s = json.load(open('ownfix_pairs_status.json'))
h = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
ok = (s['final'] and s['rows_sha256'] == h('ownfix_rows.jsonl')
      and s['s_rows_sha256'] == h('ownfix_s_rows.jsonl') and s['pairs_sha256'] == h('ownfix_pairs.json'))
files = ['ownfix_pairs_status.json', 'ownfix_pairs.json', 'ownfix_rows.jsonl', 'ownfix_s_rows.jsonl']
committed = all(subprocess.run(['git', 'ls-files', '--error-unmatch', f], capture_output=True).returncode == 0
                for f in files) and subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'] + files).returncode == 0
sys.exit(0 if ok and committed else 1)
CHECK
    stage "graphs: R and C"
    graphs ownfix_rows.jsonl ownfix_graph gemma_ownfix_features.jsonl || exit 1
    stage "analysis"
    mkdir -p results/ownfix
    $PY p6_4_analysis.py 2>&1 | tee results/ownfix/analysis.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
    stage "graphs: S (descriptive tail)"
    graphs ownfix_s_rows.jsonl ownfix_s_graph gemma_ownfix_s_features.jsonl || exit 1
    if [ -f p6_5_tail.py ]; then
        stage "tail analysis"
        $PY p6_5_tail.py 2>&1 | tee results/ownfix/tail.log
    fi
    stage "done"
fi
