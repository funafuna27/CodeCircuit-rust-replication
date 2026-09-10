#!/usr/bin/env bash
# Drive the chunked pipeline until every graph has been featurized.
#
#   generate CHUNK_SIZE graphs -> extract features -> free the .pt files
#
# A full language is ~1420 graphs (~2 days at BATCH_SIZE=64), so this is meant
# to run unattended. Every stage is resumable: script 3 skips whatever is in
# the ledger, script 4 skips (expr_id, step_number) pairs it already wrote, and
# cleanup ledgers a graph before deleting it. Graphs are written to a .partial
# name and renamed, so a kill can never leave a truncated file that later looks
# like a finished one.
#
# Usage:  nohup ./run_chunked.sh > ~/chunked_run.log 2>&1 &
#
# Exit-code contract with 3_generate_graph.py:
#   0  generated graphs normally
#   1  some attributions failed -- retry, then stop rather than drop samples
#   3  nothing left to generate (the only clean finish)

set -uo pipefail
cd "$(dirname "$0")"

# Must exceed MAX_GRAPH_ATTEMPTS in 3_generate_graph.py (default 3). A graph
# that keeps failing burns one attempt per restart; the extra pass is the one
# where script 3 gives up on it and walks past, so the chunk can finish.
MAX_RETRIES="${MAX_RETRIES:-4}"
GRAPH_DIR="${GRAPH_DIR:-python_graph}"
FEATURES_FILE="${FEATURES_FILE:-gemma_mbpp_features.jsonl}"
LEDGER="$GRAPH_DIR/done_graphs.txt"
# The machine is on loan. The features file is the only irreplaceable output
# and is a few MB, so it is copied off the working volume after every chunk.
BACKUP_DIR="${BACKUP_DIR:-/mnt/c/Users/user1/CodeCircuit-backup}"
export GRAPH_DIR FEATURES_FILE

mkdir -p "$BACKUP_DIR"

ledger_count() { [ -f "$LEDGER" ] && wc -l < "$LEDGER" || echo 0; }

while true; do
    before=$(ledger_count)
    echo "=== $(date '+%F %T') generating (ledger=$before, CHUNK_SIZE=${CHUNK_SIZE:-default}) ==="

    attempt=1
    exhausted=0
    while true; do
        python 3_generate_graph.py
        rc=$?
        if [ $rc -eq 0 ] || [ $rc -eq 3 ]; then
            break
        fi
        if [ $attempt -ge $MAX_RETRIES ]; then
            echo "!!! script 3 exited $rc on $MAX_RETRIES attempts."
            echo "!!! see $GRAPH_DIR/failed_graphs.tsv"
            exhausted=1
            break
        fi
        # Whatever finished before the failure is on disk, so a retry resumes
        # rather than restarting the chunk.
        echo "--- script 3 exit $rc, retry $attempt/$MAX_RETRIES ---"
        attempt=$((attempt + 1))
        sleep 30
    done

    # Featurize and free even when generation gave up: an aborted chunk can
    # hold ~200 graphs, ~80GB and a dozen GPU-hours, and exiting straight from
    # the retry loop would leave all of it unfeaturized, unledgered, and
    # occupying a disk the study is already tight on.
    echo "=== $(date '+%F %T') extracting features ==="
    python 4_generate_feature.py || { echo "!!! script 4 failed"; exit 1; }

    echo "=== $(date '+%F %T') freeing chunk ==="
    python chunk_cleanup.py || { echo "!!! cleanup failed"; exit 1; }

    cp "$FEATURES_FILE" "$BACKUP_DIR/" 2>/dev/null \
        && echo "backed up $FEATURES_FILE -> $BACKUP_DIR"
    df -h . | tail -1

    after=$(ledger_count)
    if [ $exhausted -eq 1 ]; then
        echo "!!! stopping after banking this chunk (ledger $before -> $after)."
        exit 1
    fi
    if [ $rc -eq 3 ]; then
        echo "=== $(date '+%F %T') ALL CHUNKS COMPLETE (ledger=$after) ==="
        break
    fi
    # Backstop against every "spins without progress" failure mode: if a whole
    # cycle ledgered nothing, another pass will do exactly the same thing.
    if [ "$after" -le "$before" ]; then
        echo "!!! a full cycle added nothing to the ledger ($before -> $after)."
        echo "!!! stopping instead of looping. Inspect $GRAPH_DIR."
        exit 1
    fi
done
