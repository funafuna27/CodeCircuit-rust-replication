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

# Consecutive script-3 failures that produced NO new graph. Resetting on
# progress is what makes this a stall detector rather than a failure quota:
# escaping one deterministic bad row already costs MAX_GRAPH_ATTEMPTS passes
# that generate nothing, so a plain per-chunk budget would leave zero tolerance
# for an ordinary transient fault during the ~11 GPU-hours that follow.
MAX_RETRIES="${MAX_RETRIES:-4}"
# Hard stop on total passes per chunk, so a cycle that keeps making a little
# progress and then dying still terminates instead of looping for days.
MAX_PASSES="${MAX_PASSES:-20}"
GRAPH_DIR="${GRAPH_DIR:-python_graph}"
FEATURES_FILE="${FEATURES_FILE:-gemma_mbpp_features.jsonl}"
LEDGER="$GRAPH_DIR/done_graphs.txt"
# The machine is on loan. The features file is the only irreplaceable output
# and is a few MB, so it is copied off the working volume after every chunk.
BACKUP_DIR="${BACKUP_DIR:-/mnt/c/Users/user1/CodeCircuit-backup}"
export GRAPH_DIR FEATURES_FILE

mkdir -p "$BACKUP_DIR"

ledger_count() { [ -f "$LEDGER" ] && wc -l < "$LEDGER" || echo 0; }
# The ledger only moves after cleanup, so it cannot measure progress inside the
# retry loop. Generated graphs can.
graph_count() { ls "$GRAPH_DIR"/*.pt 2>/dev/null | wc -l; }

while true; do
    before=$(ledger_count)
    echo "=== $(date '+%F %T') generating (ledger=$before, CHUNK_SIZE=${CHUNK_SIZE:-default}) ==="

    attempt=1
    passes=0
    exhausted=0
    while true; do
        graphs_before=$(graph_count)
        python 3_generate_graph.py
        rc=$?
        passes=$((passes + 1))
        # 0 = generated, 3 = nothing left, 4 = nothing left but some graphs
        # were permanently skipped.
        if [ $rc -eq 0 ] || [ $rc -eq 3 ] || [ $rc -eq 4 ]; then
            break
        fi

        graphs_after=$(graph_count)
        if [ "$graphs_after" -gt "$graphs_before" ]; then
            # This pass produced graphs before dying, so the failure is not a
            # stall. Forgive it and keep the budget for passes that achieve
            # nothing at all.
            echo "--- script 3 exit $rc after +$((graphs_after - graphs_before)) graphs; retrying ---"
            attempt=1
        else
            if [ $attempt -ge $MAX_RETRIES ]; then
                echo "!!! script 3 exited $rc on $MAX_RETRIES consecutive passes with no new graph."
                echo "!!! see $GRAPH_DIR/failed_graphs.tsv"
                exhausted=1
                break
            fi
            echo "--- script 3 exit $rc, no new graph, retry $attempt/$MAX_RETRIES ---"
            attempt=$((attempt + 1))
        fi

        if [ $passes -ge $MAX_PASSES ]; then
            echo "!!! $passes script-3 passes in one chunk without finishing it."
            exhausted=1
            break
        fi
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

    # This file is the only irreplaceable output, so a failed copy has to be
    # visible. Chained with && and stderr discarded, a broken drvfs mount or a
    # full C: would have shown up as nothing at all in a four-day log.
    if cp "$FEATURES_FILE" "$BACKUP_DIR/"; then
        echo "backed up $FEATURES_FILE -> $BACKUP_DIR"
    else
        echo "!!! BACKUP FAILED: could not copy $FEATURES_FILE to $BACKUP_DIR"
    fi
    df -h . | tail -1

    after=$(ledger_count)
    if [ $exhausted -eq 1 ]; then
        echo "!!! stopping after banking this chunk (ledger $before -> $after)."
        exit 1
    fi
    if [ $rc -eq 4 ]; then
        echo "=== $(date '+%F %T') FINISHED BUT INCOMPLETE (ledger=$after) ==="
        echo "!!! some graphs were permanently skipped; see $GRAPH_DIR/failed_graphs.tsv"
        echo "!!! the feature dataset is short -- do not treat this as a clean run"
        exit 4
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
