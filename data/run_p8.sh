#!/bin/bash
# Phase 8 GPU driver (final_design.md §10, 'Implementation plan'): one process per chunk of sites,
# restart rather than retry. Runs `python p8_2_run.py` with STAGE=A, then B, then C, again and again
# until each stage reports completion. Every process re-reads the ledger, so a crash costs only
# the site that was running. A site whose attribute() fails in two fresh processes is excluded
# (design §2 (c)); any other repeated failure stops the driver (exit 6) for a person.
#
# The two launch lines (P8_OUT the same for both, so one GPU clock covers P0 + P1 + main):
#   cd data && PILOT=1 bash run_p8.sh          # P1: SITES=pilot_sites.jsonl RUN_DIR=pilot/run
#   cd data && SITES=sites.jsonl bash run_p8.sh  # main run (needs a committed PREREGISTRATION_P8*.md)
# and the matching later steps (GPU idle):
#   P1:   SITES=pilot_sites.jsonl RUN_DIR=pilot/run JUDGED=pilot/judged.jsonl SUMMARY=pilot/judged.json python p8_3_judge.py
#         then p8_4_stats.py's dry run on the same pilot/ paths (p8_sim reads PILOT_RUN=$P8_OUT/pilot/run)
#   main: SITES=sites.jsonl RUN_DIR=run python p8_3_judge.py
#   STAGES="A" bash run_p8.sh                  # one stage only
# Env passed through to p8_2_run.py: SITES RUN_DIR CHUNK P0_JSON RESERVE_H C_SEC_PER_SITE P8_OUT P8_WORK
#   P8_CLOCK ORC_N EXIST_G ACCEPT_FAILED FAULT ALLOW_P0_FAIL; here: PILOT=1 (sets SITES and RUN_DIR as
#   above, refuses other values of them), PY (python), STAGES (default "A B C"), MAX_CRASH (consecutive
#   crashes before giving up, default 6), LOG (default <P8_OUT>/run_p8.log, pilot: <P8_OUT>/pilot/run_p8.log).
# Hard stop: exit 3 from stage A or B ends that stage and the driver goes on to the next stage (stage
#   B stops RESERVE_H early, by default a reserve sized for stage C; p8_2_run.reserve_hours); exit 3
#   from stage C ends the driver at exactly 28 GPU-h.
# Exit: 0 all stages complete, 3 hard stop in stage C, 4 configuration/canary/P0 problem, 5 GPU busy,
#   6 a site needs a person (p8_2_run prints what to do), 1 too many consecutive crashes.
# Never run beside another GPU job (p8_2_run.py also checks), and never run p8_3_judge.py beside it.
cd "$(dirname "$0")" || exit 1
PY=${PY:-/home/user1/miniconda3/envs/codecircuit/bin/python}
MAX_CRASH=${MAX_CRASH:-6}
OUTD=${P8_OUT:-results/intervention}
if [ "${PILOT:-0}" = "1" ]; then
  if { [ -n "$SITES" ] && [ "$SITES" != "pilot_sites.jsonl" ]; } || { [ -n "$RUN_DIR" ] && [ "$RUN_DIR" != "pilot/run" ]; }; then
    echo "PILOT=1 sets SITES=pilot_sites.jsonl RUN_DIR=pilot/run; got SITES=$SITES RUN_DIR=$RUN_DIR"; exit 4
  fi
  export SITES=pilot_sites.jsonl RUN_DIR=pilot/run
  mkdir -p "$OUTD/pilot"
  LOG=${LOG:-$OUTD/pilot/run_p8.log}
else
  if [ -z "$SITES" ]; then
    echo "SITES is not set: PILOT=1 for P1, SITES=sites.jsonl for the main run"; exit 4
  fi
  export SITES
fi
mkdir -p "$OUTD"
LOG=${LOG:-$OUTD/run_p8.log}
echo "$(date '+%F %T') run_p8.sh: SITES=$SITES RUN_DIR=${RUN_DIR:-run} P8_OUT=$OUTD STAGES=${STAGES:-A B C}" | tee -a "$LOG"
for st in ${STAGES:-A B C}; do
  crashes=0
  while true; do
    echo "$(date '+%F %T') stage $st: starting a process" | tee -a "$LOG"
    STAGE=$st "$PY" p8_2_run.py >> "$LOG" 2>&1
    rc=$?
    echo "$(date '+%F %T') stage $st: process exit $rc" | tee -a "$LOG"
    case $rc in
      0) break ;;
      10) crashes=0 ;;
      3) if [ "$st" = "C" ]; then
           echo "HARD STOP reached in stage C" | tee -a "$LOG"; exit 3
         fi
         echo "hard stop reached in stage $st: going on to the next stage" | tee -a "$LOG"; break ;;
      4|5|6) echo "stopped (exit $rc): see $LOG" | tee -a "$LOG"; exit "$rc" ;;
      *) crashes=$((crashes + 1))
         if [ "$crashes" -ge "$MAX_CRASH" ]; then
           echo "giving up after $crashes consecutive crashes in stage $st" | tee -a "$LOG"; exit 1
         fi ;;
    esac
  done
done
echo "$(date '+%F %T') all stages complete" | tee -a "$LOG"
exit 0
