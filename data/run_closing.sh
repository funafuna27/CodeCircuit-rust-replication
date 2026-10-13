#!/bin/bash
# Phase 7 driver. PREREGISTRATION_CLOSING.md §2.
#   PHASE=extract|verify|prepare|sizesim|closing bash /home/user1/CodeCircuit-rust/data/run_closing.sh
# Start it from a launcher script file (inline `wsl -- bash -lc` mangles quotes and $ on this machine).
# It reads the retained graphs only, never commits or pushes, never runs 1_code_correct.py,
# and writes nothing outside data/results/closing and ~/closing_work.
set -uo pipefail
cd /home/user1/CodeCircuit-rust/data
PY=/home/user1/miniconda3/envs/codecircuit/bin/python
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/closing
mkdir -p "$OUT" /home/user1/closing_work
PHASE=${PHASE:-}

if pgrep -f '3_generate_graph|p6_[0-9]|p7_[0-4]_' > /dev/null; then
  echo "another graph, Phase 6 or Phase 7 job is running"; exit 1
fi
avail=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)     # C: holds WSL's virtual disk
if [ "$avail" -lt 50 ]; then echo "less than 50 GB free on C: ($avail GB)"; exit 1; fi
n5=$(ls /home/user1/matched_graph_keep | grep -c '\.pt$')
n4=$(ls /home/user1/mutant_graph_keep | grep -c '\.pt$')
if [ "$n5" -ne 518 ] || [ "$n4" -ne 197 ]; then echo "retained graphs: $n5 and $n4, expected 518 and 197"; exit 1; fi
need() { if [ ! -f "$OUT/$1.ok" ]; then echo "run PHASE=$1 first"; exit 1; fi; }

log="$OUT/$PHASE.log"
echo "[$(TZ=Asia/Tokyo date '+%F %T JST')] start $PHASE; C: free ${avail} GB" | tee -a "$log"
case "$PHASE" in
  extract) WORKERS=2 "$PY" -B p7_0_extract.py ;;
  verify)  WORKERS=6 "$PY" -B p7_1_verify_p45.py ;;
  prepare) need extract; N_JOBS=1 WORKERS=16 "$PY" -B p7_2_prepare.py ;;
  sizesim) need prepare; N_JOBS=1 WORKERS=18 "$PY" -B p7_3_sizesim.py ;;
  closing) need sizesim; N_JOBS=4 "$PY" -B p7_4_closing.py ;;
  *) echo "PHASE=extract|verify|prepare|sizesim|closing"; exit 2 ;;
esac 2>&1 | tee -a "$log"
rc=${PIPESTATUS[0]}
echo "[$(TZ=Asia/Tokyo date '+%F %T JST')] end $PHASE, exit $rc" | tee -a "$log"
free -g | head -2 | tee -a "$log"
df -BG /mnt/c | tail -1 | tee -a "$log"
if [ "$rc" -eq 0 ]; then
  (cd "$OUT" && sha256sum $(ls | grep -v '\.ok$' | grep -v '\.log$')) > "$OUT/$PHASE.ok"
fi
exit "$rc"
