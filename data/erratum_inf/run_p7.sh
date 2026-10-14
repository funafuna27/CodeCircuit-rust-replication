#!/bin/bash
# wait for the Phase 4/5 reruns, then rerun Phase 7 prepare + closing on the filtered inputs
E=/home/user1/p8design/erratum
PY=/home/user1/miniconda3/envs/codecircuit/bin/python
until grep -q 'repo tracked files unchanged' $E/run_p45.out 2>/dev/null; do sleep 30; done
cd /home/user1/CodeCircuit-rust/data
echo "[$(date '+%F %T %Z')] start prepare"
WORKERS=16 $PY -B $E/e7_rerun.py prepare > $E/p7_prepare.log 2>&1; echo "[$(date '+%F %T %Z')] end prepare exit $?"
echo "[$(date '+%F %T %Z')] start closing"
$PY -B $E/e7_rerun.py closing > $E/p7_closing.log 2>&1; echo "[$(date '+%F %T %Z')] end closing exit $?"
cd /home/user1/CodeCircuit-rust && git status --short | grep -v '^??'; echo "repo tracked files unchanged if nothing above"
