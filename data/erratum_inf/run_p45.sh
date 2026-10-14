#!/bin/bash
# Erratum (inf judge bug): rerun the Phase 4 and Phase 5 analyses without the one pair whose
# survivor only returns inf. Repo files are read, never written: every OUT goes to this folder.
set -u
E=/home/user1/p8design/erratum
PY=/home/user1/miniconda3/envs/codecircuit/bin/python
cd /home/user1/CodeCircuit-rust/data || exit 1
$PY - <<'EOF'
import json
E = '/home/user1/p8design/erratum'
for src, dst, key in (('mutant_pairs.json', 'mutant_pairs_fix.json', (248, 3)),
                      ('matched_pairs.json', 'matched_pairs_fix.json', (248, 15))):
    P = json.load(open(src))
    Q = [p for p in P if (p['problem'], p['source_sample']) != key]
    assert len(Q) == len(P) - 1, (src, len(P), len(Q))
    json.dump(Q, open(f'{E}/{dst}', 'w'), indent=1)
    print(src, len(P), '->', len(Q))
EOF
run() { local name=$1; shift; echo "[$(date '+%F %T %Z')] start $name"; env "$@" > $E/$name.log 2>&1; echo "[$(date '+%F %T %Z')] end $name exit $?"; }
# determinism check: Phase 4's primary with the original pairs must reproduce results/mutant/analysis.json
run m3_orig OUT=$E/m3_orig.json $PY -B m3_mutant_analysis.py
P4="PAIRS=$E/mutant_pairs_fix.json"
run m3_fix  $P4 OUT=$E/m3_fix.json $PY -B m3_mutant_analysis.py
run m4_fix  $P4 OUT=$E/m4_fix.json $PY -B m4_mutant_centred.py
run m5_fix  $P4 OUT=$E/m5_fix.json $PY -B m5_mutant_posthoc.py
run p4_edit_fix   ROWS=mutant_rows.jsonl $P4 OUT=$E/p4_edit_fix.json $PY -B p5_5_edit_baseline.py
run p4_beyond_fix FEATURES=gemma_mutant_features.jsonl ROWS=mutant_rows.jsonl $P4 OUT=$E/p4_beyond_fix.json $PY -B p5_6_beyond_edit.py
P5="PAIRS=$E/matched_pairs_fix.json"
run p5_4_fix $P5 OUT=$E/p5_4_fix.json $PY -B p5_4_analysis.py
run p5_5_fix $P5 OUT=$E/p5_5_fix.json $PY -B p5_5_edit_baseline.py
run p5_6_fix $P5 OUT=$E/p5_6_fix.json $PY -B p5_6_beyond_edit.py
cd /home/user1/CodeCircuit-rust && git status --short | grep -v '^??' ; echo "repo tracked files unchanged if nothing above"
