"""inf audit, step 1 (CPU, read-only): which typeable tasks use the tolerant f64 check,
and how many labelled-passing items each phase's files hold in those tasks."""
import json, gzip, os, sys, collections, subprocess, tempfile
from pathlib import Path
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
sys.path.insert(0, '/home/user1/CodeCircuit-rust/data')
from mbpp_rust import load_typeable
typ, _ = load_typeable()
close = sorted(t for t, s in typ.items() if s['ret'] == 'f64' and any(c[2] for c in s['cases']))
f64 = sorted(t for t, s in typ.items() if s['ret'] == 'f64')
print('typeable', len(typ), 'f64', len(f64), 'f64 with an isclose case', len(close), close)

# the check itself, compiled: does inf pass?
RUSTC = str(Path.home() / '.cargo/bin/rustc')
src = r'''fn main(){ for v in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 5.0f64, 5.004f64] { let e = 5.0f64;
 let ok = v == e || (v - e).abs() <= 1e-3 * v.abs().max(e.abs()); println!("{:?} -> {}", v, ok); } }'''
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / 'c.rs'; p.write_text(src)
    subprocess.run([RUSTC, str(p), '-o', str(Path(td) / 'c')], check=True, capture_output=True)
    print(subprocess.run([str(Path(td) / 'c')], capture_output=True, text=True).stdout)

D = Path('/home/user1/CodeCircuit-rust/data')
def rd(p):
    op = gzip.open if str(p).endswith('.gz') else open
    with op(p, 'rt') as f:
        return [json.loads(l) for l in f if l.strip()]
for name in ['gemma_mbpp_rust_correctness.jsonl', 'within_labels.jsonl', 'mutants_judged.jsonl',
             'matched_labels_r1.jsonl', 'matched_mutants.jsonl', 'results/ownfix/data/ownfix_mutants.jsonl.gz']:
    p = D / name
    if not p.exists():
        print(name, 'MISSING'); continue
    rows = rd(p)
    keys = sorted(rows[0].keys())
    tid = 'task_id' if 'task_id' in rows[0] else ('problem' if 'problem' in rows[0] else None)
    passk = next((k for k in ['passes', 'passed', 'correct', 'label', 'pass'] if k in rows[0]), None)
    inclose = [r for r in rows if tid and int(str(r[tid])[:6]) in close] if tid else []
    npass = sum(1 for r in inclose if passk and r.get(passk) in (True, 1, 'pass', 'correct'))
    print(f'{name}: rows {len(rows)} keys {keys[:14]} | id {tid} pass-key {passk} | rows in isclose tasks {len(inclose)}, passing {npass}',
          collections.Counter(r[tid] for r in inclose).most_common(12) if inclose else '')
