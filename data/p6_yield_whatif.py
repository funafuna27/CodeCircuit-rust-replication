"""Phase 6, POST HOC and descriptive, after the gate was missed: how many pairs
relaxed rules would have given (labels and log-probabilities only, no graph).
PREREGISTRATION_OWNFIX.md §9. Changes nothing in the pre-registered result."""
import json, os, sys
from collections import Counter
os.environ['ROUND'] = '8'
sys.path.insert(0, '/home/user1/CodeCircuit-rust/data')
import p6_3_match as M

src = {(s['task_id'], s['sample']): s for s in M.read_jsonl('ownfix_sources.jsonl') if s['stable']}
lps = {M.mkey(r): r for r in M.read_jsonl('ownfix_logprob.jsonl')}
ver = json.load(open('ownfix_verify.json'))
verified = {}
for k, v in ver['mutants'].items():
    t, s, p, to = k.split('|', 3)
    verified[(int(t), int(s), int(p), to)] = v
by = {}
allr = []
for m in M.read_jsonl('ownfix_mutants.jsonl'):
    if (m['task_id'], m['sample']) not in src:
        continue
    if m.get('is_r'):
        allr.append(m)
    if M.mkey(m) in lps and all(lps[M.mkey(m)][k] is not None for k in M.LP_KEYS):
        m['lp'] = lps[M.mkey(m)]
        by.setdefault((m['task_id'], m['sample']), []).append(m)
used0 = {}
for f in ('mutant_rows.jsonl', 'matched_rows.jsonl'):
    for r in M.read_jsonl(f):
        used0.setdefault(r['problem'], set()).add(('code', M.norm(r['generated_code'])))

print('stable sources', len(src), '; sources with a stable repair', len({(m['task_id'], m['sample']) for m in allr}),
      '; stable repairs', len(allr), Counter(verified.get(M.mkey(m)) for m in allr))
withr = {(m['task_id'], m['sample']) for m in allr}
print('problems with a stable repair', len({k[0] for k in withr}))
# repairs per verification state: does the problem have passing Gemma samples?
pv = ver['problems']
print('problems with a stable repair but no passing Gemma sample to calibrate:',
      sum(1 for t, p in pv.items() if p['n_calib'] == 0), 'of', len(pv),
      '; with < 10 kept inputs:', sum(1 for t, p in pv.items() if p['n_calib'] > 0 and p['n_inputs'] < 10))


def run(caliper, allow_unverified, fallback):
    M.CALIPER = caliper
    vv = dict(verified)
    if allow_unverified:
        vv = {k: (v is not False) for k, v in verified.items()}        # None -> usable, False -> not
    pairs, n = M.build(src, by, vv, used0, fallback)
    return len(pairs), len({p['problem'] for p in pairs})


for cal in (1.0, 2.0, 3.0, 5.0):
    for unv in (False, True):
        for fb in (False, True):
            n, pr = run(cal, unv, fb)
            print(f'caliper {cal:3.1f}  unverifiable R allowed {str(unv):5}  fallback {str(fb):5}: {n:4} pairs over {pr:3} problems')
