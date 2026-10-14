"""inf audit, step 3 (exact): which analysed pairs hold a falsely passing member."""
import json, gzip
from pathlib import Path
D = Path('/home/user1/CodeCircuit-rust/data')
res = json.load(open('/home/user1/p8design/audit/audit_inf.json'))
bad = {(r['phase'], r['task_id'], r.get('sample'), r.get('pos'), r.get('to')) for r in res if r['new'][0] is not True}
bad_samples = {(t, s) for (ph, t, s, p, to) in bad if ph.endswith('sample')}
bad_mut = {(t, s, p, to) for (ph, t, s, p, to) in bad if ph.endswith('mutant')}
bad_mut_noposs = {(t, s, to) for (t, s, p, to) in bad_mut}

W = json.load(open(D / 'within_pairs.json'))
print('Phase 3 pairs with a false passing member:', [p for p in W if (p['problem'], p['pass_sample']) in bad_samples])

# Phase 4 pairs carry the survivor edit by line, not pos: match on (task, source sample, from, to, line) via mutants_judged
mj = [json.loads(l) for l in open(D / 'mutants_judged.jsonl')]
bad4 = [m for m in mj if (m['task_id'], m['sample'], m['pos'], m['to']) in bad_mut]
M4 = json.load(open(D / 'mutant_pairs.json'))
hit4 = [p for p in M4 for m in bad4 if p['problem'] == m['task_id'] and p['source_sample'] == m['sample']
        and p['survivor_edit']['line'] == m['line'] and p['survivor_edit']['from'] == m['from'] and p['survivor_edit']['to'] == m['to']]
print('Phase 4 pairs with a false survivor:', [(p['problem'], p['source_sample'], p['pass_id'], p['survivor_edit']) for p in hit4])
print('   Phase 4 sources that are false-passing samples:', [p for p in M4 if (p['problem'], p['source_sample']) in bad_samples])

M5 = json.load(open(D / 'matched_pairs.json'))
hit5 = [p for p in M5 if (p['problem'], p['source_sample'], p['survivor_edit']['pos'], p['survivor_edit']['to']) in bad_mut]
print('Phase 5 pairs with a false survivor:', [(p['problem'], p['source_sample'], p['pass_id'], p['survivor_edit']) for p in hit5])
print('   Phase 5 sources that are false-passing samples:', [p['pass_id'] for p in M5 if (p['problem'], p['source_sample']) in bad_samples])

O = json.load(open(D / 'results/ownfix/data/ownfix_pairs.json'))
hit6 = [p for p in O if (p['problem'], p['source_sample'], p['r_edit']['pos'], p['r_edit']['to']) in bad_mut]
print('Phase 6 pairs with a false repair:', [(p['problem'], p['source_sample'], p['rule'], p['tier'], p['r_edit']) for p in hit6])
print('   by rule:', {r: sum(p['rule'] == r for p in hit6) for r in ('strict', 'fallback')}, 'of', len(O), 'pairs')

V = json.load(open(D / 'results/ownfix/data/ownfix_verify.json'))['mutants']
st = [V.get(f'{t}|{s}|{p}|{to}') for (t, s, p, to) in bad_mut if t in (137, 248)]
from collections import Counter
print('Phase 6 verify status of the false repairs:', Counter(map(str, st)))
# Phase 7 survivor verification
P7 = json.load(open(D / 'results/closing/verify_p45.json'))
print('verify_p45 top-level keys:', list(P7)[:10] if isinstance(P7, dict) else type(P7))
txt = json.dumps(P7)
for p in hit5:
    print('   P7 entry mentions pass_id', p['pass_id'], ':', str(p['pass_id']) in txt)
for p in hit4:
    print('   P7 entry mentions P4 pass_id', p['pass_id'], ':', str(p['pass_id']) in txt)
