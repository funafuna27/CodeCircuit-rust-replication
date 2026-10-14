"""inf audit: Phase 6's §9 counts with the 33 inf-only repairs removed (label-level, no re-matching)."""
import json, gzip, collections
D = '/home/user1/CodeCircuit-rust/data/results/ownfix/data/'
res = json.load(open('/home/user1/p8design/audit/audit_inf.json'))
bad = {(r['task_id'], r['sample'], r['pos'], r['to']) for r in res if r['phase'] == 'P6 mutant' and r['new'][0] is not True}
V = json.load(open(D + 'ownfix_verify.json'))['mutants']
st = collections.Counter()
for k, v in V.items():
    t, s, p, to = k.split('|', 3)
    key = (int(t), int(s), int(p), to)
    st[('BAD ' if key in bad else '') + str(v)] += 1
print('verify statuses (BAD = inf-only):', dict(st))
ver = sum(v is True for v in V.values()); badv = sum(1 for k, v in V.items() if v is True and
                                                   tuple([int(x) if i < 3 else x for i, x in enumerate(k.split('|', 3))]) in bad)
lucky = sum(v is False for v in V.values()); unv = sum(v is None for v in V.values())
print(f'verified {ver} -> {ver - badv}; lucky {lucky}; unverifiable {unv}; checkable {ver + lucky} -> {ver - badv + lucky}; '
      f'lucky share {lucky / (ver + lucky):.3f} -> {lucky / (ver - badv + lucky):.3f}')
mut = [json.loads(l) for l in gzip.open(D + 'ownfix_mutants.jsonl.gz', 'rt')]
rep = [m for m in mut if m['is_r']]
print('stable repairs', len(rep), '->', len(rep) - sum((m['task_id'], m['sample'], m['pos'], m['to']) in bad for m in rep))
src_with = {(m['task_id'], m['sample']) for m in rep}
src_with2 = {(m['task_id'], m['sample']) for m in rep if (m['task_id'], m['sample'], m['pos'], m['to']) not in bad}
print('sources with a stable repair', len(src_with), '->', len(src_with2), '; problems', len({s[0] for s in src_with}), '->', len({s[0] for s in src_with2}))
O = json.load(open(D + 'ownfix_pairs.json'))
keep = [p for p in O if (p['problem'], p['source_sample'], p['r_edit']['pos'], p['r_edit']['to']) not in bad]
print('pairs', len(O), '->', len(keep), '; problems', len({p['problem'] for p in O}), '->', len({p['problem'] for p in keep}),
      '; strict', sum(p['rule'] == 'strict' for p in O), '->', sum(p['rule'] == 'strict' for p in keep))
