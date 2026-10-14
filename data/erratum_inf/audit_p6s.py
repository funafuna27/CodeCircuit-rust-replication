"""inf audit, addendum: Phase 6's own samples (ownfix_samples_r1..8) labelled passing in the 11
f64+isclose tasks, which audit_inf.py did not cover. Same two judges (audit_inf._run)."""
import json, collections, glob
from multiprocessing import Pool
import audit_inf as A
from mbpp_rust import load_typeable

if __name__ == '__main__':
    typ, _ = load_typeable()
    close = {t for t, s in typ.items() if s['ret'] == 'f64' and any(c[2] for c in s['cases'])}
    items = []
    for sf in sorted(glob.glob(str(A.D / 'ownfix_samples_r*.jsonl'))):
        lf = sf.replace('_samples_', '_labels_')
        ok = {(r['task_id'], r['sample']) for r in A.rd(lf) if r['passes'] and r['task_id'] in close}
        rnd = sf.rsplit('_r', 1)[1].split('.')[0]
        for r in A.rd(sf):
            if (r['task_id'], r['sample']) in ok:
                items.append(dict(phase='P6 sample', rnd=rnd, task_id=r['task_id'], sample=r['sample'],
                                  truncated=r.get('truncated'), code=r['generated_code']))
    print('items', len(items), collections.Counter(i['task_id'] for i in items))
    with Pool(6, initializer=A._init) as pool:
        res = pool.map(A._run, items, chunksize=1)
    json.dump(res, open('/home/user1/p8design/audit/audit_p6s.json', 'w'), indent=0)
    c = collections.Counter()
    for r in res:
        c['old reproduces pass' if r['old'][0] else 'OLD DISAGREES'] += 1
        c['still pass' if r['new'][0] else ('NOW FAILS' if r['new'][0] is False else 'error')] += 1
    print('P6 sample', dict(c))
    for r in res:
        if r['new'][0] is not True:
            print('  ', r['task_id'], 'r' + r['rnd'], r['sample'], 'truncated', r['truncated'], 'new:', r['new'])
