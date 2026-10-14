"""Erratum: compare the reruns without the inf pair with the published Phase 4/5 numbers."""
import json
R = '/home/user1/CodeCircuit-rust/data/results/'
E = '/home/user1/p8design/erratum/'

def flat(d, pre=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f'{pre}{k}.'))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[pre[:-1]] = d
    elif isinstance(d, list) and len(d) <= 4 and all(isinstance(x, (int, float)) for x in d):
        for i, x in enumerate(d):
            out[f'{pre}{i}'] = x
    return out

a, b = flat(json.load(open(R + 'mutant/analysis.json'))), flat(json.load(open(E + 'm3_orig.json')))
diff = [k for k in a if k in b and a[k] != b[k] and 'second' not in k and 'time' not in k]
print(f'determinism: m3 rerun on the original pairs vs published: {len(a)} numbers, {len(diff)} differ', diff[:8])

pairs = [('mutant/analysis.json', 'm3_fix.json'), ('mutant/centred.json', 'm4_fix.json'), ('mutant/posthoc.json', 'm5_fix.json'),
         ('mutant/edit_baseline.json', 'p4_edit_fix.json'), ('mutant/beyond_edit_posthoc.json', 'p4_beyond_fix.json'),
         ('matched/analysis.json', 'p5_4_fix.json'), ('matched/edit_baseline.json', 'p5_5_fix.json'),
         ('matched/beyond_edit.json', 'p5_6_fix.json')]
KEYS = ('A', 'p', 'ci', 'diff', 'n', 'pairs', 'mean', 'Ac', 'S1c', 'S2c', 'beta', 'share', 'killed_lower')
for pub, fix in pairs:
    a, b = flat(json.load(open(R + pub))), flat(json.load(open(E + fix)))
    print(f'\n== {pub}')
    for k in a:
        last = k.split('.')[-1]
        if k in b and (last in KEYS or any(t in k for t in ('.A.', '.p', 'ci.', '.diff', 'n_pairs', 'killed'))) \
                and isinstance(a[k], (int, float)) and len(k) < 70:
            if abs(a[k] - b[k]) > 0 or last in ('A', 'p', 'diff'):
                print(f'   {k:62s} {a[k]:+.4f} -> {b[k]:+.4f}')
