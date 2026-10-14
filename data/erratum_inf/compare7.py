"""Erratum: Phase 7 headline numbers, published (259 pairs) vs rerun without the inf pair (258)."""
import json
a = json.load(open('/home/user1/CodeCircuit-rust/data/results/closing/analysis.json'))
b = json.load(open('/home/user1/p8design/erratum/p7/out/analysis.json'))

def f(x):
    return 'None' if x is None else (f'{x:+.3f}' if isinstance(x, float) else str(x))

print('A (Phase 5 pairs):')
for k in ('C', 'G', 'P', 'PP', 'Gfree', 'Gattr', 'E', 'O', 'S2P', 'L', 'N45', 'C63', 'C+'):
    if k in a['estimates']['A'] and k in b['estimates']['A']:
        x, y = a['estimates']['A'][k], b['estimates']['A'][k]
        print(f'  {k:8s} {x["A"]:.3f} [{x["ci"][0]:.3f}, {x["ci"][1]:.3f}]  ->  {y["A"]:.3f} [{y["ci"][0]:.3f}, {y["ci"][1]:.3f}]')
print('increments:')
for k, x in a['estimates']['inc'].items():
    y = b['estimates']['inc'].get(k)
    if y is None:
        continue
    s = f'  {k:14s} raw {x["diff"]:+.3f} [{x["ci"][0]:+.3f}, {x["ci"][1]:+.3f}] -> {y["diff"]:+.3f} [{y["ci"][0]:+.3f}, {y["ci"][1]:+.3f}]'
    if 'diff_adj' in x:
        s += f' | adj {x["diff_adj"]:+.3f} [{x["ci_adj"][0]:+.3f}, {x["ci_adj"][1]:+.3f}] -> {y["diff_adj"]:+.3f} [{y["ci_adj"][0]:+.3f}, {y["ci_adj"][1]:+.3f}]'
    print(s)
print('tests:')
for fam in ('primary', 'secondary'):
    for k, x in a[fam].items():
        y = b[fam][k]
        print(f'  {k}: S {x["S"]:.3f} -> {y["S"]:.3f}; p {f(x.get("p"))} -> {f(y.get("p"))}; withdrawn {x.get("withdrawn")}; rejected {x.get("rejected")} -> {y.get("rejected")}')
for k in ('G5a_N45', 'G5b_shadow'):
    print(f'  {k}: p {a[k]["p"]:.4f} -> {b[k]["p"]:.4f}')
print('refit range G|C:', a['refit_sensitivity'].get('inc G|C'), '->', b['refit_sensitivity'].get('inc G|C'))
va, vb = a['verification'], b['verification']
print('verification counts:', va.get('counts'), '->', vb.get('counts'))
for lab in ('lucky',):
    for k in ('G', 'C', 'P'):
        x = va[lab]['verified_minus_lucky'][k]; y = vb[lab]['verified_minus_lucky'][k]
        print(f'  {k}: verified {x["A_verified"]:.3f} lucky {x["A_lucky"]:.3f} diff {x["diff"]:+.3f} {x["ci"]} -> '
              f'{y["A_verified"]:.3f} {y["A_lucky"]:.3f} {y["diff"]:+.3f} [{y["ci"][0]:+.3f}, {y["ci"][1]:+.3f}]')
print('same site:', a['same_site']['pairs'], a['same_site']['inc']['G|C'], '->', b['same_site']['pairs'], b['same_site']['inc']['G|C'])
print('phase 4 verification:', a['phase4']['verification'].get('counts'), '->', b['phase4']['verification'].get('counts'))
print('\nreading (rerun):')
for l in b['reading']:
    print('  ' + l)
print('\nreading lines that differ:')
for l in b['reading']:
    if l not in a['reading']:
        print('  NEW  ' + l)
for l in a['reading']:
    if l not in b['reading']:
        print('  OLD  ' + l)
