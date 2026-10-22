"""Gate-2 counts (pilot gradient; primary S1/S2 eligibility from tokenization only, no forward pass).
Run once on 2026-10-22 before PREREGISTRATION_P9_MUTSTATE.md was committed; its §7 quotes the output."""
import os, sys, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p9_common as C
import p9_b_analysis as A

items = C.read_jsonl(os.path.join(C.OUT_DIR, 'b_items.jsonl.gz'))
prose = {(int(r['task_id']), int(r['sample'])): r['text'] for r in C.read_jsonl(os.path.join(C.OUT_DIR, 'pool_samples.jsonl.gz'))}
out = {}

# ---- (1) pilot
F = A.load('pilot')
rows = [r for r in (A.item_tokens(it, F, prose) for it in items if it['role'] == 'pilot') if r]
cells = defaultdict(list)
for r in rows:
    for c, h, d in r['bg']:
        cells[A.dist_bin(d)].append((r['problem'], c, h))
bins = ['1-4', '5-9', '10-19', '20-39', '40+']
pil = {}
for b in range(5):
    v = cells.get(b, [])
    if not v:
        continue
    byp = defaultdict(list)
    for p, c, h in v:
        byp[p].append(c)
    lo = [c for p, c, h in v if h < 0.05]
    pil[bins[b]] = {'tokens': len(v), 'mean_c': round(float(np.mean([c for _, c, _ in v])), 4),
                    'problem_mean_c': round(float(np.mean([np.mean(x) for x in byp.values()])), 4),
                    'mean_entropy': round(float(np.mean([h for *_, h in v])), 3),
                    'mean_c_entropy_lt_0.05': round(float(np.mean(lo)), 4) if lo else None, 'n_lo': len(lo)}
out['pilot_background_by_distance'] = pil
out['pilot_span'] = {'dist_bin_q_place': dict(zip(*np.unique([bins[A.dist_bin(r['span_d'][0])] for r in rows], return_counts=True))),
                     'entropy_q_place_median': round(float(np.median([r['span_h'][0] for r in rows])), 4)}
out['pilot_span']['dist_bin_q_place'] = {k: int(v) for k, v in out['pilot_span']['dist_bin_q_place'].items()}

# ---- (2) primary, tokens only
tok = C.tokenizer()
prim = [it for it in items if it['role'] == 'primary']
n_items_read, probs_read, probs_anyread = 0, set(), set()
dq, dr, gap = [], [], []
s2 = {'items': 0, 'x_closer': 0, 'z_closer': 0, 'equal': 0, 'dx': [], 'dz': []}
for it in prim:
    p = prose[tuple(it['key'])]
    base = len(p) + 1
    S, L, x, y = it['S'], it['piece'], it['x'], it['y']
    enc = tok(p + ' ' + S, return_offsets_mapping=True)
    rec = {'ids': enc['input_ids'], 'off': enc['offset_mapping']}
    mV = A.smap(S, [('del', x['mut_pos'], L), ('ins', y['name_pos'], L)])
    inv = {v: s for s, v in enumerate(mV)}
    qp = inv.get(it['q_place'])
    if qp is None:
        continue
    iq = A.tok_at(rec, base, qp)
    eb = [x['name_pos']] + ([y['name_pos']] if y['name_pos'] < qp else [])
    start = A.tok_at(rec, base, max(eb))
    if it['reads']:
        probs_anyread.add(it['key'][0])
    rd = []
    for r in it['reads']:
        sr = inv.get(r)
        if sr is None:
            continue
        i = A.tok_at(rec, base, sr)
        if i is not None and rec['ids'][i] == rec['ids'][iq]:
            rd.append(i - start)
    if rd:
        n_items_read += 1
        probs_read.add(it['key'][0])
        dq.append(iq - start)
        dr.append(float(np.mean(rd)))
        gap.append((A.dist_bin(iq - start), [A.dist_bin(d) for d in rd]))
    ix = A.tok_at(rec, base, x['mut_pos'])
    for z in it['placebos']:
        izz = A.tok_at(rec, base, z['z_mut_pos'])
        s2['items'] += 1
        a, b = iq - ix, iq - izz
        s2['dx'].append(a); s2['dz'].append(b)
        s2['x_closer' if a < b else 'z_closer' if b < a else 'equal'] += 1

# predicted raw-S1 bias from the pilot's distance gradient (distance only; entropy ignored)
pm = {b: pil[bins[b]]['mean_c'] for b in range(5) if bins[b] in pil}
bias = [pm.get(bq, 0) - float(np.mean([pm.get(b, 0) for b in br])) for bq, br in gap]
out['primary_S1'] = {'items_any_read': sum(1 for it in prim if it['reads']), 'problems_any_read': len(probs_anyread),
                     'items_same_id_read': n_items_read, 'problems_same_id_read': len(probs_read),
                     'dist_q_median': float(np.median(dq)) if dq else None, 'dist_read_median': float(np.median(dr)) if dr else None,
                     'q_bins': {bins[k]: int(v) for k, v in zip(*np.unique([g[0] for g in gap], return_counts=True))},
                     'read_bins': {bins[k]: int(v) for k, v in zip(*np.unique([b for g in gap for b in g[1]], return_counts=True))},
                     'predicted_bias_raw_d_read_item_mean': round(float(np.mean(bias)), 4) if bias else None}
out['primary_S2'] = {'items': s2['items'], 'x_closer': s2['x_closer'], 'z_closer': s2['z_closer'], 'equal': s2['equal'],
                     'median_tokens_x_mut_to_q': float(np.median(s2['dx'])), 'median_tokens_z_mut_to_q': float(np.median(s2['dz']))}
print(json.dumps(out, indent=1, default=str))
