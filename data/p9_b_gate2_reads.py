"""Gate-2 counts: S1 eligibility variants (tokenization only, primary). PREREGISTRATION_P9_MUTSTATE.md §7."""
import os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p9_common as C
import p9_b_analysis as A
items = [i for i in C.read_jsonl(os.path.join(C.OUT_DIR, 'b_items.jsonl.gz')) if i['role'] == 'primary']
prose = {(int(r['task_id']), int(r['sample'])): r['text'] for r in C.read_jsonl(os.path.join(C.OUT_DIR, 'pool_samples.jsonl.gz'))}
tok = C.tokenizer()
cnt = defaultdict(lambda: [0, set()])
for it in items:
    p = prose[tuple(it['key'])]; base = len(p) + 1
    S, L, x, y = it['S'], it['piece'], it['x'], it['y']
    enc = tok(p + ' ' + S, return_offsets_mapping=True)
    rec = {'ids': enc['input_ids'], 'off': enc['offset_mapping']}
    inv = {v: s for s, v in enumerate(A.smap(S, [('del', x['mut_pos'], L), ('ins', y['name_pos'], L)]))}
    qp = inv.get(it['q_place'])
    if qp is None:
        continue
    iq = A.tok_at(rec, base, qp)
    eb = [x['name_pos']] + ([y['name_pos']] if y['name_pos'] < qp else [])
    start = A.tok_at(rec, base, max(eb))
    flags = defaultdict(bool)
    for r in it['reads']:
        sr = inv.get(r)
        if sr is None:
            continue
        i = A.tok_at(rec, base, sr)
        if i is None:
            continue
        same = rec['ids'][i] == rec['ids'][iq]
        far = i - start >= 20
        flags['same'] |= same; flags['any'] = True
        flags['same_far'] |= same and far; flags['any_far'] |= far
        flags['same_far10'] |= same and i - start >= 10; flags['any_far10'] |= i - start >= 10
    for k, v in flags.items():
        if v:
            cnt[k][0] += 1; cnt[k][1].add(it['key'][0])
for k in ('any', 'same', 'any_far10', 'same_far10', 'any_far', 'same_far'):
    print(f'{k:10s} items {cnt[k][0]:5d} problems {len(cnt[k][1]):4d}')
