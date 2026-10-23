"""Phase 9, part ① at gate 3: the token positions PREREGISTRATION_P9_MECH.md §2 fixes. No model is run.

For every primary and pilot item:
  D   V's tokens overlapping x's binding name at its declaration, which must be where rustc's `mut `
      suggestion for x points (V compiled again with --error-format=json); else the item is dropped
  K   the token after D (inside x's declaration statement)
  W   D through the last token that starts before the end of D's line (the whole declaration, descriptive)
  span, reads (>= 20 tokens after the later edit, before q), background (after the later edit, before q,
      not a mention of x): readouts, as part ①
Each V token is mapped to A through its last character (V and A differ only by x's `mut `); the token
ids must agree, or the item is dropped and counted. For G's items: the prefix ids (BOS through the token
before q_place) and the target id (q_place's token).

  OUT=results/p9_mech/spans.jsonl.gz   WORKERS=16
"""
import gzip
import json
import os
import resource
import sys
import tempfile
from collections import Counter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
import p9_common as C  # noqa: E402
import p9_mech_common as MC  # noqa: E402
from p9_b_analysis import smap, tok_at  # noqa: E402

WORKERS = int(os.environ.get('WORKERS', 16))


def suggestion_offsets(V):
    """Offsets in V where rustc suggests inserting `mut ` (any error's child span)."""
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    import p9_a1_diag as D
    with tempfile.TemporaryDirectory() as td:
        r = D.compile_lib(V, td)
    out = set()
    for d in r['diags']:
        if d['level'] != 'error':
            continue
        for ch in d.get('children', []):
            for s in ch.get('spans', []):
                if (s.get('repl') or '').startswith('mut ') and s.get('w') == 'code':
                    out.add(s['s'])
    return sorted(out), [d['code'] for d in r['diags'] if d['level'] == 'error']


def build(it, prose, tok, sugg):
    p = prose[tuple(it['key'])]
    base = len(p) + 1
    S, L, x, y, V, A = it['S'], it['piece'], it['x'], it['y'], it['V'], it['A']
    mV = smap(S, [('del', x['mut_pos'], L), ('ins', y['name_pos'], L)])
    mA = smap(S, [('ins', y['name_pos'], L)])
    invV = {v: s for s, v in enumerate(mV)}
    dV = mV[x['name_pos']]
    name = x['name']
    if V[dV:dV + len(name)] != name:
        return None, 'name_not_at_declaration'
    if dV not in sugg:
        return None, 'no_rustc_suggestion_at_declaration'
    eV = tok(p + ' ' + V, return_offsets_mapping=True)
    eA = tok(p + ' ' + A, return_offsets_mapping=True)
    idsV, offV, idsA, offA = eV['input_ids'], eV['offset_mapping'], eA['input_ids'], eA['offset_mapping']
    recA = {'off': offA}

    def to_a(i):
        """V token i -> A token through its last character; None when the ids differ."""
        c = offV[i][1] - 1 - base
        if c < 0:
            j = i
        else:
            s = invV.get(c)
            if s is None:
                return None
            j = tok_at(recA, base, mA[s])
        return j if j is not None and idsA[j] == idsV[i] else None

    D = [i for i, (a, b) in enumerate(offV) if a < base + dV + len(name) and b > base + dV and i > 0]
    if not D:
        return None, 'no_D_token'
    K = [max(D) + 1]
    nl = V.find('\n', dV)
    line_end = base + (nl if nl >= 0 else len(V))
    W = [i for i in range(min(D), len(idsV)) if offV[i][0] < line_end]
    jq0, jq1 = tok_at({'off': offV}, base, it['q_place']), tok_at({'off': offV}, base, it['q_op'])
    span = list(range(jq0, jq1 + 1))
    later = max([dV] + ([mV[y['name_pos']]] if mV[y['name_pos']] < it['q_place'] else []))
    start = tok_at({'off': offV}, base, later)
    xs = [(base + a, base + b) for a, b in it['x_spans']]
    reads = []
    for r in it['reads']:
        i = tok_at({'off': offV}, base, r)
        if i is not None and i - start >= MC.READ_MIN_DIST and i < jq0:
            reads.append(i)
    bg = [i for i in range(start + 1, jq0) if not any(offV[i][0] < b and offV[i][1] > a for a, b in xs)]
    out = {'id': MC.item_id(it), 'role': it['role'], 'key': it['key'], 'problem': it['key'][0], 'source': it['key'][1],
           'x': name, 'site': it['site_code'], 'idsV': idsV, 'idsA': idsA, 'jq0': jq0}
    for nm, idx in (('D', D), ('K', K), ('W', W), ('span', span), ('reads', reads), ('bg', bg)):
        a_idx = [to_a(i) for i in idx]
        if nm in ('D', 'K', 'W', 'span') and any(j is None for j in a_idx):
            return None, f'{nm}_ids_differ'
        keep = [(i, j) for i, j in zip(idx, a_idx) if j is not None]
        out[f'{nm}_V'] = [i for i, _ in keep]
        out[f'{nm}_A'] = [j for _, j in keep]
    out['target'] = idsV[jq0]
    return out, None


def main():
    items = [i for i in C.read_jsonl(os.path.join(C.OUT_DIR, 'b_items.jsonl.gz')) if i['role'] in ('primary', 'pilot')]
    prose = {(int(r['task_id']), int(r['sample'])): r['text'] for r in C.read_jsonl(os.path.join(C.OUT_DIR, 'pool_samples.jsonl.gz'))}
    with Pool(WORKERS) as pool:
        sug = pool.map(suggestion_offsets, [it['V'] for it in items], chunksize=4)
    tok = C.tokenizer()
    g_ids = {MC.item_id(it) for it in MC.g_items(items)}
    rows, drops = [], Counter()
    for it, (offs, codes) in zip(items, sug):
        r, why = build(it, prose, tok, set(offs))
        if r is None:
            drops[(it['role'], why)] += 1
            continue
        r['g_item'] = r['id'] in g_ids
        r['rustc_errors'] = codes
        rows.append(r)
    os.makedirs(MC.OUT, exist_ok=True)
    with gzip.open(MC.SPANS, 'wt') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    summ = {'items': len(items), 'kept': Counter(r['role'] for r in rows), 'dropped': {f'{a}:{b}': n for (a, b), n in drops.items()},
            'g_items': sum(r['g_item'] for r in rows),
            'problems_primary': len({r['problem'] for r in rows if r['role'] == 'primary'}),
            'D_tokens': Counter(len(r['D_V']) for r in rows), 'with_reads': sum(bool(r['reads_V']) for r in rows if r['role'] == 'primary'),
            'prefix_tokens_g': sorted(r['jq0'] for r in rows if r['g_item'])[::20]}
    summ = json.loads(json.dumps(summ, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)))
    with open(os.path.join(MC.OUT, 'spans_summary.json'), 'w') as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
