"""Phase 9, part ①: an independent recomputation of the point estimates of p9_b_analysis.py.

Written apart from p9_b_analysis.py (no import from it): character offsets by arithmetic on the two edits,
tokens by offset lookup, x's mentions by a word-boundary regex rather than the lexer, the matched null
and the hierarchy written again. Prints the primary, S1 and S2 means next to those of
results/p9_state/b_analysis_<role>.json.

  ROLE=primary  WORK=/home/user1/p9_work/forward
"""
import json
import os
import re
import sys
from collections import defaultdict
from hashlib import sha1

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_common as C  # noqa: E402

ROLE = os.environ.get('ROLE', 'primary')
WORK = os.environ.get('WORK', '/home/user1/p9_work/forward')


def records():
    z = np.load(os.path.join(WORK, f'{ROLE}.npz'))
    return {k: (z[f'ids_{i}'], z[f'off_{i}'], z[f'logp_{i}'], z[f'ent_{i}']) for i, k in enumerate(z['keys'])}


def token_at(off, c):
    hit = np.flatnonzero((off[:, 0] <= c) & (c < off[:, 1]))
    return int(hit[0]) if len(hit) else None


def shift(pos, edits):
    """S character -> edited text character; edits: (kind, at, n)."""
    out = pos
    for kind, at, n in edits:
        if kind == 'del' and pos >= at + n:
            out -= n
        if kind == 'ins' and pos >= at:
            out += n
    return out


def item(it, R, prose):
    p = prose[tuple(it['key'])]
    b = len(p) + 1
    S, L, x, y = it['S'], it['piece'], it['x'], it['y']
    eV = [('del', x['mut_pos'], L), ('ins', y['name_pos'], L)]
    eA = [('ins', y['name_pos'], L)]
    rS, rV, rA = (R[sha1((p + ' ' + it[k]).encode()).hexdigest()] for k in ('S', 'V', 'A'))
    # q in S: the V offset of q maps back through the two edits (both before q here)
    back = {shift(s, eV): s for s in range(len(S) + 1)}
    qs, qo = back.get(it['q_place']), back.get(it['q_op'])
    if qs is None or qo is None:
        return None
    t0, t1 = token_at(rS[1], b + qs), token_at(rS[1], b + qo)
    if t0 is None or t1 is None:
        return None

    def pair(i, other, edits):
        j = token_at(other[1], b + shift(int(rS[1][i][0]) - b, edits))
        return j if j is not None and other[0][j] == rS[0][i] else None
    span = list(range(t0, t1 + 1))
    js = [(pair(i, rV, eV), pair(i, rA, eA)) for i in span]
    if any(a is None or c is None for a, c in js):
        return None
    c_span = [float(rV[2][a] - rA[2][c]) for a, c in js]
    later = max([x['name_pos']] + ([y['name_pos']] if y['name_pos'] < qs else []))
    start = token_at(rS[1], b + later)
    xs = [(m.start(), m.end()) for m in re.finditer(r'\b' + re.escape(x['name']) + r'\b', S)]
    bg = []
    for i in range(start + 1, t0):
        a0, a1 = int(rS[1][i][0]) - b, int(rS[1][i][1]) - b
        if any(a0 < e and a1 > s for s, e in xs):
            continue
        a, c = pair(i, rV, eV), pair(i, rA, eA)
        if a is not None and c is not None:
            bg.append((float(rV[2][a] - rA[2][c]), float(rS[3][i]), i - start))
    reads = []
    for r in it['reads']:
        s = back.get(r)
        i = token_at(rS[1], b + s) if s is not None else None
        if i is None or i - start < 20 or i >= t0:
            continue
        a, c = pair(i, rV, eV), pair(i, rA, eA)
        if a is not None and c is not None:
            reads.append((float(rV[2][a] - rA[2][c]), float(rS[3][i]), i - start))
    dz = []
    for z in it['placebos']:
        rz = R[sha1((p + ' ' + z['text']).encode()).hexdigest()]
        ez = [('del', z['z_mut_pos'], L), ('ins', y['name_pos'], L)]
        jz = [pair(i, rz, ez) for i in span]
        if all(j is not None for j in jz):
            dz.append(float(sum(rV[2][a] for a, _ in js) - sum(rz[2][j] for j in jz)))
    return {'g': it['key'][0], 's': it['key'][1], 'x': x['name'], 'c': c_span,
            'h': [float(rS[3][i]) for i in span], 'd': [i - start for i in span], 'bg': bg, 'reads': reads, 'dz': dz}


def dbin(d):
    return 0 if d < 5 else 1 if d < 10 else 2 if d < 20 else 3 if d < 40 else 4


def main():
    items = [i for i in C.read_jsonl(os.path.join(C.OUT_DIR, 'b_items.jsonl.gz')) if i['role'] == ROLE]
    prose = {(int(r['task_id']), int(r['sample'])): r['text'] for r in C.read_jsonl(os.path.join(C.OUT_DIR, 'pool_samples.jsonl.gz'))}
    R = records()
    rows = [r for r in (item(it, R, prose) for it in items) if r]
    H = np.array([h for r in rows for _, h, _ in r['bg']])
    q = np.quantile(H, np.arange(1, 20) / 20)
    hb = lambda h: int(np.sum(h >= q))
    tot, per = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
    for r in rows:
        for c, h, d in r['bg']:
            k = (hb(h), dbin(d))
            tot[k][0] += c; tot[k][1] += 1
            per[(r['g'],) + k][0] += c; per[(r['g'],) + k][1] += 1

    def chat(g, h, d):
        e, db = hb(h), dbin(d)
        while True:
            s = tot[(e, db)][0] - per[(g, e, db)][0]
            n = tot[(e, db)][1] - per[(g, e, db)][1]
            if n >= 30 or e == 0:
                return s / n if n else 0.0
            e -= 1

    def mean3(vals):
        a = defaultdict(list)
        for r, v in vals:
            a[(r['g'], r['s'], r['x'])].append(v)
        b = defaultdict(list)
        for (g, s, _), v in a.items():
            b[(g, s)].append(np.mean(v))
        c = defaultdict(list)
        for (g, _), v in b.items():
            c[g].append(np.mean(v))
        return float(np.mean([np.mean(v) for v in c.values()])), len(c)
    prim = mean3([(r, sum(c - chat(r['g'], h, d) for c, h, d in zip(r['c'], r['h'], r['d']))) for r in rows])
    s1 = mean3([(r, (r['c'][0] - chat(r['g'], r['h'][0], r['d'][0])) - np.mean([c - chat(r['g'], h, d) for c, h, d in r['reads']]))
                for r in rows if r['reads']])
    s2 = mean3([(r, np.mean(r['dz'])) for r in rows if r['dz']])
    ref = json.load(open(os.path.join(C.OUT_DIR, f'b_analysis_{ROLE}.json')))
    out = {'items': len(rows), 'items_ref': ref['items'],
           'primary': [round(prim[0], 4), prim[1], ref['primary']['mean'], ref['primary']['problems']],
           'S1': [round(s1[0], 4), s1[1], ref['S1'].get('mean'), ref['S1'].get('problems')],
           'S2': [round(s2[0], 4), s2[1], ref['S2'].get('mean'), ref['S2'].get('problems')]}
    print(json.dumps(out))


if __name__ == '__main__':
    main()
