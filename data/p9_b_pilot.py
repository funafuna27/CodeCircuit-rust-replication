"""Phase 9, stage B: what the pilot and the behaviour gate show. Reads forward outputs of those roles only.

Pilot (problems that can never enter the primary analysis): per item, with the span = the tokens from
q_place to q_op of the first mutation site,
  D_VA   sum over the span of log p_V - log p_A   (the primary contrast's form; in the pilot y's edit lies
         after q, so at q this equals R - S: x's `mut` removed, nothing else)
  bg     the same contrast at each background token: after x's edit, before q_place, not x's name
  room   p_S and the entropy at q_place and at q_op (S side), to see whether there is mass to move
Behaviour gate (natural errors on a parameter): D_beh = sum over the span of log p_M - log p_S, where M
adds `mut ` to the parameter; its background is the same contrast at non-parameter tokens between the
signature and q. Gemma sampled S, so under no response E[D_beh] <= 0: a positive mean is conservative.

Every per-problem figure averages item -> (source, x) -> source -> problem, each level equally.
Intervals: 5000 bootstrap draws of problems; tests: problem-level sign flips (99,999 + identity).
Streams [20261022, 6] and [20261022, 7].

  WORK=/home/user1/p9_work/forward  OUT=results/p9_state/b_pilot.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_common as C  # noqa: E402
import p9_a2_swaps as W  # noqa: E402

WORK = os.environ.get('WORK', '/home/user1/p9_work/forward')
OUT = os.environ.get('OUT', os.path.join(C.OUT_DIR, 'b_pilot.json'))


def load(role):
    z = np.load(os.path.join(WORK, f'{role}.npz'))
    keys = list(z['keys'])
    return {k: {f: z[f'{f}_{i}'] for f in ('ids', 'off', 'logp', 'ent')} for i, k in enumerate(keys)}


def sha(t):
    import hashlib
    return hashlib.sha1(t.encode()).hexdigest()


def smap(S, edits):
    """S character offset -> variant offset, for edits [('del', pos, n) | ('ins', pos, n)] in S offsets."""
    out = []
    for s in range(len(S) + 1):
        v = s
        for kind, p, n in edits:
            if kind == 'del' and s >= p + n:
                v -= n
            elif kind == 'ins' and s >= p:
                v += n
        out.append(v)
    return out


def tok_at(rec, base, c):
    off = rec['off']
    pos = base + c
    for i, (a, b) in enumerate(off):
        if a <= pos < b:
            return i
    for i, (a, b) in enumerate(off):
        if a >= pos:
            return i
    return None


def hier(rows, val):
    """item -> (source, x) -> source -> problem means; returns {problem: mean}."""
    sx = defaultdict(list)
    for r in rows:
        sx[(r['problem'], r['source'], r['x'])].append(val(r))
    src = defaultdict(list)
    for (p, s, _), v in sx.items():
        src[(p, s)].append(np.mean(v))
    prob = defaultdict(list)
    for (p, _), v in src.items():
        prob[p].append(np.mean(v))
    return {p: float(np.mean(v)) for p, v in prob.items()}


def infer(per_problem, seed, alternative):
    v = np.array(list(per_problem.values()))
    rng = np.random.default_rng([20261022, seed])
    boots = [v[rng.integers(len(v), size=len(v))].mean() for _ in range(5000)]
    flips = rng.choice([-1.0, 1.0], size=(99999, len(v)))
    null = (flips * v).mean(1)
    obs = v.mean()
    if alternative == 'less':
        p = (1 + np.sum(null <= obs)) / 100000
    else:
        p = (1 + np.sum(null >= obs)) / 100000
    return {'problems': len(v), 'mean': round(float(obs), 4), 'ci95': [round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4)], 'sd_problems': round(float(v.std(ddof=1)), 4),
            'p_one_sided': round(float(p), 5), 'alternative': alternative,
            'mde80_two_sided_05': round(float(2.8 * v.std(ddof=1) / np.sqrt(len(v))), 4)}


def span_tokens(rec, base, cmap, q_place_s, q_op_s):
    a, b = tok_at(rec, base, cmap[q_place_s]), tok_at(rec, base, cmap[q_op_s])
    return list(range(a, b + 1)) if a is not None and b is not None and b >= a else []


def main():
    items = C.read_jsonl(os.path.join(C.OUT_DIR, 'b_items.jsonl.gz'))
    prose = {(int(r['task_id']), int(r['sample'])): r['text'] for r in C.read_jsonl(os.path.join(C.OUT_DIR, 'pool_samples.jsonl.gz'))}
    out = {}

    # ---------------------------------------------------------------- pilot
    F = load('pilot')
    rows, bg_rows, problems_idmismatch = [], [], 0
    for it in [i for i in items if i['role'] == 'pilot']:
        p = prose[tuple(it['key'])]
        base = len(p) + 1
        S, L, x, y = it['S'], it['piece'], it['x'], it['y']
        eV = [('del', x['mut_pos'], L), ('ins', y['name_pos'], L)]
        mV, mA, mR = smap(S, eV), smap(S, [('ins', y['name_pos'], L)]), smap(S, [('del', x['mut_pos'], L)])
        inv = {v: s for s, v in enumerate(mV)}
        qp, qo = inv[it['q_place']], inv[it['q_op']]             # q in S offsets
        recs = {k: F[sha(p + ' ' + it[k])] for k in ('S', 'V', 'A', 'R')}
        ident = lambda m: m                                       # S map
        maps = {'S': list(range(len(S) + 1)), 'V': mV, 'A': mA, 'R': mR}
        sp = {k: span_tokens(recs[k], base, maps[k], qp, qo) for k in recs}
        if len({len(v) for v in sp.values()}) != 1 or not sp['S'] or \
                any(not np.array_equal(recs[k]['ids'][sp[k]], recs['S']['ids'][sp['S']]) for k in recs):
            problems_idmismatch += 1
            continue
        lp = {k: float(np.sum(recs[k]['logp'][sp[k]])) for k in recs}
        r = {'problem': it['key'][0], 'source': it['key'][1], 'x': x['name'], 'site': it['site_code'],
             'D_VA': lp['V'] - lp['A'], 'D_RS': lp['R'] - lp['S'], 'D_VS': lp['V'] - lp['S'],
             'p_S_place': float(np.exp(recs['S']['logp'][sp['S'][0]])), 'p_S_op': float(np.exp(recs['S']['logp'][sp['S'][-1]])),
             'H_S_place': float(recs['S']['ent'][sp['S'][0]]), 'H_S_op': float(recs['S']['ent'][sp['S'][-1]]),
             'span_len': len(sp['S'])}
        # background: S tokens after x's edit and before q_place, not x's name
        x_end = x['name_pos']
        t0 = tok_at(recs['S'], base, x_end)
        xs = [(a, b) for a, b in [(s, s + len(x['name'])) for kind, t, s in C.lex(S) if kind == 'ident' and t == x['name']]]
        bg = []
        for i in range(t0 + 1, sp['S'][0]):
            a, b = recs['S']['off'][i]
            if any(a - base < xb and b - base > xa for xa, xb in xs):
                continue
            sV = tok_at(recs['V'], base, mV[a - base]) if a - base < len(mV) else None
            if sV is None or recs['V']['ids'][sV] != recs['S']['ids'][i]:
                continue
            sA = tok_at(recs['A'], base, mA[a - base])
            bg.append({'c': float(recs['V']['logp'][sV] - recs['A']['logp'][sA]), 'H_S': float(recs['S']['ent'][i]),
                       'dist': i - t0})
        r['bg_mean'] = float(np.mean([b['c'] for b in bg])) if bg else None
        r['bg_n'] = len(bg)
        r['D_minus_bg_per_token'] = (r['D_VA'] / r['span_len'] - r['bg_mean']) if bg else None
        rows.append(r)
        bg_rows += [dict(b, problem=r['problem']) for b in bg]
    pp = lambda f: hier([r for r in rows if r[f] is not None], lambda r: r[f])
    out['pilot'] = {
        'items': len(rows), 'dropped_token_mismatch': problems_idmismatch,
        'problems': len({r['problem'] for r in rows}),
        'D_VA': infer(pp('D_VA'), 6, 'less'),
        'D_VA_per_token_minus_background': infer(pp('D_minus_bg_per_token'), 6, 'less'),
        'background_mean_per_token': infer(pp('bg_mean'), 6, 'less'),
        'room': {k: [round(float(np.percentile([r[k] for r in rows], q)), 3) for q in (25, 50, 75)]
                 for k in ('p_S_place', 'p_S_op', 'H_S_place', 'H_S_op')},
        'span_len': [int(np.percentile([r['span_len'] for r in rows], q)) for q in (25, 50, 75)],
        'background_tokens_per_item': [int(np.percentile([r['bg_n'] for r in rows], q)) for q in (25, 50, 75)],
        'by_site': {s: infer(hier([r for r in rows if r['site'] == s], lambda r: r['D_VA']), 6, 'less')
                    for s in ('E0384', 'E0596') if sum(r['site'] == s for r in rows) >= 5},
    }
    if bg_rows:
        H = np.array([b['H_S'] for b in bg_rows])
        c = np.array([b['c'] for b in bg_rows])
        out['pilot']['background_by_entropy_tercile'] = [round(float(c[(H >= lo) & (H <= hi)].mean()), 4)
                                                         for lo, hi in zip(np.percentile(H, [0, 33.3, 66.7]),
                                                                           np.percentile(H, [33.3, 66.7, 100]))]

    # ---------------------------------------------------------------- behaviour gate
    F = load('behaviour')
    brows = []
    for it in [i for i in items if i['role'] == 'behaviour']:
        p = prose[tuple(it['key'])]
        base = len(p) + 1
        S, M = it['S'], it['M']
        site = {'s': it['sites'][0][0], 'e': it['sites'][0][1]}
        q_op = W.mutating_token(S, site)
        if q_op is None:
            q_op = site['s']
        mM = smap(S, [('ins', it['decl_pos'], 4)])
        rS, rM = F[sha(p + ' ' + S)], F[sha(p + ' ' + M)]
        sS = span_tokens(rS, base, list(range(len(S) + 1)), site['s'], q_op)
        sM = span_tokens(rM, base, mM, site['s'], q_op)
        if not sS or len(sS) != len(sM) or not np.array_equal(rS['ids'][sS], rM['ids'][sM]):
            continue
        D = float(np.sum(rM['logp'][sM]) - np.sum(rS['logp'][sS]))
        t0 = tok_at(rS, base, it['decl_pos'] + 2)
        bg = []
        for i in range(t0 + 1, sS[0]):
            a = rS['off'][i][0] - base
            j = tok_at(rM, base, mM[a])
            if j is not None and rM['ids'][j] == rS['ids'][i]:
                bg.append(float(rM['logp'][j] - rS['logp'][i]))
        brows.append({'problem': it['key'][0], 'source': it['key'][1], 'x': 'a', 'D': D, 'span_len': len(sS),
                      'bg': float(np.mean(bg)) if bg else None,
                      'p_S_place': float(np.exp(rS['logp'][sS[0]]))})
    out['behaviour'] = {
        'items': len(brows), 'problems': len({r['problem'] for r in brows}),
        'D_beh': infer(hier(brows, lambda r: r['D']), 7, 'greater'),
        'D_beh_per_token_minus_background': infer(hier([r for r in brows if r['bg'] is not None],
                                                       lambda r: r['D'] / r['span_len'] - r['bg']), 7, 'greater'),
        'background_mean_per_token': infer(hier([r for r in brows if r['bg'] is not None], lambda r: r['bg']), 7, 'greater'),
        'p_S_place_quartiles': [round(float(np.percentile([r['p_S_place'] for r in brows], q)), 3) for q in (25, 50, 75)],
    }
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
