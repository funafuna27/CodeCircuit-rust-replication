"""Phase 9, part ①: the analysis fixed by PREREGISTRATION_P9_MUTSTATE.md §2 (run once on ROLE=primary).

Reads the forward outputs (p9_b_forward.py) and the items (p9_b_items.py). For each item:
  span    the tokens from q_place to q_op (V's first mutation site of x)
  c_t     log p_V(t) - log p_A(t)          (A = S with y's `mut ` added; V and A share y's edit)
  bg      tokens of S after every edit that precedes q and before q_place, not x's name, whose ids match
          in V and A; c_b with S-side entropy H and distance to the edits
  c-hat   for problem g, the mean c_b of all OTHER problems' background tokens in the same cell of
          20 entropy bins (quantiles of all background entropies) x 5 distance bins (1-4, 5-9, 10-19,
          20-39, 40+); a cell with < 30 tokens is merged into its lower-entropy neighbour
  d       sum over the span of (c_t - c-hat_g(t))
Problem score: mean over items within (source, x), then x within source, then sources within problem.
Primary: one-sided sign flip, H1 mean < 0, alpha 0.05 (99,999 flips + identity, stream [20261022, 11]);
interval 5,000 bootstrap draws of problems (stream [20261022, 12]).
S1 and S2 (only if the primary rejects; Holm over the two at 0.05):
  S1  d_read = (c(q_place) - c-hat) - mean over reads of (c(read) - c-hat), reads = occurrences of x's name
      before q_place and at least 20 tokens after the later edit (rules fixed at gate 2, §2)
  S2  d_z = sum over the span of log p_V - log p_Pz
Descriptive: raw span sum, background mean, R arm, 2 x 2, q_place / q_op alone, E0384 / E0596, Phase 3
only, y in another function excluded, 20%-trimmed problem means, per-token form, the jump (c just after
the edit, just before q, at q).

  ROLE=primary|pilot  WORK=/home/user1/p9_work/forward  OUT=results/p9_state/b_analysis_<role>.json
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_common as C  # noqa: E402

ROLE = os.environ.get('ROLE', 'primary')
WORK = os.environ.get('WORK', '/home/user1/p9_work/forward')
OUT = os.environ.get('OUT', os.path.join(C.OUT_DIR, f'b_analysis_{ROLE}.json'))
ALPHA = 0.05
N_FLIP, N_BOOT = 99999, 5000
DIST_EDGES = [1, 5, 10, 20, 40, 10 ** 9]
N_HBINS, MIN_CELL = 20, 30
READ_MIN_DIST = 20


def sha(t):
    return hashlib.sha1(t.encode()).hexdigest()


def load(role):
    z = np.load(os.path.join(WORK, f'{role}.npz'))
    return {k: {f: z[f'{f}_{i}'] for f in ('ids', 'off', 'logp', 'ent')} for i, k in enumerate(z['keys'])}


def smap(S, edits):
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
    for i, (a, b) in enumerate(rec['off']):
        if a <= base + c < b:
            return i
    for i, (a, b) in enumerate(rec['off']):
        if a >= base + c:
            return i
    return None


def item_tokens(it, F, prose):
    """Per-item token data, or None (dropped) when ids disagree."""
    p = prose[tuple(it['key'])]
    base = len(p) + 1
    S, L, x, y = it['S'], it['piece'], it['x'], it['y']
    eV = [('del', x['mut_pos'], L), ('ins', y['name_pos'], L)]
    maps = {'S': list(range(len(S) + 1)), 'V': smap(S, eV), 'A': smap(S, [('ins', y['name_pos'], L)]),
            'R': smap(S, [('del', x['mut_pos'], L)])}
    inv = {v: s for s, v in enumerate(maps['V'])}
    qp, qo = inv.get(it['q_place']), inv.get(it['q_op'])
    if qp is None or qo is None:
        return None
    recs = {k: F[sha(p + ' ' + it[k])] for k in ('S', 'V', 'A', 'R')}
    pz = []
    for z in it['placebos']:
        mz = smap(S, [('del', z['z_mut_pos'], L), ('ins', y['name_pos'], L)])
        pz.append((mz, F[sha(p + ' ' + z['text'])]))

    def at(k, s, rec=None, m=None):
        return tok_at(rec or recs[k], base, (m or maps[k])[s])

    span_S = [i for i in range(at('S', qp), at('S', qo) + 1)]
    if not span_S:
        return None
    off_S = recs['S']['off']

    def s_of(i):                                  # S char offset of S token i
        return int(off_S[i][0]) - base

    def tok_in(k, i, rec=None, m=None):
        j = at(k, s_of(i), rec, m)
        r = rec or recs[k]
        return j if j is not None and r['ids'][j] == recs['S']['ids'][i] else None

    span = {k: [tok_in(k, i) for i in span_S] for k in ('V', 'A', 'R')}
    if any(j is None for v in span.values() for j in v):
        return None
    edits_before = [x['name_pos']] + ([y['name_pos']] if y['name_pos'] < qp else [])
    start = at('S', max(edits_before))
    xs = [(s, s + len(x['name'])) for kind, t, s in C.lex(S) if kind == 'ident' and t == x['name']]
    bg, bg_j = [], []
    for i in range(start + 1, span_S[0]):
        a, b = int(off_S[i][0]) - base, int(off_S[i][1]) - base
        if any(a < xb and b > xa for xa, xb in xs):
            continue
        jv, ja = tok_in('V', i), tok_in('A', i)
        if jv is None or ja is None:
            continue
        bg.append((float(recs['V']['logp'][jv] - recs['A']['logp'][ja]), float(recs['S']['ent'][i]), i - start))
        bg_j.append((i, jv, ja))
    span_c = [float(recs['V']['logp'][jv] - recs['A']['logp'][ja]) for jv, ja in zip(span['V'], span['A'])]
    span_h = [float(recs['S']['ent'][i]) for i in span_S]
    span_d = [i - start for i in span_S]
    # reads of x before q, at least READ_MIN_DIST tokens after the later edit (S1): (c, H, distance)
    reads = []
    for r in it['reads']:
        sr = inv.get(r)
        if sr is None:
            continue
        i = at('S', sr)
        if i is None or i - start < READ_MIN_DIST or i >= span_S[0]:
            continue
        jv, ja = tok_in('V', i), tok_in('A', i)
        if jv is not None and ja is not None:
            reads.append((float(recs['V']['logp'][jv] - recs['A']['logp'][ja]), float(recs['S']['ent'][i]), i - start))
    # placebos (S2)
    dz = []
    for mz, rz in pz:
        jz = [tok_in(None, i, rz, mz) for i in span_S]
        if all(j is not None for j in jz):
            dz.append(float(sum(recs['V']['logp'][jv] for jv in span['V']) - sum(rz['logp'][j] for j in jz)))
    lp = lambda k, idx: float(sum(recs[k]['logp'][j] for j in idx))
    return {'problem': it['key'][0], 'source': it['key'][1], 'x': x['name'], 'site': it['site_code'],
            'phase3': None, 'y_other_fn': it['y_other_fn'],
            'span_c': span_c, 'span_h': span_h, 'span_d': span_d, 'bg': bg, 'reads': reads, 'dz': dz,
            'bg_j': bg_j, 'q_j': (span_S[0], span['V'][0], span['A'][0]),
            'raw': sum(span_c), 'R_S': lp('R', span['R']) - lp('S', span_S), 'V_S': lp('V', span['V']) - lp('S', span_S),
            'A_S': lp('A', span['A']) - lp('S', span_S),
            'place_c': span_c[0], 'op_c': span_c[-1], 'n_span': len(span_c)}


def dist_bin(d):
    return int(np.searchsorted(DIST_EDGES, d, side='right') - 1)


def null_model(rows):
    """Leave-one-problem-out cell means. Returns f(problem, h, d) -> c-hat."""
    allh = np.array([h for r in rows for _, h, _ in r['bg']])
    edges = np.quantile(allh, np.linspace(0, 1, N_HBINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    hbin = lambda h: int(np.clip(np.searchsorted(edges, h, side='right') - 1, 0, N_HBINS - 1))
    tot = defaultdict(lambda: np.zeros(2))
    byp = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
    for r in rows:
        for c, h, d in r['bg']:
            cell = (hbin(h), dist_bin(d))
            tot[cell] += (c, 1)
            byp[r['problem']][cell] += (c, 1)

    def f(problem, h, d):
        hb, db = hbin(h), dist_bin(d)
        while True:
            s = tot[(hb, db)] - byp[problem][(hb, db)]
            if s[1] >= MIN_CELL or hb == 0:
                break
            hb -= 1
        if s[1] == 0:
            return 0.0
        return float(s[0] / s[1])
    return f, edges


def hier(rows, val):
    sx = defaultdict(list)
    for r in rows:
        v = val(r)
        if v is not None:
            sx[(r['problem'], r['source'], r['x'])].append(v)
    src = defaultdict(list)
    for (p, s, _), v in sx.items():
        src[(p, s)].append(np.mean(v))
    prob = defaultdict(list)
    for (p, _), v in src.items():
        prob[p].append(np.mean(v))
    return {p: float(np.mean(v)) for p, v in prob.items()}


def test(per_problem, stream, trim=0.0):
    v = np.array([per_problem[k] for k in sorted(per_problem)])
    if len(v) < 2:
        return {'problems': len(v)}
    if trim:
        lo, hi = np.quantile(v, [trim, 1 - trim])
        v = np.clip(v, lo, hi)
    rng = np.random.default_rng([20261022, stream])
    flips = rng.choice([-1.0, 1.0], size=(N_FLIP, len(v)))
    obs = v.mean()
    p = (1 + np.sum((flips * v).mean(1) <= obs)) / (N_FLIP + 1)
    rb = np.random.default_rng([20261022, stream + 1])
    boots = np.array([v[rb.integers(len(v), size=len(v))].mean() for _ in range(N_BOOT)])
    return {'problems': int(len(v)), 'mean': round(float(obs), 4),
            'ci95': [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            'p_one_sided_less': round(float(p), 5), 'sd': round(float(v.std(ddof=1)), 4),
            'mde80_one_sided': round(float((1.645 + 0.842) * v.std(ddof=1) / np.sqrt(len(v))), 4)}


def main():
    items = [i for i in C.read_jsonl(os.path.join(C.OUT_DIR, 'b_items.jsonl.gz')) if i['role'] == ROLE]
    samples = C.read_jsonl(os.path.join(C.OUT_DIR, 'pool_samples.jsonl.gz'))
    prose = {(int(r['task_id']), int(r['sample'])): r['text'] for r in samples}
    phase = {(int(r['task_id']), int(r['sample'])): r['phase'] for r in samples}
    F = load(ROLE)
    rows, dropped = [], 0
    for it in items:
        r = item_tokens(it, F, prose)
        if r is None:
            dropped += 1
            continue
        r['phase3'] = phase[tuple(it['key'])] == '3'
        rows.append(r)
    f, edges = null_model(rows)
    for r in rows:
        r['chat'] = [f(r['problem'], h, d) for h, d in zip(r['span_h'], r['span_d'])]
        r['d'] = float(sum(c - ch for c, ch in zip(r['span_c'], r['chat'])))
        r['d_tok'] = r['d'] / r['n_span']
        r['bg_mean'] = float(np.mean([c for c, _, _ in r['bg']])) if r['bg'] else None
        r['res_q'] = r['place_c'] - r['chat'][0]
        r['res_reads'] = float(np.mean([c - f(r['problem'], h, d) for c, h, d in r['reads']])) if r['reads'] else None
        r['d_read'] = (r['res_q'] - r['res_reads']) if r['reads'] else None
        r['d_z'] = float(np.mean(r['dz'])) if r['dz'] else None
        after = [c for c, _, d in r['bg'] if d <= 4]
        before = [c for c, _, _ in r['bg'][-5:]]
        r['jump_after'] = float(np.mean(after)) if after else None
        r['jump_before'] = float(np.mean(before)) if before else None
        r['jump_at_q'] = r['raw'] / r['n_span']
        r['jump'] = (r['jump_at_q'] - r['jump_before']) if before else None
    out = {'role': ROLE, 'items': len(rows), 'dropped_id_mismatch': dropped,
           'problems': len({r['problem'] for r in rows}), 'entropy_bin_edges': [round(float(e), 4) for e in edges[1:-1]]}
    prim = test(hier(rows, lambda r: r['d']), 11)
    out['primary'] = dict(prim, rejected=bool(prim.get('p_one_sided_less', 1) < ALPHA))
    s1 = test(hier(rows, lambda r: r['d_read']), 13)
    s2 = test(hier(rows, lambda r: r['d_z']), 15)
    # Holm over S1 and S2 at ALPHA, gated by the primary
    ps = [s1.get('p_one_sided_less', 1.0), s2.get('p_one_sided_less', 1.0)]
    rej = [False, False]
    if out['primary']['rejected']:
        lo, hi = (0, 1) if ps[0] <= ps[1] else (1, 0)
        if ps[lo] < ALPHA / 2:
            rej[lo] = True
            rej[hi] = ps[hi] < ALPHA
    out['S1'] = dict(s1, tested=out['primary']['rejected'], rejected=rej[0])
    out['S2'] = dict(s2, tested=out['primary']['rejected'], rejected=rej[1])
    desc = {
        'raw_span_c': test(hier(rows, lambda r: r['raw']), 21),
        'background_mean_per_token': test(hier(rows, lambda r: r['bg_mean']), 23),
        'R_minus_S': test(hier(rows, lambda r: r['R_S']), 25),
        'V_minus_S': test(hier(rows, lambda r: r['V_S']), 27),
        'A_minus_S': test(hier(rows, lambda r: r['A_S']), 29),
        'q_place_only': test(hier(rows, lambda r: r['place_c'] - r['chat'][0]), 31),
        'q_op_only': test(hier(rows, lambda r: r['op_c'] - r['chat'][-1]), 33),
        'per_token': test(hier(rows, lambda r: r['d_tok']), 35),
        'trimmed_20': test(hier(rows, lambda r: r['d']), 37, trim=0.2),
        'phase3_only': test(hier([r for r in rows if r['phase3']], lambda r: r['d']), 39),
        'without_y_other_fn': test(hier([r for r in rows if not r['y_other_fn']], lambda r: r['d']), 41),
        'S1_q_place_residual': test(hier([r for r in rows if r['reads']], lambda r: r['res_q']), 47),
        'S1_reads_residual': test(hier(rows, lambda r: r['res_reads']), 49),
        'jump_after_edit': test(hier(rows, lambda r: r['jump_after']), 51),
        'jump_before_q': test(hier(rows, lambda r: r['jump_before']), 53),
        'jump_at_q_per_token': test(hier(rows, lambda r: r['jump_at_q']), 55),
        'jump_at_q_minus_before': test(hier(rows, lambda r: r['jump']), 57),
    }
    for site, s in (('E0384', 43), ('E0596', 45)):
        desc[f'site_{site}'] = test(hier([r for r in rows if r['site'] == site], lambda r: r['d']), s)
    out['descriptive'] = desc
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
