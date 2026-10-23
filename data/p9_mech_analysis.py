"""Phase 9, part ① at gate 3: the analysis fixed by PREREGISTRATION_P9_MECH.md §3-§5 (run once per part).

  PART=P   patching: P1 (d_D > 0), then Holm over P2 (d_D - d_K > 0) and P3 (per-token span - reads > 0);
           descriptives (fraction restored, K, W, background, span per token minus background, sites, sweep)
  PART=G   graphs: G1 (s_graph(D) - s_GxA(D) > 0, one item per problem); descriptive shares and the
           Spearman correlations of each method's share on D with the item's patching effect
Streams [20261023, k]: P1 1-2, P2 3-4, P3 5-6, G1 11-12, the correlation difference 13, descriptives 21+.

  REHEARSAL=1: no guard, the pilot patching and the local test graphs (MECH_WORK), nothing in results/
  GRAPH_DIR: where the Colab outputs were copied (default MECH_WORK/colab/out)
"""
import glob
import gzip
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_common as C  # noqa: E402
import p9_mech_common as MC  # noqa: E402

PART = os.environ.get('PART', 'P')
REHEARSAL = os.environ.get('REHEARSAL', '0') == '1'
GRAPH_DIR = os.environ.get('GRAPH_DIR', os.path.join(MC.WORK, 'colab', 'out'))
ALPHA = 0.05


def holm(ps):
    """Holm at ALPHA over a dict name -> p; returns name -> rejected."""
    order = sorted(ps, key=lambda k: ps[k])
    out, still = {}, True
    for i, k in enumerate(order):
        still = still and ps[k] < ALPHA / (len(ps) - i)
        out[k] = bool(still)
    return out


def part_p():
    role = 'pilot' if REHEARSAL else 'primary'
    path = os.path.join(MC.WORK if REHEARSAL else MC.OUT, f'patch_{role}.jsonl.gz')
    meta = json.load(open(path.replace('.jsonl.gz', '.json')))
    if MC.sha256(path) != meta['sha256']:
        sys.exit(f'{path}: sha256 differs from the one recorded when it was written')
    rows = C.read_jsonl(path)
    out = {'items': len(rows), 'problems': len({r['problem'] for r in rows}), 'patch_meta': meta}
    out['P1'] = MC.test_greater(MC.hier(rows, lambda r: r['d_D']), 1)
    out['P1']['rejected'] = bool(out['P1'].get('p_one_sided_greater', 1) < ALPHA)
    p2 = MC.test_greater(MC.hier(rows, lambda r: r['d_D'] - r['d_K']), 3)
    p3 = MC.test_greater(MC.hier(rows, lambda r: (r['d_D'] / r['n_span'] - float(np.mean(r['d_R']))) if r['d_R'] else None), 5)
    rej = holm({'P2': p2.get('p_one_sided_greater', 1), 'P3': p3.get('p_one_sided_greater', 1)}) if out['P1']['rejected'] \
        else {'P2': False, 'P3': False}
    out['P2'] = dict(p2, tested=out['P1']['rejected'], rejected=rej['P2'])
    out['P3'] = dict(p3, tested=out['P1']['rejected'], rejected=rej['P3'])
    gap = MC.hier(rows, lambda r: r['lpA_span'] - r['lpV_span'])
    dD = MC.hier(rows, lambda r: r['d_D'])
    desc = {'gap_A_minus_V_span': MC.test_greater(gap, 21),
            'fraction_restored_D': round(float(np.mean(list(dD.values())) / np.mean(list(gap.values()))), 4),
            'fraction_restored_W': round(float(np.mean(list(MC.hier(rows, lambda r: r['d_W']).values()))
                                               / np.mean(list(gap.values()))), 4),
            'd_K': MC.test_greater(MC.hier(rows, lambda r: r['d_K']), 23),
            'd_W': MC.test_greater(MC.hier(rows, lambda r: r['d_W']), 25),
            'd_background_per_token': MC.test_greater(MC.hier(rows, lambda r: r['d_bg']), 27),
            'span_per_token_minus_background': MC.test_greater(
                MC.hier(rows, lambda r: (r['d_D'] / r['n_span'] - r['d_bg']) if r['d_bg'] is not None else None), 29),
            'reads_per_token': MC.test_greater(MC.hier(rows, lambda r: float(np.mean(r['d_R'])) if r['d_R'] else None), 31)}
    for site, s in (('E0384', 33), ('E0596', 35)):
        desc[f'P1_{site}'] = MC.test_greater(MC.hier([r for r in rows if r['site'] == site], lambda r: r['d_D']), s)
    sw = [r for r in rows if r.get('sweep')]
    if sw:
        M = np.array([r['sweep'] for r in sw])
        rb = MC.rng(41)
        B = np.array([M[rb.integers(len(M), size=len(M))].mean(0) for _ in range(2000)])
        desc['sweep_by_layer'] = {'items': len(sw), 'mean': M.mean(0).round(4).tolist(),
                                  'ci95_low': np.percentile(B, 2.5, axis=0).round(4).tolist(),
                                  'ci95_high': np.percentile(B, 97.5, axis=0).round(4).tolist()}
    out['descriptive'] = desc
    return out


def let_names(code, q_place, x):
    """Binding names of other `let` statements before q (V's code offsets)."""
    return [(m.start(2), m.end(2)) for m in re.finditer(r'\blet\s+(mut\s+)?([A-Za-z_][A-Za-z0-9_]*)', code[:q_place])
            if m.group(2) != x]


def part_g():
    spans = {r['id']: r for r in C.read_jsonl(MC.SPANS)}
    items = {MC.item_id(i): i for i in C.read_jsonl(os.path.join(C.OUT_DIR, 'b_items.jsonl.gz'))
             if i['role'] in ('primary', 'pilot')}
    prose = {(int(r['task_id']), int(r['sample'])): r['text'] for r in C.read_jsonl(os.path.join(C.OUT_DIR, 'pool_samples.jsonl.gz'))}
    tok = C.tokenizer()
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(GRAPH_DIR, 'run_*.json'))]
    want_sha = MC.sha256(os.path.join(HERE, 'p9_mech_graphs.py'))
    env_ok = bool(runs) and all('A100' in r['env']['gpu'] and r['env']['script_sha256'] == want_sha for r in runs)
    if not REHEARSAL and not env_ok:
        sys.exit('graph outputs: not all from an A100 with the committed p9_mech_graphs.py')
    patch = {}
    ppath = os.path.join(MC.WORK if REHEARSAL else MC.OUT, f"patch_{'pilot' if REHEARSAL else 'primary'}.jsonl.gz")
    if os.path.exists(ppath):
        patch = {r['id']: r for r in C.read_jsonl(ppath)}
    rows = []
    for f in sorted(glob.glob(os.path.join(GRAPH_DIR, 'graph', '*.json'))):
        g = json.load(open(f))
        x = json.load(open(f.replace(os.sep + 'graph' + os.sep, os.sep + 'gxa' + os.sep)))
        sp, it = spans[g['id']], items[g['id']]
        if not g['tokens_equal'] or g['n_pos'] != sp['jq0'] or x['n_pos'] != sp['jq0']:
            sys.exit(f"{g['id']}: inputs differ from the spans file")
        tot = np.array(g['infl_feat']) + np.array(g['infl_err']) + np.array(g['infl_tok'])
        gx = np.abs(np.array(x['gxa_by_layer'][0]))
        n = sp['jq0']
        share = lambda v, idx: float(v[idx].sum() / v[1:].sum()) if len(idx) else None
        base = len(prose[tuple(sp['key'])]) + 1
        offs = tok(prose[tuple(sp['key'])] + ' ' + it['V'], return_offsets_mapping=True)['offset_mapping']
        cover = lambda a, b: [i for i in range(1, n) if offs[i][0] < base + b and offs[i][1] > base + a]
        D = [i for i in sp['D_V'] if i < n]
        xm = sorted({i for a, b in it['x_spans'] if b <= it['q_place'] for i in cover(a, b)} - set(D))
        other = sorted({i for a, b in let_names(it['V'], it['q_place'], it['x']['name']) for i in cover(a, b)})
        rec = {'id': g['id'], 'problem': sp['problem'], 'n_pos': n,
               'graph_D': share(tot, D), 'gxa_D': share(gx, D),
               'graph_D_per_token': share(tot, D) / len(D), 'gxa_D_per_token': share(gx, D) / len(D),
               'graph_x_other_per_token': share(tot, xm) / len(xm) if xm else None,
               'gxa_x_other_per_token': share(gx, xm) / len(xm) if xm else None,
               'graph_other_lets_per_token': share(tot, other) / len(other) if other else None,
               'gxa_other_lets_per_token': share(gx, other) / len(other) if other else None,
               'graph_last': share(tot, [n - 1]), 'gxa_last': share(gx, [n - 1]),
               'd_D_patch': patch.get(g['id'], {}).get('d_D'), 'features49_ok': 'features49' in g,
               'attribute_s': g['attribute_s']}
        rows.append(rec)
    out = {'items': len(rows), 'env_ok': env_ok, 'runs': [r['env'] for r in runs]}
    per = {r['problem']: r['graph_D'] - r['gxa_D'] for r in rows}
    out['G1'] = MC.test_greater(per, 11)
    out['G1']['rejected'] = bool(out['G1'].get('p_one_sided_greater', 1) < ALPHA)
    desc = {}
    # §4: the specificity row G1 is read with: per token, D minus x's other mentions, per method and between them
    sx = [r for r in rows if r['graph_x_other_per_token'] is not None]
    spec_g = {r['problem']: r['graph_D_per_token'] - r['graph_x_other_per_token'] for r in sx}
    spec_x = {r['problem']: r['gxa_D_per_token'] - r['gxa_x_other_per_token'] for r in sx}
    desc['specificity_graph'] = MC.test_greater(spec_g, 15)
    desc['specificity_gxa'] = MC.test_greater(spec_x, 17)
    desc['specificity_graph_minus_gxa'] = MC.test_greater({p: spec_g[p] - spec_x[p] for p in spec_g}, 19)
    for k in ('graph_D', 'gxa_D', 'graph_D_per_token', 'gxa_D_per_token', 'graph_x_other_per_token', 'gxa_x_other_per_token',
              'graph_other_lets_per_token', 'gxa_other_lets_per_token', 'graph_last', 'gxa_last', 'attribute_s'):
        v = [r[k] for r in rows if r[k] is not None]
        desc[k] = {'n': len(v), 'mean': round(float(np.mean(v)), 5) if v else None,
                   'median': round(float(np.median(v)), 5) if v else None}
    from scipy.stats import spearmanr
    pr = [r for r in rows if r['d_D_patch'] is not None]
    if len(pr) > 3:
        a = np.array([r['graph_D'] for r in pr]); b = np.array([r['gxa_D'] for r in pr]); d = np.array([r['d_D_patch'] for r in pr])
        rho_g, rho_x = spearmanr(a, d)[0], spearmanr(b, d)[0]
        rb = MC.rng(13)
        diffs = []
        for _ in range(2000):
            i = rb.integers(len(pr), size=len(pr))
            diffs.append(spearmanr(a[i], d[i])[0] - spearmanr(b[i], d[i])[0])
        desc['spearman_with_patch'] = {'n': len(pr), 'graph': round(float(rho_g), 4), 'gxa': round(float(rho_x), 4),
                                       'graph_minus_gxa_ci95': [round(float(np.nanpercentile(diffs, 2.5)), 4),
                                                                round(float(np.nanpercentile(diffs, 97.5)), 4)]}
    desc['features49_ok'] = sum(r['features49_ok'] for r in rows)
    out['descriptive'] = desc
    out['rows'] = rows
    return out


def main():
    out = {'part': PART, 'rehearsal': REHEARSAL}
    if not REHEARSAL:
        out['guard'] = MC.guard()
    out.update(part_p() if PART == 'P' else part_g())
    dest = os.path.join(MC.WORK, 'rehearsal') if REHEARSAL else MC.OUT
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, f'analysis_{PART}.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    show = {k: v for k, v in out.items() if k not in ('rows', 'runs', 'patch_meta', 'guard')}
    print(json.dumps(show, indent=1)[:6000])


if __name__ == '__main__':
    main()
