"""Phase 9 gate 3: post hoc covariates of the graph-vs-patching correlation (PREREGISTRATION_P9_MECH.md §13).

§12 reported, as a registered descriptive, the Spearman correlation of each method's share on D with the same
item's patching effect d_D (graph -0.33, GxA -0.07). Reviewers of the write-up (2026-10-26) found that most of
it goes with the mutation's log p in V (lpV): when V already gives the mutation a high probability, patching can
restore little and the graph puts more on D. This script records that, post hoc: partial Spearman correlations
(ranks residualised on the covariates' ranks by least squares) for several covariate sets, with 2,000 bootstrap
draws of the 167 items (one per problem; stream [20261023, 50]). lpV may be a mediator rather than a
confounder, in which case conditioning on it removes part of what is asked; both readings are reported.

  OUT=results/p9_mech/covariates.json
"""
import gzip
import hashlib
import json
import os

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results', 'p9_mech')
OUT = os.environ.get('OUT', os.path.join(RES, 'covariates.json'))
N_BOOT = 2000


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def partial(a, b, cov):
    ra, rb = rankdata(a), rankdata(b)
    if not cov:
        return float(np.corrcoef(ra, rb)[0, 1])
    Z = np.column_stack([np.ones(len(a))] + [rankdata(c) for c in cov])
    ea = ra - Z @ np.linalg.lstsq(Z, ra, rcond=None)[0]
    eb = rb - Z @ np.linalg.lstsq(Z, rb, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])


def main():
    gpath, ppath = os.path.join(RES, 'analysis_G.json'), os.path.join(RES, 'patch_primary.jsonl.gz')
    rows = json.load(open(gpath))['rows']
    with gzip.open(ppath, 'rt') as fh:
        patch = {r['id']: r for r in (json.loads(l) for l in fh)}
    ids = [r['id'] for r in rows]
    d = np.array([patch[i]['d_D'] for i in ids])
    assert np.allclose(d, [r['d_D_patch'] for r in rows]), 'analysis_G and the patch file disagree on d_D'
    v = {'graph': np.array([r['graph_D'] for r in rows]), 'gxa': np.array([r['gxa_D'] for r in rows])}
    cov = {'lpV': np.array([patch[i]['lpV_span'] for i in ids]),
           'gap': np.array([patch[i]['lpA_span'] - patch[i]['lpV_span'] for i in ids]),
           'n_pos': np.array([r['n_pos'] for r in rows], float),
           'graph_last': np.array([r['graph_last'] for r in rows]),
           'n_span': np.array([patch[i]['n_span'] for i in ids], float)}
    sets = {'none (registered)': [], 'lpV': ['lpV'], 'gap': ['gap'], 'lpV + gap': ['lpV', 'gap'],
            'n_pos + graph_last': ['n_pos', 'graph_last'], 'all five': list(cov)}
    B = np.random.default_rng([20261023, 50]).integers(0, len(d), size=(N_BOOT, len(d)))
    out = {'post_hoc': True, 'items': len(d), 'inputs_sha256': {'analysis_G.json': sha(gpath),
                                                                 'patch_primary.jsonl.gz': sha(ppath)},
           'partial_spearman_with_d_D': {}}
    for name, keys in sets.items():
        cs = [cov[k] for k in keys]
        pt = {m: partial(x, d, cs) for m, x in v.items()}
        bs = np.array([[partial(v[m][b], d[b], [c[b] for c in cs]) for m in ('graph', 'gxa')] for b in B])
        q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
        out['partial_spearman_with_d_D'][name] = {
            'graph': round(pt['graph'], 3), 'graph_ci95': q(bs[:, 0]), 'gxa': round(pt['gxa'], 3),
            'gxa_ci95': q(bs[:, 1]), 'graph_minus_gxa': round(pt['graph'] - pt['gxa'], 3),
            'graph_minus_gxa_ci95': q(bs[:, 0] - bs[:, 1])}
        print(name, out['partial_spearman_with_d_D'][name], flush=True)
    out['spearman'] = {'d_D_vs_lpV': round(partial(d, cov['lpV'], []), 3),
                       'graph_vs_lpV': round(partial(v['graph'], cov['lpV'], []), 3),
                       'gxa_vs_lpV': round(partial(v['gxa'], cov['lpV'], []), 3)}
    out['gap'] = {'median': round(float(np.median(cov['gap'])), 4), 'items_gap_le_0': int((cov['gap'] <= 0).sum())}
    print(json.dumps({k: out[k] for k in ('spearman', 'gap')}), flush=True)
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
