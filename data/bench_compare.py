"""Compare two attribution graphs of the same prompt (.pt from Graph.to_pt), e.g. a retained graph against
one rebuilt on the same or another GPU.

Reports: input tokens equal; active features (layer, position, feature) shared, and among the shared ones
the share with bit-identical activation; selected features shared (Jaccard); logit tokens equal and the
largest logit-probability gap; on the nodes both graphs hold (selected features matched by key, error
nodes by (layer, position), tokens by position, logits by token id), the share of nonzero edges that are
not bit-identical (Phase 7's measure on the shared prefix), the relative Frobenius gap and the edge
correlation; and the share of each graph's total logit influence (compute_node_influence) carried by
the selected features both hold.

  python bench_compare.py A.pt B.pt [label]   -> one JSON line
"""
import json
import sys

import numpy as np
import torch

N_LAYERS, D_SAE = 26, 1 << 20


def logit_ids(d):
    """circuit-tracer 0.5.3 stores 'logit_targets' (LogitTarget records, or a legacy id tensor)."""
    lt = d['logit_targets'] if 'logit_targets' in d else d['logit_tokens']
    if isinstance(lt, torch.Tensor):
        return lt.long()
    return torch.tensor([t.vocab_idx for t in lt], dtype=torch.long)


def load(path):
    d = torch.load(path, weights_only=False, map_location='cpu')
    d['logit_tokens'] = logit_ids(d)
    af = d['active_features'].numpy().astype(np.int64)
    sel = d['selected_features'].numpy()
    n_pos, n_sel, n_log = len(d['input_tokens']), len(sel), len(d['logit_tokens'])
    akey = (af[:, 0] * 1000 + af[:, 1]) * D_SAE + af[:, 2]
    keys = [('f', int(k)) for k in akey[sel]] + \
           [('e', l_, p) for l_ in range(N_LAYERS) for p in range(n_pos)] + \
           [('t', p) for p in range(n_pos)] + [('l', int(t)) for t in d['logit_tokens']]
    assert len(keys) == d['adjacency_matrix'].shape[0], 'node count'
    return d, akey, keys, n_sel


def influence_share(d, keys, shared, n_sel):
    from circuit_tracer.graph import compute_node_influence
    A = d['adjacency_matrix'].float()
    n_log = len(d['logit_tokens'])
    lw = torch.zeros(A.shape[0])
    lw[-n_log:] = d['logit_probabilities'].float()
    infl = compute_node_influence(A, lw).double().numpy()
    feats = infl[:n_sel]
    mask = np.array([keys[i] in shared for i in range(n_sel)])
    return float(feats[mask].sum() / feats.sum()) if feats.sum() > 0 else None


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else ''
    a, ak, kA, sA = load(pa)
    b, bk, kB, sB = load(pb)
    out = {'label': label, 'tokens_equal': bool(torch.equal(a['input_tokens'], b['input_tokens']))}
    common, ia, ib = np.intersect1d(ak, bk, return_indices=True)
    va, vb = a['activation_values'][ia], b['activation_values'][ib]
    out['active_a'], out['active_b'] = int(len(ak)), int(len(bk))
    out['active_shared'] = int(len(common))
    out['active_bit_identical_share'] = float((va.view(torch.int16) == vb.view(torch.int16)).float().mean()) if len(common) else None
    out['active_max_abs_diff'] = float((va.float() - vb.float()).abs().max()) if len(common) else None
    fa, fb = set(kA[:sA]), set(kB[:sB])
    out['selected_jaccard'] = len(fa & fb) / len(fa | fb)
    out['logit_tokens_equal'] = bool(torch.equal(a['logit_tokens'], b['logit_tokens']))
    if out['logit_tokens_equal']:
        out['logit_prob_max_abs_diff'] = float((a['logit_probabilities'].float() - b['logit_probabilities'].float()).abs().max())
    shared = set(kA) & set(kB)
    ka = [i for i, k in enumerate(kA) if k in shared]
    pos_b = {k: i for i, k in enumerate(kB)}
    kb = [pos_b[kA[i]] for i in ka]
    ta, tb = torch.tensor(ka), torch.tensor(kb)
    Ea = a['adjacency_matrix'][ta][:, ta]
    Eb = b['adjacency_matrix'][tb][:, tb]
    nz = (Ea != 0) | (Eb != 0)
    same = Ea.view(torch.int32) == Eb.view(torch.int32) if Ea.dtype == torch.float32 else Ea == Eb
    out['shared_nodes'] = len(ka)
    out['edges_nonzero'] = int(nz.sum())
    out['edge_mismatch_share'] = float((nz & ~same).sum() / nz.sum()) if nz.any() else None
    x, y = Ea[nz].double(), Eb[nz].double()
    out['edge_rel_frobenius_gap'] = float((x - y).norm() / x.norm()) if x.norm() > 0 else None
    out['edge_corr'] = float(torch.corrcoef(torch.stack([x, y]))[0, 1]) if len(x) > 1 else None
    out['influence_on_shared_features_a'] = influence_share(a, kA, fa & fb, sA)
    out['influence_on_shared_features_b'] = influence_share(b, kB, fa & fb, sB)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
