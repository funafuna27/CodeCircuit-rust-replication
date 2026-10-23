"""Phase 9, part ① at gate 3: graphs and GxA of the mutation's first token (PREREGISTRATION_P9_MECH.md §4).

Runs on Colab (A100-SXM4-80GB; refuses another GPU unless FORCE=1, for a local test only). Inputs: ITEMS, a
JSON list of {id, ids (BOS .. the token before q_place), target (q_place's token id)}. Each item's results
are written to OUT_DIR as soon as they exist, and items already there are skipped (resume).

  gxa/<id>.json    one backward pass of the target's demeaned logit (logit - mean logit) through a plain
                   TransformerLens model (from_pretrained_no_processing: no circuit-tracer hooks, so the true
                   gradient); per layer l and position p, sum_d grad * act on blocks.{l}.hook_resid_pre
  graph/<id>.json  attribute(ids, model, attribution_targets=[target], batch_size=64, max_feature_nodes=8192,
                   offload='cpu'), as script 3 calls it; the graph is not saved. Kept: each node's influence on
                   the target (compute_node_influence) summed by position for features, errors and tokens; the
                   target's probability; the active and selected counts; the 49 features of graph_dataset.py
                   (script 4's featuriser, with the igraph override)
  run_<gpu>.json   the environment and this file's sha256

  ITEMS=mech_items.json  OUT_DIR=out  REQUIRE_GPU=A100  FORCE=0  STEP=gxa|graph|all
"""
import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8')

import hashlib  # noqa: E402
import importlib.metadata as md  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ITEMS = os.environ.get('ITEMS', 'mech_items.json')
OUT_DIR = Path(os.environ.get('OUT_DIR', 'out'))
REQUIRE_GPU = os.environ.get('REQUIRE_GPU', 'A100')
FORCE = os.environ.get('FORCE', '0') == '1'
STEP = os.environ.get('STEP', 'all')
LIMIT = int(os.environ.get('LIMIT', 0))
N_LAYERS = 26


def env():
    out = {'gpu': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
           'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'script_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}
    for p in ('circuit-tracer', 'transformers', 'transformer-lens', 'huggingface_hub', 'numpy', 'igraph'):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def gxa_step(items):
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained_no_processing('google/gemma-2-2b-it', dtype=torch.bfloat16).to('cuda')
    model.eval()
    for p in model.parameters():                 # gradients are needed for activations only
        p.requires_grad_(False)
    names = [f'blocks.{l}.hook_resid_pre' for l in range(N_LAYERS)]
    for it in items:
        path = OUT_DIR / 'gxa' / f"{it['id']}.json"
        if path.exists():
            continue
        t0 = time.time()
        acts = {}

        def keep(t, hook):
            if hook.name == names[0]:
                t = t.detach().requires_grad_(True)   # the root of the backward pass (input x gradient)
            else:
                t.retain_grad()
            acts[hook.name] = t
            return t
        ids = torch.tensor(it['ids'], device='cuda')[None]
        logits = model.run_with_hooks(ids, fwd_hooks=[(n, keep) for n in names])
        last = logits[0, -1].float()
        target = last[it['target']] - last.mean()
        target.backward()
        gxa = [(acts[n].grad[0].float() * acts[n][0].float()).sum(-1).tolist() for n in names]
        prob = float(torch.softmax(last, -1)[it['target']])
        write(path, {'id': it['id'], 'n_pos': len(it['ids']), 'gxa_by_layer': gxa, 'target_logit_demeaned': float(target),
                     'target_prob': prob, 'seconds': round(time.time() - t0, 2)})
        del logits, acts
    del model
    torch.cuda.empty_cache()


def graph_step(items):
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.graph import compute_node_influence
    sys.path.insert(0, str(Path(__file__).parent))
    import graph_dataset   # noqa: F401  (applies the igraph override at import, as script 4 runs)
    feat = graph_dataset.GraphDataset([], feature_type='advanced_graph_features', use_cpu=True)
    model = ReplacementModel.from_pretrained('google/gemma-2-2b-it', 'gemma', dtype=torch.bfloat16).to('cuda')
    attribute(prompt='fn add(a: i64, b: i64) -> i64 { a + b }', model=model, batch_size=64, max_feature_nodes=8192,
              offload='cpu', verbose=False)                      # untimed warm-up, as the bench
    for it in items:
        path = OUT_DIR / 'graph' / f"{it['id']}.json"
        if path.exists():
            continue
        torch.cuda.empty_cache()
        t0 = time.time()
        g = attribute(prompt=it['ids'], model=model, attribution_targets=torch.tensor([it['target']]),
                      batch_size=64, max_feature_nodes=8192, offload='cpu', verbose=False)
        attr_s = time.time() - t0
        A = g.adjacency_matrix.float().cpu()
        n_pos = len(g.input_tokens)
        sel = g.selected_features.cpu().numpy()
        n_sel = len(sel)
        n_log = len(g.logit_targets)
        N = A.shape[0]
        if N != n_sel + (N_LAYERS + 1) * n_pos + n_log:
            raise SystemExit(f"{it['id']}: node count {N}")
        lw = torch.zeros(N)
        lw[N - n_log:] = g.logit_probabilities.float().cpu()
        infl = compute_node_influence(A, lw).double().numpy()
        pos_f = g.active_features[sel, 1].cpu().numpy() if n_sel else np.zeros(0, int)
        by_feat = np.bincount(pos_f, weights=infl[:n_sel], minlength=n_pos)
        e0, t0n = n_sel, n_sel + N_LAYERS * n_pos
        by_err = infl[e0:t0n].reshape(N_LAYERS, n_pos).sum(0)
        by_tok = infl[t0n:t0n + n_pos]
        rec = {'id': it['id'], 'n_pos': n_pos, 'tokens_equal': g.input_tokens.tolist() == it['ids'],
               'target_prob': float(g.logit_probabilities[0]), 'active': int(g.active_features.shape[0]),
               'selected': n_sel, 'infl_feat': by_feat.tolist(), 'infl_err': by_err.tolist(), 'infl_tok': by_tok.tolist(),
               'attribute_s': round(attr_s, 1)}
        try:
            rec['features49'] = feat._extract_advanced_features(g).tolist()
        except Exception as exc:                                  # recorded, not fatal
            rec['features49_error'] = f'{type(exc).__name__}: {exc}'
        write(path, rec)
        print(json.dumps({k: rec[k] for k in ('id', 'n_pos', 'attribute_s', 'target_prob', 'selected')}), flush=True)
        del g, A


def main():
    gpu = torch.cuda.get_device_name(0)
    if REQUIRE_GPU and REQUIRE_GPU not in gpu and not FORCE:
        sys.exit(f'GPU is {gpu!r}, not {REQUIRE_GPU}: stop here to save compute units')
    items = json.load(open(ITEMS))
    if LIMIT:
        items = items[:LIMIT]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    e = env()
    write(OUT_DIR / f"run_{gpu.replace(' ', '_')}_{int(time.time())}.json", {'env': e, 'items': len(items), 'step': STEP})
    if STEP in ('gxa', 'all'):
        gxa_step(items)
    if STEP in ('graph', 'all'):
        graph_step(items)
    print('done', len(list((OUT_DIR / 'gxa').glob('*.json'))), 'gxa,', len(list((OUT_DIR / 'graph').glob('*.json'))), 'graphs')


if __name__ == '__main__':
    main()
