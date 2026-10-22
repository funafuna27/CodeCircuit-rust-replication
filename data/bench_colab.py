"""Time attribute() on a few fixed prompts with the study's call, on any GPU (Colab or the local 5070).

The call is script 3's, unchanged: BATCH_SIZE=64, max_feature_nodes=8192, max_n_logits=10,
desired_logit_prob=0.95, offload='cpu', the allocator's GC threshold 0.8, empty_cache before each graph.
One short untimed warm-up graph first. Per graph: attribute() seconds, to_pt seconds, peak VRAM, the
graph's sizes. Writes <OUT_DIR>/bench_<gpu>.json, and the graphs as .pt when SAVE=1 (for
bench_compare.py against the retained graphs).

  PROMPTS=bench_prompts.json  OUT_DIR=out  SAVE=1  FORCE=0 (refuse a GPU whose name lacks REQUIRE_GPU)
  REQUIRE_GPU=A100 (empty: any)
"""
import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8')

import importlib.metadata as md  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

PROMPTS = os.environ.get('PROMPTS', 'bench_prompts.json')
OUT_DIR = Path(os.environ.get('OUT_DIR', 'out'))
SAVE = os.environ.get('SAVE', '1') == '1'
REQUIRE_GPU = os.environ.get('REQUIRE_GPU', 'A100')
FORCE = os.environ.get('FORCE', '0') == '1'
CALL = dict(max_n_logits=10, desired_logit_prob=0.95, batch_size=64, max_feature_nodes=8192,
            offload='cpu', verbose=False)


def versions():
    out = {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda}
    for p in ('transformers', 'transformer-lens', 'huggingface_hub', 'einops', 'safetensors', 'tokenizers', 'numpy'):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def main():
    gpu = torch.cuda.get_device_name(0)
    if REQUIRE_GPU and REQUIRE_GPU not in gpu and not FORCE:
        sys.exit(f'GPU is {gpu!r}, not {REQUIRE_GPU}: stop here to save compute units (FORCE=1 runs anyway)')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from circuit_tracer import ReplacementModel, attribute
    prompts = json.load(open(PROMPTS))
    t0 = time.time()
    model = ReplacementModel.from_pretrained('google/gemma-2-2b-it', 'gemma', dtype=torch.bfloat16).to('cuda')
    load_s = time.time() - t0
    torch.cuda.empty_cache()
    t0 = time.time()
    attribute(prompt='fn add(a: i64, b: i64) -> i64 { a + b }', model=model, **CALL)
    warm_s = time.time() - t0
    rows = []
    for p in prompts:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        g = attribute(prompt=p['text'], model=model, **CALL)
        torch.cuda.synchronize()
        attr_s = time.time() - t0
        save_s = None
        if SAVE:
            t1 = time.time()
            g.to_pt(OUT_DIR / p['graph'])
            save_s = time.time() - t1
        rows.append({'row': p['row'], 'graph': p['graph'], 'tokens': p['tokens'], 'attribute_s': round(attr_s, 1),
                     'to_pt_s': round(save_s, 1) if save_s is not None else None,
                     'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                     'active_features': int(g.active_features.shape[0]), 'selected_features': int(len(g.selected_features)),
                     'n_logits': int(len(g.logit_tokens))})
        print(json.dumps(rows[-1]), flush=True)
        del g
    out = {'gpu': gpu, 'capability': list(torch.cuda.get_device_capability(0)),
           'vram_gib': round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1),
           'cpu_count': os.cpu_count(), 'versions': versions(), 'call': CALL,
           'alloc_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'), 'model_load_s': round(load_s, 1),
           'warmup_s': round(warm_s, 1), 'graphs': rows,
           'attribute_s_total': round(sum(r['attribute_s'] for r in rows), 1)}
    tag = gpu.replace(' ', '_').replace('/', '_')
    with open(OUT_DIR / f'bench_{tag}.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'graphs'}, indent=1))


if __name__ == '__main__':
    main()
