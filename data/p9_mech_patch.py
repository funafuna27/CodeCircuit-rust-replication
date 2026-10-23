"""Phase 9, part ① at gate 3: activation patching from A into V (PREREGISTRATION_P9_MECH.md §3).

The ReplacementModel's TransformerLens model loaded as script 3 loads it (circuit-tracer 0.5.3.dev1+g6018ed8d3
from site-packages, the local RTX 5070), run as a plain model, batch 1. Per item (spans.jsonl.gz):
  A        run once; blocks.{l}.hook_resid_pre cached at every layer
  V        unpatched
  V|D      D's positions replaced at every layer by A's cached values at D's A positions; likewise V|K, V|W
  sweep    (G items only) D replaced from layer l on (a hook at layer l only), l = 0..25
Readouts (float32 log-softmax): the summed log p of the span tokens; each read's log p; the mean log p of the
background tokens. d = patched - unpatched.

  ROLE=primary (refuses unless the pre-registration and the scripts are committed and clean) | pilot
  (rehearsal: no guard, writes to MECH_WORK)
"""
import gzip
import json
import os
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8')

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_common as C  # noqa: E402
import p9_mech_common as MC  # noqa: E402

ROLE = os.environ.get('ROLE', 'pilot')


def main():
    import torch
    import p9_m1_extract as X
    guard = MC.guard() if ROLE == 'primary' else None
    rows = [r for r in C.read_jsonl(MC.SPANS) if r['role'] == ROLE]
    model = X.load_model()
    lib = X.library()
    names = [f'blocks.{l}.hook_resid_pre' for l in range(MC.N_LAYERS)]

    def lp_of(logits, ids, idx):
        logq = torch.log_softmax(logits[0].float(), dim=-1)
        return [float(logq[i - 1, ids[i]]) for i in idx]

    def patched(idsV_t, posV, posA, cache, layers):
        def mk(name):
            def hook(resid, hook):
                resid[:, posV, :] = cache[name][:, posA, :].to(resid.dtype)
                return resid
            return hook
        return model.run_with_hooks(idsV_t, fwd_hooks=[(names[l], mk(names[l])) for l in layers])

    out, secs = [], []
    with torch.inference_mode():
        for n, r in enumerate(rows, 1):
            t0 = time.time()
            idsV, idsA = r['idsV'], r['idsA']
            tV = torch.tensor(idsV, device='cuda')[None]
            tA = torch.tensor(idsA, device='cuda')[None]
            logitsA, cacheA = model.run_with_cache(tA, names_filter=lambda nm: nm in names)
            cache = {nm: cacheA[nm] for nm in names}
            del cacheA
            logitsV = model(tV)

            def read(logits):
                sp = lp_of(logits, idsV, r['span_V'])
                rd = lp_of(logits, idsV, r['reads_V']) if r['reads_V'] else []
                bg = lp_of(logits, idsV, r['bg_V']) if r['bg_V'] else []
                return float(np.sum(sp)), rd, (float(np.mean(bg)) if bg else None)
            base = read(logitsV)
            rec = {'id': r['id'], 'problem': r['problem'], 'source': r['source'], 'x': r['x'], 'site': r['site'],
                   'g_item': r['g_item'], 'n_span': len(r['span_V']), 'n_reads': len(r['reads_V']),
                   'lpV_span': base[0], 'lpA_span': float(np.sum(lp_of(logitsA, idsA, r['span_A'])))}
            all_layers = list(range(MC.N_LAYERS))
            for tag in ('D', 'K', 'W'):
                res = read(patched(tV, r[f'{tag}_V'], r[f'{tag}_A'], cache, all_layers))
                rec[f'd_{tag}'] = res[0] - base[0]
                if tag == 'D':
                    rec['d_R'] = [a - b for a, b in zip(res[1], base[1])]
                    rec['d_bg'] = (res[2] - base[2]) if base[2] is not None else None
            if r['g_item']:
                rec['sweep'] = [read(patched(tV, r['D_V'], r['D_A'], cache, [l]))[0] - base[0] for l in all_layers]
            del cache, logitsA, logitsV
            out.append(rec)
            secs.append(time.time() - t0)
            if n % 250 == 0 or n == len(rows):
                print(f'  {ROLE} {n}/{len(rows)}  median {np.median(secs):.3f} s per item', flush=True)
    dest = MC.OUT if ROLE == 'primary' else MC.WORK
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, f'patch_{ROLE}.jsonl.gz')
    with gzip.open(path, 'wt') as fh:
        for rec in out:
            fh.write(json.dumps(rec) + '\n')
    meta = {'role': ROLE, 'items': len(out), 'seconds_total': round(float(np.sum(secs)), 1),
            's_median': round(float(np.median(secs)), 3), 'max_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'library': lib, 'guard': guard,
            'sha256': MC.sha256(path)}
    with open(path.replace('.jsonl.gz', '.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
