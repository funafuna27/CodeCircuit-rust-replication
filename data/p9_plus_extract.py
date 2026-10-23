"""Phase 9, part ②: one forward pass per program (PREREGISTRATION_P9_PLUS.md §3).

The call of part ③ (p9_m1_extract.forward_one: the ReplacementModel loaded as script 3 loads it, then
model.setup_attribution on BOS + the text's tokens; installed circuit-tracer 0.5.3.dev1+g6018ed8d3; local RTX
5070), on text = the task's prose + ' ' + the program's code. Kept per program, as part ③ keeps them: token
ids and offsets; log p and entropy per token; the final position's top-1, top-2 and entropy; f0, f3, f4; the
posmean over all positions and over code positions (CSR). No label is read.

  PLUS_WORK=~/p9_work/plus   TIMING=n (the n longest texts, timing only; no guard, nothing saved)
Refuses unless the pre-registration and the scripts are committed and clean (TIMING excepted).
"""
import json
import os
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8')   # script 3's

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_plus_common as PC  # noqa: E402

TIMING = int(os.environ.get('TIMING', 0))
D_SAE = 16384


def save(keys, recs, meta):
    def csr(field):
        ind = [r[field][0] for r in recs]
        dat = [r[field][1] for r in recs]
        return {f'{field}_indices': np.concatenate(ind), f'{field}_data': np.concatenate(dat),
                f'{field}_indptr': np.r_[0, np.cumsum([len(i) for i in ind])]}
    flat = {f: np.concatenate([r[f] for r in recs]) for f in ('ids', 'off', 'logp', 'ent')}
    tptr = np.r_[0, np.cumsum([len(r['ids']) for r in recs])]
    sal = np.full((len(recs), 10), -1, np.int32)
    salp = np.zeros((len(recs), 10), np.float32)
    for i, r in enumerate(recs):
        sal[i, :len(r['sal_ids'])] = r['sal_ids']
        salp[i, :len(r['sal_p'])] = r['sal_p']
    os.makedirs(PC.WORK, exist_ok=True)
    path = os.path.join(PC.WORK, 'extract_plus.npz')
    np.savez(path + '.tmp.npz', keys=np.array(keys), tptr=tptr, **flat,
             final=np.array([r['final'] for r in recs]), f=np.array([r['f'] for r in recs]),
             n_code=np.array([r['n_code'] for r in recs]), n_prose=np.array([r['n_prose'] for r in recs]),
             sal_ids=sal, sal_p=salp, shape=np.array([len(recs), 26 * D_SAE]), **csr('pm'), **csr('pm_code'))
    os.replace(path + '.tmp.npz', path)
    meta['sha256'] = PC.sha256(path)
    with open(path.replace('.npz', '.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    return path


def main():
    import torch
    import p9_m1_common as M
    import p9_m1_extract as X
    guard = None if TIMING else PC.guard()
    items = [{'id': r['id'], 'text': r['text'], 'prose': r['prose']} for r in PC.population()]
    if TIMING:
        items = sorted(items, key=lambda r: (-len(r['text']), r['id']))[:TIMING]
    tok = M.tokenizer()
    model = X.load_model()
    recs, secs = [], []
    for n, it in enumerate(items, 1):
        torch.cuda.synchronize()
        t0 = time.time()
        r = X.forward_one(model, tok, it['text'], it['prose'])
        torch.cuda.synchronize()
        secs.append(time.time() - t0)
        for k in ('lay', 'pos', 'fid', 'vals'):
            del r[k]
        if r['n_code'] < 1:
            sys.exit(f"{it['id']}: no code token")
        recs.append(r)
        if n % 100 == 0 or n == len(items):
            print(f'  plus {n}/{len(items)}  median {np.median(secs):.3f} s per text', flush=True)
    meta = {'texts': len(items), 'seconds_total': round(float(np.sum(secs)), 1),
            's_median': round(float(np.median(secs)), 3), 'max_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'library': X.library(), 'guard': guard}
    if TIMING:
        print(json.dumps(meta))
        return
    meta['path'] = save([it['id'] for it in items], recs, meta)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
