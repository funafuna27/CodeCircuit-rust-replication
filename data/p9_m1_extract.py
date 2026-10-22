"""Phase 9, part ③ (M1): one forward pass per text, forward only (PREREGISTRATION_P9_M1.md §3).

The ReplacementModel loaded as script 3 loads it, and model.setup_attribution(input_ids): the call
attribute() makes first, whose activations Phase 8 found bitwise equal to the stored graphs'. Per text:
  posmean       every active transcoder feature summed over positions (layer x 16384 + feature, float64
                from the stored bf16 values, np.unique/np.bincount in the coalesced order), divided by the
                number of positions including BOS, float32: p7_0_extract.py lines 137-150
  posmean_code  the same over code positions only (tokens starting at or after the space before the
                code), divided by their number
  f0, f3, f4    as Phase 7 read them from graphs: the active-feature count; the salient logit set of
                AttributionTargets(None, logits[0, -1], W_U, tokenizer, 10, 0.95) on the GPU, its
                probabilities moved to the CPU in bf16, f3 = the first, f4 = -sum p log(p + 1e-8)
  logp, ent     per token: log p of the token given its prefix and the entropy of that distribution
                (float32 log-softmax of the model's logits)
  final         at the last position: top-1 and top-2 probability and the entropy (float32)
No logits are written.

  ROLE=matched|rust|python   TIMING=n (the n longest texts, timing only)   M1_WORK=~/p9_work/m1
The rust and python roles refuse unless the pre-registration and the M1 scripts are committed and clean
and gate X1 passed (results/p9_m1/gates_x1.json).
"""
import json
import os
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8')   # script 3's

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_m1_common as M  # noqa: E402

ROLE = os.environ.get('ROLE', 'matched')
TIMING = int(os.environ.get('TIMING', 0))
SAVE_TIMING = os.environ.get('SAVE_TIMING', '0') == '1'     # smoke test only: keep a TIMING run's records
D_SAE = 16384


def texts_for(role):
    if role == 'matched':
        rows = M.read_jsonl(os.path.join(HERE, 'matched_rows.jsonl'))
        return [{'key': int(r['task_id']), 'text': r['text'], 'prose': r['prose']} for r in rows]
    return [{'key': r['row_index'], 'text': r['text'], 'prose': r['prose']} for r in M.load_arm(role)]


def library():
    """The circuit-tracer that built every graph: the installed 0.5.3.dev1+g6018ed8d3, not the repo's copy."""
    import importlib.metadata
    import circuit_tracer
    loc = os.path.dirname(circuit_tracer.__file__)
    ver = importlib.metadata.version('circuit-tracer')
    if 'site-packages' not in loc or ver != '0.5.3.dev1+g6018ed8d3':
        sys.exit(f'circuit_tracer from {loc} ({ver}): not the installed package that built the graphs')
    return {'circuit_tracer': loc, 'version': ver}


def load_model():
    import torch
    library()
    from circuit_tracer import ReplacementModel
    model = ReplacementModel.from_pretrained('google/gemma-2-2b-it', 'gemma', dtype=torch.bfloat16).to('cuda')
    return model


def posmean(lay, fid, val64, n):
    key = lay.astype(np.int64) * D_SAE + fid
    uk, inv = np.unique(key, return_inverse=True)
    return uk.astype(np.int32), (np.bincount(inv, weights=val64) / n).astype(np.float32)


def forward_one(model, tok, text, prose):
    """Everything §3 needs from one text. Returns (record, ctx-free)."""
    import torch
    from circuit_tracer.attribution.targets import AttributionTargets
    enc = tok(text, return_offsets_mapping=True)
    ids = enc['input_ids']
    ct = model.ensure_tokenized(text).tolist()
    if ct != ids:
        raise SystemExit(f'tokenisation differs from circuit-tracer\'s: {text[:60]!r}')
    input_ids = torch.tensor(ids, dtype=torch.long, device='cuda')
    with torch.no_grad():
        ctx = model.setup_attribution(input_ids)
        am = ctx.activation_matrix
        lay, pos, fid = (a.cpu().numpy() for a in am.indices())
        vals = am.values().cpu()
        f0 = float(am._nnz())
        targets = AttributionTargets(None, ctx.logits[0, -1], model.unembed.W_U, model.tokenizer,
                                     max_n_logits=10, desired_logit_prob=0.95)
        lp = targets.logit_probabilities.cpu()
        f3 = lp[0].item()
        f4 = (-torch.sum(lp * torch.log(lp + 1e-8))).item()
        sal_ids = [t.vocab_idx for t in targets.logit_targets]
        logq = torch.log_softmax(ctx.logits[0].float(), dim=-1)
        ent = -(logq.exp() * logq).sum(-1)
        idt = input_ids[1:, None]
        logp = logq[:-1].gather(1, idt)[:, 0]
        top2 = logq[-1].exp().topk(2).values
        rec = {'ids': np.array(ids, np.int32), 'off': np.array(enc['offset_mapping'], np.int32),
               'logp': np.concatenate([[np.nan], logp.cpu().numpy()]).astype(np.float32),
               'ent': np.concatenate([[np.nan], ent[:-1].cpu().numpy()]).astype(np.float32),
               'final': np.array([top2[0].item(), top2[1].item(), ent[-1].item()], np.float32),
               'f': np.array([f0, f3, f4], np.float64), 'sal_ids': np.array(sal_ids, np.int32),
               'sal_p': lp.float().numpy()}
    del ctx, am, targets, logq, ent
    torch.cuda.empty_cache()
    n = len(ids)
    val64 = vals.float().numpy().astype(np.float64)
    rec['pm'] = posmean(lay, fid, val64, n)
    code_pos = np.array([o[0] >= len(prose) for o in enc['offset_mapping']])
    code_pos[0] = False
    m = code_pos[pos]
    n_code = int(code_pos.sum())
    rec['pm_code'] = posmean(lay[m], fid[m], val64[m], max(n_code, 1))
    rec['n_code'], rec['n_prose'] = n_code, int(n - 1 - n_code)
    rec['lay'], rec['pos'], rec['fid'], rec['vals'] = lay, pos, fid, vals   # for gate X2's 94 graphs
    return rec


def save(role, keys, recs, meta):
    def csr(field):
        ind = [r[field][0] for r in recs]
        dat = [r[field][1] for r in recs]
        return {f'{field}_indices': np.concatenate(ind), f'{field}_data': np.concatenate(dat),
                f'{field}_indptr': np.r_[0, np.cumsum([len(i) for i in ind])]}
    flat = {}
    for f in ('ids', 'off', 'logp', 'ent'):
        flat[f] = np.concatenate([r[f] for r in recs])
    tptr = np.r_[0, np.cumsum([len(r['ids']) for r in recs])]
    sal = np.full((len(recs), 10), -1, np.int32)
    salp = np.zeros((len(recs), 10), np.float32)
    for i, r in enumerate(recs):
        sal[i, :len(r['sal_ids'])] = r['sal_ids']
        salp[i, :len(r['sal_p'])] = r['sal_p']
    os.makedirs(M.WORK, exist_ok=True)
    path = os.path.join(M.WORK, f'extract_{role}.npz')
    np.savez(path + '.tmp.npz', keys=np.array(keys), tptr=tptr, **flat,
             final=np.array([r['final'] for r in recs]), f=np.array([r['f'] for r in recs]),
             n_code=np.array([r['n_code'] for r in recs]), n_prose=np.array([r['n_prose'] for r in recs]),
             sal_ids=sal, sal_p=salp, shape=np.array([len(recs), 26 * D_SAE]), **csr('pm'), **csr('pm_code'))
    os.replace(path + '.tmp.npz', path)
    meta['sha256'] = M.sha256(path)
    with open(path.replace('.npz', '.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    return path


def main():
    import torch
    guard = None
    if not TIMING:
        guard = M.guard()
        if ROLE != 'matched':
            g = os.path.join(M.OUT, 'gates_x1.json')
            if not os.path.exists(g) or not json.load(open(g)).get('extractor_ok'):
                sys.exit('refused: gate X1 has not passed')
    items = texts_for(ROLE)
    tok = M.tokenizer()
    if TIMING:
        items = sorted(items, key=lambda r: (-len(r['text']), r['key']))[:TIMING]
    model = load_model()
    recs, secs = [], []
    for n, it in enumerate(items, 1):
        torch.cuda.synchronize()
        t0 = time.time()
        r = forward_one(model, tok, it['text'], it['prose'])
        torch.cuda.synchronize()
        secs.append(time.time() - t0)
        for k in ('lay', 'pos', 'fid', 'vals'):
            del r[k]
        recs.append(r)
        if n % 250 == 0 or n == len(items):
            print(f'  {ROLE} {n}/{len(items)}  median {np.median(secs):.3f} s per text', flush=True)
    meta = {'role': ROLE, 'texts': len(items), 'seconds_total': round(float(np.sum(secs)), 1),
            's_median': round(float(np.median(secs)), 3), 'max_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'library': library(), 'guard': guard}
    if TIMING and not SAVE_TIMING:
        print(json.dumps(meta))
        return
    if TIMING and (ROLE != 'matched' or 'p9_work/m1' in os.path.abspath(M.WORK)):
        sys.exit('SAVE_TIMING is for Phase 5 texts in a scratch M1_WORK only')
    meta['path'] = save(ROLE, [it['key'] for it in items], recs, meta)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
