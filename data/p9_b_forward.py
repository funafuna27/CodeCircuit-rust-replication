"""Phase 9: teacher-forced log-probabilities and entropies for every text of a role.

As Phases 5-6 (p5_2_logprob): Gemma-2-2b-it in bf16, eager attention (Gemma-2's attention soft-capping
applies; the final-logit soft-cap is in the HF forward), one sequence per forward (batch 1: Phase 8
found batch-64 rows deviate), log_softmax in float32. The scored text is prose + ' ' + code, as every
phase graphed. Per token i: logp[i] = log p(token i | tokens < i) (logp[0] = nan), ent[i] = the entropy
(nats) of the distribution that predicted token i, off[i] = the token's (start, end) characters.

Roles: pilot, behaviour (allowed now); primary (refused unless a PREREGISTRATION_P9*.md is committed
and unchanged in the working tree). TIMING=n scores only the first n texts and reports seconds.

  ROLE=pilot  ITEMS=results/p9_state/b_items.jsonl.gz  WORK=/home/user1/p9_work/forward
"""
import glob
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
import p9_common as C  # noqa: E402

ROLE = os.environ.get('ROLE', 'pilot')
ITEMS = os.environ.get('ITEMS', os.path.join(C.OUT_DIR, 'b_items.jsonl.gz'))
WORK = os.environ.get('WORK', '/home/user1/p9_work/forward')
TIMING = int(os.environ.get('TIMING', 0))


def prereg_guard():
    repo = os.path.dirname(HERE)
    files = sorted(glob.glob(os.path.join(repo, 'PREREGISTRATION_P9*.md')))
    if not files:
        sys.exit('primary role refused: no PREREGISTRATION_P9*.md')
    for f in files:
        rel = os.path.relpath(f, repo)
        logged = subprocess.run(['git', '-C', repo, 'log', '-1', '--format=%H', '--', rel],
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', repo, 'status', '--porcelain', '--', rel],
                               capture_output=True, text=True).stdout.strip()
        if not logged or dirty:
            sys.exit(f'primary role refused: {rel} is not committed or has uncommitted changes')
    return {os.path.basename(f): hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16] for f in files}


def texts_for(items, prose):
    """Distinct texts of the role, keyed by sha1 of the full text."""
    out = {}
    for it in items:
        p = prose[tuple(it['key'])]
        codes = [it['S'], it['M']] if it['role'] == 'behaviour' else \
            [it['S'], it['V'], it['A'], it['R']] + [z['text'] for z in it['placebos']]
        for code in codes:
            t = p + ' ' + code
            out[hashlib.sha1(t.encode()).hexdigest()] = t
    return out


def main():
    guard = prereg_guard() if ROLE == 'primary' else None
    items = [it for it in C.read_jsonl(ITEMS) if it['role'] == ROLE]
    prose = {(int(r['task_id']), int(r['sample'])): r['text'] for r in C.read_jsonl(os.path.join(C.OUT_DIR, 'pool_samples.jsonl.gz'))}
    texts = texts_for(items, prose)
    keys = sorted(texts)
    os.makedirs(WORK, exist_ok=True)
    out_path = os.path.join(WORK, f'{ROLE}{"_timing" if TIMING else ""}.npz')
    if TIMING:
        rng = np.random.default_rng([20261022, 5])
        keys = [keys[i] for i in sorted(rng.choice(len(keys), min(TIMING, len(keys)), replace=False))]
    print(f'role {ROLE}: {len(items)} items, {len(keys)} distinct texts' + (f'; guard {guard}' if guard else ''), flush=True)

    import torch
    from transformers import AutoModelForCausalLM
    snaps = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/*/'))
    assert len(snaps) == 1, snaps
    tok = C.tokenizer()
    model = AutoModelForCausalLM.from_pretrained(snaps[0], torch_dtype=torch.bfloat16,
                                                 attn_implementation='eager').to('cuda').eval()
    res, secs = {}, []
    with torch.no_grad():
        for n, k in enumerate(keys, 1):
            enc = tok(texts[k], return_offsets_mapping=True, return_tensors='pt')
            ids = enc['input_ids'].to('cuda')
            torch.cuda.synchronize()
            t0 = time.time()
            logits = model(ids).logits[0].float()
            logq = torch.log_softmax(logits, dim=-1)
            ent = -(logq.exp() * logq).sum(-1)
            lp = logq[:-1].gather(1, ids[0, 1:, None])[:, 0]
            torch.cuda.synchronize()
            secs.append(time.time() - t0)
            res[k] = {'ids': ids[0].cpu().numpy().astype(np.int32),
                      'off': enc['offset_mapping'][0].numpy().astype(np.int32),
                      'logp': np.concatenate([[np.nan], lp.cpu().numpy()]).astype(np.float32),
                      'ent': np.concatenate([[np.nan], ent[:-1].cpu().numpy()]).astype(np.float32)}
            if n % 500 == 0 or n == len(keys):
                print(f'  {n}/{len(keys)}  median {np.median(secs) * 1000:.1f} ms per text', flush=True)
    np.savez_compressed(out_path, keys=np.array(keys),
                        **{f'{f}_{i}': res[k][f] for i, k in enumerate(keys) for f in ('ids', 'off', 'logp', 'ent')})
    lens = [len(res[k]['ids']) for k in keys]
    summary = {'role': ROLE, 'texts': len(keys), 'seconds_total': round(float(np.sum(secs)), 1),
               'ms_median': round(float(np.median(secs)) * 1000, 1), 'ms_p95': round(float(np.percentile(secs, 95)) * 1000, 1),
               'tokens_median': int(np.median(lens)), 'tokens_max': int(np.max(lens)),
               'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'guard': guard}
    with open(out_path.replace('.npz', '.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
