"""Find the largest batch_size that fits in 12GB, and what it buys.

A full Python run is 6-9 days at batch_size=16, so a config that halves
per-graph cost is worth days. The upstream default is 64; it was cut to 16 for
this card's VRAM. Two things say that cut is the main slowdown and may be
larger than necessary:

  - offload='cpu' is NOT the culprit. cpu_offload_module() is a one-time
    module.to('cpu') after each phase, with a single reload at the end -- no
    per-batch round trip -- so offload=None would cost VRAM and buy no speed.
  - batch_size 8 vs 16 peaked at the same ~11.8GB, so VRAM here is dominated
    by batch-independent terms (weights, activation matrix; the big edge_matrix
    is allocated on CPU). Larger batches may well fit.

One model load, many timed calls: batch_size and offload are per-call arguments
to attribute(), so configs are compared without paying for reloads. Graphs are
not written to disk -- only wall-clock and peak VRAM are reported.
"""

import argparse
import json
import time

import torch
from circuit_tracer import ReplacementModel, attribute

MODEL_NAME = 'google/gemma-2-2b-it'
TRANSCODER_NAME = 'gemma'
DATA_FILE_PATH = 'gemma_mbpp_incremental.jsonl'

MAX_N_LOGITS = 10
DESIRED_LOGIT_PROB = 0.95
MAX_FEATURE_NODES = 8192

# Row index -> seconds measured under (batch_size=16, offload='cpu'), so every
# config below has a baseline to beat.
BASELINE = {3: 88, 6: 402, 8: 756}

CONFIGS = [
    (16, 'cpu'),    # current production setting, re-measured as a control
    (32, 'cpu'),
    (64, 'cpu'),    # upstream default
    (128, 'cpu'),   # only reached if 64 fits comfortably
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, nargs='+', default=[3],
                    help='row indices of gemma_mbpp_incremental.jsonl to time')
    ap.add_argument('--batch-sizes', type=int, nargs='+', default=None,
                    help='override the batch sizes to compare')
    args = ap.parse_args()

    configs = ([(b, 'cpu') for b in args.batch_sizes]
               if args.batch_sizes else CONFIGS)

    rows = [json.loads(l) for l in open(DATA_FILE_PATH, encoding='utf-8')]
    model = ReplacementModel.from_pretrained(
        MODEL_NAME, TRANSCODER_NAME, dtype=torch.bfloat16).to('cuda')

    results = []
    for idx in args.rows:
        prompt = rows[idx]['text']
        n_tok = len(model.tokenizer.encode(prompt))
        print(f"\n===== row {idx}: {len(prompt)} chars, {n_tok} tokens, "
              f"baseline {BASELINE.get(idx, '?')}s =====", flush=True)

        for batch_size, offload in configs:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            tag = f"batch={batch_size:<3} offload={str(offload):<5}"

            t0 = time.time()
            try:
                attribute(
                    prompt=prompt,
                    model=model,
                    max_n_logits=MAX_N_LOGITS,
                    desired_logit_prob=DESIRED_LOGIT_PROB,
                    batch_size=batch_size,
                    max_feature_nodes=MAX_FEATURE_NODES,
                    offload=offload,
                    verbose=False,
                )
            except Exception as exc:
                print(f"  {tag} FAILED after {time.time()-t0:.0f}s: "
                      f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)
                torch.cuda.empty_cache()
                continue

            el = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            base = BASELINE.get(idx)
            speed = f"  ({base/el:.2f}x vs baseline)" if base else ""
            print(f"  {tag} {el:7.1f}s  peak {peak:5.2f} GiB{speed}", flush=True)
            results.append((idx, n_tok, batch_size, str(offload), el, peak))

    print("\n===== summary =====")
    print(f"{'row':>4} {'tok':>5} {'batch':>6} {'offload':>8} {'sec':>8} {'peakGiB':>8}")
    for r in results:
        print(f"{r[0]:>4} {r[1]:>5} {r[2]:>6} {r[3]:>8} {r[4]:>8.1f} {r[5]:>8.2f}")


if __name__ == '__main__':
    main()
