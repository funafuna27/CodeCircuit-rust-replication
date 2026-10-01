"""Phase 3, step 0: K sampled solutions per problem. PREREGISTRATION_WITHIN.md §2.

The model, prompt, extraction and cleaning are imported from
0_generate_responses_rust.py rather than copied, so the only thing that differs
from Phase 2 is the decoding: temperature 0.8, top_p 0.95, K per problem.

Resumable: (task, sample) pairs already in OUTPUT are not drawn again. Each
problem's draw is seeded on its own (SEED_BASE + task_id, + 1000 for the gate's
second draw), so the samples do not depend on the order or on a restart.

  K=10  SAMPLE_OFFSET=0 (first draw) | 10 (second draw, samples 10-19)
  OUTPUT=within_samples.jsonl  LIMIT=0 (every problem)
"""
import importlib.util
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mbpp_rust import load_typeable  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'phase2_gen', os.path.join(HERE, '0_generate_responses_rust.py'))
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

K = int(os.environ.get('K', 10))
TEMPERATURE = float(os.environ.get('TEMPERATURE', 0.8))
TOP_P = float(os.environ.get('TOP_P', 0.95))
SEED_BASE = int(os.environ.get('SEED_BASE', 20261001))
SAMPLE_OFFSET = int(os.environ.get('SAMPLE_OFFSET', 0))
OUTPUT = os.environ.get('OUTPUT', 'within_samples.jsonl')
LIMIT = int(os.environ.get('LIMIT', 0))
N_PROBLEMS = 334


def main():
    typeable, _ = load_typeable()
    tids = sorted(t for t, s in typeable.items() if s['graded_all_tests'])
    if len(tids) != N_PROBLEMS:
        raise SystemExit(f"expected {N_PROBLEMS} strict problems, got {len(tids)}")
    if LIMIT:
        tids = tids[:LIMIT]

    done = set()
    if os.path.exists(OUTPUT):
        with open(OUTPUT, encoding='utf-8') as f:
            done = {(int(r['task_id']), int(r['sample']))
                    for r in (json.loads(l) for l in f if l.strip())}
    todo = [t for t in tids
            if not all((t, SAMPLE_OFFSET + k) in done for k in range(K))]
    print(f"{len(tids)} problems, K={K}, samples {SAMPLE_OFFSET}..{SAMPLE_OFFSET + K - 1}, "
          f"T={TEMPERATURE}, top_p={TOP_P}; {len(tids) - len(todo)} already drawn")

    tok = AutoTokenizer.from_pretrained(gen.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        gen.MODEL_NAME, device_map='auto', torch_dtype=torch.bfloat16)
    stop = model.generation_config.eos_token_id
    stop = set(stop if isinstance(stop, (list, tuple)) else [stop])

    t0 = time.time()
    with open(OUTPUT, 'a', encoding='utf-8') as f:
        for n, tid in enumerate(todo, 1):
            sig = typeable[tid]
            inputs = tok(gen.build_prompt(sig), return_tensors='pt').to(model.device)
            torch.manual_seed(SEED_BASE + tid + (1000 if SAMPLE_OFFSET else 0))
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=gen.MAX_NEW_TOKENS, do_sample=True,
                    temperature=TEMPERATURE, top_p=TOP_P, num_return_sequences=K,
                    pad_token_id=tok.eos_token_id)
            start = inputs['input_ids'].shape[1]
            for k in range(K):
                new = out[k][start:].tolist()
                end = next((i for i, t in enumerate(new) if t in stop), None)
                n_new = len(new) if end is None else end + 1
                raw = tok.decode(new[:n_new], skip_special_tokens=True)
                f.write(json.dumps({
                    'task_id': tid, 'sample': SAMPLE_OFFSET + k,
                    'text': gen.to_rust_prose(sig['prompt']),
                    'signature': sig['signature'],
                    'generated_code': gen.clean_output(raw),
                    'n_new_tokens': n_new,
                    'truncated': end is None and n_new >= gen.MAX_NEW_TOKENS,
                    'temperature': TEMPERATURE, 'top_p': TOP_P,
                }, ensure_ascii=False) + '\n')
            f.flush()
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} problems, {el / 60:.1f} min, "
                      f"~{el / n * (len(todo) - n) / 60:.0f} min left", flush=True)


if __name__ == '__main__':
    main()
