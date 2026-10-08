"""Phase 5, step 0: one more round of sampled solutions. PREREGISTRATION_MATCHED.md §2.

Round r (1..4) draws samples 20r .. 20r+19 for every strict problem that has at
least one eligible passing sample among Phase 3's samples 0-19 (passes, not
truncated, prose + ' ' + code <= 140 Gemma tokens). The draw is w0_sample_rust's
-- Phase 3's model, prompt, extraction, decoding (temperature 0.8, top_p 0.95) and
tidy cleaning -- with the round's own seed, torch.manual_seed(20261001 + task_id
+ 1000 r). (w0 itself seeds every non-zero offset with + 1000, which is right for
its one second draw but would repeat it for the next.) Round 1's seed is the one
Phase 3 pre-registered for samples 20-39, which were never drawn.

Resumable: (task, sample) pairs already in the round's file are not drawn again.

  ROUND=1..4  OUTPUT=matched_samples_r{ROUND}.jsonl
"""
import importlib.util
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mbpp_rust import load_typeable  # noqa: E402
from within_clean import tidy  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'phase2_gen', os.path.join(HERE, '0_generate_responses_rust.py'))
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

ROUND = int(os.environ['ROUND'])
K, TEMPERATURE, TOP_P, SEED_BASE, TOKEN_CAP = 20, 0.8, 0.95, 20261001, 140
OUTPUT = os.environ.get('OUTPUT', f'matched_samples_r{ROUND}.jsonl')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    if not 1 <= ROUND <= 4:
        raise SystemExit(f"ROUND must be 1..4, got {ROUND}")
    offset = 20 * ROUND
    typeable, _ = load_typeable()
    tok = AutoTokenizer.from_pretrained(gen.MODEL_NAME)
    samples = read_jsonl('within_samples.jsonl')
    labels = {(r['task_id'], r['sample']): r for r in read_jsonl('within_labels.jsonl')}
    tids = sorted({s['task_id'] for s in samples
                   if not s['truncated'] and labels[(s['task_id'], s['sample'])]['passes']
                   and len(tok(s['text'] + ' ' + s['generated_code'])['input_ids']) <= TOKEN_CAP})
    if any(not typeable[t]['graded_all_tests'] for t in tids):
        raise SystemExit("a problem outside Phase 3's strict set")

    done = set()
    if os.path.exists(OUTPUT):
        done = {(r['task_id'], r['sample']) for r in read_jsonl(OUTPUT)}
    todo = [t for t in tids if not all((t, offset + k) in done for k in range(K))]
    print(f"round {ROUND}: {len(tids)} problems, samples {offset}..{offset + K - 1}, "
          f"T={TEMPERATURE}, top_p={TOP_P}; {len(tids) - len(todo)} already drawn")

    model = AutoModelForCausalLM.from_pretrained(
        gen.MODEL_NAME, device_map='auto', torch_dtype=torch.bfloat16)
    stop = model.generation_config.eos_token_id
    stop = set(stop if isinstance(stop, (list, tuple)) else [stop])

    t0 = time.time()
    with open(OUTPUT, 'a', encoding='utf-8') as f:
        for n, tid in enumerate(todo, 1):
            sig = typeable[tid]
            inputs = tok(gen.build_prompt(sig), return_tensors='pt').to(model.device)
            torch.manual_seed(SEED_BASE + tid + 1000 * ROUND)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=gen.MAX_NEW_TOKENS, do_sample=True,
                    temperature=TEMPERATURE, top_p=TOP_P, num_return_sequences=K,
                    pad_token_id=tok.eos_token_id)
            start = inputs['input_ids'].shape[1]
            name = re.search(r'fn\s+(\w+)', sig['signature']).group(1)
            for k in range(K):
                new = out[k][start:].tolist()
                end = next((i for i, t in enumerate(new) if t in stop), None)
                n_new = len(new) if end is None else end + 1
                raw = tok.decode(new[:n_new], skip_special_tokens=True)
                code2 = gen.clean_output(raw)
                f.write(json.dumps({
                    'task_id': tid, 'sample': offset + k, 'round': ROUND,
                    'text': gen.to_rust_prose(sig['prompt']),
                    'signature': sig['signature'],
                    'generated_code': tidy(code2, name),
                    'code_phase2': code2, 'raw': raw, 'tidy': True,
                    'n_new_tokens': n_new,
                    'truncated': end is None and n_new >= gen.MAX_NEW_TOKENS,
                    'temperature': TEMPERATURE, 'top_p': TOP_P,
                }, ensure_ascii=False) + '\n')
            f.flush()
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} problems, {el / 60:.1f} min, "
                      f"~{el / n * (len(todo) - n) / 60:.0f} min left", flush=True)


if __name__ == '__main__':
    main()
