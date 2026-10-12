"""Phase 6, step 0: one more round of sampled solutions. PREREGISTRATION_OWNFIX.md §2.

Round r (1..8) draws samples 20r .. 20r+19 for every strict problem that has at
least one eligible failing sample among Phase 3's samples 0-19 (compiles, fails
through a wrong value or a panic, not truncated, prose + ' ' + code <= 140
tokens), skipping problems that already hold 5 pairs (ownfix_pairs.json) and
(task, sample) pairs drawn before by any phase. Phase 3's model, prompt,
decoding (T 0.8, top_p 0.95) and tidy cleaning; seed 20261001 + task_id +
1000 r, the scheme of p5_0_sample (Phase 5 drew round 1 for 164 problems with
the same seeds, so those samples are reused, not redrawn).

  ROUND=1..8  OUTPUT=ownfix_samples_r{ROUND}.jsonl
"""
import glob
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5_0_sample as p5  # noqa: E402  (the draw; its ROUND env is read at import)
from p6_1_repairs import failure_ok  # noqa: E402

ROUND = int(os.environ['ROUND'])
OUTPUT = os.environ.get('OUTPUT', f'ownfix_samples_r{ROUND}.jsonl')


def main():
    import re
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not 1 <= ROUND <= 8:
        raise SystemExit(f"ROUND must be 1..8, got {ROUND}")
    offset = 20 * ROUND
    typeable, _ = p5.load_typeable()
    tok = AutoTokenizer.from_pretrained(p5.gen.MODEL_NAME)
    labels = {(r['task_id'], r['sample']): r for r in p5.read_jsonl('within_labels.jsonl')}
    tids = sorted({s['task_id'] for s in p5.read_jsonl('within_samples.jsonl')
                   if not s['truncated'] and failure_ok(labels[(s['task_id'], s['sample'])])
                   and len(tok(s['text'] + ' ' + s['generated_code'])['input_ids']) <= p5.TOKEN_CAP})
    full = set()
    if os.path.exists('ownfix_pairs.json'):
        per = Counter(p['problem'] for p in json.load(open('ownfix_pairs.json', encoding='utf-8')))
        full = {t for t, n in per.items() if n >= 5}
    drawn = set()
    for f in glob.glob('*_samples*.jsonl'):
        drawn |= {(r['task_id'], r['sample']) for r in p5.read_jsonl(f)}
    todo = [t for t in tids if t not in full and not all((t, offset + k) in drawn for k in range(p5.K))]
    print(f"round {ROUND}: {len(tids)} problems with a failing source, {len(full)} already full; "
          f"{len(todo)} to draw (samples {offset}..{offset + p5.K - 1})")
    if not todo:
        return
    model = AutoModelForCausalLM.from_pretrained(
        p5.gen.MODEL_NAME, device_map='auto', torch_dtype=torch.bfloat16)
    stop = model.generation_config.eos_token_id
    stop = set(stop if isinstance(stop, (list, tuple)) else [stop])
    t0 = time.time()
    with open(OUTPUT, 'a', encoding='utf-8') as f:
        for n, tid in enumerate(todo, 1):
            sig = typeable[tid]
            inputs = tok(p5.gen.build_prompt(sig), return_tensors='pt').to(model.device)
            torch.manual_seed(p5.SEED_BASE + tid + 1000 * ROUND)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=p5.gen.MAX_NEW_TOKENS, do_sample=True,
                    temperature=p5.TEMPERATURE, top_p=p5.TOP_P, num_return_sequences=p5.K,
                    pad_token_id=tok.eos_token_id)
            start = inputs['input_ids'].shape[1]
            name = re.search(r'fn\s+(\w+)', sig['signature']).group(1)
            for k in range(p5.K):
                new = out[k][start:].tolist()
                end = next((i for i, t in enumerate(new) if t in stop), None)
                n_new = len(new) if end is None else end + 1
                raw = tok.decode(new[:n_new], skip_special_tokens=True)
                code2 = p5.gen.clean_output(raw)
                f.write(json.dumps({
                    'task_id': tid, 'sample': offset + k, 'round': ROUND, 'phase': 6,
                    'text': p5.gen.to_rust_prose(sig['prompt']), 'signature': sig['signature'],
                    'generated_code': p5.tidy(code2, name), 'code_phase2': code2, 'raw': raw,
                    'tidy': True, 'n_new_tokens': n_new,
                    'truncated': end is None and n_new >= p5.gen.MAX_NEW_TOKENS,
                    'temperature': p5.TEMPERATURE, 'top_p': p5.TOP_P}, ensure_ascii=False) + '\n')
            f.flush()
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} problems, {el / 60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
