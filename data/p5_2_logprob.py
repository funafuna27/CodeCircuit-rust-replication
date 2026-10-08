"""Phase 5, step 2: the model's surprise at every survivor and killed mutant of a
usable source. PREREGISTRATION_MATCHED.md §2.

As Phase 4's m2_logprob: Gemma-2-2b-it in bf16, teacher-forced on the text the
graph is computed on (prose + ' ' + code); the code is every token that starts at
or after the space before it. One change: eager attention, so Gemma-2's
attention-logit soft-capping applies, as it does in the graphs (m2 used sdpa,
which omits it; PREREGISTRATION_MUTANT.md §8). Recorded per mutant:

  code_logprob   summed log-probability of the code tokens (the matching variable)
  edit_logprob   log-probability of the token that holds the edit
  min_logprob    the lowest log-probability among the code tokens
  mean_entropy   mean entropy (nats) of the next-token distribution at the
                 positions that predict a code token

Incremental: mutants already in LOGPROB are skipped. Only usable sources (>= 1
survivor and >= 1 killed) are scored; mutants that do not compile are not.

  SOURCES=matched_sources.jsonl  MUTANTS=matched_mutants.jsonl  LOGPROB=matched_logprob.jsonl
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SOURCES = os.environ.get('SOURCES', 'matched_sources.jsonl')
MUTANTS = os.environ.get('MUTANTS', 'matched_mutants.jsonl')
LOGPROB = os.environ.get('LOGPROB', 'matched_logprob.jsonl')
MODEL_NAME = 'google/gemma-2-2b-it'


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def mkey(m):
    return (m['task_id'], m['sample'], m['pos'], m['to'])


def main():
    usable = {(s['task_id'], s['sample']): s for s in read_jsonl(SOURCES)
              if s['n_survive'] and s['n_killed']}
    todo = [m for m in read_jsonl(MUTANTS)
            if (m['task_id'], m['sample']) in usable and m['compiles'] and m['passes'] is not None]
    done = set()
    if os.path.exists(LOGPROB):
        done = {mkey(r) for r in read_jsonl(LOGPROB)}
    todo = [m for m in todo if mkey(m) not in done]
    print(f"{len(usable)} usable sources; {len(done)} mutants scored before, {len(todo)} to score")
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
    with open(LOGPROB, 'a', encoding='utf-8') as f:
        for i, m in enumerate(todo, 1):
            prose = usable[(m['task_id'], m['sample'])]['prose']
            enc = tok(prose + ' ' + m['code'], return_offsets_mapping=True, return_tensors='pt')
            ids = enc['input_ids'].to('cuda')
            offs = enc['offset_mapping'][0].tolist()
            with torch.no_grad():
                logp = torch.log_softmax(model(ids).logits[0, :-1].float(), dim=-1)
            lp = logp.gather(1, ids[0, 1:, None])[:, 0]
            ent = -(logp.exp() * logp).sum(-1)
            code = [t for t in range(1, ids.shape[1]) if offs[t][0] >= len(prose)]
            edit_at = len(prose) + 1 + m['pos']
            edit = [t for t in range(1, ids.shape[1]) if offs[t][0] <= edit_at < offs[t][1]]
            f.write(json.dumps({
                'task_id': m['task_id'], 'sample': m['sample'], 'pos': m['pos'], 'to': m['to'],
                'code_logprob': float(lp[[t - 1 for t in code]].sum()),
                'edit_logprob': float(lp[edit[0] - 1]) if edit else None,
                'min_logprob': float(lp[[t - 1 for t in code]].min()),
                'mean_entropy': float(ent[[t - 1 for t in code]].mean()),
            }) + '\n')
            if i % 500 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"wrote {LOGPROB}")


if __name__ == '__main__':
    main()
