"""Phase 4, step 2: how surprising each member's code is to the model.
PREREGISTRATION_MUTANT.md §3.

Both members of a pair carry one edit the model did not write, and a killed
edit may simply be the less expected one. For every row of ROWS this records
the summed log-probability Gemma-2-2b-it assigns to the code, teacher-forced on
the same text the graph is computed on (prose + ' ' + code): the sum over the
tokens that start at or after the space before the code. The prose is the same
for both members of a pair, so it is left out. Also recorded: the log-probability
of the token that holds the edit.

  ROWS=mutant_rows.jsonl  OUTPUT=mutant_logprob.jsonl
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROWS = os.environ.get('ROWS', 'mutant_rows.jsonl')
OUTPUT = os.environ.get('OUTPUT', 'mutant_logprob.jsonl')
MODEL_NAME = 'google/gemma-2-2b-it'


def main():
    with open(ROWS, encoding='utf-8') as f:
        rows = [json.loads(l) for l in f if l.strip()]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map='auto', torch_dtype=torch.bfloat16)
    model.eval()
    out = []
    for r in rows:
        enc = tok(r['text'], return_offsets_mapping=True, return_tensors='pt')
        ids = enc['input_ids'].to(model.device)
        offs = enc['offset_mapping'][0].tolist()
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        lp = torch.log_softmax(logits[:-1], dim=-1).gather(1, ids[0, 1:, None])[:, 0].cpu()
        code_from = len(r['prose'])                # the space before the code
        edit_at = len(r['prose']) + 1 + r['edit']['pos']
        code_lp, edit_lp = 0.0, None
        for t in range(1, ids.shape[1]):
            s, e = offs[t]
            if s >= code_from:
                code_lp += float(lp[t - 1])
            if s <= edit_at < e:
                edit_lp = float(lp[t - 1])
        out.append({'task_id': r['task_id'], 'code_logprob': code_lp, 'edit_logprob': edit_lp})
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o) + '\n')
    miss = sum(o['edit_logprob'] is None for o in out)
    print(f"wrote {len(out)} rows -> {OUTPUT}; edit token not located in {miss}")


if __name__ == '__main__':
    main()
