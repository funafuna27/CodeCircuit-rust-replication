"""Phase 6, step 2: the model's surprise at S, at its repairs R and at its
controls C. PREREGISTRATION_OWNFIX.md §2.

As p5_2_logprob (Gemma-2-2b-it, bf16, eager attention so soft-capping applies,
teacher-forced on prose + ' ' + code; the code is every token that starts at or
after the space before it). For every source with a stable repair: every R, and
every live mutant that fails with S's FAIL/PANIC line (the controls, and the
fallback's candidates). Recorded per mutant:

  code_logprob, edit_logprob, min_logprob, mean_entropy   as in Phase 5
  tokens_to_end   code tokens after the token that holds the edit
and, from S's own forward pass at the token that holds the edit position:
  site_entropy    entropy (nats) of the distribution that predicts S's token there
  orig_logprob    log-probability of S's token there
  s_top           S's token was the most probable one there (a "belief" rather
                  than a "slip")
  own_top         the mutant's own new token is the most probable one at its
                  position, in its own forward pass (descriptive)

Incremental: mutants already in LOGPROB are skipped.

  SOURCES=ownfix_sources.jsonl  MUTANTS=ownfix_mutants.jsonl  LOGPROB=ownfix_logprob.jsonl
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SOURCES = os.environ.get('SOURCES', 'ownfix_sources.jsonl')
MUTANTS = os.environ.get('MUTANTS', 'ownfix_mutants.jsonl')
LOGPROB = os.environ.get('LOGPROB', 'ownfix_logprob.jsonl')
MODEL_NAME = 'google/gemma-2-2b-it'


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def mkey(m):
    return (m['task_id'], m['sample'], m['pos'], m['to'])


def main():
    usable = {(s['task_id'], s['sample']): s for s in read_jsonl(SOURCES)
              if s['stable'] and s['n_repairs']}
    # every repair, and every live mutant that fails like S: the strict controls
    # and the fallback's candidates (p6_3)
    todo = [m for m in read_jsonl(MUTANTS)
            if (m['task_id'], m['sample']) in usable
            and (m['is_r'] or (m['live'] and m['compiles'] and m['passes'] is False
                               and m.get('error') == usable[(m['task_id'], m['sample'])]['error']))]
    done = {mkey(r) for r in read_jsonl(LOGPROB)} if os.path.exists(LOGPROB) else set()
    todo = [m for m in todo if mkey(m) not in done]
    print(f"{len(usable)} usable sources; {len(done)} mutants scored before, {len(todo)} to score")
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()

    def forward(prose, code):
        enc = tok(prose + ' ' + code, return_offsets_mapping=True, return_tensors='pt')
        ids = enc['input_ids'].to('cuda')
        offs = enc['offset_mapping'][0].tolist()
        with torch.no_grad():
            logp = torch.log_softmax(model(ids).logits[0, :-1].float(), dim=-1)
        lp = logp.gather(1, ids[0, 1:, None])[:, 0]
        ent = -(logp.exp() * logp).sum(-1)
        top = logp.argmax(-1) == ids[0, 1:]
        code = [t for t in range(1, ids.shape[1]) if offs[t][0] >= len(prose)]
        return lp, ent, top, offs, code

    def holder(offs, at):
        return next((t for t in range(1, len(offs)) if offs[t][0] <= at < offs[t][1]), None)

    s_cache = {}
    with open(LOGPROB, 'a', encoding='utf-8') as f:
        for i, m in enumerate(todo, 1):
            src = usable[(m['task_id'], m['sample'])]
            prose = src['prose']
            key = (m['task_id'], m['sample'])
            if key not in s_cache:
                s_cache.clear()
                s_cache[key] = forward(prose, src['code'])
            s_lp, s_ent, s_top, s_offs, _ = s_cache[key]
            lp, ent, top, offs, code = forward(prose, m['code'])
            at = len(prose) + 1 + m['pos']
            t_m, t_s = holder(offs, at), holder(s_offs, at)
            idx = [t - 1 for t in code]
            f.write(json.dumps({
                'task_id': m['task_id'], 'sample': m['sample'], 'pos': m['pos'], 'to': m['to'],
                'code_logprob': float(lp[idx].sum()),
                'edit_logprob': float(lp[t_m - 1]) if t_m else None,
                'min_logprob': float(lp[idx].min()),
                'mean_entropy': float(ent[idx].mean()),
                'tokens_to_end': (len(code) - 1 - code.index(t_m)) if t_m in code else None,
                'site_entropy': float(s_ent[t_s - 1]) if t_s else None,
                'orig_logprob': float(s_lp[t_s - 1]) if t_s else None,
                's_top': bool(s_top[t_s - 1]) if t_s else None,
                'own_top': bool(top[t_m - 1]) if t_m else None,
            }) + '\n')
            if i % 500 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"wrote {LOGPROB}")


if __name__ == '__main__':
    main()
