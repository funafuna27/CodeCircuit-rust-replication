"""Phase 3, step 2: eligibility, pairing and the gate. PREREGISTRATION_WITHIN.md §2.

  eligible   compiles, not truncated, prompt + whole solution <= 140 tokens
  pair       per problem with an eligible pass AND an eligible (compiling) fail,
             the pass x fail combination closest in tokens; ties to lowest indices
  cap        > 150 pairs: the 150 with the smallest token difference
  gate       >= 70 pairs, else exit code 2 (run_within.sh then draws once more)

Writes the graph pipeline's DATA_FILE -- one row per paired sample, under the
id task_id * 100 + sample, passing member first -- and the pair list.

  SAMPLES=within_samples.jsonl  LABELS=within_labels.jsonl
  OUT_ROWS=within_rows.jsonl    OUT_PAIRS=within_pairs.json
"""
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

SAMPLES = os.environ.get('SAMPLES', 'within_samples.jsonl')
LABELS = os.environ.get('LABELS', 'within_labels.jsonl')
OUT_ROWS = os.environ.get('OUT_ROWS', 'within_rows.jsonl')
OUT_PAIRS = os.environ.get('OUT_PAIRS', 'within_pairs.json')
TOKEN_CAP, MIN_PAIRS, MAX_PAIRS = 140, 70, 150
MODEL_NAME = 'google/gemma-2-2b-it'


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    samples = {(int(r['task_id']), int(r['sample'])): r for r in read_jsonl(SAMPLES)}
    labels = {(int(r['task_id']), int(r['sample'])): r for r in read_jsonl(LABELS)}
    if set(samples) != set(labels):
        sys.exit(f"samples and labels disagree on {len(set(samples) ^ set(labels))} keys")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    by_task = {}
    n_trunc = n_long = n_nocomp = 0
    for key, s in samples.items():
        lab = labels[key]
        text = s['text'] + ' ' + s['generated_code']   # the Phase 2 last-row text
        if s['truncated']:
            n_trunc += 1
            continue
        if not lab['compiles']:
            n_nocomp += 1
            continue
        n = len(tok(text)['input_ids'])
        if n > TOKEN_CAP:
            n_long += 1
            continue
        by_task.setdefault(key[0], []).append(
            {'sample': key[1], 'passes': bool(lab['passes']), 'tokens': n,
             'text': text, 'code': s['generated_code']})

    pairs = []
    for tid in sorted(by_task):
        ok = [e for e in by_task[tid] if e['passes']]
        bad = [e for e in by_task[tid] if not e['passes']]
        if not ok or not bad:
            continue
        p, q = min(((p, q) for p in ok for q in bad),
                   key=lambda pq: (abs(pq[0]['tokens'] - pq[1]['tokens']),
                                   pq[0]['sample'], pq[1]['sample']))
        pairs.append({'problem': tid, 'pass': p, 'fail': q,
                      'dtok': p['tokens'] - q['tokens']})

    n_tasks = len({k[0] for k in samples})
    n_mixed = len(pairs)
    if len(pairs) > MAX_PAIRS:
        pairs = sorted(pairs, key=lambda x: (abs(x['dtok']), x['problem']))[:MAX_PAIRS]
        pairs.sort(key=lambda x: x['problem'])

    print(f"{len(samples)} samples over {n_tasks} problems")
    print(f"  excluded: truncated {n_trunc}, not compiling {n_nocomp}, "
          f"over {TOKEN_CAP} tokens {n_long}")
    print(f"  eligible: {sum(len(v) for v in by_task.values())} samples over {len(by_task)} problems")
    print(f"  mixed problems (an eligible pass and an eligible fail): {n_mixed}")
    print(f"  pairs kept: {len(pairs)}" + (f" (capped at {MAX_PAIRS})" if n_mixed > MAX_PAIRS else ""))
    if pairs:
        d = np.array([x['dtok'] for x in pairs])
        print(f"  token difference pass - fail: |d| median {np.median(np.abs(d)):.0f}, "
              f"max {np.abs(d).max()}, mean signed {d.mean():+.2f}")
        tp = np.array([x['pass']['tokens'] for x in pairs])
        print(f"  tokens per member: median {np.median(tp):.0f}, range {tp.min()}..{tp.max()}")

    with open(OUT_PAIRS, 'w', encoding='utf-8') as f:
        json.dump([{'problem': x['problem'], 'pass_sample': x['pass']['sample'],
                    'fail_sample': x['fail']['sample'],
                    'pass_id': x['problem'] * 100 + x['pass']['sample'],
                    'fail_id': x['problem'] * 100 + x['fail']['sample'],
                    'pass_tokens': x['pass']['tokens'], 'fail_tokens': x['fail']['tokens']}
                   for x in pairs], f, indent=1)
    with open(OUT_ROWS, 'w', encoding='utf-8') as f:
        for x in pairs:
            for e, lab in ((x['pass'], 1), (x['fail'], 0)):
                f.write(json.dumps({
                    'task_id': x['problem'] * 100 + e['sample'], 'text': e['text'],
                    'generated_code': e['code'], 'label': lab,
                    'problem': x['problem'], 'sample': e['sample'], 'tokens': e['tokens'],
                }, ensure_ascii=False) + '\n')
    print(f"wrote {OUT_PAIRS} and {OUT_ROWS} ({2 * len(pairs)} rows)")

    if len(pairs) < MIN_PAIRS:
        print(f"GATE NOT MET: {len(pairs)} < {MIN_PAIRS} pairs")
        sys.exit(2)
    print(f"GATE MET: {len(pairs)} >= {MIN_PAIRS} pairs")


if __name__ == '__main__':
    main()
