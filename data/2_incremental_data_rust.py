"""Expand each Rust generation into one row per cumulative line prefix.

Same output schema as 2_incremental_data.py -- task_id / text / generated_code /
label -- so 3_generate_graph.py consumes it unchanged.

The label differs in kind, and that is the point of the Rust arm. Upstream's
label is an LLM judge's per-line 1/0, whose step-function shape its own prompt
mandates. Here the label is whether the whole generation compiled and passed
every MBPP assert, broadcast to all of that task's rows. So it is constant
within a task, which has two consequences worth stating up front:

  - every AUROC must be computed at the TASK level, with one aggregated
    prediction per held-out task. A row-level AUROC weights each task by its
    line count, i.e. by its solution length, and length correlates with the
    label;
  - the effective sample size is the number of tasks, not the number of rows.

n_pass / n_assert and `compiles` are carried through so a graded or
compile-only target can be tried without re-running anything.
"""
import json
import os

RESPONSES = os.environ.get('RESPONSES', 'gemma_mbpp_rust_responses.jsonl')
LABELS = os.environ.get('LABELS', 'gemma_mbpp_rust_correctness.jsonl')
OUTPUT = os.environ.get('OUTPUT', 'gemma_mbpp_rust_incremental.jsonl')

# Drop generations that did not compile. They are the Rust analogue of the 16
# markdown-fence corruptions on the Python arm: the graph features predicted
# "did this survive extraction" at 0.9995 AUROC there, which looks like signal
# about correctness and is not. Set KEEP_UNCOMPILABLE=1 to study them instead.
KEEP_UNCOMPILABLE = os.environ.get('KEEP_UNCOMPILABLE', '0') == '1'


def read_jsonl(path):
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def main():
    responses = {int(r['task_id']): r for r in read_jsonl(RESPONSES)}
    labels = {int(r['task_id']): r for r in read_jsonl(LABELS)}
    missing = set(responses) ^ set(labels)
    if missing:
        raise SystemExit(f"responses and labels disagree on {len(missing)} task ids: "
                         f"{sorted(missing)[:10]}")

    records, skipped, n_trunc = [], [], 0
    for tid in sorted(responses):
        resp, lab = responses[tid], labels[tid]
        if resp.get('truncated'):
            n_trunc += 1
        if not lab['compiles'] and not KEEP_UNCOMPILABLE:
            skipped.append(tid)
            continue
        code = resp['generated_code']
        lines = code.split('\n')
        for i in range(1, len(lines) + 1):
            records.append({
                'task_id': tid,
                'text': resp['text'] + ' ' + '\n'.join(lines[:i]),
                'generated_code': '\n'.join(lines[:i]),
                'label': int(lab['passes']),
                'compiles': bool(lab['compiles']),
                'n_pass': lab['n_pass'],
                'n_assert': lab['n_assert'],
            })

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    kept = sorted({r['task_id'] for r in records})
    pos = sum(1 for t in kept if labels[t]['passes'])
    lens = sorted(len(r['text']) for r in records)
    print(f"wrote {len(records)} rows over {len(kept)} tasks -> {OUTPUT}")
    print(f"  positive tasks: {pos}/{len(kept)} ({pos/len(kept)*100:.1f}%)")
    print(f"  positive rows:  {sum(r['label'] for r in records)}/{len(records)}")
    print(f"  rows per task:  mean {len(records)/len(kept):.2f}")
    if skipped:
        print(f"  dropped {len(skipped)} generations that do not compile "
              f"(KEEP_UNCOMPILABLE=1 to keep them)")
    if n_trunc:
        print(f"  WARNING: {n_trunc} generations hit the token cap; a truncated "
              f"generation cannot compile and is almost certainly among the dropped")
    print(f"  prompt chars: median {lens[len(lens)//2]}, "
          f"p90 {lens[int(len(lens)*0.9)]}, max {lens[-1]}")
    for thr in (450, 550, 650, 800, 1000):
        over = sum(1 for n in lens if n > thr)
        print(f"    MAX_PROMPT_CHARS={thr:5}: drops {over:5} rows "
              f"({over/len(lens)*100:5.1f}%)")
    print("  Python dropped 27 of 1447 rows (1.9%) at 550. Pick the Rust "
          "threshold to match that rate, not that number.")


if __name__ == '__main__':
    main()
