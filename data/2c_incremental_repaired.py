"""Incremental rows for the repaired Python generations, for a top-up graph run.

The 16 fence-corrupted problems need their graphs rebuilt on the repaired code.
Everything else in the Python arm is untouched, so this emits rows only for the
problems whose generated_code actually changed.

The `label` field carries the EXECUTION label broadcast to every row, not the
per-line judge score. Regenerating the judge score would need the OpenAI API,
which this fork does not call, and the analysis does not use it: 5_problem_level.py
reads labels from a correctness file keyed by task_id, and script 3 only copies
`label` through to `step_labels`.
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESPONSES = os.environ.get('RESPONSES', str(HERE / 'gemma_mbpp_responses_repaired.jsonl'))
LABELS = os.environ.get('LABELS', str(HERE / 'gemma_mbpp_python_correctness_repaired.jsonl'))
OUTPUT = os.environ.get('OUTPUT', str(HERE / 'gemma_mbpp_incremental_repaired.jsonl'))
ONLY_REPAIRED = os.environ.get('ONLY_REPAIRED', '1') == '1'


def read_jsonl(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    resp = read_jsonl(RESPONSES)
    labels = {int(r['task_id']): int(bool(r['passes'])) for r in read_jsonl(LABELS)}
    if ONLY_REPAIRED:
        resp = [r for r in resp if r.get('repaired')]

    records = []
    for r in sorted(resp, key=lambda x: int(x['task_id'])):
        tid = int(r['task_id'])
        lines = r['generated_code'].split('\n')
        for i in range(1, len(lines) + 1):
            records.append({
                'task_id': tid,
                'text': r['text'] + ' ' + '\n'.join(lines[:i]),
                'generated_code': '\n'.join(lines[:i]),
                'label': labels[tid],
            })

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    tasks = sorted({r['task_id'] for r in records})
    pos = sum(labels[t] for t in tasks)
    print(f"wrote {len(records)} rows over {len(tasks)} problems -> {OUTPUT}")
    print(f"  problems: {tasks}")
    print(f"  positive: {pos}/{len(tasks)}")
    print(f"  NOTE: run 3_generate_graph.py against this with GRAPH_DIR=python_graph_repaired")
    print(f"        and FEATURES_FILE=gemma_mbpp_features_repaired.jsonl, then merge.")


if __name__ == '__main__':
    main()
