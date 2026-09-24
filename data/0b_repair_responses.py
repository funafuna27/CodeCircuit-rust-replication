"""Apply the fence-extraction fix to generations already on disk.

The fix in 0_generate_responses.py only affects future runs. The shipped
gemma_mbpp_responses.jsonl predates it: 16 of 256 generations still carry
unstripped markdown fences and do not parse, and 147 of the 1420 feature rows
contain a literal fence. Regenerating from the model is unnecessary -- decoding
was greedy and the prompts are unchanged, and both repair rules are pure text
transformations over the cleaned output, so applying them here is identical to
having generated with them.

Writes a NEW file rather than editing in place. The pre-registration locks a
240-problem baseline measured on the current data, and silently changing what
that baseline was measured on is exactly the freedom that document exists to
remove. The repaired file is an input to a future top-up graph run, not a
retroactive edit.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = os.environ.get('INPUT', str(HERE / 'gemma_mbpp_responses.jsonl'))
OUTPUT = os.environ.get('OUTPUT', str(HERE / 'gemma_mbpp_responses_repaired.jsonl'))

# Reuse the shipped extractor rather than restating it, so the two can never
# drift. Loading by path avoids importing the module, which would load gemma.
src = (HERE / '0_generate_responses.py').read_text(encoding='utf-8')
ns = {'re': __import__('re'), 'ast': ast}
exec(src[src.index('FENCE_LINE = '):src.index('def clean_output')], ns)
extract_code, parses = ns['extract_code'], ns['parses']


def repair(code):
    """Fences only. The stray-semicolon and brace-balance rules in the Rust
    generator are Rust-specific: the semicolon comes from ending a prompt on a
    bare signature, which the Python prompt does not do, and Python has no
    braces to balance."""
    return extract_code(code).strip()


def main():
    rows = []
    with open(INPUT, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    before_bad = [r['task_id'] for r in rows if not parses(r['generated_code'])]
    changed, after_bad, line_delta = [], [], {}
    for r in rows:
        old = r['generated_code']
        new = repair(old)
        if new != old.strip():
            changed.append(int(r['task_id']))
            line_delta[int(r['task_id'])] = (
                len(old.splitlines()), len(new.splitlines()))
            r['generated_code'] = new
            r['repaired'] = True
        r['parses'] = parses(r['generated_code'])
        if not r['parses']:
            after_bad.append(int(r['task_id']))

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"{len(rows)} generations")
    print(f"  did not parse before: {len(before_bad)}  {sorted(before_bad)}")
    print(f"  did not parse after:  {len(after_bad)}   {sorted(after_bad)}")
    print(f"  repaired:             {len(changed)}")
    print(f"\nwrote {OUTPUT}")
    if line_delta:
        tot_old = sum(a for a, _ in line_delta.values())
        tot_new = sum(b for _, b in line_delta.values())
        print(f"\nline counts on the repaired problems: {tot_old} -> {tot_new}")
        print("  (one feature row per cumulative line, so this is the graph cost)")
        for t in sorted(line_delta):
            a, b = line_delta[t]
            print(f"    task {t:4}: {a:3} -> {b:3} lines")


if __name__ == '__main__':
    main()
