"""Apply a token cap to an incremental file, and say exactly what it removed.

Why a token cap rather than the character filter upstream uses: graph cost is
superlinear in prompt TOKENS -- peak allocation crosses a 12 GB card around
130-140 tokens and spills to host RAM over PCIe -- so a handful of long rows
dominate the bill. Measured on the Rust arm, rows past 180 tokens are 4.6% of
the data and 21.6% of the compute, and capping at 140 takes the run from 13.5
days to 7.6 without losing a single problem.

It is only free if it is applied to BOTH arms and does not move the result.
Checked before adopting it: Python's problem-level AUROC is 0.7324 [0.6686,
0.7955] uncapped and 0.7269 [0.6637, 0.7873] at 140 tokens -- a shift of 0.006
against a CI half-width of 0.06, with all 256 problems retained.

Evaluation is at the problem level, and the Python arm showed the signal is
between problems rather than within them (deleting every within-task comparison
moved the row-level AUROC by -0.0010), so what matters is that every problem
survives with enough rows to aggregate, not that every prefix survives.
"""
import json
import os

MODEL_NAME = os.environ.get('MODEL_NAME', 'google/gemma-2-2b-it')
INPUT = os.environ.get('INPUT', 'gemma_mbpp_rust_incremental.jsonl')
OUTPUT = os.environ.get('OUTPUT', 'gemma_mbpp_rust_incremental_capped.jsonl')
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', 140))
MAX_CHARS = int(os.environ.get('MAX_CHARS', 605))


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    rows = []
    with open(INPUT, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"{len(rows)} rows in {INPUT}")

    for r in rows:
        r['_tok'] = len(tok(r['text'])['input_ids'])

    fits = lambda r: r['_tok'] <= MAX_TOKENS and len(r['text']) <= MAX_CHARS
    kept = [r for r in rows if fits(r)]
    dropped = [r for r in rows if not fits(r)]

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for r in kept:
            f.write(json.dumps({k: v for k, v in r.items() if k != '_tok'},
                               ensure_ascii=False) + '\n')

    def tasks(rs):
        out = {}
        for r in rs:
            out.setdefault(r['task_id'], []).append(r['label'])
        return out

    t_all, t_kept = tasks(rows), tasks(kept)
    lost = sorted(set(t_all) - set(t_kept))
    counts = [len(v) for v in t_kept.values()]
    pos_all = sum(v[0] for v in t_all.values())
    pos_kept = sum(v[0] for v in t_kept.values())

    print(f"\nkept {len(kept)} rows over {len(t_kept)} problems -> {OUTPUT}")
    print(f"  dropped {len(dropped)} rows ({len(dropped)/len(rows)*100:.2f}%)")
    print(f"  problems lost entirely: {len(lost)} {lost if lost else ''}")
    print(f"  rows per problem: min {min(counts)}, median "
          f"{sorted(counts)[len(counts)//2]}, max {max(counts)}")
    print(f"  problems with fewer than 2 rows: {sum(1 for c in counts if c < 2)}")
    print(f"  positive problems: {pos_all}/{len(t_all)} "
          f"({pos_all/len(t_all)*100:.1f}%) -> {pos_kept}/{len(t_kept)} "
          f"({pos_kept/len(t_kept)*100:.1f}%)")
    if dropped:
        dpos = sum(r['label'] for r in dropped) / len(dropped)
        apos = sum(r['label'] for r in rows) / len(rows)
        print(f"  dropped rows are {dpos*100:.1f}% positive vs {apos*100:.1f}% overall")
        print("  (long prefixes skew to failures; the same cap must be applied to "
              "the other arm at analysis time or the arms differ in preprocessing)")


if __name__ == '__main__':
    main()
