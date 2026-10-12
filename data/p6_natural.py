"""Phase 6, descriptive: pairs that Gemma itself produced. PREREGISTRATION_OWNFIX.md §3.

  natural pairs   two Gemma samples of one problem, both compiling, not
                  truncated and <= 140 tokens with the same Gemma token count,
                  whose lexer tokens differ in exactly one position, one passing
                  every assert and one failing through a wrong value or a panic.
                  Every whitespace variant of a code is tried. Counted as
                  combinations, and as member-unique pairs (greedy, problem and
                  sample order, no code twice): the §3 threshold of 40 pairs over
                  20 problems applies to the member-unique count.
  doubly natural  the sum of r_doubly_natural in ownfix_pairs.json (§2's
                  definition: R equals, whitespace aside, any passing sample
                  Gemma drew for the problem).

Counts only; reads labels, never graphs.  OUT=ownfix_natural.json
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m1_make_mutants as m1  # noqa: E402
from p6_1_repairs import failure_ok  # noqa: E402

OUT = os.environ.get('OUT', 'ownfix_natural.json')
PAIRS = os.environ.get('OUT_PAIRS', 'ownfix_pairs.json')


def norm(c):
    return re.sub(r'\s+', '', c)


def main():
    tok = m1.AutoTokenizer.from_pretrained(m1.MODEL_NAME)
    files = [('within_samples.jsonl', 'within_labels.jsonl')]
    for pat in ('matched_samples_r*.jsonl', 'ownfix_samples_r*.jsonl'):
        files += [(s, s.replace('_samples_', '_labels_')) for s in sorted(glob.glob(pat))]
    # problem -> normalised code -> {(tokens, code)} (every whitespace variant)
    passing, failing = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    for sf, lf in files:
        labels = {(r['task_id'], r['sample']): r for r in m1.read_jsonl(lf)}
        for s in m1.read_jsonl(sf):
            lab = labels[(s['task_id'], s['sample'])]
            if s['truncated'] or not lab['compiles']:
                continue
            n = len(tok(s['text'] + ' ' + s['generated_code'])['input_ids'])
            if n > m1.TOKEN_CAP:
                continue
            if lab['passes']:
                passing[s['task_id']][norm(s['generated_code'])].add((n, s['generated_code']))
            elif failure_ok(lab):
                failing[s['task_id']][norm(s['generated_code'])].add((n, s['generated_code']))
    combos, unique = [], []
    for tid in sorted(failing):
        used = set()
        for fk in sorted(failing[tid]):
            for pk in sorted(passing.get(tid, {})):
                hit = False
                for nf, cf in failing[tid][fk]:
                    tf = [t for _, t, _ in m1.lex(cf)]
                    for np_, cp in passing[tid][pk]:
                        if np_ != nf:
                            continue
                        tp = [t for _, t, _ in m1.lex(cp)]
                        if len(tp) == len(tf) and sum(a != b for a, b in zip(tp, tf)) == 1:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    combos.append((tid, fk, pk))
                    if fk not in used and pk not in used:
                        used |= {fk, pk}
                        unique.append((tid, fk, pk))
    res = {'natural_combinations': len(combos), 'natural_unique_pairs': len(unique),
           'natural_unique_problems': len({p[0] for p in unique}),
           'distinct_failing_codes': sum(len(v) for v in failing.values()),
           'distinct_passing_codes': sum(len(v) for v in passing.values())}
    if os.path.exists(PAIRS):
        pairs = json.load(open(PAIRS, encoding='utf-8'))
        res['pairs'] = len(pairs)
        res['doubly_natural'] = sum(bool(p.get('r_doubly_natural')) for p in pairs)
    print(json.dumps(res))
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
