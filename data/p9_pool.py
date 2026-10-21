"""Phase 9: the pool of every Rust sample Gemma drew in Phases 3, 5 and 6, with its stored label.

  Phase 3   within_samples.jsonl / within_labels.jsonl        samples 0-19, 334 problems
  Phase 5   matched_samples_r1.jsonl / matched_labels_r1.jsonl samples 20-39, its 89 problems
  Phase 6   ownfix_samples_r{1..8}.jsonl / ownfix_labels_r*    samples 20r..20r+19

Phase 6 reused Phase 5's round 1 with the same seeds, so a (task, sample) drawn twice must carry
identical code; that is checked, and the first copy is kept. Writes pool_samples.jsonl.gz and
pool_labels.jsonl.gz (rows as stored, plus 'phase').
"""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_common as C  # noqa: E402

SOURCES = [('3', 'within_samples.jsonl', 'within_labels.jsonl'),
           ('5', 'matched_samples_r1.jsonl', 'matched_labels_r1.jsonl')] + \
          [('6', f'ownfix_samples_r{r}.jsonl', f'ownfix_labels_r{r}.jsonl') for r in range(1, 9)]


def main():
    samples, labels, clash, repeat = {}, {}, 0, 0
    for phase, sf, lf in SOURCES:
        s_rows = C.read_jsonl(os.path.join(HERE, sf))
        l_rows = {(int(r['task_id']), int(r['sample'])): r for r in C.read_jsonl(os.path.join(HERE, lf))}
        assert len(l_rows) == len(s_rows), (sf, len(s_rows), len(l_rows))
        for r in s_rows:
            k = (int(r['task_id']), int(r['sample']))
            if k in samples:
                repeat += 1
                clash += samples[k]['generated_code'] != r['generated_code']
                continue
            samples[k] = dict(r, phase=phase)
            labels[k] = dict(l_rows[k], phase=phase)
        print(f'{sf}: {len(s_rows)} rows; pool now {len(samples)}')
    print(f'repeated (task, sample): {repeat}, with different code: {clash}')
    if clash:
        sys.exit('a repeated sample has different code')
    os.makedirs(C.OUT_DIR, exist_ok=True)
    for name, d in (('pool_samples.jsonl.gz', samples), ('pool_labels.jsonl.gz', labels)):
        with gzip.open(os.path.join(C.OUT_DIR, name), 'wt', encoding='utf-8') as f:
            for k in sorted(d):
                f.write(json.dumps(d[k], ensure_ascii=False) + '\n')
    print(f'{len(samples)} samples over {len({k[0] for k in samples})} problems')


if __name__ == '__main__':
    main()
