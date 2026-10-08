"""Phase 5, step 1: the source solutions and their judged one-token mutants.
PREREGISTRATION_MATCHED.md §2.

  sources  every eligible passing sample -- passes, not truncated, prose + ' ' +
           code <= 140 Gemma tokens -- of Phase 3 (within_samples.jsonl, samples
           0-19) and of Phase 5's extra rounds (matched_samples_r*.jsonl), except
           Phase 4's 99 pair sources. Within a problem, a sample whose code with
           all whitespace removed equals that of a Phase 4 pair source or of an
           earlier kept sample (in sample order) is a duplicate and is dropped.
  mutants  Phase 4's (m1_make_mutants: sites, families, the exact token-count
           filter, the judge under an 8 GiB address-space limit), imported
           unchanged. Every source is mutated and judged -- not only until its
           problem has one usable source, as in Phase 4.

Incremental: a source already in SOURCES is not judged again, so each round
only judges the sources it adds. Appends to SOURCES (one row per kept source)
and MUTANTS (one row per judged mutant).

  SOURCES=matched_sources.jsonl  MUTANTS=matched_mutants.jsonl  WORKERS=6
  P4_ONLY=1 uses Phase 4's pair sources instead (pipeline tests only; Phase 5
  excludes them)
"""
import glob
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m1_make_mutants as m1  # noqa: E402

SOURCES = os.environ.get('SOURCES', 'matched_sources.jsonl')
MUTANTS = os.environ.get('MUTANTS', 'matched_mutants.jsonl')
WORKERS = int(os.environ.get('WORKERS', 6))
P4_ONLY = os.environ.get('P4_ONLY', '0') == '1'
LIMIT = int(os.environ.get('LIMIT', 0))              # first LIMIT problems (tests)


def norm(code):
    return re.sub(r'\s+', '', code)


def load_samples():
    """(task, sample) -> (sample row, label row), Phase 3's and every round's."""
    out = {}
    files = [('within_samples.jsonl', 'within_labels.jsonl')]
    for s in sorted(glob.glob('matched_samples_r*.jsonl')):
        lab = s.replace('matched_samples_', 'matched_labels_')
        if not os.path.exists(lab):
            raise SystemExit(f"{s} has no labels ({lab}); run w1_label_samples.py first")
        files.append((s, lab))
    for sf, lf in files:
        labels = {(r['task_id'], r['sample']): r for r in m1.read_jsonl(lf)}
        for r in m1.read_jsonl(sf):
            key = (r['task_id'], r['sample'])
            if key in out:
                raise SystemExit(f"sample {key} drawn twice")
            if key not in labels:
                raise SystemExit(f"sample {key} has no label")
            out[key] = (r, labels[key])
    return out


def main():
    if not Path(m1.judge_mod.RUSTC).exists():
        sys.exit(f"rustc not found at {m1.judge_mod.RUSTC}")
    samples = load_samples()
    tok = m1.AutoTokenizer.from_pretrained(m1.MODEL_NAME)
    ntok = lambda text: len(tok(text)['input_ids'])
    with open('mutant_pairs.json', encoding='utf-8') as f:
        p4 = {(p['problem'], p['source_sample']) for p in json.load(f)}

    kept, seen = [], {(t, norm(samples[(t, s)][0]['generated_code'])) for t, s in p4}
    for key in sorted(samples):
        s, lab = samples[key]
        if s['truncated'] or not lab['passes']:
            continue
        if P4_ONLY != (key in p4):
            continue
        n = ntok(s['text'] + ' ' + s['generated_code'])
        if n > m1.TOKEN_CAP:
            continue
        if not P4_ONLY:
            dkey = (key[0], norm(s['generated_code']))
            if dkey in seen:
                continue
            seen.add(dkey)
        kept.append((key, n))
    if LIMIT:
        allowed = sorted({k[0] for k, _ in kept})[:LIMIT]
        kept = [(k, n) for k, n in kept if k[0] in allowed]

    done = set()
    if os.path.exists(SOURCES):
        done = {(r['task_id'], r['sample']) for r in m1.read_jsonl(SOURCES)}
    new = [(k, n) for k, n in kept if k not in done]
    print(f"{len(kept)} sources over {len({k[0] for k, _ in kept})} problems; "
          f"{len(done)} judged before, {len(new)} new")

    batch = []
    for (tid, smp), n in new:
        s = samples[(tid, smp)][0]
        for m in m1.mutants(s['generated_code']):
            if ntok(s['text'] + ' ' + m['code']) == n:
                batch.append({'task_id': tid, 'sample': smp, 'tokens': n, **m})
    print(f"  {len(batch)} token-preserving mutants to judge")
    with Pool(WORKERS, initializer=m1._init) as pool:
        verdicts = pool.map(m1._judge, [(b['task_id'], b['code']) for b in batch], chunksize=4)

    by_src = {}
    with open(MUTANTS, 'a', encoding='utf-8') as f:
        for b, v in zip(batch, verdicts):
            b.update(v)
            by_src.setdefault((b['task_id'], b['sample']), []).append(b)
            f.write(json.dumps(b, ensure_ascii=False) + '\n')
    usable = 0
    with open(SOURCES, 'a', encoding='utf-8') as f:
        for (tid, smp), n in new:
            ms = by_src.get((tid, smp), [])
            sv = sum(1 for m in ms if m['compiles'] and m['passes'])
            kl = sum(1 for m in ms if m['compiles'] and m['passes'] is False)
            usable += bool(sv and kl)
            f.write(json.dumps({'task_id': tid, 'sample': smp, 'tokens': n,
                                'code': samples[(tid, smp)][0]['generated_code'],
                                'prose': samples[(tid, smp)][0]['text'],
                                'n_mutants': len(ms), 'n_survive': sv, 'n_killed': kl,
                                'judge_failures': sum(1 for m in ms if m['compiles'] is None)},
                               ensure_ascii=False) + '\n')
    comp = [b for b in batch if b['compiles']]
    print(f"  judged {len(batch)}: {len(comp)} compile, "
          f"{sum(1 for b in comp if b['passes'])} survive, "
          f"{sum(1 for b in comp if b['passes'] is False)} killed, "
          f"{sum(1 for b in batch if b['compiles'] is None)} judge failures")
    print(f"  new usable sources (>= 1 survivor and >= 1 killed): {usable} of {len(new)}")


if __name__ == '__main__':
    main()
