"""Phase 9, A2 summary: how many `mut` relocations exist, and what they look like. Descriptive.

  counts    sources, the necessity of each `mut`, violation and legal candidates by y's kind,
            with the Gemma token multiset equal or not; sources and problems
  tokens    token-count and multiset agreement with the source
  position  token distances: x's declaration -> q, y's `mut` -> q; y before or after x; where q
            sits in the text; the 140-token cap
  q         the violating token's pieces and rustc's codes
  examples  20 candidates (violations and legal, fixed seed) written out for reading by eye

  CANDS=results/p9_state/a2_candidates_<pool>.jsonl.gz  (one or more, comma-separated)
"""
import json
import os
import statistics
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_common as C  # noqa: E402

CANDS = os.environ.get('CANDS', os.path.join(C.OUT_DIR, 'a2_candidates_all.jsonl.gz')).split(',')
TAG = os.environ.get('TAG', 'all')
OUT = os.path.join(C.OUT_DIR, f'a2_summary_{TAG}.json')
SEED = [20261022, 1]


def q4(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return xs
    return [round(v, 1) for v in statistics.quantiles(xs, n=4)] + [min(xs), max(xs)]


def main():
    rows = [r for path in CANDS for r in C.read_jsonl(path)]
    out, log = {}, []
    pr = lambda *a: log.append(' '.join(str(x) for x in a))
    nec = Counter(n['class'] for r in rows for n in r['necessity'])
    out['sources'] = {'unique_compiling': len(rows), 'problems': len({r['key'][0] for r in rows}),
                      'with_mut_let_or_for': sum(bool(r['necessity']) for r in rows),
                      'necessity': dict(nec), 'compiles_run': sum(r['n_compiles'] for r in rows),
                      'by_phase': dict(Counter(r['phase'] for r in rows))}
    pr('sources:', out['sources'])

    cands = [dict(c, key=tuple(r['key']), phase=r['phase'], passes=r['passes'], n_tok_src=r['n_tok_src'])
             for r in rows for c in r['cands']]
    out['candidates_tried'] = dict(Counter(f"{c['type']} ok={c['ok']}" for c in cands).most_common())
    ok = [c for c in cands if c['ok']]
    table = {}
    for typ in ('violation', 'legal'):
        for ykinds, label in ((('let',), 'y let'), (('let', 'for'), 'y let/for'), (('let', 'for', 'param'), 'y any')):
            for ms in (False, True):
                cs = [c for c in ok if c['type'] == typ and c['y']['kind'] in ykinds and (c['same_multiset'] or not ms)]
                table[f'{typ} | {label} | {"multiset equal" if ms else "any"}'] = {
                    'candidates': len(cs), 'sources': len({c['key'] for c in cs}),
                    'problems': len({c['key'][0] for c in cs}),
                    'sources_le140': len({c['key'] for c in cs if c['n_tok'] <= 140}),
                    'problems_le140': len({c['key'][0] for c in cs if c['n_tok'] <= 140}),
                    'passing_sources': len({c['key'] for c in cs if c['passes']}),
                    # both edits before q: the prefixes at q have the same tokens and length
                    'sources_aligned': len({c['key'] for c in cs if typ == 'violation'
                                            and c.get('tok_q') is not None and c['tok_y_mut'] < c['tok_q']}),
                    'problems_aligned_le140': len({c['key'][0] for c in cs if typ == 'violation'
                                                   and c.get('tok_q') is not None and c['tok_y_mut'] < c['tok_q']
                                                   and c['n_tok'] <= 140})}
    out['table'] = table
    pr('\ncandidates that rustc labels as intended (ok):')
    pr(f'  {"set":<46}{"cands":>7}{"srcs":>6}{"probs":>6}{"srcs<=140":>10}{"probs<=140":>11}{"pass srcs":>10}'
       f'{"srcs y<q":>9}{"probs y<q<=140":>15}')
    for k, v in table.items():
        pr(f'  {k:<46}{v["candidates"]:>7}{v["sources"]:>6}{v["problems"]:>6}{v["sources_le140"]:>10}'
           f'{v["problems_le140"]:>11}{v["passing_sources"]:>10}{v["sources_aligned"]:>9}{v["problems_aligned_le140"]:>15}')
    pr('  tried (type, ok):', out['candidates_tried'])

    out['tokens'] = {typ: {'same_len': round(np.mean([c['same_len'] for c in ok if c['type'] == typ]), 3),
                           'same_multiset': round(np.mean([c['same_multiset'] for c in ok if c['type'] == typ]), 3),
                           'n': sum(c['type'] == typ for c in ok),
                           'mut_piece': dict(Counter(c['tok_mut_piece'] for c in ok if c['type'] == typ)),
                           'same_multiset_by_y_kind': {k: round(np.mean([c['same_multiset'] for c in ok
                                                                         if c['type'] == typ and c['y']['kind'] == k]), 3)
                                                       for k in ('let', 'for', 'param')
                                                       if any(c['type'] == typ and c['y']['kind'] == k for c in ok)}}
                     for typ in ('violation', 'legal') if any(c['type'] == typ for c in ok)}
    pr('\ntoken agreement with the source:', json.dumps(out['tokens']))

    v = [c for c in ok if c['type'] == 'violation']
    if v:
        out['position'] = {
            'x_decl_to_q_tokens': q4([c['tok_q'] - c['tok_x_decl'] for c in v if c.get('tok_q') is not None]),
            'y_mut_to_q_tokens': q4([c['tok_q'] - c['tok_y_mut'] for c in v if c.get('tok_q') is not None]),
            'y_after_q': sum(c['tok_y_mut'] > c['tok_q'] for c in v if c.get('tok_q') is not None),
            'y_before_x': sum(c['y']['name_pos'] < c['x']['name_pos'] for c in v),
            'q_share_of_text': q4([round(c['tok_q'] / c['n_tok'], 2) for c in v if c.get('tok_q') is not None]),
            'n_tok': q4([c['n_tok'] for c in v]),
            'q_op_missing': sum(c.get('tok_q_op') is None for c in v),
        }
        out['q'] = {'errors': dict(Counter('+'.join(c['errors']) for c in v).most_common(6)),
                    'q_piece': dict(Counter(c.get('tok_q_piece') for c in v).most_common(12)),
                    'q_op_piece': dict(Counter(c.get('tok_q_op_piece') for c in v).most_common(15)),
                    'site_label': dict(Counter(c['sites'][0]['label'] for c in v).most_common(5))}
        pr('\nposition (violations):', json.dumps(out['position']))
        pr('q:', json.dumps(out['q'], ensure_ascii=False))

    # natural errors are counted in A1; here, the per-problem spread of violation sources
    per_prob = Counter(c['key'][0] for c in {(c['key'], c['key'][0]): c for c in v}.values())
    out['violation_sources_per_problem'] = q4(list(per_prob.values()))
    pr('violation sources per problem (quartiles, min, max):', out['violation_sources_per_problem'])

    # 20 examples to read, at most one per source, stratified by y's kind:
    # violations 10 with y a let/for binding, 4 with y a parameter; legal 4 and 2
    rng = np.random.default_rng(SEED)
    ex = []
    for typ, kinds, n in (('violation', ('let', 'for'), 10), ('violation', ('param',), 4),
                          ('legal', ('let', 'for'), 4), ('legal', ('param',), 2)):
        pool = {}
        for c in ok:
            if c['type'] == typ and c['y']['kind'] in kinds:
                pool.setdefault(c['key'], []).append(c)
        keys = sorted(pool)
        for i in rng.choice(len(keys), min(n, len(keys)), replace=False):
            cs = pool[keys[i]]
            ex.append(cs[int(rng.integers(len(cs)))])
    with open(os.path.join(C.OUT_DIR, f'a2_examples_{TAG}.txt'), 'w', encoding='utf-8') as f:
        for c in ex:
            f.write('=' * 78 + '\n')
            f.write(f"{c['type']}  problem {c['key'][0]} sample {c['key'][1]}  x `{c['x']['name']}` ({c['x']['kind']}) "
                    f"-> y `{c['y']['name']}` ({c['y']['kind']})  errors {c['errors']}  tokens {c['n_tok']} "
                    f"same multiset {c['same_multiset']}\n")
            if c['type'] == 'violation':
                f.write(f"q: {c.get('tok_q_piece')!r} (token {c.get('tok_q')}), op {c.get('tok_q_op_piece')!r}; "
                        f"site: {c['sites'][0]['label']}\n")
            f.write(c['code'] + '\n')
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    with open(OUT.replace('.json', '.log'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log) + '\n')
    print('\n'.join(log))


if __name__ == '__main__':
    main()
