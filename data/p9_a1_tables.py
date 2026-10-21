"""Phase 9, A1 tables: what rustc's diagnostics say about Phase 3's 6,680 samples.

Reads p9_a1_diag.py's output, the samples and their stored labels. Descriptive only; no model output.

  T0  reproduction: the harness build's exit status against the stored `compiles` label, and its
      first error code against the stored first error
  T1  lib mode against harness mode
  T2  error family x location of the primary span (target signature / body / other code / harness)
  T3  per family: diagnostics, samples, problems; share with a secondary span in the code (a
      non-primary span, or a help/note child's span), and how far it is from the primary span
  T4  duplicates (the study's key: code with whitespace removed, within a problem)
  T5  natural mutability / move / borrow errors per problem, raw and deduplicated, with the samples
      in which borrow checking could have been masked by a type or resolution error
  T6  unused_mut warnings in compiling samples
  T7  where in the body the mutability and move errors sit (quintiles of the body)

  DIAG=results/p9_state/a1_diag.jsonl.gz  OUT=results/p9_state/a1_tables.json
"""
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
import p9_common as C  # noqa: E402

DIAG = os.environ.get('DIAG', os.path.join(C.OUT_DIR, 'a1_diag.jsonl.gz'))
SAMPLES = os.environ.get('SAMPLES', os.path.join(HERE, 'within_samples.jsonl'))
LABELS = os.environ.get('LABELS', os.path.join(HERE, 'within_labels.jsonl'))
OUT = os.environ.get('OUT', os.path.join(C.OUT_DIR, 'a1_tables.json'))
BCK = ('mutability', 'move', 'borrow')


def stored_code(err):
    m = re.match(r'error\[(E\d+)\]', err or '')
    return m.group(1) if m else None


def main():
    diag = {(r['task_id'], r['sample']): r for r in C.read_jsonl(DIAG)}
    samples = {(int(r['task_id']), int(r['sample'])): r for r in C.read_jsonl(SAMPLES)}
    labels = {(int(r['task_id']), int(r['sample'])): r for r in C.read_jsonl(LABELS)}
    assert set(diag) == set(samples) == set(labels), 'sample sets differ'
    import p8_judge
    sigs = p8_judge.typeable()
    out, log = {}, []
    pr = lambda *a: log.append(' '.join(str(x) for x in a))

    # ---------------------------------------------------------------- T0, T1
    agree = first_agree = first_n = timeouts = 0
    cross = Counter()
    for k, r in diag.items():
        lab = labels[k]
        comp = C.truthy(lab['compiles'])
        h, l = r['harness'], r['lib']
        timeouts += bool(h.get('timeout')) + bool(l.get('timeout'))
        agree += (h['rc'] == 0) == comp
        if not comp and stored_code(lab['error']):
            first_n += 1
            fh = next((d['code'] for d in h['diags'] if d['level'] == 'error'), None)
            first_agree += fh == stored_code(lab['error'])
        cross[(l['rc'] == 0, h['rc'] == 0)] += 1
    out['T0'] = {'n': len(diag), 'compiles_agree': agree, 'first_code_agree': first_agree,
                 'first_code_n': first_n, 'timeouts': timeouts}
    out['T1'] = {f'lib_{"ok" if a else "err"}__harness_{"ok" if b else "err"}': v for (a, b), v in sorted(cross.items())}
    pr('T0 reproduction: harness exit status = stored compiles in', agree, 'of', len(diag),
       f'| first error code agrees in {first_agree} of {first_n} | timeouts {timeouts}')
    pr('T1 lib x harness:', out['T1'])

    # ---------------------------------------------------------------- per-diagnostic records
    seen = defaultdict(set)              # task -> ws keys already seen (sample order)
    dup = {}
    for k in sorted(samples):
        key = C.ws_key(samples[k]['generated_code'])
        dup[k] = key in seen[k[0]]
        seen[k[0]].add(key)

    recs, per_sample = [], {}
    for k in sorted(diag):
        code = samples[k]['generated_code']
        ext = C.extent(code, sigs[k[0]]['name'])
        fams = []
        for mode in ('lib', 'harness'):
            for d in diag[k][mode]['diags']:
                if d['level'] != 'error':
                    continue
                fam = C.family(d['code'], 'error')
                prim = next((s for s in d['spans'] if s['p']), None)
                if prim is None:
                    loc = 'none'
                elif prim['w'] == 'code':
                    loc = C.where(prim['s'], ext)
                else:
                    loc = prim['w']
                sec = [s for s in d['spans'] if not s['p'] and s['w'] == 'code']
                kid = [s for c in d['children'] for s in c['spans'] if s['w'] == 'code']
                cause = sec + kid
                dist = None
                if prim is not None and prim['w'] == 'code' and cause:
                    near = min(cause, key=lambda s: abs(s['s'] - prim['s']))
                    dist = {'chars': prim['s'] - near['s'],
                            'lines': code.count('\n', 0, prim['s']) - code.count('\n', 0, near['s'])}
                rel = None
                if loc == 'body' and ext:
                    rel = (prim['s'] - ext['open']) / max(1, ext['close'] - ext['open'])
                # where rustc suggests adding `mut`: the binding's declaration (a parameter or a let)
                decl = next((C.where(s['s'], ext) for c in d['children'] for s in c['spans']
                             if s['w'] == 'code' and 'mut' in (s.get('repl') or '')), None)
                recs.append({'k': k, 'mode': mode, 'code': d['code'], 'fam': fam, 'loc': loc,
                             'sec': bool(sec), 'kid': bool(kid), 'dist': dist, 'rel': rel, 'dup': dup[k],
                             'decl': decl})
                if mode == 'lib':
                    fams.append(fam)
        per_sample[k] = fams

    lib = [r for r in recs if r['mode'] == 'lib']
    harness_only = [r for r in recs if r['mode'] == 'harness' and r['loc'] in ('harness', 'head')]

    # ---------------------------------------------------------------- T2
    locs = ('sig', 'body', 'other', 'none')
    t2 = {f: {l: sum(1 for r in lib if r['fam'] == f and r['loc'] == l) for l in locs} for f in C.FAMILIES}
    t2 = {f: v for f, v in t2.items() if sum(v.values())}
    t2['harness_part'] = dict(Counter(r['fam'] for r in harness_only))
    out['T2'] = t2
    pr('\nT2 lib-mode errors: family x location of the primary span (counts of diagnostics)')
    pr(f'  {"family":<11}' + ''.join(f'{l:>7}' for l in locs) + f'{"total":>8}')
    for f, v in t2.items():
        if f == 'harness_part':
            continue
        pr(f'  {f:<11}' + ''.join(f'{v[l]:>7}' for l in locs) + f'{sum(v.values()):>8}')
    pr('  harness-mode errors whose primary span is in the harness:', t2['harness_part'])

    # ---------------------------------------------------------------- T3
    t3 = {}
    for f in [f for f in C.FAMILIES if any(r['fam'] == f for r in lib)]:
        rs = [r for r in lib if r['fam'] == f]
        ds = [r['dist'] for r in rs if r['dist']]
        t3[f] = {
            'diags': len(rs), 'samples': len({r['k'] for r in rs}), 'problems': len({r['k'][0] for r in rs}),
            'samples_dedup': len({r['k'] for r in rs if not r['dup']}),
            'secondary_share': round(sum(r['sec'] for r in rs) / len(rs), 3),
            'child_share': round(sum(r['kid'] for r in rs) / len(rs), 3),
            'either_share': round(sum(r['sec'] or r['kid'] for r in rs) / len(rs), 3),
            'cause_before_primary_share': round(sum(d['chars'] > 0 for d in ds) / len(ds), 3) if ds else None,
            'median_line_gap': statistics.median(d['lines'] for d in ds) if ds else None,
            'codes': dict(Counter(r['code'] for r in rs).most_common(8)),
        }
    out['T3'] = t3
    pr('\nT3 lib-mode errors per family')
    pr(f'  {"family":<11}{"diags":>7}{"samples":>8}{"dedup":>7}{"probs":>6}{"sec":>7}{"child":>7}'
       f'{"either":>7}{"before":>7}{"gap":>5}  codes')
    for f, v in t3.items():
        pr(f'  {f:<11}{v["diags"]:>7}{v["samples"]:>8}{v["samples_dedup"]:>7}{v["problems"]:>6}'
           f'{v["secondary_share"]:>7}{v["child_share"]:>7}{v["either_share"]:>7}'
           f'{str(v["cause_before_primary_share"]):>7}{str(v["median_line_gap"]):>5}  {v["codes"]}')

    # ---------------------------------------------------------------- T4
    comp = {k for k in samples if C.truthy(labels[k]['compiles'])}
    t4 = {'samples': len(samples), 'unique': sum(not v for v in dup.values()),
          'compiling': len(comp), 'compiling_unique': sum(not dup[k] for k in comp),
          'failing': len(samples) - len(comp), 'failing_unique': sum(not dup[k] for k in samples if k not in comp),
          'passing': sum(C.truthy(labels[k]['passes']) for k in samples),
          'passing_unique': sum(C.truthy(labels[k]['passes']) and not dup[k] for k in samples)}
    for f in BCK:
        ks = {r['k'] for r in lib if r['fam'] == f}
        t4[f'{f}_samples'] = len(ks)
        t4[f'{f}_unique'] = sum(not dup[k] for k in ks)
    t4['dup_rate'] = round(1 - t4['unique'] / t4['samples'], 3)
    out['T4'] = t4
    pr('\nT4 duplicates (whitespace removed, within a problem):', t4)

    # ---------------------------------------------------------------- T5
    t5 = {}
    for f in BCK + ('any_bck',):
        fs = BCK if f == 'any_bck' else (f,)
        ks = {k for k, fams in per_sample.items() if any(x in fs for x in fams)}
        ku = {k for k in ks if not dup[k]}
        per_task = Counter(k[0] for k in ku)
        t5[f] = {'samples': len(ks), 'samples_dedup': len(ku), 'problems': len(per_task),
                 'per_problem_dedup_quartiles': (statistics.quantiles(per_task.values(), n=4)
                                                 if len(per_task) > 1 else None),
                 'top_problems': per_task.most_common(10)}
    failing = [k for k in samples if k not in comp]
    masked = [k for k in failing if any(x in ('syntax', 'resolve', 'type', 'pattern') for x in per_sample[k])]
    pure = [k for k in failing if per_sample[k] and all(x in BCK + ('lifetime',) for x in per_sample[k])]
    t5['failing'] = len(failing)
    t5['failing_with_syntax_resolve_type_error'] = len(masked)
    t5['failing_only_borrowck_errors'] = len(pure)
    t5['failing_only_borrowck_errors_dedup'] = sum(not dup[k] for k in pure)
    t5['bck_alongside_type_error'] = sum(1 for k in failing if any(x in BCK for x in per_sample[k])
                                         and any(x in ('resolve', 'type', 'pattern') for x in per_sample[k]))
    mut_recs = [r for r in lib if r['fam'] == 'mutability']
    t5['mutability_decl_site'] = dict(Counter(r['decl'] or 'no suggestion' for r in mut_recs))
    t5['mutability_decl_site_samples_dedup'] = {
        site: len({r['k'] for r in mut_recs if (r['decl'] or 'no suggestion') == site and not r['dup']})
        for site in t5['mutability_decl_site']}
    if 'phase' in next(iter(labels.values())):
        t5['any_bck_by_phase_dedup'] = dict(Counter(
            labels[k]['phase'] for k, fams in per_sample.items() if any(x in BCK for x in fams) and not dup[k]))
        t5['samples_by_phase'] = dict(Counter(labels[k]['phase'] for k in samples))
    out['T5'] = t5
    pr('\nT5 natural borrow-check errors (lib mode)')
    pr('  mutability errors by the site rustc suggests making mutable (diagnostics):',
       t5['mutability_decl_site'], '| samples, dedup:', t5['mutability_decl_site_samples_dedup'])
    if 'any_bck_by_phase_dedup' in t5:
        pr('  any borrow-check error, dedup samples by phase:', t5['any_bck_by_phase_dedup'],
           '| samples by phase:', t5['samples_by_phase'])
    for f in BCK + ('any_bck',):
        v = t5[f]
        pr(f'  {f:<11} samples {v["samples"]} (dedup {v["samples_dedup"]}) over {v["problems"]} problems; '
           f'dedup per problem quartiles {v["per_problem_dedup_quartiles"]}; top {v["top_problems"][:5]}')
    pr(f'  failing samples {len(failing)}: with a syntax/resolve/type/pattern error {len(masked)} '
       f'(borrow checking possibly masked); only borrow-check errors {len(pure)} '
       f'(dedup {t5["failing_only_borrowck_errors_dedup"]}); borrow-check error beside a type error '
       f'{t5["bck_alongside_type_error"]}')

    # ---------------------------------------------------------------- T6
    um = Counter()
    for k in comp:
        n = sum(1 for d in diag[k]['lib']['diags'] if d['level'] == 'warning' and d['code'] == 'unused_mut')
        if n:
            um[k] = n
    out['T6'] = {'compiling': len(comp), 'with_unused_mut': len(um),
                 'with_unused_mut_dedup': sum(not dup[k] for k in um),
                 'problems': len({k[0] for k in um}), 'total_warnings': sum(um.values()),
                 'passing_with_unused_mut_dedup': sum(not dup[k] and C.truthy(labels[k]['passes']) for k in um)}
    pr('\nT6 unused_mut in compiling samples:', out['T6'])

    # ---------------------------------------------------------------- T7
    t7 = {}
    for f in BCK:
        rel = [r['rel'] for r in lib if r['fam'] == f and r['rel'] is not None]
        t7[f] = [sum(1 for x in rel if i / 5 <= x < (i + 1) / 5 or (i == 4 and x >= 1)) for i in range(5)]
    out['T7'] = t7
    pr('\nT7 primary span in the body, by fifth of the body:', t7)

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    with open(OUT.replace('.json', '.log'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log) + '\n')
    print('\n'.join(log))


if __name__ == '__main__':
    main()
