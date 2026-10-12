"""Phase 6, step 3: the pairs (R, C), the caps, and the decision to draw another
round. PREREGISTRATION_OWNFIX.md §2.

R: a stable repair whose extra-input verification (p6_1b) is True.
C, strict: output-identical to S on the asserts and on a live site (p6_1).
C, fallback (only if the strict pairs miss the gate after the last round): any
   live mutant that fails with S's FAIL/PANIC line. The fallback is a second
   pass after the strict selection, over the sources left without a pair; it
   adds pairs and never displaces a strict one, and within it an
   output-identical control comes first whatever its tier.

Candidates per source: every (R, C) with the same number of characters,
|code log-probability R - C| <= CALIPER = 1.0 nat, and a tier: 0, C at R's
site with another replacement; 1, C makes R's replacement (same from -> to)
elsewhere; 2, C's edit in R's family with the same change in length, elsewhere;
3 (fallback only), any family. Distance d = |d code lp| + |d site entropy|
+ |d original-token lp| + |d tokens to the end| / 10, all R minus C. Order:
(fallback: output-identical first), tier, d, R's position and new token, C's
position and new token -- deterministic.

Per problem, sources in order of their best candidate (then sample); each takes
its first candidate none of whose keys is already used in the problem: the pair
key (R's from -> to, R's edited line with whitespace removed), R's code and C's
code (whitespace removed). The codes of Phase 4's and Phase 5's graph members
for that problem are used from the start. At most CAP_PER_PROBLEM = 5 pairs per
problem, shared by both passes. More than MAX_PAIRS = 300 in all: the strict
pairs are kept first, and the rest drawn with default_rng(SEED).

Every run writes OUT_STATUS (round, rule, final, counts, and the sha256 of the
rows files); run_ownfix.sh builds graphs only from final, unchanged rows.

Exit codes for run_ownfix.sh: 0 pairs final (>= TARGET = 250, or ROUND =
MAX_ROUND = 8 and >= GATE = 150 pairs over >= 50 problems); 3 draw round
ROUND + 1; 2 gate not met after the last round, even with the fallback.

Ids: problem * 100000 + sample * 10 + 0 (R), 1 (C), 2 (S, the descriptive tail).
  ROUND=0..8  SOURCES MUTANTS LOGPROB VERIFY OUT_PAIRS=ownfix_pairs.json
  OUT_ROWS=ownfix_rows.jsonl  OUT_S_ROWS=ownfix_s_rows.jsonl
"""
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m1_make_mutants import lex  # noqa: E402

ROUND = int(os.environ['ROUND'])
SOURCES = os.environ.get('SOURCES', 'ownfix_sources.jsonl')
MUTANTS = os.environ.get('MUTANTS', 'ownfix_mutants.jsonl')
LOGPROB = os.environ.get('LOGPROB', 'ownfix_logprob.jsonl')
VERIFY = os.environ.get('VERIFY', 'ownfix_verify.json')
OUT_PAIRS = os.environ.get('OUT_PAIRS', 'ownfix_pairs.json')
OUT_ROWS = os.environ.get('OUT_ROWS', 'ownfix_rows.jsonl')
OUT_S_ROWS = os.environ.get('OUT_S_ROWS', 'ownfix_s_rows.jsonl')
OUT_STATUS = os.environ.get('OUT_STATUS', 'ownfix_pairs_status.json')
CALIPER, CAP_PER_PROBLEM, MAX_PAIRS = 1.0, 5, 300
TARGET, GATE, GATE_PROBLEMS, MAX_ROUND = 250, 150, 50, 8
SEED = 20261012
LP_KEYS = ('code_logprob', 'edit_logprob', 'min_logprob', 'mean_entropy', 'tokens_to_end',
           'site_entropy', 'orig_logprob', 's_top', 'own_top')
CONTEXTS = ('if', 'while', 'for', 'match', 'return', 'let')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def norm(code):
    return re.sub(r'\s+', '', code)


def mkey(m):
    return (m['task_id'], m['sample'], m['pos'], m['to'])


def context(code, pos):
    """The construct an edit sits in: walking back from the edited token at its
    own brace depth, the first of if/while/for/match/return/let, else 'other'
    (a ';' or the start of the enclosing block comes first)."""
    toks = lex(code)
    i = next(k for k, (_, _, s) in enumerate(toks) if s == pos)
    depth = 0
    for kind, t, _ in reversed(toks[:i]):
        if kind == 'punct' and t == '}':
            depth += 1
        elif kind == 'punct' and t == '{':
            if depth == 0:
                return 'other'
            depth -= 1
        elif depth == 0 and kind == 'punct' and t == ';':
            return 'other'
        elif depth == 0 and kind == 'ident' and t in CONTEXTS:
            return t
    return 'other'


def tier(r, c, fallback):
    if c['pos'] == r['pos']:
        return 0
    if (c['from'], c['to']) == (r['from'], r['to']):
        return 1
    if c['family'] == r['family'] and len(c['to']) - len(c['from']) == len(r['to']) - len(r['from']):
        return 2
    return 3 if fallback else None


def dist(r, c):
    a, b = r['lp'], c['lp']
    return (abs(a['code_logprob'] - b['code_logprob']) + abs(a['site_entropy'] - b['site_entropy'])
            + abs(a['orig_logprob'] - b['orig_logprob'])
            + abs(a['tokens_to_end'] - b['tokens_to_end']) / 10)


def edited_line(m):
    code = m['code']
    start = code.rfind('\n', 0, m['pos']) + 1
    end = code.find('\n', m['pos'])
    return norm(code[start:None if end == -1 else end])


def candidates(sources, by_src, verified, strict):
    """Per problem, [(source order key, sample, sorted candidates)]. strict: the
    strict candidates (output-identical, live C at tier 0-2); otherwise the
    fallback's own (every other live C failing like S, tier 0-3), with an
    output-identical control first whatever its tier."""
    per_problem = {}
    for key, ms in sorted(by_src.items()):
        s = sources[key]
        rs = [m for m in ms if m['is_r'] and verified.get(mkey(m)) is True]
        cs = [m for m in ms if m['compiles'] and m['passes'] is False and m.get('error') == s['error']
              and m['live']]
        cands = []
        for r in rs:
            for c in cs:
                if len(c['code']) != len(r['code']):
                    continue
                if abs(r['lp']['code_logprob'] - c['lp']['code_logprob']) > CALIPER:
                    continue
                t = tier(r, c, not strict)
                if t is None:
                    continue
                is_strict = c['is_c'] and t <= 2
                if is_strict != strict:
                    continue
                cands.append(((0 if c['is_c'] else 1, t, dist(r, c), r['pos'], r['to'], c['pos'], c['to']), r, c))
        if cands:
            cands.sort(key=lambda x: x[0])
            per_problem.setdefault(key[0], []).append((cands[0][0][:3] + (key[1],), key[1], cands))
    return per_problem


def select(per_problem, used, took, k, pairs, skip=frozenset()):
    """Greedy, problem by problem: each source in order takes its first candidate
    none of whose keys is used yet in the problem, until the problem holds
    CAP_PER_PROBLEM pairs."""
    for tid in sorted(per_problem):
        u = used.setdefault(tid, set())
        for _, smp, cands in sorted(per_problem[tid], key=lambda x: x[0]):
            if took.get(tid, 0) == CAP_PER_PROBLEM:
                break
            if (tid, smp) in skip:
                continue
            for order, r, c in cands:
                keys = {('pair', r['from'], r['to'], edited_line(r)), ('code', norm(r['code'])),
                        ('code', norm(c['code']))}
                if keys & u:
                    continue
                u |= keys
                pairs.append({'problem': tid, 'sample': smp, 'k': k, 'tier': order[1],
                              'dist': order[2], 'r': r, 'c': c})
                took[tid] = took.get(tid, 0) + 1
                break


def build(sources, by_src, verified, used0, fallback):
    """The strict selection; with fallback, a second pass that adds pairs for the
    sources left without one, sharing the used keys and the per-problem cap, so
    that no strict pair is ever displaced."""
    used = {t: set(v) for t, v in used0.items()}
    took, pairs = {}, []
    select(candidates(sources, by_src, verified, True), used, took, 0, pairs)
    if fallback:
        paired = {(p['problem'], p['sample']) for p in pairs}
        select(candidates(sources, by_src, verified, False), used, took, 1, pairs, skip=paired)
    n_found = len(pairs)
    if n_found > MAX_PAIRS:
        strict = [p for p in pairs if p['k'] == 0]
        rest = [p for p in pairs if p['k'] == 1]
        rng = np.random.default_rng(SEED)
        if len(strict) >= MAX_PAIRS:
            keep = [strict[i] for i in sorted(rng.choice(len(strict), MAX_PAIRS, replace=False))]
        else:
            keep = strict + [rest[i] for i in sorted(rng.choice(len(rest), MAX_PAIRS - len(strict), replace=False))]
        pairs = sorted(keep, key=lambda p: (p['problem'], p['sample']))
    return pairs, n_found


def sha256(path):
    import hashlib
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    if os.path.exists(OUT_STATUS):
        os.remove(OUT_STATUS)
    sources = {(s['task_id'], s['sample']): s for s in read_jsonl(SOURCES) if s['stable']}
    lps = {mkey(r): r for r in read_jsonl(LOGPROB)}
    ver = json.load(open(VERIFY, encoding='utf-8')) if os.path.exists(VERIFY) else {'mutants': {}}
    verified = {}
    for k, v in ver['mutants'].items():
        t, s, pos, to = k.split('|', 3)                 # the new token may itself be '||'
        verified[(int(t), int(s), int(pos), to)] = v
    by_src, all_r = {}, []
    for m in read_jsonl(MUTANTS):
        if (m['task_id'], m['sample']) not in sources:
            continue
        if m.get('is_r'):
            all_r.append(m)
        if mkey(m) in lps and all(lps[mkey(m)][k] is not None for k in LP_KEYS):
            m['lp'] = lps[mkey(m)]
            by_src.setdefault((m['task_id'], m['sample']), []).append(m)
    used0 = {}
    for f in ('mutant_rows.jsonl', 'matched_rows.jsonl'):        # Phase 4 and 5 graph members
        if os.path.exists(f):
            for r in read_jsonl(f):
                used0.setdefault(r['problem'], set()).add(('code', norm(r['generated_code'])))
    # Gemma's own samples, to flag a pair member that Gemma itself wrote
    gemma_pass, gemma_any = {}, {}
    for sf in ['within_samples.jsonl'] + sorted(glob.glob('matched_samples_r*.jsonl')) + sorted(glob.glob('ownfix_samples_r*.jsonl')):
        lf = sf.replace('_samples', '_labels')
        if not os.path.exists(lf):
            continue
        labels = {(r['task_id'], r['sample']): r for r in read_jsonl(lf)}
        for s in read_jsonl(sf):
            n = norm(s['generated_code'])
            gemma_any.setdefault(s['task_id'], set()).add(n)
            if labels[(s['task_id'], s['sample'])]['passes']:
                gemma_pass.setdefault(s['task_id'], set()).add(n)

    pairs, n_found = build(sources, by_src, verified, used0, False)
    rule = 'strict'
    n_prob = len({p['problem'] for p in pairs})
    final = len(pairs) >= TARGET or (ROUND >= MAX_ROUND and len(pairs) >= GATE and n_prob >= GATE_PROBLEMS)
    if ROUND >= MAX_ROUND and not final:
        pairs, n_found = build(sources, by_src, verified, used0, True)
        rule = 'fallback'
        n_prob = len({p['problem'] for p in pairs})
        final = len(pairs) >= GATE and n_prob >= GATE_PROBLEMS

    vr = Counter(verified.get(mkey(m)) for m in all_r)
    fam_all = Counter(m['family'] for m in all_r if verified.get(mkey(m)) is True)
    print(f"round {ROUND} ({rule}): {len(sources)} stable sources; stable repairs {len(all_r)}: verified "
          f"{vr[True]}, failed the extra inputs {vr[False]}, unverifiable {vr[None]}; {n_found} pairs after "
          f"the per-problem cap" + (f", {len(pairs)} kept" if n_found > len(pairs) else "") + f"; {n_prob} problems")
    print("  verified repairs by family (all sources): " + ", ".join(f"{k} {v}" for k, v in fam_all.most_common()))
    if pairs:
        d = np.array([p['r']['lp']['code_logprob'] - p['c']['lp']['code_logprob'] for p in pairs])
        per = Counter(p['problem'] for p in pairs)
        print("  pairs per problem: " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(per.values()).items())))
        print("  tiers: " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(p['tier'] for p in pairs).items()))
              + f"; strict {sum(p['k'] == 0 for p in pairs)}")
        print(f"  |R - C| code log-probability: median {np.median(np.abs(d)):.3f}; R less probable in "
              f"{np.mean(d < 0):.3f}; beliefs (S's token was the model's top choice at R's site) "
              f"{np.mean([p['r']['lp']['s_top'] for p in pairs]):.3f}")
        print("  pair R families: " + ", ".join(f"{k} {v}" for k, v in Counter(p['r']['family'] for p in pairs).most_common()))

    with open(OUT_PAIRS, 'w', encoding='utf-8') as f:
        json.dump([{'problem': p['problem'], 'source_sample': p['sample'],
                    'rule': 'strict' if p['k'] == 0 else 'fallback', 'tier': p['tier'], 'dist': p['dist'],
                    'r_id': p['problem'] * 100000 + p['sample'] * 10,
                    'c_id': p['problem'] * 100000 + p['sample'] * 10 + 1,
                    's_id': p['problem'] * 100000 + p['sample'] * 10 + 2,
                    'tokens': p['r']['tokens'], 's_error': sources[(p['problem'], p['sample'])]['error'],
                    's_n_pass': sources[(p['problem'], p['sample'])]['n_pass'],
                    'c_output_identical': p['c']['output_identical'], 'c_diff': p['c'].get('diff'),
                    'c_extra_identical': ver.get('controls', {}).get(
                        f"{p['c']['task_id']}|{p['c']['sample']}|{p['c']['pos']}|{p['c']['to']}"),
                    'r_doubly_natural': norm(p['r']['code']) in gemma_pass.get(p['problem'], set()),
                    'c_natural': norm(p['c']['code']) in gemma_any.get(p['problem'], set()),
                    'r_edit': {k: p['r'][k] for k in ('pos', 'line', 'from', 'to', 'family')},
                    'c_edit': {k: p['c'][k] for k in ('pos', 'line', 'from', 'to', 'family')}}
                   for p in pairs], f, indent=1)
    with open(OUT_ROWS, 'w', encoding='utf-8') as f, open(OUT_S_ROWS, 'w', encoding='utf-8') as g:
        for p in pairs:
            s = sources[(p['problem'], p['sample'])]
            base = p['problem'] * 100000 + p['sample'] * 10
            for m, lab, member in ((p['r'], 1, 0), (p['c'], 0, 1)):
                f.write(json.dumps({
                    'task_id': base + member, 'text': s['prose'] + ' ' + m['code'], 'prose': s['prose'],
                    'generated_code': m['code'], 'label': lab, 'problem': p['problem'],
                    'sample': p['sample'], 'tokens': m['tokens'], 'tier': p['tier'],
                    'rule': 'strict' if p['k'] == 0 else 'fallback',
                    'output_identical': m.get('output_identical'),
                    'edit': {k: m[k] for k in ('pos', 'line', 'from', 'to', 'family')},
                    'context': context(m['code'], m['pos']),
                    'surprise': {k: m['lp'][k] for k in LP_KEYS}}, ensure_ascii=False) + '\n')
            g.write(json.dumps({'task_id': base + 2, 'text': s['prose'] + ' ' + s['code'], 'prose': s['prose'],
                                'generated_code': s['code'], 'label': 0, 'problem': p['problem'],
                                'sample': p['sample'], 'tokens': s['tokens']}, ensure_ascii=False) + '\n')
    print(f"wrote {OUT_PAIRS}, {OUT_ROWS} ({2 * len(pairs)} rows) and {OUT_S_ROWS}")
    with open(OUT_STATUS + '.tmp', 'w', encoding='utf-8') as f:
        json.dump({'round': ROUND, 'rule': rule, 'final': bool(final), 'n_pairs': len(pairs),
                   'n_problems': n_prob, 'rows_sha256': sha256(OUT_ROWS),
                   's_rows_sha256': sha256(OUT_S_ROWS), 'pairs_sha256': sha256(OUT_PAIRS)}, f, indent=1)
    os.replace(OUT_STATUS + '.tmp', OUT_STATUS)

    if final:
        print(f"PAIRS FINAL ({rule}): {len(pairs)} over {n_prob} problems, round {ROUND}")
        sys.exit(0)
    if ROUND < MAX_ROUND:
        print(f"BELOW TARGET: {len(pairs)} < {TARGET}; draw round {ROUND + 1}")
        sys.exit(3)
    print(f"GATE NOT MET: {len(pairs)} pairs over {n_prob} problems after round {ROUND}, even with the fallback")
    sys.exit(2)


if __name__ == '__main__':
    main()
