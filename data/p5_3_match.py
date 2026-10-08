"""Phase 5, step 3: surprise-matched pairs, the cap, and the decision to draw
another round. PREREGISTRATION_MATCHED.md §2.

  candidates  per usable source, every (survivor, killed mutant) whose code
              log-probabilities differ by at most CALIPER = 1.0 nat. If any
              candidate has both edits in the same family, only those are kept.
  order       |difference|, then the survivor's position, the killed mutant's
              position, and the replacement tokens -- deterministic.
  per problem sources in order of their best candidate (then sample index); each
              takes its first candidate whose two codes are not already used in
              that problem -- neither by an earlier pair nor by a Phase 4 pair
              member -- until the problem has CAP_PER_PROBLEM = 5 pairs.
  overall     more than MAX_PAIRS = 300: 300 drawn at random, default_rng(SEED).

Exit code, read by run_matched.sh: 0 when the pairs are final -- at least TARGET
= 250, or ROUND = 4 and at least GATE = 150; 3 when fewer than TARGET and ROUND
< 4 (draw the next round); 2 when ROUND = 4 and fewer than GATE (gate not met).

Writes OUT_PAIRS and the graph pipeline's DATA_FILE (OUT_ROWS): one row per
member, id problem * 10000 + sample * 10, + 0 for the survivor and + 1 for the
killed mutant, survivor first.

  ROUND=0..4  SOURCES=matched_sources.jsonl  MUTANTS=matched_mutants.jsonl
  LOGPROB=matched_logprob.jsonl  OUT_PAIRS=matched_pairs.json  OUT_ROWS=matched_rows.jsonl
"""
import json
import os
import sys
from collections import Counter

import numpy as np

ROUND = int(os.environ['ROUND'])
SOURCES = os.environ.get('SOURCES', 'matched_sources.jsonl')
MUTANTS = os.environ.get('MUTANTS', 'matched_mutants.jsonl')
LOGPROB = os.environ.get('LOGPROB', 'matched_logprob.jsonl')
OUT_PAIRS = os.environ.get('OUT_PAIRS', 'matched_pairs.json')
OUT_ROWS = os.environ.get('OUT_ROWS', 'matched_rows.jsonl')
CALIPER, CAP_PER_PROBLEM, TARGET, GATE, MAX_PAIRS, MAX_ROUND = 1.0, 5, 250, 150, 300, 4
SEED = 20261008


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def mkey(m):
    return (m['task_id'], m['sample'], m['pos'], m['to'])


def candidates(sv, kl):
    out = [(abs(s['lp']['code_logprob'] - k['lp']['code_logprob']), s, k)
           for s in sv for k in kl
           if abs(s['lp']['code_logprob'] - k['lp']['code_logprob']) <= CALIPER]
    same = [c for c in out if c[1]['family'] == c[2]['family']]
    out = same or out
    out.sort(key=lambda c: (c[0], c[1]['pos'], c[2]['pos'], c[1]['to'], c[2]['to']))
    return out


def main():
    sources = {(s['task_id'], s['sample']): s for s in read_jsonl(SOURCES)}
    lps = {mkey(r): r for r in read_jsonl(LOGPROB)}
    by_src = {}
    for m in read_jsonl(MUTANTS):
        if not m['compiles'] or m['passes'] is None or mkey(m) not in lps:
            continue
        m['lp'] = lps[mkey(m)]
        by_src.setdefault((m['task_id'], m['sample']), ([], []))[0 if m['passes'] else 1].append(m)
    used = {}
    if os.path.exists('mutant_rows.jsonl'):                       # Phase 4's pair members
        for r in read_jsonl('mutant_rows.jsonl'):
            used.setdefault(r['problem'], set()).add(r['generated_code'])

    per_problem = {}
    n_usable = n_within = 0
    for key, (sv, kl) in sorted(by_src.items()):
        if not sv or not kl:
            continue
        n_usable += 1
        c = candidates(sv, kl)
        if c:
            n_within += 1
            per_problem.setdefault(key[0], []).append((c[0][0], key[1], c))
    pairs = []
    for tid in sorted(per_problem):
        u = used.setdefault(tid, set())
        took = 0
        for best, smp, cands in sorted(per_problem[tid], key=lambda t: (t[0], t[1])):
            if took == CAP_PER_PROBLEM:
                break
            pick = next((c for c in cands if c[1]['code'] not in u and c[2]['code'] not in u), None)
            if pick is None:
                continue
            u.update((pick[1]['code'], pick[2]['code']))
            pairs.append({'problem': tid, 'sample': smp, 'delta': pick[1]['lp']['code_logprob']
                          - pick[2]['lp']['code_logprob'], 'survivor': pick[1], 'killed': pick[2]})
            took += 1
    n_found = len(pairs)
    if n_found > MAX_PAIRS:
        keep = np.random.default_rng(SEED).choice(n_found, MAX_PAIRS, replace=False)
        pairs = [pairs[i] for i in sorted(keep)]

    print(f"round {ROUND}: {len(sources)} sources, {n_usable} usable, {n_within} with a pair "
          f"within {CALIPER} nat; {n_found} pairs after the cap of {CAP_PER_PROBLEM} per problem"
          + (f", {len(pairs)} kept at random" if n_found > MAX_PAIRS else ""))
    if pairs:
        d = np.array([p['delta'] for p in pairs])
        per = Counter(p['problem'] for p in pairs)
        print(f"  {len(per)} problems; pairs per problem: "
              + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(per.values()).items())))
        print(f"  |survivor - killed| code log-probability: median {np.median(np.abs(d)):.3f}, "
              f"IQR {np.percentile(np.abs(d), 25):.3f}-{np.percentile(np.abs(d), 75):.3f}; "
              f"killed less probable in {np.mean(d > 0):.3f}")
        sf = np.mean([p['survivor']['family'] == p['killed']['family'] for p in pairs])
        ss = np.mean([p['survivor']['pos'] == p['killed']['pos'] for p in pairs])
        dch = np.array([len(p['survivor']['code']) - len(p['killed']['code']) for p in pairs])
        print(f"  same family {sf:.3f}, same site {ss:.3f}; chars survivor - killed mean "
              f"{dch.mean():+.3f}, nonzero in {int((dch != 0).sum())}")
        tp = np.array([p['survivor']['tokens'] for p in pairs])
        print(f"  tokens per member: median {np.median(tp):.0f}, range {tp.min()}..{tp.max()}")

    with open(OUT_PAIRS, 'w', encoding='utf-8') as f:
        json.dump([{'problem': p['problem'], 'source_sample': p['sample'],
                    'pass_id': p['problem'] * 10000 + p['sample'] * 10,
                    'fail_id': p['problem'] * 10000 + p['sample'] * 10 + 1,
                    'tokens': p['survivor']['tokens'], 'delta_logprob': p['delta'],
                    'survivor_edit': {k: p['survivor'][k] for k in ('pos', 'line', 'from', 'to', 'family')},
                    'killed_edit': {k: p['killed'][k] for k in ('pos', 'line', 'from', 'to', 'family')},
                    'killed_error': p['killed']['error'],
                    'same_family': p['survivor']['family'] == p['killed']['family']}
                   for p in pairs], f, indent=1)
    with open(OUT_ROWS, 'w', encoding='utf-8') as f:
        for p in pairs:
            prose = sources[(p['problem'], p['sample'])]['prose']
            for m, lab, member in ((p['survivor'], 1, 0), (p['killed'], 0, 1)):
                f.write(json.dumps({
                    'task_id': p['problem'] * 10000 + p['sample'] * 10 + member,
                    'text': prose + ' ' + m['code'], 'prose': prose, 'generated_code': m['code'],
                    'label': lab, 'problem': p['problem'], 'sample': p['sample'],
                    'tokens': m['tokens'],
                    'edit': {k: m[k] for k in ('pos', 'line', 'from', 'to', 'family')},
                    'surprise': {k: m['lp'][k] for k in
                                 ('code_logprob', 'edit_logprob', 'min_logprob', 'mean_entropy')},
                }, ensure_ascii=False) + '\n')
    print(f"wrote {OUT_PAIRS} and {OUT_ROWS} ({2 * len(pairs)} rows)")

    if len(pairs) >= TARGET or (ROUND >= MAX_ROUND and len(pairs) >= GATE):
        print(f"PAIRS FINAL: {len(pairs)} (target {TARGET}, gate {GATE}, round {ROUND})")
        sys.exit(0)
    if ROUND < MAX_ROUND:
        print(f"BELOW TARGET: {len(pairs)} < {TARGET}; draw round {ROUND + 1}")
        sys.exit(3)
    print(f"GATE NOT MET: {len(pairs)} < {GATE} after round {ROUND}")
    sys.exit(2)


if __name__ == '__main__':
    main()
