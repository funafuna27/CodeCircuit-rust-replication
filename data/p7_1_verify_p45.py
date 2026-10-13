"""Phase 7, step 1: are Phase 4's and Phase 5's survivors correct beyond MBPP's
asserts? PREREGISTRATION_CLOSING.md §2 (verification) and §3 (gate G4).

Phase 6's extra inputs and judge (p6_1b_verify, imported, unchanged): each
assert's arguments varied one at a time, MBPP's Python reference for the
expected values, and a calibration that keeps an input only if the problem's
passing Gemma samples (whitespace-distinct, sample order) all return the
reference's value on it.

  rule A   Phase 6's: the first CALIB = 10 passing samples calibrate. Gate G4:
           on the problems Phase 6 verified, its kept-input counts must be
           reproduced (exit 3 otherwise). Its survivor verdicts are descriptive.
  rule B   the verdicts: the pair's source and survivor (whitespace removed) are
           left out of the calibration samples, which would otherwise verify the
           source by construction and pass its blind spots to the survivor.
  verdict  verified (right on every kept input), lucky (a wrong value or a panic
           on one; the inputs are recorded), error (no PASS line, a timeout or an
           abort), unverifiable (fewer than 10 candidate inputs, no calibration
           sample that compiles and finishes, or fewer than 10 kept inputs).
           The source's verdict comes from its own run on all candidate inputs,
           restricted to the pair's kept inputs.
  descriptive  whether the survivor returns the source's values on the kept
           inputs (p6_1's differential harness)

Gemma's Rust is compiled and run only in Pool workers, each under an 8 GiB
address-space limit (m1._init), the judge's time limits, with an empty
temporary directory as its working directory. MBPP's references run in the same
workers (so under the same limit): python -I, their own temporary directory,
20 s. No label of the main analysis changes.

  WORKERS=6  OUT=results/closing/verify_p45.json  PROBLEMS=t1,t2 (tests only)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m1_make_mutants as m1  # noqa: E402
import p6_1b_verify as v  # noqa: E402
from p6_1_repairs import _diff, load_samples, norm  # noqa: E402

WORKERS = int(os.environ.get('WORKERS', 6))
OUT = os.environ.get('OUT', os.path.join(os.environ.get('OUT_DIR', 'results/closing'), 'verify_p45.json'))
ONLY = {int(t) for t in os.environ.get('PROBLEMS', '').split(',') if t}
P6_VERIFY = 'ownfix_verify.json'


def _init(parent):
    m1._init()
    os.chdir(tempfile.mkdtemp(prefix='cwd_', dir=parent))


def _ref(args):
    task, sig, inputs = args
    return v.reference_values(task, sig, inputs)


def _judge(args):
    return v._judge_cases(args)


def kept(cand, fsets):
    """cand minus the union of the failure sets of the samples that ran (None
    = did not compile or finish: left out, as p6_1b does); None if none ran."""
    ran = [f for f in fsets if f is not None]
    if not ran:
        return None
    bad = set().union(*ran)
    return [i for i in range(len(cand)) if i not in bad]


def main():
    t0 = time.time()
    if not os.path.exists(m1.judge_mod.RUSTC):
        sys.exit(f"rustc not found at {m1.judge_mod.RUSTC}")
    rustc = subprocess.run([m1.judge_mod.RUSTC, '--version'], capture_output=True, text=True).stdout.strip()
    from datasets import load_dataset
    ds = load_dataset('mbpp', 'sanitized')
    tasks = {int(t['task_id']): t for s in ('train', 'test', 'validation', 'prompt') for t in ds[s]}
    typeable, _ = m1.load_typeable()
    samples = load_samples()
    passing = {}
    for (t, smp), (s, lab) in sorted(samples.items()):
        if lab['passes'] and not s['truncated']:
            passing.setdefault(t, {}).setdefault(norm(s['generated_code']), s['generated_code'])

    rows5 = {r['task_id']: r for r in m1.read_jsonl('matched_rows.jsonl')}
    src5 = {(r['task_id'], r['sample']): r for r in m1.read_jsonl('matched_sources.jsonl')}
    rows4 = {r['task_id']: r for r in m1.read_jsonl('mutant_rows.jsonl')}
    feats4 = {json.loads(l)['expr_id'] for l in open('gemma_mutant_features.jsonl', encoding='utf-8') if l.strip()}
    targets = []
    for p in json.load(open('matched_pairs.json', encoding='utf-8')):
        targets.append({'phase': 5, 'problem': p['problem'], 'source_sample': p['source_sample'],
                        'pass_id': p['pass_id'], 'fail_id': p['fail_id'], 'has_graphs': True,
                        'survivor': rows5[p['pass_id']]['generated_code'],
                        'source': src5[(p['problem'], p['source_sample'])]['code']})
    for p in json.load(open('mutant_pairs.json', encoding='utf-8')):
        targets.append({'phase': 4, 'problem': p['problem'], 'source_sample': p['source_sample'],
                        'pass_id': p['pass_id'], 'fail_id': p['fail_id'],
                        'has_graphs': p['pass_id'] in feats4 and p['fail_id'] in feats4,
                        'survivor': rows4[p['pass_id']]['generated_code'],
                        'source': samples[(p['problem'], p['source_sample'])][0]['generated_code']})
    if ONLY:
        targets = [x for x in targets if x['problem'] in ONLY]
    problems = sorted({x['problem'] for x in targets})
    print(f"{len(targets)} pairs over {len(problems)} problems; {rustc}; {WORKERS} workers", flush=True)

    scratch = tempfile.mkdtemp(prefix='p7verify_')
    with Pool(WORKERS, initializer=_init, initargs=(scratch,)) as pool:
        # references, in the limited workers
        refs = pool.map(_ref, [(tasks[t], typeable[t], v.inputs_for(typeable[t])) for t in problems])
        cand = {}
        for t, ref in zip(problems, refs):
            is_close = any(c[2] for c in typeable[t]['cases'])
            cand[t] = [(a, val, is_close) for a, val in ref]
        # calibration sets, then every code that needs a failure set on all candidates
        pool_t = {t: list(passing.get(t, {}).values()) for t in problems}
        for x in targets:
            t = x['problem']
            out = {norm(x['source']), norm(x['survivor'])}
            x['calibB'] = [c for c in pool_t[t] if norm(c) not in out][:v.CALIB]
            x['src_in_calibA'] = norm(x['source']) in {norm(c) for c in pool_t[t][:v.CALIB]}
            x['surv_equals_passing'] = norm(x['survivor']) in passing.get(t, {})
        need = {}
        for t in problems:                           # rule A runs as Phase 6 ran it, on every problem
            for c in pool_t[t][:v.CALIB]:
                need[(t, c)] = None
        for x in targets:
            t = x['problem']
            if len(cand[t]) < v.MIN_INPUTS:
                continue
            for c in x['calibB'] + [x['source']]:
                need[(t, c)] = None
        keys = list(need)
        fs = pool.map(_judge, [(t, c, cand[t]) for t, c in keys], chunksize=2)
        fail = dict(zip(keys, fs))
        print(f"  references and {len(keys)} calibration/source runs: {time.time() - t0:.0f} s", flush=True)

        # rule A per problem (gate G4) and rule B per pair
        keptA = {}
        for t in problems:
            calib = pool_t[t][:v.CALIB]
            k = kept(cand[t], [fail[(t, c)] for c in calib]) if calib else None
            keptA[t] = k if k is not None else []
        jobs, jobs_a = [], []
        for n, x in enumerate(targets):
            t = x['problem']
            x['n_cand'] = len(cand[t])
            if len(cand[t]) < v.MIN_INPUTS:
                x['keptB'] = None
                x['why_unverifiable'] = 'fewer than 10 candidate inputs'
                continue
            fsB = [fail[(t, c)] for c in x['calibB']]
            x['n_calibB'], x['n_calibB_none'] = len(fsB), sum(f is None for f in fsB)
            kB = kept(cand[t], fsB) if fsB else None
            x['keptB'] = kB
            if kB is None:
                x['why_unverifiable'] = 'no calibration sample left, or none compiles and finishes'
            elif len(kB) < v.MIN_INPUTS:
                x['why_unverifiable'] = 'fewer than 10 kept inputs'
            else:
                jobs.append((n, (t, x['survivor'], [cand[t][i] for i in kB])))
            if len(keptA[t]) >= v.MIN_INPUTS:
                jobs_a.append((n, (t, x['survivor'], [cand[t][i] for i in keptA[t]])))
        res_b = pool.map(_judge, [j for _, j in jobs], chunksize=2)
        res_a = pool.map(_judge, [j for _, j in jobs_a], chunksize=2)
        diffs = pool.map(_diff, [(targets[n]['problem'], targets[n]['source'], targets[n]['survivor'],
                                  [cand[targets[n]['problem']][i][0] for i in targets[n]['keptB']])
                                 for n, _ in jobs], chunksize=2)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"  survivors judged: {time.time() - t0:.0f} s", flush=True)

    def verdict(f):
        return 'error' if f is None else ('verified' if not f else 'lucky')

    for x in targets:
        x['survivor_verdict'] = 'unverifiable'
        x['source_verdict'] = 'unverifiable'
        x['survivor_verdict_ruleA'] = 'unverifiable'
    for (n, (t, _, cases)), f, d in zip(jobs, res_b, diffs):
        x = targets[n]
        x['survivor_verdict'] = verdict(f)
        x['survivor_fail_args'] = [repr(cases[i][0]) for i in sorted(f)] if f else []
        x['equivalent_on_kept'] = d
        fsrc = fail[(t, x['source'])]
        x['source_verdict'] = 'error' if fsrc is None else (
            'lucky' if set(fsrc) & set(x['keptB']) else 'verified')
    for (n, _), f in zip(jobs_a, res_a):
        targets[n]['survivor_verdict_ruleA'] = verdict(f)

    # gate G4: rule A against Phase 6
    p6 = json.load(open(P6_VERIFY, encoding='utf-8'))['problems']
    g4 = {'compared': 0, 'count_mismatch': [], 'exact_cases_match': 0}
    for t in problems:
        if str(t) not in p6:
            continue
        g4['compared'] += 1
        mine = [cand[t][i] for i in keptA[t]]
        if len(mine) != p6[str(t)]['n_inputs']:
            g4['count_mismatch'].append((t, len(mine), p6[str(t)]['n_inputs']))
        g4['exact_cases_match'] += repr(mine) == p6[str(t)]['cases']
    g4['passed'] = not g4['count_mismatch']

    summ = {}
    for ph in (5, 4):
        xs = [x for x in targets if x['phase'] == ph and x['has_graphs']]
        if not xs:
            continue
        cnt = lambda key: {k: sum(x[key] == k for x in xs) for k in ('verified', 'lucky', 'error', 'unverifiable')}
        table = {f'{a}|{b}': sum(x['source_verdict'] == a and x['survivor_verdict'] == b for x in xs)
                 for a in ('verified', 'lucky', 'error', 'unverifiable')
                 for b in ('verified', 'lucky', 'error', 'unverifiable')}
        summ[ph] = {'pairs': len(xs), 'problems': len({x['problem'] for x in xs}),
                    'survivor': cnt('survivor_verdict'), 'source': cnt('source_verdict'),
                    'survivor_ruleA': cnt('survivor_verdict_ruleA'), 'source_x_survivor': {k: n for k, n in table.items() if n},
                    'equivalent_on_kept': {k: sum(x.get('equivalent_on_kept') == k for x in xs)
                                           for k in sorted({str(x.get('equivalent_on_kept')) for x in xs})},
                    'why_unverifiable': {k: sum(x.get('why_unverifiable') == k for x in xs)
                                         for k in sorted({x['why_unverifiable'] for x in xs if 'why_unverifiable' in x})}}
        print(f"  Phase {ph}: {summ[ph]['pairs']} pairs: survivors {summ[ph]['survivor']}; sources "
              f"{summ[ph]['source']}; rule A survivors {summ[ph]['survivor_ruleA']}", flush=True)
    out = {'rustc': rustc, 'workers': WORKERS, 'seconds': time.time() - t0, 'only': sorted(ONLY),
           'gate_G4': g4, 'summary': summ,
           'problems': {str(t): {'n_cand': len(cand[t]), 'n_pool': len(pool_t[t]),
                                 'calibA': [norm(c) for c in pool_t[t][:v.CALIB]],
                                 'n_keptA': len(keptA[t])} for t in problems},
           'pairs': [{k: x.get(k) for k in ('phase', 'problem', 'source_sample', 'pass_id', 'fail_id', 'has_graphs',
                                             'n_cand', 'n_calibB', 'n_calibB_none', 'keptB', 'why_unverifiable',
                                             'survivor_verdict', 'survivor_fail_args', 'source_verdict',
                                             'survivor_verdict_ruleA', 'equivalent_on_kept', 'src_in_calibA',
                                             'surv_equals_passing')} for x in targets]}
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    with open(OUT + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    os.replace(OUT + '.tmp', OUT)
    print(f"G4: rule A compared on {g4['compared']} problems; count mismatches {g4['count_mismatch']}; "
          f"identical kept inputs {g4['exact_cases_match']}; {'passed' if g4['passed'] else 'FAILED'}; "
          f"{time.time() - t0:.0f} s", flush=True)
    sys.exit(0 if g4['passed'] else 3)


if __name__ == '__main__':
    main()
