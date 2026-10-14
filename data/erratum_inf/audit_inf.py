"""inf audit, step 2 (CPU; writes only into this folder).
Every item a phase labelled passing in the 11 f64+isclose tasks is judged twice:
with the study's judge unchanged (must reproduce the stored label) and with the tolerant
branch made finite-aware (|v-e| <= 1e-3*max(|v|,|e|) only when v and e are both finite;
v == e unchanged). The bug can only create false passes, so failing items are not re-judged."""
import json, gzip, os, sys, collections, tempfile, resource, importlib.util
from pathlib import Path
from multiprocessing import Pool
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
D = Path('/home/user1/CodeCircuit-rust/data')
sys.path.insert(0, str(D))
from mbpp_rust import load_typeable, rust_literal

_spec = importlib.util.spec_from_file_location('judge_mod', str(D / '1_rust_correct.py'))
judge_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judge_mod)
OLD_CASE = judge_mod.build_case

def fixed_case(i, sig, args, expected, is_close):
    if not (is_close and sig['ret'] == 'f64'):
        return OLD_CASE(i, sig, args, expected, is_close)
    call_args = ', '.join(rust_literal(a, owned=(sig['params'][j] != '&str')) for j, a in enumerate(args))
    call = f"{sig['name']}({call_args})"
    exp = rust_literal(expected, owned=True)
    check = (f"match r {{ Ok(v) => {{ let e = {exp}; "
             f"if v == e || (v.is_finite() && e.is_finite() && (v - e).abs() <= 1e-3 * v.abs().max(e.abs())) "
             f"{{ p += 1; }} else {{ println!(\"FAIL {i}\"); }} }}, "
             f"Err(_) => println!(\"PANIC {i}\") }}")
    return (f"    let r = std::panic::catch_unwind(|| {call});\n    {check};\n")

_typ = None
def _init():
    global _typ
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    _typ, _ = load_typeable()

def _run(item):
    tid, code = item['task_id'], item['code']
    out = {}
    for mode, bc in (('old', OLD_CASE), ('new', fixed_case)):
        judge_mod.build_case = bc
        try:
            with tempfile.TemporaryDirectory(prefix='p8audit_') as td:
                r = judge_mod.judge(code, _typ[tid], Path(td))
            out[mode] = (r['passes'], r['n_pass'], r.get('error', ''))
        except Exception as e:
            out[mode] = (None, None, repr(e)[:100])
    judge_mod.build_case = OLD_CASE
    return {**{k: v for k, v in item.items() if k != 'code'}, **out}

def rd(p):
    op = gzip.open if str(p).endswith('.gz') else open
    with op(p, 'rt') as f:
        return [json.loads(l) for l in f if l.strip()]

if __name__ == '__main__':
    typ, _ = load_typeable()
    close = {t for t, s in typ.items() if s['ret'] == 'f64' and any(c[2] for c in s['cases'])}
    items = []
    # Phase 2: one response per problem
    lab = {r['task_id']: r for r in rd(D / 'gemma_mbpp_rust_correctness.jsonl')}
    for r in rd(D / 'gemma_mbpp_rust_responses.jsonl'):
        if r['task_id'] in close and lab.get(r['task_id'], {}).get('passes'):
            items.append(dict(phase='P2 response', task_id=r['task_id'], sample=None, code=r['generated_code']))
    # Phase 3 samples (also the sources of Phase 4 mutants) and Phase 5 round-1 samples
    for samples, labels, ph in (('within_samples.jsonl', 'within_labels.jsonl', 'P3 sample'),
                                ('matched_samples_r1.jsonl', 'matched_labels_r1.jsonl', 'P5 sample')):
        ok = {(r['task_id'], r['sample']) for r in rd(D / labels) if r['passes'] and r['task_id'] in close}
        for r in rd(D / samples):
            if (r['task_id'], r['sample']) in ok:
                items.append(dict(phase=ph, task_id=r['task_id'], sample=r['sample'], code=r['generated_code']))
    # Phase 4 / 5 / 6 mutants labelled passing (survivors, repairs)
    for f, ph in (('mutants_judged.jsonl', 'P4 mutant'), ('matched_mutants.jsonl', 'P5 mutant'),
                  ('results/ownfix/data/ownfix_mutants.jsonl.gz', 'P6 mutant')):
        for r in rd(D / f):
            if r['task_id'] in close and r['passes']:
                items.append(dict(phase=ph, task_id=r['task_id'], sample=r['sample'], pos=r['pos'],
                                  frm=r['from'], to=r['to'], code=r['code']))
    print('items', len(items), collections.Counter(i['phase'] for i in items))
    with Pool(6, initializer=_init) as pool:
        res = pool.map(_run, items, chunksize=1)
    json.dump(res, open('/home/user1/p8design/audit/audit_inf.json', 'w'), indent=0)
    by = collections.defaultdict(collections.Counter)
    for r in res:
        o, n = r['old'][0], r['new'][0]
        by[r['phase']]['old reproduces pass' if o else 'OLD DISAGREES'] += 1
        by[r['phase']]['still pass' if n else ('NOW FAILS' if n is False else 'error')] += 1
    for ph, c in by.items():
        print(ph, dict(c))
    for r in res:
        if r['new'][0] is not True:
            print('  ', r['phase'], r['task_id'], r.get('sample'), r.get('pos'), r.get('frm'), '->', r.get('to'), 'new:', r['new'])
