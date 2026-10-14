"""Erratum (inf judge bug): rerun Phase 7's prepare + closing analysis without the pair whose
survivor only returns inf (Phase 5: 2480150/2480151; Phase 4: 24800/24801).

Nothing in the repository is written. Inputs are copied (filtered) into this folder:
OUT_DIR = erratum/p7/out, WORK = erratum/p7/work. The committed scripts run unchanged except:
  - p7_common.load_pairs drops the pair after its own checks,
  - p7_2's reproduction gates (G1, C+) are recorded but cannot stop the run (they reproduce
    the 259-pair numbers by design),
  - p7_4's guard (clean tree, input hashes, sizesim sha) is replaced by a provenance note.
The size simulation and its null bands (c0) are reused from the 259-pair run."""
import os, sys, json, shutil
import numpy as np
E7 = '/home/user1/p8design/erratum/p7'
DATA = '/home/user1/CodeCircuit-rust/data'
SRC_OUT = os.path.join(DATA, 'results', 'closing')
os.makedirs(E7 + '/out', exist_ok=True)
os.makedirs(E7 + '/work', exist_ok=True)
os.environ['OUT_DIR'] = E7 + '/out'
os.environ['WORK'] = E7 + '/work'
os.environ.pop('TEST', None)
sys.path.insert(0, DATA)
os.chdir(DATA)
DROP = {5: (2480150, 2480151), 4: (24800, 24801)}
TAG = {5: 'matched', 4: 'mutant'}

# ---- inputs, filtered
from scipy.sparse import csr_matrix
for f in ('extract.json', 'sizesim.json', 'verify_p45.json'):
    shutil.copy(os.path.join(SRC_OUT, f), os.path.join(E7, 'out', f))
for ph, tag in TAG.items():
    drop = set(DROP[ph])
    rows = [json.loads(l) for l in open(os.path.join(SRC_OUT, f'local_{tag}.jsonl'), encoding='utf-8') if l.strip()]
    keep = [r for r in rows if r['expr_id'] not in drop]
    assert len(keep) == len(rows) - 2
    with open(os.path.join(E7, 'out', f'local_{tag}.jsonl'), 'w', encoding='utf-8') as fh:
        for r in keep:
            fh.write(json.dumps(r) + '\n')
    z = np.load(os.path.join('/home/user1/closing_work', f'posmean_{tag}.npz'))
    X = csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
    ids = z['ids']
    m = ~np.isin(ids, list(drop))
    assert (~m).sum() == 2
    Y = X[np.flatnonzero(m)].tocsr()
    extra = {k: z[k] for k in z.files if k not in ('data', 'indices', 'indptr', 'shape', 'ids')}
    np.savez(os.path.join(E7, 'work', f'posmean_{tag}.npz'), data=Y.data, indices=Y.indices, indptr=Y.indptr,
             shape=np.array(Y.shape), ids=ids[m], **extra)
    print(f'{tag}: local {len(rows)} -> {len(keep)} rows; posmean {X.shape} -> {Y.shape}; extra keys {list(extra)}', flush=True)

import p7_common as pc
assert pc.OUT == E7 + '/out' and pc.WORK == E7 + '/work'
_orig_load = pc.load_pairs

def load_pairs(phase):
    P = _orig_load(phase)
    k = [j for j, p in enumerate(P['pairs']) if (p['pass_id'], p['fail_id']) == DROP[phase]]
    assert len(k) == 1, (phase, k)
    k = k[0]
    keep_pair = np.ones(P['n'], bool); keep_pair[k] = False
    keep_mem = np.repeat(keep_pair, 2)
    assert set(P['ids'][~keep_mem].tolist()) == set(DROP[phase])
    Q = dict(P)
    Q['pairs'] = [p for j, p in enumerate(P['pairs']) if j != k]
    Q['ids'], Q['groups'] = P['ids'][keep_mem], P['groups'][keep_mem]
    Q['pprob'] = P['pprob'][keep_pair]
    Q['problems'] = np.array(sorted(set(Q['groups'].tolist())))
    Q['n'] = P['n'] - 1
    return Q
pc.load_pairs = load_pairs

STAGE = sys.argv[1]
if STAGE == 'prepare':
    import p7_2_prepare as p2
    _g1, _cp = p2.gate_g1, p2.gate_cplus
    def gate_g1(phase, P, B):
        ok, g = _g1(phase, P, B)
        g['erratum'] = 'recorded only: the 259-pair reproduction gate cannot hold on 258 pairs'
        return True, g
    def gate_cplus(P, B, tte, ctx):
        ok, g = _cp(P, B, tte, ctx)
        g['erratum'] = f'recorded only (reproduces reference.json: {ok})'
        return True, g
    p2.gate_g1, p2.gate_cplus = gate_g1, gate_cplus
    p2.main()
elif STAGE == 'closing':
    import p7_4_closing as p4
    def guard(paths):
        if os.path.exists(paths['out']):
            raise SystemExit('erratum analysis.json exists; remove it first')
        return {'erratum': 'inf-judge erratum rerun without pairs 2480150/2480151 (P5) and 24800/24801 (P4); '
                           'guard replaced; sizesim and its bands reused from the 259-pair run',
                'head': os.popen('git -C /home/user1/CodeCircuit-rust rev-parse HEAD').read().strip()}
    p4.guard = guard
    p4.main()
else:
    raise SystemExit('stage: prepare | closing')
