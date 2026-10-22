"""Phase 9, part ③ (M1): the gates of PREREGISTRATION_P9_M1.md §5.

  STEP=x1   after `ROLE=matched p9_m1_extract.py`: the forward-only posmean against Phase 7's posmean of the
            518 retained Phase 5 graphs (byte for byte; the npz's sha256 first), f0/f3/f4 against their
            stored columns, and X3 (the code log p, minimum log p and mean entropy against Phase 5's p5_2
            values). Writes results/p9_m1/gates_x1.json.
  STEP=x2   after the rust and python extractions: R (stored G reproduces Phase 2's AUROCs, and the seeded
            out-of-fold function equals 5_problem_level's at the defaults), f0/f3/f4 against the stored G
            columns 0/3/4 on every row, and the 94 surviving Phase 1-2 graphs rebuilt by setup_attribution
            (tokens, active features and values, logit targets and probabilities, bit for bit; reported).
            Writes results/p9_m1/gates_x2.json.
"""
import glob
import importlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9_m1_common as M  # noqa: E402

STEP = os.environ.get('STEP', 'x1')
POSMEAN = os.path.expanduser('~/closing_work/posmean_matched.npz')
POSMEAN_SHA = 'ed083690c164a29f1ff095ff8b7530ebbc70446245c3844cea6bc61dae5dc0d4'




def row_csr(z, field, i):
    a, b = z[f'{field}_indptr'][i], z[f'{field}_indptr'][i + 1]
    return z[f'{field}_indices'][a:b], z[f'{field}_data'][a:b]


def bf16_ulp(x):
    x = abs(float(x))
    return 2.0 ** (np.floor(np.log2(x)) - 7) if x > 0 else 2.0 ** -133


def x1():
    out = {'guard': M.guard()}
    if M.sha256(POSMEAN) != POSMEAN_SHA:
        sys.exit('posmean_matched.npz: sha256 differs from PREREGISTRATION_CLOSING.md')
    with np.load(POSMEAN) as f:
        ref = {k: f[k] for k in f.files}
    z, meta = M.load_extract('matched', out['guard'])
    out['extract_sha256'] = meta['sha256']
    keys = z['keys'].tolist()
    pos_of = {int(k): i for i, k in enumerate(keys)}
    rows = {int(r['task_id']): r for r in M.read_jsonl(os.path.join(HERE, 'matched_rows.jsonl'))}
    feats = {int(r['expr_id']): r['feature'] for r in M.read_jsonl(os.path.join(HERE, 'gemma_matched_features.jsonl'))}
    eq_pm, eq_f, bad_pm, bad_f = 0, 0, [], []
    for j, e in enumerate(ref['ids'].tolist()):
        i = pos_of[int(e)]
        ri, rd = ref['indices'][ref['indptr'][j]:ref['indptr'][j + 1]], ref['data'][ref['indptr'][j]:ref['indptr'][j + 1]]
        ci, cd = row_csr(z, 'pm', i)
        if np.array_equal(ri, ci) and np.array_equal(rd.view(np.uint32), cd.view(np.uint32)):
            eq_pm += 1
        else:
            bad_pm.append(int(e))
        stored = [feats[int(e)][k] for k in (0, 3, 4)]
        if list(z['f'][i]) == stored:
            eq_f += 1
        else:
            bad_f.append({'id': int(e), 'got': list(map(float, z['f'][i])), 'stored': stored})
    # X3: code log p, min log p, mean entropy against p5_2 (HF eager) on the same texts
    d_lp, d_min, d_ent = [], [], []
    tptr = z['tptr']
    for e, i in pos_of.items():
        s = rows[e]['surprise']
        a, b = tptr[i], tptr[i + 1]
        off, lp, en = z['off'][a:b], z['logp'][a:b], z['ent'][a:b]
        prose = rows[e]['prose']
        code = [t for t in range(1, b - a) if off[t][0] >= len(prose)]
        d_lp.append(abs(float(lp[code].sum()) - s['code_logprob']) / len(code))
        d_min.append(abs(float(lp[code].min()) - s['min_logprob']))
        d_ent.append(abs(float(en[code].mean()) - s['mean_entropy']))
    x3 = {'per_token_code_logprob_mean_abs': round(float(np.mean(d_lp)), 5),
          'min_logprob_mean_abs': round(float(np.mean(d_min)), 5), 'mean_entropy_mean_abs': round(float(np.mean(d_ent)), 5)}
    out.update({'rows': len(ref['ids']), 'posmean_equal': eq_pm, 'posmean_differ': bad_pm[:20],
                'f0f3f4_equal': eq_f, 'f0f3f4_differ': bad_f[:20], 'x3': x3})
    out['p_ok'] = eq_pm == len(ref['ids'])
    out['f_ok'] = eq_f == len(ref['ids'])
    # Deviation 1 (PREREGISTRATION_P9_M1.md §10): X3 is judged on the two averaged quantities; the minimum
    # log p (one token, so one bf16-rounded logit) is reported, not judged.
    out['c_ok'] = x3['per_token_code_logprob_mean_abs'] < 0.02 and x3['mean_entropy_mean_abs'] < 0.02
    out['extractor_ok'] = bool(out['c_ok'])        # C is needed for the primary; P only if p_ok
    os.makedirs(M.OUT, exist_ok=True)
    with open(os.path.join(M.OUT, 'gates_x1.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


def gate_f(arm, guard):
    """f0/f3/f4 recomputed against stored G columns 0/3/4 on every row of the arm (§5 X2)."""
    rows = M.load_arm(arm)
    z, meta = M.load_extract(arm, guard)
    pos_of = {int(k): i for i, k in enumerate(z['keys'].tolist())}
    d0, d34 = [], []
    for r in rows:
        f = z['f'][pos_of[r['row_index']]]
        g = [r['G'][0], r['G'][3], r['G'][4]]
        if f[0] != g[0]:
            d0.append({'row': r['row_index'], 'task': r['task_id'], 'got': float(f[0]), 'stored': g[0]})
        for k in (1, 2):
            if f[k] != g[k]:
                d34.append({'row': r['row_index'], 'task': r['task_id'], 'col': (3, 4)[k - 1], 'got': float(f[k]),
                            'stored': g[k], 'ulps': abs(float(f[k]) - g[k]) / bf16_ulp(g[k])})
    n = len(rows)
    f0_rows = {d['row'] for d in d0}
    f0_ok = not d0 or (len(f0_rows) <= 0.01 * n and all(abs(d['got'] - d['stored']) <= 0.001 * d['stored'] for d in d0))
    f34_rows = {d['row'] for d in d34}
    f34_ok = not d34 or (len(f34_rows) <= 0.01 * n and all(d['ulps'] <= 1 for d in d34))
    return {'extract_sha256': meta['sha256'], 'rows': n, 'f0_differ': len(f0_rows), 'f34_differ': len(f34_rows), 'f0_examples': d0[:10],
            'f34_examples': d34[:10], 'f0_ok': bool(f0_ok), 'f34_ok': bool(f34_ok),
            'flagged_problems': sorted({d['task'] for d in d0}), 'arm_ok': bool(f0_ok and f34_ok)}


def gate_r():
    """Stored G reproduces Phase 2's AUROCs; the seeded oof equals 5_problem_level's at the defaults."""
    os.environ.setdefault('N_JOBS', '4')
    pl = importlib.import_module('5_problem_level')
    tok = M.tokenizer()
    out, ok = {}, True
    arm_rows = {a: M.load_arm(a) for a in M.ARMS}
    for pop, (want_all, want_0) in M.GATE_R.items():
        P = M.load_pop(pop, tok, arm_rows[M.POPS[pop][0]])
        s = M.oof(P['G'], P['groups'], P['y_row'], P['problems'], n_jobs=4)
        if pop == 'rust_strict':
            ref = pl.oof_problem_scores(P['G'], P['groups'], P['y_row'], P['problems'])
            out['oof_equals_5_problem_level'] = bool(np.array_equal(s, ref))
            ok &= out['oof_equals_5_problem_level']
        m0 = P['step'] == 0
        p0 = np.array(sorted(set(P['groups'][m0].tolist())))
        s0 = M.oof(P['G'][m0], P['groups'][m0], P['y_row'][m0], p0, n_jobs=4)
        y0 = np.array([P['y_prob'][list(P['problems']).index(p)] for p in p0])
        a, a0 = round(M.auc(P['y_prob'], s), 4), round(M.auc(y0, s0), 4)
        out[pop] = {'rows': len(P['rows']), 'problems': len(P['problems']), 'positive': int(P['y_prob'].sum()),
                    'auroc_all': a, 'auroc_step0': a0, 'want': [want_all, want_0]}
        ok &= (a == want_all and a0 == want_0)
    out['r_ok'] = bool(ok)
    return out


def gate_94():
    """The surviving Phase 1-2 graphs rebuilt by setup_attribution (reported, §5)."""
    import torch
    import p9_m1_extract as X
    model = X.load_model()
    tok = M.tokenizer()
    res, skipped = [], []
    for arm, d in (('rust', 'rust_graph'), ('python', 'python_graph')):
        rows = {r['row_index']: r for r in M.load_arm(arm)}
        for p in sorted(glob.glob(os.path.join(HERE, d, 'graph_*.pt'))):
            row = int(os.path.basename(p).split('_')[1])
            if row not in rows:
                skipped.append(os.path.basename(p))
                continue
            g = torch.load(p, weights_only=False, map_location='cpu')
            r = X.forward_one(model, tok, rows[row]['text'], rows[row]['prose'])
            af = g['active_features'].numpy()
            same_tok = g['input_tokens'].tolist() == r['ids'].tolist()
            same_idx = af.shape[0] == len(r['lay']) and np.array_equal(af[:, 0], r['lay']) and \
                np.array_equal(af[:, 1], r['pos']) and np.array_equal(af[:, 2], r['fid'])
            same_val = same_idx and torch.equal(g['activation_values'].view(torch.int16), r['vals'].view(torch.int16))
            lt = g['logit_targets'] if 'logit_targets' in g else g['logit_tokens']
            lt_ids = lt.tolist() if isinstance(lt, torch.Tensor) else [t.vocab_idx for t in lt]
            same_lt = lt_ids == r['sal_ids'].tolist()
            same_lp = same_lt and torch.equal(g['logit_probabilities'].float(), torch.from_numpy(r['sal_p']))
            res.append({'arm': arm, 'graph': os.path.basename(p), 'tokens': same_tok, 'active': same_idx,
                        'values': bool(same_val), 'logit_targets': same_lt, 'logit_probs': bool(same_lp)})
    summ = {k: sum(x[k] for x in res) for k in ('tokens', 'active', 'values', 'logit_targets', 'logit_probs')}
    per_arm = {a: sum(x['arm'] == a for x in res) for a in ('rust', 'python')}
    return {'graphs': len(res), 'per_arm': per_arm, 'all_94_compared': per_arm == {'rust': 50, 'python': 44},
            'skipped': skipped, 'equal': summ, 'differ': [x for x in res if not all(
        x[k] for k in ('tokens', 'active', 'values', 'logit_targets', 'logit_probs'))][:20]}


def x2():
    out = {'guard': M.guard(), 'R': gate_r()}
    for arm in M.ARMS:
        out[f'F_{arm}'] = gate_f(arm, out['guard'])
    out['graphs94'] = gate_94()
    with open(os.path.join(M.OUT, 'gates_x2.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: (v if k != 'graphs94' else {'equal': v['equal'], 'graphs': v['graphs']}) for k, v in out.items()}, indent=1))


if __name__ == '__main__':
    {'x1': x1, 'x2': x2}[STEP]()
