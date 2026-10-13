"""Phase 6, version 2: the reference values fixed from Phase 5 before any Phase 6
graph exists. PREREGISTRATION_OWNFIX.md §3.

  c        the sd (ddof 1) over Phase 5's 259 pairs of log(1 + nodes) of the
           survivor minus that of the killed member; Phase 6's z is divided by it
  beta_5   §3's model fitted on Phase 5's pairs, the survivor as the passing
           member, with the baseline terms Phase 5 recorded (no site entropy,
           no original-token log-probability); its CR2 interval and score test
           (stream 8)
  forests  Phase 5's forest on its 259 pairs' centred graph features
           (RandomForestClassifier(100, random_state=42)), and the same on the
           centred edit identity (p5_5's encoding), frozen with joblib; their
           sha256 are recorded

Phase 5 did not record tokens to the end; it is recounted here from the rows'
text with the tokenizer, as p6_2 counts it. The syntactic context is p6_3's.

  OUT_DIR=results/ownfix
"""
import hashlib
import json
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p6_stats as st  # noqa: E402
from p5_5_edit_baseline import encode  # noqa: E402
os.environ.setdefault('ROUND', '0')           # p6_3_match reads it at import; only context() is used
from p6_3_match import context  # noqa: E402

OUT_DIR = os.environ.get('OUT_DIR', 'results/ownfix')
P5_MEMBER = ('code_logprob', 'edit_logprob', 'min_logprob', 'mean_entropy', 'tokens_to_end')
SEED = 20261012


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')
    rows = {r['task_id']: r for r in read_jsonl('matched_rows.jsonl')}
    feats = {r['expr_id']: r['feature'] for r in read_jsonl('gemma_matched_features.jsonl')}
    pairs = json.load(open('matched_pairs.json', encoding='utf-8'))
    assert len(pairs) == 259 and all(p['pass_id'] in feats and p['fail_id'] in feats for p in pairs)

    def prep(r):
        enc = tok(r['text'], return_offsets_mapping=True)
        offs = enc['offset_mapping']
        code = [t for t in range(1, len(offs)) if offs[t][0] >= len(r['prose'])]
        at = len(r['prose']) + 1 + r['edit']['pos']
        tm = next(t for t in range(1, len(offs)) if offs[t][0] <= at < offs[t][1])
        r = dict(r)
        r['surprise'] = dict(r['surprise'], tokens_to_end=len(code) - 1 - code.index(tm))
        r['context'] = context(r['generated_code'], r['edit']['pos'])
        return r

    rs = [prep(rows[p['pass_id']]) for p in pairs]
    ks = [prep(rows[p['fail_id']]) for p in pairs]
    groups = np.array([p['problem'] for p in pairs])
    nodes_s = np.array([feats[p['pass_id']][1] for p in pairs])
    nodes_k = np.array([feats[p['fail_id']][1] for p in pairs])
    raw = np.log1p(nodes_s) - np.log1p(nodes_k)
    c = float(raw.std(ddof=1))
    z = st.graph_z(nodes_s, nodes_k, c)
    B, pen, names, dropped = st.baseline(rs, ks, member=P5_MEMBER)
    beta5, ci5 = st.cr2_interval(z, B, pen, groups)
    S5, p5 = st.score_test(z, B, pen, groups, np.random.default_rng([SEED, 8]))
    print(f"c = {c:.6f}; beta_5 = {beta5:+.4f}, CR2 95% [{ci5[0]:+.4f}, {ci5[1]:+.4f}]; "
          f"score test S_U {S5:+.3f}, p {p5:.4f}; {B.shape[1]} baseline columns, dropped {len(dropped)}")

    # frozen forests, fitted on all of Phase 5's pairs (centred within each pair)
    X = np.array([f for p in pairs for f in (feats[p['pass_id']], feats[p['fail_id']])], float)
    E = np.array([encode(rows[i]) for p in pairs for i in (p['pass_id'], p['fail_id'])], float)
    y = np.tile([1, 0], len(pairs))

    def centred(M):
        return M - np.repeat((M[0::2] + M[1::2]) / 2, 2, axis=0)

    out = {'c': c, 'beta_5': beta5, 'beta_5_ci': list(ci5), 'beta_5_S_U': S5, 'beta_5_p': p5,
           'baseline_columns': names, 'dropped': dropped}
    for name, M in (('graph', X), ('edit', E)):
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4).fit(centred(M), y)
        path = os.path.join(OUT_DIR, f'frozen_{name}_forest.joblib')
        joblib.dump(rf, path)
        out[f'frozen_{name}_forest_sha256'] = sha256(path)
    with open(os.path.join(OUT_DIR, 'reference.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ('baseline_columns', 'dropped')}, indent=1))


if __name__ == '__main__':
    main()
