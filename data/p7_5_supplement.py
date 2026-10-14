"""Phase 7, step 5 (after p7_4; PREREGISTRATION_CLOSING.md §8, deviation 2):
two figures §5's readings need that p7_4 did not print.

  A_lucky's 95% interval, per signal: §5 reads "A_lucky's interval covers 0.5";
           p7_4 gave the verified - lucky contrast but not this interval.
           Problem draws over the lucky pairs, stream 9 (key 3).
  A(G) - A(G_free)'s interval to more digits (p7_4 printed +0.000).
  the verified-only increments against the all-pair ones, as paired differences
           over the same problem draws (§5), stream 9 (key 5).

The forests are p7_4's: Phase 3's, seeds 100 + rep and 42, so the out-of-fold
scores are p7_4's exactly. Descriptive only; no test.

  OUT_DIR=results/closing  ->  supplement.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7_common as pc  # noqa: E402
from p7_4_closing import forest_blocks, estimates, load_blocks  # noqa: E402

KEYS = ('G', 'C', 'E', 'S2P', 'P', 'Gfree', 'L', 'G+C', 'L+C', 'Lact+C')


def main():
    P = pc.load_pairs(5)
    B, _ = load_blocks('matched')
    o = pc.orientation(P)
    perm = pc.member_perm(o)
    a = json.load(open(os.path.join(pc.OUT, 'analysis.json'), encoding='utf-8'))
    r = estimates(forest_blocks(B, 5), perm, P['groups'], P['problems'], None, KEYS)
    for k in ('G', 'C', 'Gfree'):
        if abs(float(r[k].mean()) - a['estimates']['A'][k]['A']) > 1e-12:
            raise SystemExit(f"{k}: scores differ from p7_4's")
    v = json.load(open(os.path.join(pc.OUT, 'verify_p45.json'), encoding='utf-8'))
    by = {(x['phase'], x['pass_id']): x['survivor_verdict'] for x in v['pairs']}
    surv = np.array([by[(5, p['pass_id'])] for p in P['pairs']])
    z = o * B['z']
    r['z (killed has more nodes)'] = (z < 0) + 0.5 * (z == 0)
    out = {'deviation': 2, 'lucky': {}, 'verified': {}}
    for lab, mask, key in (('lucky', surv == 'lucky', 3), ('verified', surv == 'verified', 4)):
        d = pc.boot_draws(P['pprob'][mask], pc.rng(9, key))
        out[lab] = {k: {'A': float(x[mask].mean()), 'ci': pc.boot_ci(x[mask], d), 'pairs': int(mask.sum()),
                        'problems': int(len(np.unique(P['pprob'][mask])))} for k, x in r.items()}
    # §5: the verified-only increment against the all-pair one, as a paired difference over the same draws
    g = pc.rng(9, 5)
    probs = np.unique(P['pprob'])
    members = {p: np.flatnonzero(P['pprob'] == p) for p in probs}
    V = surv == 'verified'
    out['increment_verified_minus_all'] = {}
    for name, (x, y) in (('G|C', ('G+C', 'C')), ('L|C', ('L+C', 'C'))):
        inc = r[x] - r[y]
        ds = []
        for _ in range(5000):
            sel = np.concatenate([members[p] for p in g.choice(probs, len(probs))])
            sv = sel[V[sel]]
            if len(sv):
                ds.append(inc[sv].mean() - inc[sel].mean())
        out['increment_verified_minus_all'][name] = {
            'verified': float(inc[V].mean()), 'all': float(inc.mean()), 'diff': float(inc[V].mean() - inc.mean()),
            'ci': [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]}
    draws = pc.boot_draws(P['pprob'], pc.rng(1))
    dg = r['G'] - r['Gfree']
    m = np.array([dg[s].mean() for s in draws])
    out['G-Gfree'] = {'diff': float(dg.mean()), 'ci': [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
                      'share_of_draws_at_or_below_0': float(np.mean(m <= 0))}
    pc.write_json(os.path.join(pc.OUT, 'supplement.json'), out)
    for lab in ('lucky', 'verified'):
        for k, x in out[lab].items():
            print(f"  A_{lab:8} {k:28} {x['A']:.4f} [{x['ci'][0]:.4f}, {x['ci'][1]:.4f}]  "
                  f"({x['pairs']} pairs, {x['problems']} problems)")
    for name, x in out['increment_verified_minus_all'].items():
        print(f"  increment {name}: verified {x['verified']:+.4f}, all {x['all']:+.4f}, "
              f"difference {x['diff']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}]")
    g = out['G-Gfree']
    print(f"  A(G) - A(Gfree) {g['diff']:+.6f} [{g['ci'][0]:+.6f}, {g['ci'][1]:+.6f}]; "
          f"draws at or below 0: {g['share_of_draws_at_or_below_0']:.4f}")
    print(f"supplement.json sha256 {pc.sha256(os.path.join(pc.OUT, 'supplement.json'))}")


if __name__ == '__main__':
    main()
