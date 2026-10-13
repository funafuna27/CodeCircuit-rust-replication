"""Phase 7, step 4: the closing comparison, run once on the real labels.
PREREGISTRATION_CLOSING.md §3 (tests, estimates) and §5 (readings).

Order: G5 (controls) -> primary family H1, H2 (weighted Holm) -> secondary S1-S3
(Holm) -> descriptive T -> forest estimates and increments -> refit and fold
sensitivity -> verification readout -> Phase 4 (descriptive) -> readings.

Refuses to run unless the gates passed, the size simulation's sha256 equals the
one recorded in PREREGISTRATION_CLOSING.md §8, its input hashes equal the current
files, every Phase 7 file is committed and unchanged, and no analysis.json
exists (a rerun needs RERUN_DEVIATION=<text>, which is written into the output).

  TEST=none|f1|b1  synthetic orientation (never the real labels), optionally with
                   the killed member's feature 1 or b1 raised by one within-pair
                   sd; no guard; writes analysis_test_<TEST>.json
  N_JOBS=4 (w3)  OUT_DIR=results/closing
"""
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7_common as pc  # noqa: E402
import w3_within_analysis as w3  # noqa: E402

TEST = os.environ.get('TEST', '')
RERUN = os.environ.get('RERUN_DEVIATION', '')
ALT = int(os.environ.get('ALT_SEEDS', 10))           # tests may use fewer; the real run uses 10
if not TEST and ALT != 10:
    raise SystemExit("ALT_SEEDS is for tests only")
REPO = os.path.dirname(HERE)
TRACKED = ['data/p7_common.py', 'data/p7_0_extract.py', 'data/p7_1_verify_p45.py', 'data/p7_2_prepare.py',
           'data/p7_3_sizesim.py', 'data/p7_4_closing.py', 'data/run_closing.sh', 'PREREGISTRATION_CLOSING.md']
L_NAMES = ('b1', 'b2', 'b3', 'b4', 'a1', 'a2', 'a3', 'a4')
B_SIGNS = 9999
MIN_SUB = (40, 20)          # pairs, problems: a subset's figures
MIN_SIDE = (20, 10)         # each side of the verified - lucky contrast
MIN_WITHIN = 10             # problems holding both kinds
SINGLE = ['C', 'E', 'S2P', 'S2AGN', 'O', 'G', 'Gfree', 'Gattr', 'P', 'PP', 'P_layer', 'Lact', 'Lattr', 'L',
          'N45', 'C63', 'E+edit_lp']
COMBO = ['G+C', 'L+C', 'N45+C', 'L+G+C', 'Lact+C', 'P+f0+C', 'G+P+f0+C', 'G+PP', 'Gfree+C', 'G+C63', 'L+C63',
         'C+', 'G+C+', 'L+C+']
INC = {'G|C': ('G+C', 'C'), 'L|C': ('L+C', 'C'), 'N45|C': ('N45+C', 'C'), 'L|G+C': ('L+G+C', 'G+C'),
       'Lattr|Lact+C': ('L+C', 'Lact+C'), 'G|P+f0+C': ('G+P+f0+C', 'P+f0+C'), 'G|PP': ('G+PP', 'PP'),
       'Gattr|Gfree+C': ('G+C', 'Gfree+C'), 'G-Gfree': ('G', 'Gfree'), 'G|C63': ('G+C63', 'C63'),
       'L|C63': ('L+C63', 'C63'), 'G|C+': ('G+C+', 'C+'), 'L|C+': ('L+C+', 'C+'), 'P-G': ('P', 'G'),
       'PP-G': ('PP', 'G')}


def guard(paths):
    if TEST:
        return {}
    g = json.load(open(paths['gates'], encoding='utf-8'))
    ex = json.load(open(paths['extract'], encoding='utf-8'))
    prep = json.load(open(paths['prepare'], encoding='utf-8'))
    sz = json.load(open(paths['sizesim'], encoding='utf-8'))
    why = []
    if g.get('G1') != 'passed':
        why.append('G1 did not pass')
    if ex.get('gate_G2') != 'passed' or ex.get('mode') != 'full' or ex.get('limit'):
        why.append('G2 did not pass on the full extraction')
    if prep.get('mhat_cols_limit'):
        why.append('prepare.json comes from a test run')
    if sz['params']['test_run']:
        why.append('sizesim.json comes from a test run')
    prereg = open(os.path.join(REPO, 'PREREGISTRATION_CLOSING.md'), encoding='utf-8').read()
    m = re.search(r'sizesim sha256: `?([0-9a-f]{64})', prereg)
    if not m or m.group(1) != pc.sha256(paths['sizesim']):
        why.append('the sizesim sha256 in PREREGISTRATION_CLOSING.md §8 differs from sizesim.json')
    for k, h in sz['inputs'].items():
        if pc.sha256(os.path.join(pc.OUT, k)) != h:
            why.append(f'{k} changed after the size simulation')
    ls = subprocess.run(['git', '-C', REPO, 'ls-files', '--error-unmatch', *TRACKED], capture_output=True, text=True)
    st = subprocess.run(['git', '-C', REPO, 'status', '--porcelain', '--', *TRACKED], capture_output=True, text=True)
    if ls.returncode != 0 or st.stdout.strip():
        why.append(f'Phase 7 files not committed or changed: {ls.stderr.strip()} {st.stdout.strip()}')
    if os.path.exists(paths['out']) and not RERUN:
        why.append('analysis.json exists; a rerun needs RERUN_DEVIATION')
    if why:
        raise SystemExit('refusing to run: ' + '; '.join(why))
    head = subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    return {'head': head, 'files': {f: pc.sha256(os.path.join(REPO, f)) for f in TRACKED}}


def load_blocks(tag):
    b = np.load(os.path.join(pc.OUT, f'blocks_{tag}.npz'))
    nz = np.load(os.path.join(pc.OUT, f'nuisance_{tag}.npz'))
    return {k: b[k] for k in b.files}, {k: nz[k] for k in nz.files}


def cond_sets(B):
    out = {'C': B['C'], 'CF': np.column_stack([B['C'], B['f0']]),
           'PP': np.column_stack([B['P'], B['f0'], B['O']]), 'C63': B.get('C63'),
           'CL': np.column_stack([B['C'], B['Ln']]), 'CG': np.column_stack([B['C'], B['G43n']])}
    if 'tte' in B:
        out['CPL'] = np.column_stack([B['C'], B['tte'], B['ctx']])
    return out


def forest_blocks(B, phase):
    G, C = B['G49'], B['C']
    f = {'C': C, 'E': B['E'], 'S2P': B['S2P'], 'O': B['O'], 'G': G, 'Gfree': B['GFREE'], 'Gattr': B['GATTR46'],
         'P': B['P'], 'PP': np.column_stack([B['P'], B['f0'], B['O']]), 'P_layer': B['P_layer'],
         'Lact': B['Lact'], 'Lattr': B['Lattr'], 'L': B['L'], 'N45': B['N45'],
         'E+edit_lp': np.column_stack([B['E'], B['S2P'][:, 1]]),
         'S2AGN': np.delete(B['S2P'], 1, axis=1)}
    f.update({'G+C': np.column_stack([G, C]), 'L+C': np.column_stack([B['L'], C]),
              'N45+C': np.column_stack([B['N45'], C]), 'L+G+C': np.column_stack([B['L'], G, C]),
              'Lact+C': np.column_stack([B['Lact'], C]), 'P+f0+C': np.column_stack([B['P'], B['f0'], C]),
              'G+P+f0+C': np.column_stack([G, B['P'], B['f0'], C]), 'G+PP': np.column_stack([G, f['PP']]),
              'Gfree+C': np.column_stack([B['GFREE'], C])})
    if phase == 5:
        f.update({'C63': B['C63'], 'G+C63': np.column_stack([G, B['C63']]), 'L+C63': np.column_stack([B['L'], B['C63']])})
        if 'tte' in B:
            cp = np.column_stack([C, B['tte'], B['ctx']])
            f.update({'C+': cp, 'G+C+': np.column_stack([G, cp]), 'L+C+': np.column_stack([B['L'], cp])})
    return f


def run_test(ph, nu, s, b, o, version, eps, pprob, problems, rw=False):
    db = o[:, None] * nu[f'{s}|{b}|db']
    rf, lin = nu[f'{s}|{b}|rf'], nu[f'{s}|{b}|lin']
    mh = o[:, None] * (rf if version == 1 else (rf + lin) / 2)
    r = pc.gcm(ph, db, mh, pprob, problems, eps=eps, rw=rw)
    r['version'] = version
    return r


def name_cols(block, cols):
    names = {'G43': [f'f{j}' for j in pc.G43], 'GATTR': [f'f{j}' for j in pc.GATTR],
             'G45': [f'f{j}' for j in pc.G45], 'L': list(L_NAMES), 'G43sh': [f'sh{j}' for j in pc.G43],
             'N45': [f'n{j}' for j in range(45)], 'z': ['z']}[block]
    return [names[c] for c in cols]


def estimates(F, perm, groups, problems, draws, keys, perm_seed=None, rf_seed=42):
    y = np.tile([1, 0], len(perm) // 2)
    r = {}
    for k in keys:
        s = pc.oof_seeded(pc.centred(F[k][perm]), y, groups[perm], problems, perm_seed, rf_seed)
        r[k] = pc.correct(s)
    return r


def summarise(r, draws, bands=None):
    out = {'A': {k: {'A': float(v.mean()), 'ci': pc.boot_ci(v, draws)} for k, v in r.items()}, 'inc': {}}
    for name, (a, b) in INC.items():
        if a not in r or b not in r:
            continue
        d = r[a] - r[b]
        e = {'diff': float(d.mean()), 'ci': pc.boot_ci(d, draws)}
        if bands and name in bands:
            bd = bands[name]
            e.update({'c0': bd['c0'], 'band': [bd['lo'], bd['hi']], 'diff_adj': e['diff'] - bd['c0'],
                      'ci_adj': [e['ci'][0] - bd['c0'], e['ci'][1] - bd['c0']]})
        out['inc'][name] = e
    return out


def reading(res, sz):
    lines = []
    prim = res['primary']
    pw = sz['power']
    power = {'H1': f"power of H1 at 0.035: {pw.get('z@-0.461', {}).get('H1@0.035', float('nan')):.2f} for a "
                   f"beta_5-sized z, {pw.get('z@-0.2305', {}).get('H1@0.035', float('nan')):.2f} at half of it",
             'H2': f"power of H2 at 0.015: {pw.get('L@0.04', {}).get('H2@0.015', float('nan')):.2f} for +0.04 "
                   f"oracle accuracy"}
    fmt = lambda ci: f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
    for h, inc in (('H1', 'G|C'), ('H2', 'L|C')):
        rej = prim[h]['rejected']
        e = res['estimates']['inc'][inc]
        lo, hi = e.get('ci_adj', e['ci'])
        if rej and lo > 0:
            lines.append(f"{h}: rejected; the {inc} increment, corrected for dilution, is above 0: it carries "
                         f"information beyond C and the forest ranks better with it. A second look at analysed "
                         f"pairs: new problems are needed to confirm it.")
        elif rej:
            lines.append(f"{h}: rejected; dependence beyond C in {prim[h].get('named', [])}; a gain in ranking "
                         f"is not shown (corrected {inc} interval [{lo:+.3f}, {hi:+.3f}]). A second look at "
                         f"analysed pairs.")
        elif hi < 0.05:
            lines.append(f"{h}: not rejected; corrected {inc} upper bound {hi:+.3f} < 0.05: beyond C it adds less "
                         f"than 0.05 in within-pair accuracy on these pairs.")
        else:
            lines.append(f"{h}: not rejected; corrected {inc} interval [{lo:+.3f}, {hi:+.3f}] reaches 0.05: not "
                         f"detected, a gain up to its upper bound is not excluded ({power[h]}).")
        if rej and set(prim[h].get('named', [])) <= {'f0'}:
            lines.append(f"{h}: the columns named are attribution-free (f0): read as forward-pass information.")
    r1, r2 = prim['H1']['rejected'], prim['H2']['rejected']
    dl = res['descriptive_T'].get('L|CG', {})
    lines.append({(True, True): "together: information beyond C in the graph's summary and in the local quantities",
                  (False, True): "together: L shows dependence beyond C; G43 is not shown to (at the power of §6). "
                                 f"L given C and G43: max |T| {dl.get('S', float('nan')):.2f} (descriptive); "
                                 "'the summary is too coarse' only if that holds up",
                  (True, False): "together: dependence beyond C in the graph's summary; the local quantities are "
                                 "not shown to add",
                  (False, False): "together: the closing negative: on these pairs the graph is not shown to know "
                                  "more than the edit's identity and the output distribution, at the power of §6"}
                 [(r1, r2)])
    s1 = res['secondary']['S1']
    if s1['rejected']:
        lines.append(f"S1: rejected with T {s1['T'][0]:+.2f}: " + (
            "beta_5's lead survives a nonlinear adjustment for C; graph size carries information beyond C "
            "(a second test on the pairs that produced the lead, not a confirmation)" if s1['T'][0] < 0 else
            "the opposite sign to beta_5; reported, not interpreted"))
    else:
        lines.append("S1: not rejected: beta_5's result is not reproduced under a nonlinear adjustment for C "
                     "(its baseline differed; power in sizesim.json)")
    s2, s3 = res['secondary']['S2'], res['secondary']['S3']
    lines.append("S2: " + ("rejected: the attribution-dependent columns carry what the forward-pass probe P+ does not"
                           if s2['rejected'] else "not rejected: beyond P+ the attribution is not shown to add; "
                           f"A(PP) - A(G) {res['estimates']['inc']['PP-G']['diff']:+.3f} "
                           f"{fmt(res['estimates']['inc']['PP-G']['ci'])} (no equivalence is claimed)"))
    lines.append("S3: " + ("rejected: the attribution-dependent columns add to C and f0; feature 1 is a known lead"
                           if s3['rejected'] else "not rejected: attribution is not shown to be needed beyond C and f0"))
    g = res['estimates']['inc']['G-Gfree']
    if g['ci'][0] > 0:
        lines.append(f"A(G) - A(Gfree) {g['diff']:+.3f} {fmt(g['ci'])}: above 0: the attribution-dependent "
                     "columns contribute to Ac")
    elif g['ci'][1] < 0:
        lines.append(f"A(G) - A(Gfree) {g['diff']:+.3f} {fmt(g['ci'])}: below 0: the three attribution-free "
                     "features rank better than all 49; at this n the other 46 columns dilute the forest")
    elif g['diff'] <= 0.03:
        lines.append(f"A(G) - A(Gfree) {g['diff']:+.3f} {fmt(g['ci'])}: three attribution-free features "
                     "come within 0.03 of Ac at this n (not an equivalence)")
    else:
        lines.append(f"A(G) - A(Gfree) {g['diff']:+.3f} {fmt(g['ci'])}: undecided")
    return lines


def verification(phase, P, perm, r_sf, z_sf, groups, problems, g_seed):
    path = os.path.join(pc.OUT, 'verify_p45.json')
    if not os.path.exists(path):
        return {'status': 'not run'}
    v = json.load(open(path, encoding='utf-8'))
    if not v['gate_G4']['passed']:
        return {'status': 'G4 failed: not used'}
    by = {(x['phase'], x['pass_id']): x for x in v['pairs']}
    surv = np.array([by[(phase, p['pass_id'])]['survivor_verdict'] for p in P['pairs']])
    pprob = P['pprob']
    out = {'status': 'used', 'counts': {k: int((surv == k).sum()) for k in ('verified', 'lucky', 'error', 'unverifiable')}}
    signals = dict(r_sf)
    signals['z (killed has more nodes)'] = (z_sf < 0) + 0.5 * (z_sf == 0)

    def ok(mask, need):
        return mask.sum() >= need[0] and len(np.unique(pprob[mask])) >= need[1]

    V = surv == 'verified'
    for lab, Lm in (('lucky', surv == 'lucky'), ('lucky_or_error', np.isin(surv, ['lucky', 'error']))):
        sec = {}
        if ok(V, MIN_SUB):
            dV = pc.boot_draws(pprob[V], g_seed(0))
            sec['verified_only'] = {k: {'A': float(x[V].mean()), 'ci': pc.boot_ci(x[V], dV)} for k, x in signals.items()}
            sec['verified_only_inc'] = {n: {'diff': float((signals[a] - signals[b])[V].mean()),
                                            'ci': pc.boot_ci((signals[a] - signals[b])[V], dV)}
                                        for n, (a, b) in INC.items() if a in signals and b in signals}
        if ok(V, MIN_SIDE) and ok(Lm, MIN_SIDE):
            g = g_seed(1)
            probs = np.unique(pprob)
            members = {p: np.flatnonzero(pprob == p) for p in probs}
            con = {}
            for k, x in signals.items():
                d = []
                for _ in range(5000):
                    sel = np.concatenate([members[p] for p in g.choice(probs, len(probs))])
                    a, b = sel[V[sel]], sel[Lm[sel]]
                    if len(a) and len(b):
                        d.append(x[a].mean() - x[b].mean())
                con[k] = {'A_verified': float(x[V].mean()), 'A_lucky': float(x[Lm].mean()),
                          'diff': float(x[V].mean() - x[Lm].mean()),
                          'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
            sec['verified_minus_lucky'] = con
            both = [p for p in probs if V[pprob == p].any() and Lm[pprob == p].any()]
            if len(both) >= MIN_WITHIN:
                sec['within_problem'] = {k: float(np.mean([x[(pprob == p) & V].mean() - x[(pprob == p) & Lm].mean()
                                                           for p in both])) for k, x in signals.items()}
                sec['within_problem_n'] = len(both)
        out[lab] = sec
    return out, surv


def main():
    t0 = time.time()
    paths = {k: os.path.join(pc.OUT, f) for k, f in (('gates', 'gates.json'), ('extract', 'extract.json'),
             ('prepare', 'prepare.json'), ('sizesim', 'sizesim.json'),
             ('out', 'analysis.json' if not TEST else f'analysis_test_{TEST}.json'))}
    prov = guard(paths)
    sz = json.load(open(paths['sizesim'], encoding='utf-8'))
    gates = json.load(open(paths['gates'], encoding='utf-8'))
    P = pc.load_pairs(5)
    B, nu = load_blocks('matched')
    if B['ids'].tolist() != P['ids'].tolist():
        raise SystemExit("blocks_matched: member order differs from load_pairs'")
    if TEST:
        o = np.random.default_rng(777).choice([-1, 1], P['n'])           # never the real labels
        if TEST in ('f1', 'b1'):
            # raise the synthetic killed member's value by one within-pair sd, in every block that holds it
            def bump(M, j, member_sd):
                M = M.copy()
                killed = 2 * np.arange(P['n']) + (o > 0)                  # second member if o = +1
                M[killed, j] += member_sd
                return M
            if TEST == 'f1':
                k = pc.G43.index(1)
                B['G43n'] = bump(B['G43n'], k, pc.dpair(B['G43n'])[:, k].std())
                nu['C|G43|db'] = pc.dpair(B['G43n'])
                B['G49'] = bump(B['G49'], 1, pc.dpair(B['G49'])[:, 1].std())
                f1 = np.log1p(B['G49'][:, 1])
                B['z'] = (f1[0::2] - f1[1::2]) / pc.C_REF
                nu['C|z|db'] = B['z'][:, None]
                kk = pc.GATTR.index(1)
                for s in ('CF', 'PP'):
                    nu[f'{s}|GATTR|db'] = pc.dpair(bump(B['GATTRn'], kk, pc.dpair(B['GATTRn'])[:, kk].std()))
            else:
                B['Ln'] = bump(B['Ln'], 0, pc.dpair(B['Ln'])[:, 0].std())
                nu['C|L|db'] = pc.dpair(B['Ln'])
                B['L'] = bump(B['L'], 0, pc.dpair(B['L'])[:, 0].std())
                B['Lattr'] = B['L'][:, :4]
    else:
        o = pc.orientation(P)
    perm = pc.member_perm(o)
    groups, problems, pprob = P['groups'], P['problems'], P['pprob']
    eps = pc.rng(2).choice([-1.0, 1.0], size=(B_SIGNS, len(problems)))
    draws = pc.boot_draws(pprob, pc.rng(1))
    versions, withdrawn = sz['versions'], sz['withdrawn']
    res = {'provenance': prov, 'test': TEST, 'rerun_deviation': RERUN, 'versions': versions, 'withdrawn': withdrawn}

    # p-hat on every conditioning set (passer first)
    Zs = cond_sets(B)
    ver = {s: versions.get(s, versions['C']) for s in Zs}
    ph = {}
    for s, Z in Zs.items():
        ph[s] = pc.phat(pc.centred(Z[perm]), pc.live_levels(Z), groups[perm], problems, version=ver[s])[0]
    print(f"p-hat fitted on {sorted(ph)}: {time.time() - t0:.0f} s", flush=True)

    def T(s, b, rw=False):
        r = run_test(ph[s], nu, s, b, o, ver[s], eps, pprob, problems, rw)
        r['names'] = name_cols(b, r['cols'])
        return r

    # G5: controls
    res['G5a_N45'] = T('C', 'N45')
    res['G5b_shadow'] = T('C', 'G43sh')
    print(f"G5a N45|C p {res['G5a_N45']['p']:.4f}; G5b shadow|C p {res['G5b_shadow']['p']:.4f}", flush=True)
    if res['G5a_N45']['p'] < 0.01 and not TEST and 'audit' not in RERUN.lower():
        pc.write_json(paths['out'] + '.audit.json', res)
        print("G5a: p < 0.01; audit the pipeline before reading anything (exit 4)")
        sys.exit(4)
    res['G5b_note'] = ('the shadow control rejects at 0.01: nuisance bias is suspected; read H1 with it'
                       if res['G5b_shadow']['p'] < 0.01 else '')

    # primary family
    prim = {'H1': T('C', 'G43', rw=True), 'H2': T('C', 'L', rw=True)}
    p1 = None if 'H1' in withdrawn.get('C', []) else prim['H1']['p']
    p2 = None if 'H2' in withdrawn.get('C', []) else prim['H2']['p']
    r1, r2 = pc.weighted_holm2(p1, p2)
    for h, rej in (('H1', r1), ('H2', r2)):
        prim[h]['rejected'] = rej
        prim[h]['withdrawn'] = (p1 if h == 'H1' else p2) is None
        if rej:
            prim[h]['named'] = [n for n, q in zip(prim[h]['names'], prim[h]['rw_p']) if q <= 0.05]
    res['primary'] = prim
    print(f"H1 G43|C: max|T| {prim['H1']['S']:.3f} p {prim['H1']['p']:.4f} rejected {r1}; "
          f"H2 L|C: max|T| {prim['H2']['S']:.3f} p {prim['H2']['p']:.4f} rejected {r2}", flush=True)

    # secondary family
    sec = {'S1': T('C', 'z'), 'S2': T('PP', 'GATTR'), 'S3': T('CF', 'GATTR')}
    wd = {'S1': 'S1' in withdrawn.get('C', []), 'S2': 'S2' in withdrawn.get('PP', []),
          'S3': 'S3' in withdrawn.get('CF', [])}
    rej = pc.holm([None if wd[k] else sec[k]['p'] for k in ('S1', 'S2', 'S3')])
    for k, rr in zip(('S1', 'S2', 'S3'), rej):
        sec[k]['rejected'], sec[k]['withdrawn'] = rr, wd[k]
    res['secondary'] = sec
    print("secondary: " + "; ".join(f"{k} max|T| {sec[k]['S']:.3f} p {sec[k]['p']:.4f} rejected {sec[k]['rejected']}"
                                    for k in sec), flush=True)

    # descriptive T (no p is read)
    desc = {'z|CF': ('CF', 'z'), 'L|CG': ('CG', 'L'), 'G43|CL': ('CL', 'G43'), 'G45|C63': ('C63', 'G45'),
            'L|C63': ('C63', 'L'), 'z|C63': ('C63', 'z'), 'G43|CPL': ('CPL', 'G43'), 'L|CPL': ('CPL', 'L'),
            'z|CPL': ('CPL', 'z')}
    res['descriptive_T'] = {}
    for k, (s, b) in desc.items():
        if s in ph and f'{s}|{b}|db' in nu:
            r = run_test(ph[s], nu, s, b, o, ver[s], None, pprob, problems)
            r['names'] = name_cols(b, r['cols'])
            res['descriptive_T'][k] = r

    # p-hat refits with other fold and forest seeds (descriptive ranges of T and p)
    alt = []
    for j in range(ALT):
        ps = (lambda jj: (lambda rep: [pc.ROOT, 8, jj, rep]))(j)
        rs = int(pc.rng(8, j).integers(2 ** 31))
        phj = pc.phat(pc.centred(B['C'][perm]), pc.live_levels(B['C']), groups[perm], problems,
                      version=ver['C'], perm_seed=ps, rf_seed=rs)[0]
        alt.append({h: run_test(phj, nu, 'C', b, o, ver['C'], eps, pprob, problems)
                    for h, b in (('H1', 'G43'), ('H2', 'L'), ('S1', 'z'))})
    res['phat_seed_sensitivity'] = {h: {'S': [min(a[h]['S'] for a in alt), max(a[h]['S'] for a in alt)],
                                        'p': [min(a[h]['p'] for a in alt), max(a[h]['p'] for a in alt)]}
                                    for h in ('H1', 'H2', 'S1')}
    print(f"tests done: {time.time() - t0:.0f} s", flush=True)

    # forest estimates
    F = forest_blocks(B, 5)
    keys = [k for k in SINGLE + COMBO if k in F]
    r = estimates(F, perm, groups, problems, draws, keys)
    res['estimates'] = summarise(r, draws, sz['bands'])
    ss = B['same_site'].astype(bool)
    dss = pc.boot_draws(pprob[ss], pc.rng(1, 1))
    res['same_site'] = {'pairs': int(ss.sum()), 'problems': int(len(np.unique(pprob[ss]))),
                        'inc': {n: {'diff': float((r[a] - r[b])[ss].mean()), 'ci': pc.boot_ci((r[a] - r[b])[ss], dss)}
                                for n, (a, b) in (('G|C', ('G+C', 'C')), ('L|C', ('L+C', 'C')))}}
    print(f"estimates done: {time.time() - t0:.0f} s", flush=True)
    sens = []
    for j in range(ALT):
        ps = (lambda jj: (lambda rep: [pc.ROOT, 8, jj, rep]))(j)
        rs = int(pc.rng(8, j).integers(2 ** 31))
        rj = estimates(F, perm, groups, problems, draws, keys, ps, rs)
        sens.append({**{f'A {k}': float(v.mean()) for k, v in rj.items()},
                     **{f'inc {n}': float((rj[a] - rj[b]).mean()) for n, (a, b) in INC.items() if a in rj and b in rj}})
    main_pts = {**{f'A {k}': res['estimates']['A'][k]['A'] for k in res['estimates']['A']},
                **{f'inc {n}': res['estimates']['inc'][n]['diff'] for n in res['estimates']['inc']}}
    res['refit_sensitivity'] = {k: {'main': main_pts[k], 'min': min(s[k] for s in sens),
                                    'median': float(np.median([s[k] for s in sens])), 'max': max(s[k] for s in sens),
                                    'main_outside_10_90': not (np.percentile([s[k] for s in sens], 10) <= main_pts[k]
                                                               <= np.percentile([s[k] for s in sens], 90))}
                                for k in main_pts}
    print(f"refit sensitivity done: {time.time() - t0:.0f} s", flush=True)

    # verification readout
    z_sf = o * B['z']
    vr = verification(5, P, perm, {k: r[k] for k in ('G', 'C', 'E', 'S2P', 'P', 'Gfree', 'L', 'G+C', 'L+C', 'Lact+C')},
                      z_sf, groups, problems, lambda k: pc.rng(9, k))
    if isinstance(vr, tuple):
        vr, surv = vr
        V = surv == 'verified'
        if V.sum() >= MIN_SUB[0] and len(np.unique(pprob[V])) >= MIN_SUB[1]:
            # refitted on the verified pairs alone (sensitivity), and H1's T there with all-pair nuisances
            idx = np.flatnonzero(V)
            mem = np.r_[[2 * i for i in idx], [2 * i + 1 for i in idx]]
            mem.sort()
            Pv_prob = np.array(sorted(set(pprob[V].tolist())))
            yv = np.tile([1, 0], len(idx))
            rv = {}
            for k in ('C', 'G+C', 'L+C'):
                s = pc.oof_seeded(pc.centred(F[k][perm][mem]), yv, groups[perm][mem], Pv_prob)
                rv[k] = pc.correct(s)
            dV = pc.boot_draws(pprob[V], pc.rng(9, 2))
            vr['refit_on_verified'] = {n: {'diff': float((rv[a] - rv[b]).mean()), 'ci': pc.boot_ci(rv[a] - rv[b], dV)}
                                       for n, (a, b) in (('G|C', ('G+C', 'C')), ('L|C', ('L+C', 'C')))}
            mh = nu['C|G43|rf'] if ver['C'] == 1 else (nu['C|G43|rf'] + nu['C|G43|lin']) / 2
            h1v = pc.gcm(ph['C'][V], (o[:, None] * nu['C|G43|db'])[V], (o[:, None] * mh)[V], pprob[V], Pv_prob)
            vr['H1_T_on_verified'] = {'S': h1v['S'], 'T': h1v['T'], 'names': name_cols('G43', h1v['cols'])}
    res['verification'] = vr
    print(f"verification readout: {vr.get('status')}; {time.time() - t0:.0f} s", flush=True)

    # Phase 4, descriptive
    P4 = pc.load_pairs(4)
    B4, nu4 = load_blocks('mutant')
    o4 = np.random.default_rng(778).choice([-1, 1], P4['n']) if TEST else pc.orientation(P4)
    perm4 = pc.member_perm(o4)
    draws4 = pc.boot_draws(P4['pprob'], pc.rng(14))
    F4 = forest_blocks(B4, 4)
    keys4 = [k for k in ('C', 'E', 'O', 'G', 'Gfree', 'Gattr', 'P', 'PP', 'L', 'Lact', 'N45', 'G+C', 'L+C', 'N45+C',
                         'Gfree+C', 'G+PP', 'L+G+C', 'Lact+C') if k in F4]
    r4 = estimates(F4, perm4, P4['groups'], P4['problems'], draws4, keys4)
    ph4 = pc.phat(pc.centred(B4['C'][perm4]), pc.live_levels(B4['C']), P4['groups'][perm4], P4['problems'])[0]
    res['phase4'] = {'estimates': summarise(r4, draws4),
                     'T': {h: {k: v for k, v in run_test(ph4, nu4, 'C', b, o4, 1, None, P4['pprob'],
                                                         P4['problems']).items() if k in ('S', 'T', 'cols')}
                           for h, b in (('H1', 'G43'), ('H2', 'L'), ('S1', 'z'))},
                     'note': 'C4 = E + sdpa code and edit-token log-probabilities + chars + f3, f4; no minimum '
                             'log-probability or mean entropy; unmatched; 73 of 98 problems are Phase 5s. '
                             'Biased towards the graph; never read as support.'}
    v4 = verification(4, P4, perm4, {k: r4[k] for k in ('G', 'C', 'E', 'P', 'Gfree', 'L', 'G+C', 'L+C') if k in r4},
                      o4 * B4['z'], P4['groups'], P4['problems'], lambda k: pc.rng(10, k))
    res['phase4']['verification'] = v4[0] if isinstance(v4, tuple) else v4
    res['reading'] = reading(res, sz)
    res['gates'] = {'G1': gates.get('G1'), 'cplus_ok': gates.get('cplus_ok')}
    res['seconds'] = time.time() - t0
    pc.write_json(paths['out'], res)
    print("\n=== reading (PREREGISTRATION_CLOSING.md §5) ===")
    for line in res['reading']:
        print("  " + line)
    print(f"\n{os.path.basename(paths['out'])} sha256 {pc.sha256(paths['out'])}; {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
