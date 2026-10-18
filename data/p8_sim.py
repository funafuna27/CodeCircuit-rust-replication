"""Phase 8: size and power of the registered sign-flip test on the real site structure, and the
post-pilot alpha rule (final_design.md §5, §8). Generalises /home/user1/p8design/final/g3_sim.py
(size) and g4_pow03.py (power). CPU only. Reads no outcome of any arm: the structure is the belief
sites' problems and bugs and the committed HF BAN table; the pilot enters only through q, the
label-free share of pilot sites at which G's lexeme differs from BAN's.

Every simulated dataset is scored by p8_4_stats.signflip_p, the analysis's own test on its integer
statistic (X_g = L / n_g * the problem's summed differences): exact when the number of discordant
problems m <= 24, else 99,999 sign flips from stream 0 plus the identity.

Structure  STRUCT=sites (default): the kind == 'belief' rows of $P8_OUT/sites.jsonl (problem, bug, and
             the HF BAN status: verified -> 1, any other known label -> 0, unknown -> -1; see hf_ban()).
             SUBSET=all (default) | tier1.
           STRUCT=design: TESTS ONLY. The design-stage structure behind §2 and g3/g4 (146 sites,
             31 problems, 94 bugs, HF BAN 79/17/50), rebuilt read-only from data_agent/sites.jsonl,
             final/g1_inf.json and ownfix_sources.jsonl.gz exactly as g3_sim.py builds it.
Nulls (E[D] = 0), as g3
  A  unconstrained: a site is a rescue (+1) or a break (-1); 20% of problems are type 1 (F).
     hom: rescue q/2 and break q/2 everywhere; skew+: type 1 rescues only, type 2 breaks only;
     skew-: type 1 breaks only, type 2 rescues only; q is the expected discordant share.
  B  structured: rescues only at BAN-fail sites, breaks only at BAN-fix sites; unknown BAN
     outcomes drawn per bug at ufix; sym / asym- (20% of problems only break) / asym+ (20% only
     rescue), rates balanced so E[D] = 0 using fixfrac, the expected equal-weight BAN fix share.
  A site's uniform is its bug's with probability rho (within-bug sharing), else its own.
  MODE=size   g3's grid: A q 0.05/.10/.20/.30/.40 x rho 0/.5/.9/1; B ufix .3/.5/.7 x b1 .1/.3/.6/1
              x rho .7/1 (132 cells).
  MODE=alpha  q from the pilot; grid Q* = {q-0.1, q-0.05, q, q+0.05, q+0.1} within [0.01, 0.99];
              A at every q in Q* x rho 0/.5/.9/1; B with b1 set so that the expected discordant share
              is each q in Q* (sym 0.6 b1 fixfrac, asym- 2F b1 fixfrac, asym+ 2F b1 (1-fixfrac))
              x ufix .3/.5/.7 x rho .7/1. A null keeps E[D] = 0 only while every rate is <= 1, so a q
              (A skew: q <= 2F) or b1 (b1_max) beyond that is lowered to the maximum and marked
              'capped' (g3's min(1, .) capped one side only and would have measured power there);
              cells that coincide after this run once. Every cell reports E_D and its realised share.
              Rule (§5): alpha = 0.03 unless the worst size at 0.03 exceeds 0.05; then the largest of
              0.025, 0.02 whose worst size is <= 0.05; if neither, 0.02 flagged 'no_allowed_alpha_holds'.
              q = (# pilot sites where lexeme(G's t*) != lexeme(BAN's t*)) / (# pilot sites with both
              outputs, neither excluded); a site where G did not flip keeps S's own lexeme, which is
              never BAN's (BAN's token is outside E), so it counts as differing (G's Y is 0 there and
              the site can be discordant). Lexemes: decode S's ids with ids[h] := t* (BOS dropped), cut
              the prose, p8_common.lexeme_key (p8_4_stats.splice_key). Refuses to run once a main
              site's stage-B output exists (a file in $P8_OUT/run/B or $RUN_DIR/B whose site_id is in
              $P8_OUT/sites.jsonl; pilot files never count; AFTER_STAGE_B=1 overrides): alpha is fixed
              before any main graph. The pilot's run dir: env PILOT_RUN, else whichever one of
              $P8_OUT/pilot/run, $P8_OUT/run, $RUN_DIR holds pilot-site files.
              Futility (design §9), in alpha.json 'futility': G's label-free flip count over the 30
              pilot sites (run-side 'flipped', a t* in E counts as no flip as in p8_3; excluded sites
              and sites without a stage-B file count as not flipped); fires when < 15.
              fixfrac is computed exactly (known sites as they are, unknown at ufix), where g3 took
              the mean of 400 simulated BAN draws.
Power (MODE=power), as g4: rescues only at BAN-fail sites (rate r), breaks only at BAN-fix sites
  (rate b), unknown BAN imputed per bug at 0.5, problem logit-normal sd, outcomes shared within a bug
  (rho 1). S2 = G > BAN; S3 with GxA = G's outcome at a bug w.p. kappa, else BAN's; S1 against RND
  fixing at 0.15 per bug; all three = the sequence. Grid b 0/.05/.10 x r .25/.4/.6/.8/.95 x sd 0/1,
  kappa .3/.5/.8. The 80% MDE of S2 per (b, sd, alpha) by linear interpolation in E[D].

Usage (cwd data/)
  MODE=alpha python p8_sim.py      -> $P8_OUT/alpha.json      (after the pilot, before any main graph)
  MODE=size  python p8_sim.py      -> $P8_OUT/sim_size.json   (g3's grid on the real structure)
  MODE=power python p8_sim.py      -> $P8_OUT/sim_power.json  (read by p8_4 for the MDE)
  MODE=selftest P8_OUT=<scratch> python p8_sim.py
Env  STRUCT=sites|design  SUBSET=all|tier1  N_DS=2000  N_POW=1000  WORKERS=8
     OVERWRITE_ALPHA=0  AFTER_STAGE_B=0  Q=<float> (test override of the pilot q)
     PILOT_SITES=pilot_sites.jsonl  PILOT_RUN=<searched> (holds A/<site>.json with BAN and
     B/<site>.json with G)  RUN_DIR (p8_2's)  Relative paths are under P8_OUT, as in p8_2/p8_3.
A run with STRUCT=design, SUBSET=tier1, N_DS < 2000 (g3's 2000 is the default), N_POW < 1000 or Q
set is a test run: its output says test_run true, and p8_4 refuses such an alpha.json. A real
alpha.json is never overwritten unless OVERWRITE_ALPHA=1. (The old FORCE is refused.)
Selftest (SPEC test B): structure, the analysis's own p on simulated data, the alpha rule, the
pilot q on a real tokenization, g3's 132 size cells and g4's 30 power cells against their logs,
MODE=alpha as a test run and as a real run on a site table in p8_1's schema. Writes only in P8_OUT.
Streams: size cells rng(4, part, cell) (part 1/2 = g3 grid A/B, 3/4 = alpha grid A/B), power cells
rng(5, cell); the test's own flips rng(0) (inside p8_4_stats).
"""
import glob
import gzip
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p8_common as pc  # noqa: E402
import p8_4_stats as st  # noqa: E402

MODE = os.environ.get('MODE', 'size')
STRUCT = os.environ.get('STRUCT', 'sites')
SUBSET = os.environ.get('SUBSET', 'all')
N_DS = int(os.environ.get('N_DS', 2000))
N_POW = int(os.environ.get('N_POW', 1000))
WORKERS = int(os.environ.get('WORKERS', 8))
if 'FORCE' in os.environ:
    raise SystemExit('FORCE is gone: OVERWRITE_ALPHA=1 overwrites a real alpha.json, AFTER_STAGE_B=1 runs although '
                     'main stage-B outputs exist; set the one you mean')
OVERWRITE_ALPHA = os.environ.get('OVERWRITE_ALPHA', '0') == '1'
AFTER_STAGE_B = os.environ.get('AFTER_STAGE_B', '0') == '1'
Q_ENV = os.environ.get('Q')


def under_out(v):
    """p8_2/p8_3's rule for env paths: absolute as given, relative under P8_OUT."""
    return v if os.path.isabs(v) else os.path.join(pc.OUT, v)


PILOT_SITES = under_out(os.environ.get('PILOT_SITES', 'pilot_sites.jsonl'))
PILOT_RUN = under_out(os.environ['PILOT_RUN']) if os.environ.get('PILOT_RUN') else None   # None: searched
FUTILITY_MIN, FUTILITY_N = 15, 30          # design §9: G flips at < 15 of the 30 pilot sites -> no main graphs

DESIGN = '/home/user1/p8design'
DA_SITES = f'{DESIGN}/data_agent/sites.jsonl'
G1_JSON = f'{DESIGN}/final/g1_inf.json'
G3_LOG = f'{DESIGN}/final/g3_sim.log'
G4_LOG = f'{DESIGN}/final/g4_pow03.log'

ALPHAS = (0.05, 0.04, 0.03, 0.025, 0.02)
POWER_ALPHAS = (0.03, 0.025, 0.02)
F = 0.2
P_RND = 0.15
A_MODES = ('hom', 'skew+', 'skew-')
B_MODES = ('sym', 'asym-', 'asym+')
A_RHO = (0.0, 0.5, 0.9, 1.0)
B_RHO = (0.7, 1.0)
UFIX = (0.3, 0.5, 0.7)
G3_Q = (0.05, 0.10, 0.20, 0.30, 0.40)
G3_B1 = (0.1, 0.3, 0.6, 1.0)
POW_B = (0.0, 0.05, 0.10)
POW_R = (0.25, 0.4, 0.6, 0.8, 0.95)
POW_SD = (0.0, 1.0)
KAPPAS = (0.3, 0.5, 0.8)

S = {}                                   # the structure, per process


# ---------------------------------------------------------------- structure
HF_LABELS = ('verified', 'lucky', 'fail', 'unverifiable', 'no_compile')


def hf_ban(row):
    """The committed HF BAN status of a site (p8_1's 'hf_runner_up' = {known, to, label, p}): 1 when
    the runner-up is known and verified, 0 when it is known with any other label (Y would be 0, as in
    g3), -1 when it is unknown (drawn per bug at ufix / 0.5). A known runner-up without a label is
    counted as unknown and reported."""
    hr = row.get('hf_runner_up')
    if not isinstance(hr, dict) or 'known' not in hr:
        raise SystemExit(f"{row.get('site_id')}: no 'hf_runner_up' {{known, label}} in sites.jsonl")
    if not hr['known']:
        return -1, 'unknown'
    if hr.get('label') is None:
        return -1, 'known without a label'
    if hr['label'] not in HF_LABELS:
        raise SystemExit(f"{row.get('site_id')}: hf_runner_up label {hr['label']!r}")
    return int(hr['label'] == 'verified'), hr['label']


def make_structure(task, bug, ban, info):
    tasks = sorted(set(task))
    ti = {t: i for i, t in enumerate(tasks)}
    bugs = sorted(set(bug), key=str)
    bi = {b: i for i, b in enumerate(bugs)}
    site_task = np.array([ti[t] for t in task])
    site_bug = np.array([bi[b] for b in bug])
    G, N = len(tasks), len(task)
    nsite = np.bincount(site_task, minlength=G)
    L = st.lcm(nsite)
    W = np.zeros((N, G), dtype=np.int64)
    W[np.arange(N), site_task] = L // nsite[site_task]
    ban = np.array(ban, dtype=np.int64)
    return {'N': N, 'G': G, 'NB': len(bugs), 'L': L, 'W': W, 'site_task': site_task, 'site_bug': site_bug,
            'nsite': nsite, 'ban_known': ban, 'problems': tasks,
            'info': {**info, 'sites': N, 'problems': G, 'bugs': len(bugs), 'L': L,
                     'per_problem_sorted': sorted(nsite.tolist(), reverse=True),
                     'hf_ban_fix_fail_unknown': [int((ban == 1).sum()), int((ban == 0).sum()), int((ban == -1).sum())]}}


def sites_structure(subset):
    p = os.path.join(pc.OUT, 'sites.jsonl')
    if not os.path.exists(p):
        raise SystemExit(f'{p} does not exist yet; STRUCT=design is for tests only')
    rows = [r for r in pc.read_jsonl(p) if r['kind'] == 'belief']
    if subset == 'tier1':
        rows = [r for r in rows if r['tier'] == 1]
    bans = [hf_ban(r) for r in rows]
    return make_structure([r['problem'] for r in rows], [r['bug'] for r in rows], [b for b, _ in bans],
                          {'source': 'sites.jsonl', 'subset': subset, 'sites_sha256': pc.sha256(p),
                           'hf_runner_up_labels': dict(sorted(Counter(w for _, w in bans).items()))})


def design_structure():
    """g3_sim.py's structure (tests only): belief sites kept by g1_inf, bug = (task, edited line
    without whitespace, wrong id), HF BAN = the runner-up's verified status, non-finite repairs removed."""
    rows = pc.read_jsonl(DA_SITES)
    with open(G1_JSON, encoding='utf-8') as f:
        g1 = json.load(f)
    bad = {tuple(x) for x in g1['bad']}
    kept = {tuple(x) for x in g1['kept_sites']}
    src = {}
    with gzip.open(os.path.join(pc.OWNFIX, 'ownfix_sources.jsonl.gz'), 'rt', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            src[(d['task_id'], d['sample'])] = d
    B = [r for r in rows if r['belief'] and (r['task_id'], r['sample'], r['pos']) in kept]

    def bugkey(r):
        code = src[(r['task_id'], r['sample'])]['code']
        line = code.splitlines()[code[:r['pos']].count('\n')]
        return (r['task_id'], ''.join(line.split()), r['wrong_id'])
    ban = []
    for r in B:
        gv = {v['to'] for v in r['verified'] if (r['task_id'], r['sample'], r['pos'], v['to']) not in bad}
        ban.append((1 if r['next_token'] in gv else 0) if r['next_token_known'] else -1)
    return make_structure([r['task_id'] for r in B], [bugkey(r) for r in B], ban,
                          {'source': 'design (g3_sim.py inputs; TESTS ONLY)',
                           'inputs_sha256': {p: pc.sha256(p) for p in (DA_SITES, G1_JSON)}})


def load_structure():
    if STRUCT == 'design':
        return design_structure()
    if STRUCT == 'sites':
        return sites_structure(SUBSET)
    raise SystemExit(f'STRUCT={STRUCT}?')


def _init(struct):
    S.update(struct)


# ---------------------------------------------------------------- simulation pieces (g3, g4)
def pvals(d):
    """d: (nds, N) integer site differences. The registered p, m and D of each dataset."""
    X = d.astype(np.int64) @ S['W']
    p = np.array([st.signflip_p(x) for x in X])
    return p, (X != 0).sum(axis=1), X.sum(axis=1) / (S['L'] * S['G'])


def shared_u(g, nds, rho):
    ub = g.random((nds, S['NB']))
    us = g.random((nds, S['N']))
    return np.where(g.random((nds, S['N'])) < rho, ub[:, S['site_bug']], us)


def ban_draw(g, nds, ufix):
    bb = (g.random((nds, S['NB'])) < ufix).astype(np.int64)
    k = S['ban_known'][None]
    return np.where(k >= 0, k, bb[:, S['site_bug']])


def fixfrac(ufix):
    """Expected equal-weight BAN fix share: unknown sites at ufix."""
    e = np.where(S['ban_known'] >= 0, S['ban_known'], ufix).astype(float)
    return float(np.mean(np.bincount(S['site_task'], weights=e, minlength=S['G']) / S['nsite']))


def rates_A(mode, q):
    """((rescue, break) of type-1 problems, (rescue, break) of type-2 problems), as g3. E[D] = 0 needs
    every rate <= 1 (no cap): q <= q_max_A(mode)."""
    if q > q_max_A(mode) + 1e-12:
        raise ValueError(f'A {mode}: q {q} > {q_max_A(mode)} would cap a rate and break E[D] = 0')
    if mode == 'hom':
        return (q / 2, q / 2), (q / 2, q / 2)
    if mode == 'skew+':
        c = q / (2 * (1 - F))
        return ((1 - F) * c / F, 0.0), (0.0, c)
    if mode == 'skew-':
        return (0.0, q / (2 * F)), (q / (2 * (1 - F)), 0.0)
    raise ValueError(mode)


def q_max_A(mode):
    """The largest discordant share A's mode can carry with every rate <= 1 (skew: F's problems carry half)."""
    return 1.0 if mode == 'hom' else 2 * F


def b1_for_q(mode, q, ff):
    """B's b1 whose expected discordant share is q."""
    return {'sym': q / (0.6 * ff), 'asym-': q / (2 * F * ff), 'asym+': q / (2 * F * (1 - ff))}[mode]


def b1_max(mode, ff):
    """The largest b1 at which every rate of B's mode is <= 1. Above it g3's min(1, .) would cap one side
    only and E[D] would no longer be 0 (the cell would measure power, not size)."""
    fl = 1 - ff
    return {'sym': min(1 / 0.3, fl / (0.3 * ff)), 'asym-': min(1.0, (1 - F) * fl / (F * ff)),
            'asym+': min(1.0, (1 - F) * ff / (F * fl))}[mode]


def summarise(cell, d, p, m, D):
    return {**cell, 'E_D': float(D.mean()), 'z_E_D': float(D.mean() / max(D.std(ddof=1) / np.sqrt(len(D)), 1e-12)),
            'm_median': float(np.median(m)), 'discordant_share': float((d != 0).mean()),
            'size': {str(a): float(np.mean(p <= a)) for a in ALPHAS}, 'n_datasets': len(p)}


def cell_A(c):
    g, nds = pc.rng(4, *c['key']), c['nds']
    (r1, b1), (r2, b2) = rates_A(c['mode'], c['q'])
    typ = g.random((nds, S['G'])) < F
    r = np.where(typ, r1, r2)[:, S['site_task']]
    b = np.where(typ, b1, b2)[:, S['site_task']]
    u = shared_u(g, nds, c['rho'])
    d = (u < r).astype(np.int64) - ((u >= r) & (u < r + b)).astype(np.int64)
    return summarise(c, d, *pvals(d))


def cell_B(c):
    g, nds = pc.rng(4, *c['key']), c['nds']
    ban = ban_draw(g, nds, c['ufix'])
    ff = fixfrac(c['ufix'])
    fl = 1 - ff
    typ = (g.random((nds, S['G'])) < F)[:, S['site_task']]
    b1 = min(c['b1'], b1_max(c['mode'], ff))          # g3's grid never needs this on the design structure
    if c['mode'] == 'sym':
        bb = np.full((nds, S['N']), 0.3 * b1)
        rr = np.full((nds, S['N']), 0.3 * b1 * ff / fl)
    elif c['mode'] == 'asym-':
        bb = np.where(typ, b1, 0.0)
        rr = np.where(typ, 0.0, F * b1 * ff / ((1 - F) * fl))
    else:
        rr = np.where(typ, b1, 0.0)
        bb = np.where(typ, 0.0, F * b1 * fl / ((1 - F) * ff))
    assert max(np.max(bb), np.max(rr)) <= 1 + 1e-12
    u = shared_u(g, nds, c['rho'])
    d = np.where(ban == 1, -(u < bb).astype(np.int64), (u < rr).astype(np.int64))
    return summarise({**c, 'fixfrac': ff, 'b1_used': b1, 'b1_capped': bool(b1 < c['b1'])}, d, *pvals(d))


def rate(p, z):
    if p <= 0:
        return np.zeros_like(z)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + z)))


def cell_C(c):
    g, nds = pc.rng(5, *c['key']), c['nds']
    G, N = S['G'], S['N']
    ban = ban_draw(g, nds, 0.5)
    z = g.normal(0, c['sd'], (nds, G))
    rg = rate(min(c['r'], 0.999), z)[:, S['site_task']]
    bg = rate(c['b'], z)[:, S['site_task']] if c['b'] > 0 else np.zeros((nds, N))
    u = shared_u(g, nds, 1.0)
    yg = np.where(ban == 1, 1 - (u < bg), (u < rg)).astype(np.int64)
    p2, m2, D2 = pvals(yg - ban)
    ub = g.random((nds, S['NB']))[:, S['site_bug']]
    p3, D3 = {}, {}
    for kappa in KAPPAS:
        p3[kappa], _, D3[kappa] = pvals(yg - np.where(ub < kappa, yg, ban))
    yr = (g.random((nds, S['NB'])) < P_RND).astype(np.int64)[:, S['site_bug']]
    p1, _, D1 = pvals(yg - yr)
    out = {**c, 'E_D_S2': float(D2.mean()), 'm_median_S2': float(np.median(m2)), 'E_D_S1': float(D1.mean()),
           'E_D_S3': {str(k): float(D3[k].mean()) for k in KAPPAS}, 'power': {}}
    for a in POWER_ALPHAS:
        out['power'][str(a)] = {
            'S1': float(np.mean(p1 <= a)), 'S2': float(np.mean(p2 <= a)),
            'S2&S3': {str(k): float(np.mean((p2 <= a) & (p3[k] <= a))) for k in KAPPAS},
            'all3': {str(k): float(np.mean((p1 <= a) & (p2 <= a) & (p3[k] <= a))) for k in KAPPAS}}
    return out


def run_cells(fn, cells, struct):
    if WORKERS <= 1:
        _init(struct)
        return [fn(c) for c in cells]
    with Pool(WORKERS, initializer=_init, initargs=(struct,)) as pool:
        return pool.map(fn, cells, chunksize=1)


# ---------------------------------------------------------------- grids
def g3_cells(nds):
    A = [{'section': 'A', 'mode': mo, 'q': q, 'rho': rho} for mo in A_MODES for q in G3_Q for rho in A_RHO]
    B = [{'section': 'B', 'ufix': uf, 'mode': mo, 'b1': b1, 'rho': rho}
         for uf in UFIX for mo in B_MODES for b1 in G3_B1 for rho in B_RHO]
    return ([{**c, 'key': (1, i), 'nds': nds} for i, c in enumerate(A)],
            [{**c, 'key': (2, i), 'nds': nds} for i, c in enumerate(B)])


def q_grid(q):
    return sorted({round(min(0.99, max(0.01, q + dq)), 4) for dq in (-0.1, -0.05, 0.0, 0.05, 0.1)})


def alpha_cells(qs, nds, struct):
    """The alpha rule's grid. A q (or B's b1) beyond what a mode can carry with every rate <= 1 is
    lowered to that maximum (a null must keep E[D] = 0); cells that coincide after this are run once,
    with every q they stand for listed in q_targets."""
    _init(struct)
    A, B = {}, {}
    for mo in A_MODES:
        for q in qs:
            qe = round(min(q, q_max_A(mo)), 12)
            for rho in A_RHO:
                A.setdefault((mo, qe, rho), {'section': 'A', 'mode': mo, 'q': qe, 'rho': rho, 'q_targets': [],
                                             'capped': False})
                A[(mo, qe, rho)]['q_targets'].append(q)
                A[(mo, qe, rho)]['capped'] |= qe < q
    for uf in UFIX:
        ff = fixfrac(uf)
        for mo in B_MODES:
            for q in qs:
                b1 = round(min(b1_for_q(mo, q, ff), b1_max(mo, ff)), 12)
                for rho in B_RHO:
                    c = B.setdefault((uf, mo, b1, rho), {'section': 'B', 'ufix': uf, 'mode': mo, 'b1': b1, 'rho': rho,
                                                         'q_targets': [], 'capped': False})
                    c['q_targets'].append(q)
                    c['capped'] |= b1 < b1_for_q(mo, q, ff) - 1e-12
    return ([{**c, 'key': (3, i), 'nds': nds} for i, c in enumerate(A.values())],
            [{**c, 'key': (4, i), 'nds': nds} for i, c in enumerate(B.values())])


def power_cells(nds):
    C = [{'b': b, 'r': r, 'sd': sd} for b in POW_B for r in POW_R for sd in POW_SD]
    return [{**c, 'key': (i,), 'nds': nds} for i, c in enumerate(C)]


def worst_of(rows):
    w = {}
    for a in ALPHAS:
        top = max(rows, key=lambda x: x['size'][str(a)])
        w[str(a)] = {'size': top['size'][str(a)], 'cell': {k: top[k] for k in top if k in
                     ('section', 'mode', 'q', 'q_targets', 'capped', 'rho', 'ufix', 'b1', 'b1_used', 'E_D')}}
    return w


def alpha_rule(worst):
    """§5: worst = {alpha: worst size}. Returns (alpha, reason, flag)."""
    if worst[0.03] <= 0.05:
        return 0.03, f'worst size at 0.03 is {worst[0.03]:.3f} <= 0.05: alpha stays 0.03', False
    for a in (0.025, 0.02):
        if worst[a] <= 0.05:
            return a, f'worst size at 0.03 is {worst[0.03]:.3f} > 0.05; {a} is the largest allowed alpha with worst size {worst[a]:.3f} <= 0.05', False
    return 0.02, (f'worst size at 0.03 is {worst[0.03]:.3f} > 0.05 and no allowed alpha holds size 0.05 '
                  f'(0.025: {worst[0.025]:.3f}, 0.02: {worst[0.02]:.3f}); alpha set to the floor 0.02'), True


def mde80(cells):
    """Per (b, sd, alpha): the E[D] at which S2 power reaches 0.8, linear between grid cells."""
    out = {}
    for a in POWER_ALPHAS:
        for b in POW_B:
            for sd in POW_SD:
                pts = sorted((c['E_D_S2'], c['power'][str(a)]['S2']) for c in cells if c['b'] == b and c['sd'] == sd)
                key = f'alpha={a} b={b} sd={sd:g}'
                hit = next((i for i, (_, pw) in enumerate(pts) if pw >= 0.8), None)
                if hit is None:
                    out[key] = f'> {pts[-1][0]:.3f} (power {pts[-1][1]:.2f} at the largest D)'
                elif hit == 0:
                    out[key] = f'<= {pts[0][0]:.3f}'
                else:
                    (d0, p0), (d1, p1) = pts[hit - 1], pts[hit]
                    out[key] = round(d0 + (0.8 - p0) * (d1 - d0) / (p1 - p0), 4)
    return out


# ---------------------------------------------------------------- the pilot's q
splice_key = st.splice_key          # lexeme_key of S with ids[h] := t (shared with p8_4's HF-BAN check)


def _fname(sid):
    return sid.replace('|', '_') + '.json'


def _read(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def pilot_run_dir(sites_path):
    """The pilot's RUN_DIR: env PILOT_RUN, else whichever of $P8_OUT/pilot/run, $P8_OUT/run and env
    RUN_DIR (p8_2's default is $P8_OUT/run) holds stage-A or stage-B files of pilot sites; two such
    directories are refused as ambiguous."""
    if PILOT_RUN:
        return PILOT_RUN, 'env PILOT_RUN'
    ids = [r['site_id'] for r in pc.read_jsonl(sites_path)]
    cands = [os.path.join(pc.OUT, 'pilot', 'run'), os.path.join(pc.OUT, 'run')]
    if os.environ.get('RUN_DIR'):
        cands.append(under_out(os.environ['RUN_DIR']))
    cands = list(dict.fromkeys(os.path.normpath(c) for c in cands))
    hit = [c for c in cands if any(os.path.exists(os.path.join(c, stg, _fname(s))) for s in ids for stg in ('A', 'B'))]
    if not hit:
        raise SystemExit(f'no pilot run outputs in {cands}; set PILOT_RUN')
    if len(hit) > 1:
        raise SystemExit(f'pilot run outputs in more than one directory {hit}; set PILOT_RUN')
    return hit[0], f'searched {cands}'


def main_stage_b_outputs(pilot_ids):
    """Stage-B files of MAIN sites in $P8_OUT/run/B and env RUN_DIR/B. A file is a main site's when its
    site_id is in $P8_OUT/sites.jsonl, or (no sites.jsonl yet) when it is not a pilot site. Pilot files,
    wherever they were written, never count."""
    p = os.path.join(pc.OUT, 'sites.jsonl')
    main_ids = {r['site_id'] for r in pc.read_jsonl(p)} if os.path.exists(p) else None
    dirs = [os.path.join(pc.OUT, 'run')] + ([under_out(os.environ['RUN_DIR'])] if os.environ.get('RUN_DIR') else [])
    found = []
    for d in dict.fromkeys(os.path.normpath(x) for x in dirs):
        for f in sorted(glob.glob(os.path.join(d, 'B', '*.json'))):
            sid = _read(f).get('site_id')
            if (sid in main_ids) if main_ids is not None else (sid not in pilot_ids):
                found.append(f)
    return found


def pilot_q(sites_path, run_dir, tok=None):
    """q (module docstring) and the futility rule (design §9) from the pilot's run outputs, label-free.
    Futility: G flips = pilot sites whose stage-B G FLIP has flipped true and a t* outside E (a t* whose
    splice has S's lexer stream is scored no_flip, as p8_3 does; an undecodable splice counts as a flip,
    as p8_3 scores it unscorable, not no_flip). Excluded sites and sites without a stage-B file count as
    not flipped; the denominator is every site of the pilot table (30). Fires when flips < 15."""
    sites = {r['site_id']: r for r in pc.read_jsonl(sites_path)}
    tok = tok or pc.tokenizer()
    out = {'pilot_sites': len(sites), 'sites_sha256': pc.sha256(sites_path), 'run_dir': run_dir,
           'missing': [], 'excluded': [], 'undecodable': [], 'differ': [], 'same': []}
    fut = {'rule': f'G flips at fewer than {FUTILITY_MIN} of the {FUTILITY_N} pilot sites -> build no main graphs; '
                   'run BAN, GxA and NULL descriptively (design §9)',
           'counted_as_not_flipped': 'excluded sites, sites without a stage-B file, flips whose t* is in E',
           'flipped': [], 'no_flip': [], 't_star_in_E': [], 'excluded': [], 'no_stage_B': [], 'undecodable_flip': []}
    for sid in sorted(sites):
        pa, pb = os.path.join(run_dir, 'A', _fname(sid)), os.path.join(run_dir, 'B', _fname(sid))
        a = _read(pa) if os.path.exists(pa) else None
        b = _read(pb) if os.path.exists(pb) else None
        kS = pc.lexeme_key(sites[sid]['code'])
        kg = None                                    # G's lexeme (S's own when G did not flip)
        if b is None:
            fut['no_stage_B'].append(sid)
        elif b.get('excluded') or (a or {}).get('excluded'):
            fut['excluded'].append(sid)
        else:
            gg = b['G']
            try:
                kg = splice_key(sites[sid], gg['t_star_id'], tok) if gg.get('flipped') else kS
            except ValueError as e:
                out['undecodable'].append(f'{sid}: G: {e}')
                fut['undecodable_flip'].append(sid)
                fut['flipped'].append(sid)
            else:
                (fut['no_flip'] if not gg.get('flipped') else fut['t_star_in_E'] if kg == kS else fut['flipped']).append(sid)
        if a is None or b is None:
            out['missing'].append(sid)
            continue
        if a.get('excluded') or b.get('excluded'):
            out['excluded'].append(sid)
            continue
        ban = a['BAN']
        if not ban.get('flipped') or ban.get('t_star_id') is None:
            raise SystemExit(f'{sid}: BAN has no token')
        if kg is None:
            continue                                 # G's splice was undecodable (listed)
        try:
            kb = splice_key(sites[sid], ban['t_star_id'], tok)
        except ValueError as e:
            out['undecodable'].append(f'{sid}: BAN: {e}')
            continue
        out['differ' if kg != kb else 'same'].append(sid)
    fut.update(n_flips=len(fut['flipped']), denominator=len(sites), pilot_size_is_30=len(sites) == FUTILITY_N,
               fires=len(fut['flipped']) < FUTILITY_MIN)
    out['futility'] = fut
    n = len(out['differ']) + len(out['same'])
    if not n:
        raise SystemExit('no pilot site has both G and BAN outputs')
    out.update(q=len(out['differ']) / n, numerator=len(out['differ']), denominator=n)
    return out


# ---------------------------------------------------------------- modes
def header(struct, test_run):
    return {'created': time.strftime('%Y-%m-%d %H:%M:%S JST'), 'test_run': test_run,
            'script_sha256': pc.sha256(os.path.abspath(__file__)),
            'stats_sha256': pc.sha256(os.path.join(HERE, 'p8_4_stats.py')),
            'common_sha256': pc.sha256(os.path.join(HERE, 'p8_common.py')),
            'structure': struct['info'], 'sites_sha256': struct['info'].get('sites_sha256'),
            'params': {'STRUCT': STRUCT, 'SUBSET': SUBSET, 'N_DS': N_DS, 'N_POW': N_POW, 'F': F, 'P_RND': P_RND}}


def mode_size(struct):
    t0 = time.time()
    A, B = g3_cells(N_DS)
    rows = run_cells(cell_A, A, struct) + run_cells(cell_B, B, struct)
    test_run = STRUCT != 'sites' or SUBSET != 'all' or N_DS < 2000
    out = {**header(struct, test_run), 'grid': "g3's", 'cells': rows, 'worst_size': worst_of(rows), 'seconds': time.time() - t0}
    pc.write_json(os.path.join(pc.OUT, 'sim_size.json'), out)
    print('worst size over A and B:', {a: round(v['size'], 3) for a, v in out['worst_size'].items()})
    return out


def mode_alpha(struct):
    t0 = time.time()
    dest = os.path.join(pc.OUT, 'alpha.json')
    if os.path.exists(dest) and not OVERWRITE_ALPHA and not _read(dest).get('test_run', True):
        raise SystemExit(f'{dest} holds a real alpha; it is never overwritten (OVERWRITE_ALPHA=1 to override)')
    pilot_ids = {r['site_id'] for r in pc.read_jsonl(PILOT_SITES)} if os.path.exists(PILOT_SITES) else set()
    built = main_stage_b_outputs(pilot_ids)
    if built and not AFTER_STAGE_B:
        raise SystemExit(f'{len(built)} main stage-B outputs exist (e.g. {built[0]}): alpha is fixed before any main '
                         'graph is built (§5); AFTER_STAGE_B=1 to override')
    if Q_ENV is not None:
        qinfo = {'q': float(Q_ENV), 'source': 'env Q (test override)',
                 'futility': {'computed': False, 'why': 'Q override (test run): no pilot outputs read'}}
    else:
        rdir, how = pilot_run_dir(PILOT_SITES)
        qinfo = {**pilot_q(PILOT_SITES, rdir), 'source': 'pilot run outputs', 'run_dir_found_by': how}
        fu = qinfo['futility']
        print(f"futility (design §9): G flips at {fu['n_flips']} of {fu['denominator']} pilot sites "
              f"(excluded {len(fu['excluded'])}, no stage B {len(fu['no_stage_B'])}, t* in E {len(fu['t_star_in_E'])}) -> "
              + ('FIRES: build no main graphs; run BAN, GxA and NULL descriptively' if fu['fires'] else 'does not fire'))
    q = qinfo['q']
    qs = q_grid(q)
    A, B = alpha_cells(qs, N_DS, struct)
    rows = run_cells(cell_A, A, struct) + run_cells(cell_B, B, struct)
    worst = worst_of(rows)
    alpha, why, flag = alpha_rule({float(a): v['size'] for a, v in worst.items()})
    test_run = STRUCT != 'sites' or SUBSET != 'all' or N_DS < 2000 or Q_ENV is not None
    out = {**header(struct, test_run), 'alpha': alpha, 'rule': why, 'no_allowed_alpha_holds': flag,
           'futility': qinfo['futility'], 'built_main_stage_b_overridden': built, 'q': qinfo, 'q_grid': qs, 'worst_size': worst, 'n_cells': len(rows),
           'capped_cells': sum(bool(c['capped']) for c in rows), 'max_abs_E_D': max(abs(c['E_D']) for c in rows),
           'max_abs_z_E_D': max(abs(c['z_E_D']) for c in rows), 'cells': rows, 'seconds': time.time() - t0}
    pc.write_json(dest, out)
    print(f"q = {q:.3f} ({qinfo['source']}); grid {qs}; {len(rows)} cells ({out['capped_cells']} capped), "
          f"max |E[D]| {out['max_abs_E_D']:.4f} (max |z| {out['max_abs_z_E_D']:.2f})")
    print('worst size:', {a: round(v['size'], 3) for a, v in worst.items()})
    print(f'alpha = {alpha}: {why}' + ('  [TEST RUN]' if test_run else ''))
    if out['futility'].get('fires'):
        print('FUTILITY STOP FIRES (alpha.json futility.fires = true): build no main graphs (design §9)')
    return out


def mode_power(struct):
    t0 = time.time()
    try:
        alpha, ainfo = st.get_alpha(pc.OUT)
    except SystemExit as e:
        alpha, ainfo = pc.ALPHA, {'value': pc.ALPHA, 'source': f'p8_common.ALPHA (alpha.json not usable: {e})'}
    rows = run_cells(cell_C, power_cells(N_POW), struct)
    test_run = STRUCT != 'sites' or SUBSET != 'all' or N_POW < 1000
    m = mde80(rows)
    out = {**header(struct, test_run), 'alpha': ainfo, 'cells': rows, 'mde80_all': m,
           'mde80': {k: v for k, v in m.items() if k.startswith(f'alpha={alpha} ')}, 'seconds': time.time() - t0}
    pc.write_json(os.path.join(pc.OUT, 'sim_power.json'), out)
    for c in rows:
        pw = c['power'][str(alpha)]
        print(f"b={c['b']:.2f} r={c['r']:.2f} sd={c['sd']:.0f} E[D]={c['E_D_S2']:+.3f} m_med={c['m_median_S2']:4.1f} "
              f"S1={pw['S1']:.3f} S2={pw['S2']:.3f} | " + ' '.join(f"k={k}: D3={c['E_D_S3'][k]:.3f} S2&S3={pw['S2&S3'][k]:.3f}"
                                                               for k in pw['S2&S3']))
    print('80% MDE of S2 at alpha', alpha, out['mde80'])
    return out


# ---------------------------------------------------------------- selftest
def parse_g3():
    cells = {}
    for line in open(G3_LOG, encoding='utf-8'):
        mA = re.match(r'^A (\S+)\s+q=([\d.]+) rho=([\d.]+)', line)
        mB = re.match(r'^B (\S+)\s+u=([\d.]+) b1=([\d.]+) rho=([\d.]+)', line)
        if not (mA or mB):
            continue
        size = {a: float(v) for a, v in re.findall(r'(0\.\d+):(\d\.\d+)', line)}
        if mA:
            cells[('A', mA[1], float(mA[2]), float(mA[3]))] = size
        else:
            cells[('B', mB[1], float(mB[2]), float(mB[3]), float(mB[4]))] = size
    return cells


def parse_g4():
    cells = {}
    for line in open(G4_LOG, encoding='utf-8'):
        m = re.match(r'^b=([\d.]+) r=([\d.]+) sd=(\d)\s+E\[D\]=([+-][\d.]+).*?S2@\.03=([\d.]+)', line)
        if m:
            j = {float(k): float(v) for k, v in re.findall(r'k=([\d.]+): D3=[\d.]+ S2&S3@\.03=([\d.]+)', line)}
            cells[(float(m[1]), float(m[2]), float(m[3]))] = {'E_D': float(m[4]), 'S2': float(m[5]), 'S2&S3': j}
    return cells


def z(a, b, n1, n2):
    """Two-proportion z; the pooled p is kept inside [1/(n1+n2), 1-1/(n1+n2)] (both 0 or both 1 give 0)."""
    e = 1 / (n1 + n2)
    p = min(max((a * n1 + b * n2) / (n1 + n2), e), 1 - e)
    return (a - b) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))


def selftest():
    if os.path.abspath(pc.OUT) == os.path.abspath(os.path.join(HERE, 'results', 'intervention')):
        raise SystemExit('selftest writes files: set P8_OUT to a scratch directory')
    global STRUCT, Q_ENV, N_DS, PILOT_SITES, PILOT_RUN, OVERWRITE_ALPHA
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'pass' if cond else 'FAIL'}] {name} {detail}", flush=True)

    t0 = time.time()
    print('1. the design-stage structure (fallback, tests only)')
    struct = design_structure()
    info = struct['info']
    check('146 sites, 31 problems, 94 bugs, L 207480', (info['sites'], info['problems'], info['bugs'], info['L']) ==
          (146, 31, 94, 207480), str((info['sites'], info['problems'], info['bugs'], info['L'])))
    check('per-problem counts equal §2', info['per_problem_sorted'] == st.DESIGN_COUNTS)
    check('HF BAN fix/fail/unknown 79/17/50', info['hf_ban_fix_fail_unknown'] == [79, 17, 50], str(info['hf_ban_fix_fail_unknown']))

    print('2. the simulation scores datasets with the analysis\'s own statistic and p')
    _init(struct)
    g = np.random.default_rng(3)                                   # selftest draws only
    d = g.integers(-1, 2, (40, struct['N'])) * (g.random((40, struct['N'])) < 0.15)
    p, m, D = pvals(d)
    same = True
    prob = struct['site_task'].tolist()
    for i in range(len(d)):
        _, _, X, L = st.problem_X(d[i].tolist(), prob)
        t = st.signflip(X)
        same &= np.array_equal(X, d[i] @ struct['W']) and abs(t['p'] - p[i]) < 1e-15 and t['m'] == m[i] \
            and abs(t['T'] / (L * len(X)) - D[i]) < 1e-15
    check('d @ W equals p8_4_stats.problem_X, and p, m, D agree (40 datasets)', same,
          f'(m range {int(m.min())}-{int(m.max())})')

    print('3. the alpha rule')
    cases = [({0.03: 0.045, 0.025: 0.04, 0.02: 0.035}, 0.03, False), ({0.03: 0.05, 0.025: 0.04, 0.02: 0.03}, 0.03, False),
             ({0.03: 0.055, 0.025: 0.048, 0.02: 0.04}, 0.025, False), ({0.03: 0.06, 0.025: 0.052, 0.02: 0.049}, 0.02, False),
             ({0.03: 0.07, 0.025: 0.06, 0.02: 0.055}, 0.02, True)]
    check('five hand cases', all(alpha_rule(w)[0] == a and alpha_rule(w)[2] == f for w, a, f in cases))
    check('q grid', q_grid(0.3) == [0.2, 0.25, 0.3, 0.35, 0.4] and q_grid(0.05) == [0.01, 0.05, 0.1, 0.15],
          f'{q_grid(0.3)} {q_grid(0.05)}')

    print("4. the pilot's q: decode-splice-lex on a real tokenization")
    tok = pc.tokenizer()
    prose = 'Write a function to find the larger of two integers.'
    code = 'fn larger(a: i32, b: i32) -> i32 {\n    if a > b { a } else { b }\n}'
    enc = tok(prose + ' ' + code, return_offsets_mapping=True)
    ids, offs = enc['input_ids'], enc['offset_mapping']
    at = len(prose) + 1 + code.index(' > ') + 1
    h = next(t for t in range(1, len(offs)) if offs[t][0] <= at < offs[t][1])
    one = lambda s: tok.encode(s, add_special_tokens=False)
    t_lt, t_lt_ns, t_ge = one(' <'), one('<'), one(' >=')
    check('test tokens are single pieces, h holds " >"', all(len(x) == 1 for x in (t_lt, t_lt_ns, t_ge)) and
          tok.decode([ids[h]]) == ' >', f'piece {tok.decode([ids[h]])!r}')
    pdir = os.path.join(pc.OUT, 'pilot_fixture')
    for stg in ('A', 'B'):
        os.makedirs(os.path.join(pdir, 'run', stg), exist_ok=True)
    fl = lambda arm, t: {'arm': arm, 'flipped': t is not None, 't_star_id': t, 'k_star': 1 if t is not None else None}
    if os.path.exists(pdir):
        shutil.rmtree(pdir)
    for stg in ('A', 'B'):
        os.makedirs(os.path.join(pdir, 'run', stg))
    # site: (G's t*, stage-B exclusion, has a stage-B file); 1|6|1's G 'flips' to S's own token (t* in E)
    plan = {'1|1|1': (t_lt[0], None, True), '1|2|1': (t_lt_ns[0], None, True), '1|3|1': (t_ge[0], None, True),
            '1|4|1': (None, None, True), '1|5|1': (t_ge[0], 'c: attribute failed twice', True),
            '1|6|1': (ids[h], None, True), '1|7|1': (t_lt[0], None, False)}
    with open(os.path.join(pdir, 'pilot_sites.jsonl'), 'w', encoding='utf-8') as f:
        for sid, (tg, exc, hasb) in plan.items():
            f.write(json.dumps({'site_id': sid, 'kind': 'pilot', 'prose': prose, 'code': code, 'ids': ids, 'h': h}) + '\n')
            fn = sid.replace('|', '_') + '.json'
            with open(os.path.join(pdir, 'run', 'A', fn), 'w', encoding='utf-8') as g2:
                json.dump({'site_id': sid, 'excluded': None, 'BAN': fl('BAN', t_lt[0])}, g2)
            if hasb:
                with open(os.path.join(pdir, 'run', 'B', fn), 'w', encoding='utf-8') as g2:
                    json.dump({'site_id': sid, 'excluded': exc, 'G': fl('G', tg)}, g2)
    qi = pilot_q(os.path.join(pdir, 'pilot_sites.jsonl'), os.path.join(pdir, 'run'), tok)
    check("q = 3/5 (same piece; '<' vs ' <' same lexeme; ' >=' differs; no flip and t* in E differ; excluded and "
          "no stage B dropped)",
          qi['q'] == 0.6 and qi['same'] == ['1|1|1', '1|2|1'] and qi['differ'] == ['1|3|1', '1|4|1', '1|6|1']
          and qi['excluded'] == ['1|5|1'] and qi['missing'] == ['1|7|1'], f"q {qi['q']} same {qi['same']} differ {qi['differ']}")
    fu = qi['futility']
    check('futility: 3 flips of 7 (no flip, t* in E, excluded, no stage B are not flips); fires (< 15)',
          fu['flipped'] == ['1|1|1', '1|2|1', '1|3|1'] and fu['no_flip'] == ['1|4|1'] and fu['t_star_in_E'] == ['1|6|1']
          and fu['excluded'] == ['1|5|1'] and fu['no_stage_B'] == ['1|7|1'] and fu['denominator'] == 7 and fu['fires']
          and not fu['pilot_size_is_30'], f"({fu['n_flips']} of {fu['denominator']})")
    check('env paths: relative under P8_OUT, absolute as given',
          under_out('pilot_sites.jsonl') == os.path.join(pc.OUT, 'pilot_sites.jsonl') and under_out('/x/y') == '/x/y')

    print("5. size: g3's 132 cells on the design structure (N_DS = 2000), against final/g3_sim.log")
    A, B = g3_cells(2000)
    rows = run_cells(cell_A, A, struct) + run_cells(cell_B, B, struct)
    ref = parse_g3()
    zs, worst_ref = [], {}
    for r in rows:
        k = ('A', r['mode'], r['q'], r['rho']) if r['section'] == 'A' else ('B', r['mode'], r['ufix'], r['b1'], r['rho'])
        for a in ALPHAS:
            zs.append(z(r['size'][str(a)], ref[k][str(a)], 2000, 2000))
            if ref[k][str(a)] > worst_ref.get(a, (-1,))[0]:
                worst_ref[a] = (ref[k][str(a)], k, r['size'][str(a)])
    zs = np.array(zs)
    check('every one of the 132 cells is in the log', len(ref) == 132 and len(rows) == 132)
    check("no cell of g3's grid needs a cap on the design structure (same nulls as g3)",
          not any(r.get('b1_capped') for r in rows))
    mine = worst_of(rows)
    for a in ALPHAS:
        v, k, my = worst_ref[a]
        zz = z(my, v, 2000, 2000)
        check(f"alpha {a}: g3's worst cell {k}", abs(zz) <= 3, f'g3 {v:.3f}, here {my:.3f} (z {zz:+.2f}); '
              f"worst here {mine[str(a)]['size']:.3f} at {mine[str(a)]['cell']}")
    check('all 660 cell x alpha comparisons: |z| > 3 in <= 2%', np.mean(np.abs(zs) > 3) <= 0.02,
          f'(mean z {zs.mean():+.2f}, sd {zs.std():.2f}, |z|>3: {int((np.abs(zs) > 3).sum())}, max |z| {np.abs(zs).max():.2f})')

    print("6. power: g4's 30 cells (N_POW = 1000), against final/g4_pow03.log")
    prow = run_cells(cell_C, power_cells(1000), struct)
    ref4 = parse_g4()
    z2, z3, dd = [], [], []
    for c in prow:
        rr = ref4[(c['b'], c['r'], c['sd'])]
        z2.append(z(c['power']['0.03']['S2'], rr['S2'], 1000, 1000))
        dd.append(c['E_D_S2'] - rr['E_D'])
        for k in KAPPAS:
            z3.append(z(c['power']['0.03']['S2&S3'][str(k)], rr['S2&S3'][k], 1000, 1000))
    z2, z3 = np.array(z2), np.array(z3)
    check('S2 power at 0.03: 30 cells within MC error', np.max(np.abs(z2)) <= 3.5,
          f'(max |z| {np.abs(z2).max():.2f}, mean z {z2.mean():+.2f}; max |E[D] diff| {np.abs(dd).max():.3f})')
    check('S2&S3 power at 0.03: 90 cells within MC error', np.mean(np.abs(z3) > 3) <= 0.02,
          f'(max |z| {np.abs(z3).max():.2f}, mean z {z3.mean():+.2f})')
    m80 = mde80(prow)
    print('     80% MDE of S2 at 0.03:', {k: v for k, v in m80.items() if k.startswith('alpha=0.03 ')})

    print('7. MODE=alpha end to end (STRUCT=design, Q=0.2, N_DS=400: a test run)')
    STRUCT, Q_ENV, N_DS = 'design', '0.2', 400
    dest = os.path.join(pc.OUT, 'alpha.json')
    if os.path.exists(dest):
        os.remove(dest)
    out = mode_alpha(struct)
    check('alpha.json written, marked test_run, q grid 0.1..0.3', out['test_run'] and out['q_grid'] == [0.1, 0.15, 0.2, 0.25, 0.3]
          and os.path.exists(dest), f"alpha {out['alpha']}")
    check('every null cell keeps E[D] = 0 within MC error (max |z| < 4)', out['max_abs_z_E_D'] < 4,
          f"(max |E[D]| {out['max_abs_E_D']:.4f}, max |z| {out['max_abs_z_E_D']:.2f}; "
          f"{out['capped_cells']} of {out['n_cells']} cells capped)")
    try:
        st.get_alpha(pc.OUT)
        check('p8_4 refuses the test alpha.json', False)
    except SystemExit:
        check('p8_4 refuses the test alpha.json', True)

    print("8. STRUCT=sites reads p8_1's schema (hf_runner_up) and rebuilds the same structure")
    rows = []
    for i in range(struct['N']):
        b = int(struct['ban_known'][i])
        rows.append({'site_id': f'{i}|0|0', 'kind': 'belief', 'problem': struct['problems'][struct['site_task'][i]],
                     'bug': f"bug{struct['site_bug'][i]}", 'tier': 1, 'order': i,
                     'hf_runner_up': {'known': b >= 0, 'to': 'x' if b >= 0 else None,
                                      'label': ('verified' if b == 1 else ['fail', 'lucky'][i % 2]) if b >= 0 else None}})
    rows += [{'site_id': f'{i}|9|9', 'kind': 'slip', 'problem': 1, 'bug': 'slip', 'hf_runner_up': {'known': False}}
             for i in range(3)]
    with open(os.path.join(pc.OUT, 'sites.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    s2 = sites_structure('all')
    part = lambda s: sorted(sorted(np.flatnonzero(s['site_bug'] == b).tolist()) for b in range(s['NB']))
    check('same sites, W, HF BAN and bug partition as the design structure (slips dropped)',
          s2['N'] == struct['N'] and np.array_equal(s2['W'], struct['W']) and np.array_equal(s2['ban_known'], struct['ban_known'])
          and part(s2) == part(struct), f"(labels {s2['info']['hf_runner_up_labels']})")

    print('9. MODE=alpha as the real run does it: STRUCT=sites, N_DS=2000, q from the pilot outputs (fixture of 7)')
    STRUCT, Q_ENV, N_DS = 'sites', None, 2000
    PILOT_SITES, PILOT_RUN = os.path.join(pdir, 'pilot_sites.jsonl'), os.path.join(pdir, 'run')
    os.remove(dest)
    out = mode_alpha(s2)
    a_used = st.get_alpha(pc.OUT)[0] if not out['test_run'] else None
    check('not a test run; q 0.6 from the pilot; grid 0.5..0.7; capped cells present; E[D] = 0 within MC error; '
          'futility recorded',
          not out['test_run'] and out['q']['q'] == 0.6 and out['q_grid'] == [0.5, 0.55, 0.6, 0.65, 0.7]
          and out['capped_cells'] > 0 and out['max_abs_z_E_D'] < 4 and out['futility']['fires'] is True,
          f"({out['n_cells']} cells, {out['capped_cells']} capped, max |E[D]| {out['max_abs_E_D']:.4f} "
          f"(|z| {out['max_abs_z_E_D']:.2f}); worst size at 0.03 {out['worst_size']['0.03']['size']:.3f})")
    check('the rule picked an allowed alpha and p8_4 uses it', out['alpha'] in (0.03, 0.025, 0.02) and a_used == out['alpha'],
          f"(alpha {out['alpha']}: {out['rule']})")
    try:
        mode_alpha(s2)
        check('a real alpha.json is never overwritten without OVERWRITE_ALPHA', False)
    except SystemExit as e:
        check('a real alpha.json is never overwritten without OVERWRITE_ALPHA', 'never overwritten' in str(e))

    print("10. the stage-B guard and the pilot run dir: p8_2's default RUN_DIR ($P8_OUT/run) used by the pilot")
    mrun = os.path.join(pc.OUT, 'run')
    if os.path.exists(mrun):
        shutil.rmtree(mrun)
    shutil.copytree(os.path.join(pdir, 'run'), mrun)
    PILOT_RUN = None
    pids = {r['site_id'] for r in pc.read_jsonl(PILOT_SITES)}
    rd, how = pilot_run_dir(PILOT_SITES)
    check('pilot files in $P8_OUT/run: found there, and they do not trip the stage-B guard',
          os.path.normpath(rd) == os.path.normpath(mrun) and main_stage_b_outputs(pids) == [], f'({how})')
    with open(os.path.join(mrun, 'B', '0_0_0.json'), 'w', encoding='utf-8') as f:
        json.dump({'site_id': '0|0|0', 'excluded': None}, f)
    OVERWRITE_ALPHA = True                         # past the overwrite refusal, to reach the guard
    try:
        mode_alpha(s2)
        check("a main site's stage-B file stops MODE=alpha (separately from the overwrite flag)", False)
    except SystemExit as e:
        check("a main site's stage-B file stops MODE=alpha (separately from the overwrite flag)",
              'main stage-B outputs exist' in str(e) and len(main_stage_b_outputs(pids)) == 1, f'({str(e)[:70]}...)')
    finally:
        OVERWRITE_ALPHA = False
        shutil.rmtree(mrun)
    print(f"selftest {'PASSED' if ok else 'FAILED'} ({time.time() - t0:.0f} s)")
    if not ok:
        sys.exit(1)


def main():
    if MODE == 'selftest':
        selftest()
        return
    struct = load_structure()
    print(f"structure: {json.dumps({k: v for k, v in struct['info'].items() if k != 'per_problem_sorted'})}")
    print(f"per problem {struct['info']['per_problem_sorted']}")
    {'size': mode_size, 'alpha': mode_alpha, 'power': mode_power}[MODE](struct)


if __name__ == '__main__':
    main()
