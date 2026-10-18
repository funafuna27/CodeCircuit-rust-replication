"""Phase 8, step 4: the analysis (final_design.md §5, §6; SPEC 'analysis') from the judged outcomes.
CPU only: no model, no judge. Labels enter only through judged.jsonl (p8_3).

Reads   $P8_OUT/sites.jsonl          site table (p8_1): kind, problem, tier, order, rule_b, family, h
        $P8_OUT/judged.jsonl         one row per (site, arm): Y, label, how, lexeme_is_ban, neutralizing
        $P8_OUT/alpha.json           the post-pilot alpha (p8_sim MODE=alpha), if present
        $P8_OUT/run/{A,B,C}/*.json   run outputs (p8_2): descriptive secondaries only, if present
        $P8_OUT/sim_power.json       the 80% MDE of S2 (p8_sim MODE=power), if present
Writes  $P8_OUT/analysis.json

CONFIRMATORY (the only part of analysis.json that can reject anything: 'confirmatory')
  analysis set  belief sites in completed rounds, minus excluded sites; never replaced.
                A site's round is its rank by `order` within its (tier, problem). Round (tier, j) is
                complete when each of its sites has judged rows for G, RND1, BAN and GxA, or is
                excluded. A site is excluded only by a design §2 (a)-(d) reason in its run/A or run/B
                file (p8_2 tags them 'a:'..'d:'); every stage's reason is reported. A primary row
                labelled 'excluded' without such a reason (a stage-C failure, 'A:/B: failed in two
                fresh processes', no run file) stops a real run: labels may not remove a site.
                A dry run drops such sites and lists them under unregistered_exclusions.
  statistic     problem g with n_g analysis sites: X_g = (L / n_g) * sum_i (Y_a,i - Y_b,i), where
                L = lcm{n_g} (207480 for the §2 counts); T = sum_g X_g is an integer and
                D = T / (L G) is the equal-weight mean over problems of each problem's mean difference.
  p             one-sided problem-level sign flip, P(sum_g s_g X_g >= T) with s uniform on {-1,1}^G;
                m = #{g : X_g != 0} is reported. m <= 24: exact, counting all 2^m sign vectors of the
                discordant problems (meet in the middle; the selftest checks it against literal
                enumeration). m > 24: (1 + #{b : T_b >= T}) / 100000 over 99,999 sign vectors of all
                G problems in ascending problem id, s = 1 - 2 * rng(0).integers(0, 2, (99999, G)),
                rng(0) = default_rng([ROOT, 0]); the identity is the +1.
  sequence      S1 G > RND1, S2 G > BAN, S3 G > GxA (p8_common.PRIMARY_ORDER), each at alpha; the
                sequence stops at the first p > alpha. Untested steps carry their p only as
                'p_descriptive_not_tested'.
  alpha         p8_common.ALPHA (0.03); alpha.json's alpha (0.025 or 0.02) when p8_sim's post-pilot
                rule lowered it. An alpha.json from a test run, or built on another sites.jsonl, is refused.

DESCRIPTIVE (everything under 'descriptive'; nothing there can change S1-S3)
  arm rates (equal weight over problems and per site, every arm in judged.jsonl incl. gen:*; Y = 1
  iff 'verified', so 'unverifiable' - a label p8_3 writes that SPEC's list lacks - scores 0), flip
  rates; RND-k draws ('RNDk:<draw>' rows) pooled per site; the decomposition D(G-BAN) = -NF + (R - Bk)
  (checked exactly with fractions); the rescue/break table of every ordered arm against BAN; the
  landing table (no flip / BAN's lexeme / other family member / outside the family / unscorable);
  P(fix | both flip) for G against GxA; sensitivity analyses of S1-S3 (tier 1 only; without rule-B
  sites; unscorable sites excluded; neutralizing verified rows scored 0 on both arms; both arms'
  rows scored by the lexeme rule - a no-flip row has no token and leaves the subset; sqrt(n)
  problem weights); from the run outputs: k*, scan/replay disagreements, G's first feature
  (layer >= 18 at h-1; from stage B's e_top at every site, flipped or not), margin terciles,
  faithfulness, RND-k, collateral (shares of sites: G > 0; RND-k > 0 in most draws), generation stops,
  run/judge flip agreement; program fix, existence (judged arms named *exist*, else the run's
  'existence' fields), family, slips (NULL), the committed HF table against the TL run (HF
  runner-up x TL BAN, TL BAN's lexeme = HF runner-up's, slips' HF fix set vs NULL; the lexeme
  check loads the tokenizer), exclusions with reasons, the §6 predictions against their ranges,
  the 80% MDE from sim_power.json.
  Intervals: percentile, 5000 bootstrap draws of whole problems, rng(1, G) for a set of G problems.

Usage (cwd data/)
  python p8_4_stats.py                                   the analysis -> $P8_OUT/analysis.json
  KIND=pilot SITES=$P8_OUT/pilot_sites.jsonl JUDGED=<pilot judged.jsonl> RUN_DIR=<pilot run dir> \
    ANALYSIS=<scratch>/analysis_dry.json python p8_4_stats.py      the dry run on the P1 outputs
  MODE=selftest P8_OUT=/home/user1/p8design/impl/B/selftest python p8_4_stats.py
Env  P8_OUT (p8_common)  SITES=$P8_OUT/sites.jsonl  JUDGED=$P8_OUT/judged.jsonl  RUN_DIR=$P8_OUT/run
     KIND=belief (the rows that form the analysis set)  ANALYSIS=$P8_OUT/analysis.json  N_BOOT=5000
     Relative SITES/JUDGED/RUN_DIR/ANALYSIS are under P8_OUT (p8_2/p8_3's rule); absolute ones as given.
  Any of SITES/JUDGED/RUN_DIR/KIND/ANALYSIS changed, or N_BOOT != 5000, makes a dry run: analysis.json
  says dry_run true, and an alpha.json built on another site table falls back to p8_common.ALPHA
  (a real run refuses it).
"""
import glob
import itertools
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from functools import reduce

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p8_common as pc  # noqa: E402

MODE = os.environ.get('MODE', 'run')
N_BOOT = int(os.environ.get('N_BOOT', 5000))
ENV_PATHS = {k: os.environ.get(k) for k in ('SITES', 'JUDGED', 'RUN_DIR', 'ANALYSIS')}
KIND = os.environ.get('KIND', 'belief')
N_FLIPS = 99999                    # registered: 99,999 flips plus the identity
EXACT_MAX_M = 24                   # registered: exact enumeration when m <= 24
PRIMARY_ARMS = ('G', 'RND1', 'BAN', 'GxA')
STAGE_B_ARMS = ('G', 'G_LP', 'G1', 'G_mid', 'RND1', 'RND2', 'RND3', 'RND4', 'RND5', 'ORC')
STAGE_A_ARMS = ('BAN', 'NULL', 'GxA', 'DE', 'CONTRAST')
ALLOWED_ALPHA = (0.03, 0.025, 0.02)
REGISTERED_EXCLUSIONS = ('a:', 'b:', 'c:', 'd:')   # p8_2's tags for design §2 (a)-(d); case matters ('B:' is a stage)
LABELS = ('verified', 'lucky', 'fail', 'no_compile', 'unverifiable', 'unscorable', 'no_flip', 'excluded')
# 'unverifiable' (stable pass, no calibrated extra inputs; p8_3 writes it) is not in SPEC's list; it scores 0.
LANDING = ('no_flip', 'ban_lexeme', 'other_family', 'outside_family', 'unscorable')
DESIGN_COUNTS = [19, 13, 13, 12, 10, 8, 8, 8, 7, 7, 5, 5, 4, 4, 3, 2, 2, 2, 2] + [1] * 12   # §2
DESCRIPTIVE = 'Descriptive only: nothing in this block can change S1-S3.'


# ---------------------------------------------------------------- the registered test
def lcm(ns):
    return reduce(lambda a, b: a * b // math.gcd(a, b), sorted({int(n) for n in ns}), 1)


def problem_X(d, prob):
    """d: per-site integer differences Y_a - Y_b; prob: per-site problem ids.
    Returns (problems ascending, n_g, X_g int64, L) with X_g = d_g * (L / n_g)."""
    s, n = defaultdict(int), Counter()
    for di, p in zip(d, prob):
        s[p] += int(di)
        n[p] += 1
    probs = sorted(n)
    L = lcm(n.values())
    X = np.array([s[g] * (L // n[g]) for g in probs], dtype=np.int64)
    return probs, np.array([n[g] for g in probs], dtype=np.int64), X, L


def _signed_sums(v):
    """All 2^len(v) sums sum_j s_j v_j, s in {-1,1}^len(v)."""
    out = np.zeros(1, dtype=v.dtype)
    for x in v:
        out = np.concatenate([out + x, out - x])
    return out


def count_ge(x, T, tol=0):
    """#{s in {-1,1}^len(x) : s.x >= T - tol}, exactly (meet in the middle: every sign vector of
    the first half against the sorted sums of the second)."""
    x = np.asarray(x)
    h = len(x) // 2
    sa, sb = _signed_sums(x[:h]), np.sort(_signed_sums(x[h:]))
    return int((len(sb) - np.searchsorted(sb, T - tol - sa, side='left')).sum())


def count_ge_literal(x, T):
    """The same count by literal enumeration of every sign vector (selftest only)."""
    return sum(1 for s in itertools.product((1, -1), repeat=len(x)) if int(np.dot(s, x)) >= T)


_FLIPS = {}


def flips(G):
    """The registered Monte Carlo sign vectors for G problems: rng(0) = default_rng([ROOT, 0])."""
    if G not in _FLIPS:
        _FLIPS[G] = 1 - 2 * pc.rng(0).integers(0, 2, size=(N_FLIPS, G), dtype=np.int64)
    return _FLIPS[G]


def p_mc(X, tol=0):
    Ts = flips(len(X)) @ X
    return (1 + int((Ts >= X.sum() - tol).sum())) / (N_FLIPS + 1)


def signflip_p(X):
    """The registered p only (the fast path p8_sim uses for every simulated dataset). X: int64 per problem."""
    nz = X[X != 0]
    if len(nz) <= EXACT_MAX_M:
        return count_ge(nz, X.sum()) / 2.0 ** len(nz)
    return p_mc(X)


def signflip(X, float_stat=False, both=False):
    """The registered test with its bookkeeping. float_stat: X is real (sqrt(n) weights, descriptive);
    ties then count within 1e-9 relative. both: also the other p (exact for m > 24 is a cross-check)."""
    X = np.asarray(X, dtype=np.float64 if float_stat else np.int64)
    m = int(np.count_nonzero(X))
    T = X.sum()
    tol = 1e-9 * (1.0 + float(np.abs(X).sum())) if float_stat else 0
    out = {'G': len(X), 'm': m, 'T': float(T) if float_stat else int(T)}
    if m <= EXACT_MAX_M or (both and m <= 40):
        c = count_ge(X[X != 0], T, tol)
        out.update(p_exact=c / 2 ** m, exact_count=c, exact_of=2 ** m)
    if m > EXACT_MAX_M or both:
        out['p_mc'] = p_mc(X, tol)
    if m <= EXACT_MAX_M:
        out.update(p=out['p_exact'], how=f'exact over 2^{m} sign vectors', min_attainable_p=2.0 ** -m)
    else:
        out.update(p=out['p_mc'], how='Monte Carlo: 99,999 sign vectors from rng(0) plus the identity',
                   min_attainable_p=1 / (N_FLIPS + 1))
    return out


# ---------------------------------------------------------------- bootstrap (descriptive)
_BOOT = {}


def boot_idx(G):
    """5000 resamples of G problems with replacement, rng(1, G)."""
    if G not in _BOOT:
        _BOOT[G] = pc.rng(1, G).integers(0, G, size=(N_BOOT, G))
    return _BOOT[G]


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def per_problem(sids, val, prob):
    """Per-problem sums of val(sid) and site counts, problems ascending."""
    s, n = defaultdict(float), Counter()
    for x in sids:
        s[prob[x]] += val(x)
        n[prob[x]] += 1
    probs = sorted(n)
    return probs, np.array([s[g] for g in probs]), np.array([n[g] for g in probs], dtype=float)


def summ(sids, val, prob, boot=True):
    """Equal-weight (over problems) and per-site means of val over sids, with problem-bootstrap CIs."""
    if not sids:
        return {'n_sites': 0, 'n_problems': 0, 'equal_weight': None, 'per_site': None}
    probs, s, n = per_problem(sids, val, prob)
    out = {'n_sites': len(sids), 'n_problems': len(probs), 'sum': float(s.sum()),
           'equal_weight': float(np.mean(s / n)), 'per_site': float(s.sum() / n.sum())}
    if boot and len(probs) > 1:
        idx = boot_idx(len(probs))
        out['equal_weight_ci95'] = ci((s / n)[idx].mean(axis=1))
        out['per_site_ci95'] = ci(s[idx].sum(axis=1) / n[idx].sum(axis=1))
    return out


# ---------------------------------------------------------------- lexemes of single-token splices
def splice_key(site, t, tok):
    """lexeme_key of S's code with ids[h] := t: decode(ids[1:]) (special tokens kept as text, no
    clean-up) after prose + ' ', as p8_1_sites.decode_code. Raises ValueError when the site's ids do
    not decode to its text or the splice reaches into the prose."""
    ids, h = list(site['ids']), site['h']
    pre = site['prose'] + ' '
    if ids[0] != tok.bos_token_id:
        raise ValueError('ids[0] is not BOS')
    dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if dec(ids[1:]) != pre + site['code']:
        raise ValueError('decoding the ids does not reproduce the text')
    new = dec(ids[1:h] + [int(t)] + ids[h + 1:])
    if not new.startswith(pre):
        raise ValueError('the splice changed the prose')
    return pc.lexeme_key(new[len(pre):])


# ---------------------------------------------------------------- inputs
def paths(out):
    """Input and output paths: env SITES / JUDGED / RUN_DIR / ANALYSIS, else the defaults under out.
    dry: anything but the defaults (or KIND != belief, N_BOOT != 5000)."""
    def at(k, default):                  # p8_2/p8_3's rule: absolute as given, relative under P8_OUT
        v = ENV_PATHS[k] or default
        return v if os.path.isabs(v) else os.path.join(out, v)
    p = {'sites': at('SITES', 'sites.jsonl'), 'judged': at('JUDGED', 'judged.jsonl'), 'run_dir': at('RUN_DIR', 'run'),
         'analysis': at('ANALYSIS', 'analysis.json'),
         'alpha': os.path.join(out, 'alpha.json'), 'sim_power': os.path.join(out, 'sim_power.json'), 'kind': KIND}
    p['dry_run'] = (any(ENV_PATHS.values()) or KIND != 'belief' or N_BOOT != 5000)
    return p


def load(P):
    sites = pc.read_jsonl(P['sites'])
    S = {r['site_id']: r for r in sites}
    if len(S) != len(sites):
        raise SystemExit('sites.jsonl: duplicate site_id')
    J, bad = {}, []
    for r in pc.read_jsonl(P['judged']):
        k = (r['site_id'], r['arm'])
        if k in J:
            raise SystemExit(f'judged.jsonl: duplicate row {k}')
        if r['site_id'] not in S:
            raise SystemExit(f'judged.jsonl: site {r["site_id"]} is not in sites.jsonl')
        if r['label'] not in LABELS or r['Y'] not in (0, 1) or (r['Y'] == 1) != (r['label'] == 'verified'):
            bad.append(k)
        J[k] = r
    if bad:
        raise SystemExit(f'judged.jsonl: {len(bad)} rows break Y = 1 iff label verified, e.g. {bad[:3]}')
    run = {}
    for st in ('A', 'B', 'C'):
        run[st] = {}
        for p in sorted(glob.glob(os.path.join(P['run_dir'], st, '*.json'))):
            with open(p, encoding='utf-8') as f:
                d = json.load(f)
            run[st][d['site_id']] = d
    return S, J, run


def get_alpha(out, sites_path=None, dry_run=False):
    """alpha for every step: p8_common.ALPHA, or alpha.json's (p8_sim MODE=alpha) when it exists. A real
    run refuses an alpha.json from a test run or built on another site table; a dry run falls back to
    p8_common.ALPHA instead (and says why)."""
    p = os.path.join(out, 'alpha.json')
    if not os.path.exists(p):
        return pc.ALPHA, {'value': pc.ALPHA, 'source': 'p8_common.ALPHA (no alpha.json)'}
    with open(p, encoding='utf-8') as f:
        a = json.load(f)
    sp = sites_path or os.path.join(out, 'sites.jsonl')
    sh = pc.sha256(sp) if os.path.exists(sp) else None
    why = ('it comes from a test run (or does not say)' if a.get('test_run', True) else
           f'the site table {sp} does not exist' if sh is None else
           f'it was computed on another site table ({a.get("sites_sha256")} != {sh})' if a.get('sites_sha256') != sh else None)
    if why and dry_run:
        return pc.ALPHA, {'value': pc.ALPHA, 'source': f'p8_common.ALPHA (dry run; {p} not used: {why})'}
    if why:
        raise SystemExit(f'{p}: {why}; refusing it')
    v = float(a['alpha'])
    if not any(abs(v - x) < 1e-12 for x in ALLOWED_ALPHA):
        raise SystemExit(f'{p}: alpha {v} is not one of {ALLOWED_ALPHA}')
    return v, {'value': v, 'source': 'alpha.json (p8_sim post-pilot rule)', 'sha256': pc.sha256(p),
               'q': a.get('q'), 'worst_size': a.get('worst_size'), 'rule': a.get('rule'), 'futility': a.get('futility')}


def stage_exclusions(run, sid):
    """{stage: reason} of every run file of the site that carries a non-null 'excluded'."""
    return {st: run[st][sid]['excluded'] for st in ('A', 'B', 'C') if (run[st].get(sid) or {}).get('excluded')}


def registered_reason(ex):
    """The design §2 (a)-(d) reason of a site, from its stage-A or stage-B file only, or None. p8_2 writes
    these reasons as 'a: ...', 'b: ...', 'c: ...', 'd: ...'; anything else ('A:/B:/C: failed in two fresh
    processes', any stage-C reason) is not a registered exclusion."""
    for st in ('A', 'B'):
        if ex.get(st, '').startswith(REGISTERED_EXCLUSIONS):
            return ex[st]
    return None


def analysis_set(S, J, run, kind='belief'):
    """Belief sites in completed rounds, minus excluded ones (module docstring). kind='pilot' for the
    dry run on the pilot table. A site is excluded only by a registered reason in its stage-A/B file;
    a primary row labelled 'excluded' without one is an 'unregistered exclusion' (returned in info;
    analyse() stops a real run on it, a dry run drops the site and reports it)."""
    belief = [r for r in S.values() if r['kind'] == kind]
    status, reasons, unreg, stale = {}, {}, {}, []
    for r in belief:
        sid = r['site_id']
        rows = [J.get((sid, a)) for a in PRIMARY_ARMS]
        ex = stage_exclusions(run, sid)
        reg = registered_reason(ex)
        if reg:
            status[sid] = 'excluded'
            if any(x is not None and x['label'] != 'excluded' for x in rows):
                stale.append(sid)
        elif any(x is not None and x['label'] == 'excluded' for x in rows):
            status[sid] = 'unregistered exclusion'
            unreg[sid] = {'stage_reasons': ex or 'no run file says excluded',
                          'excluded_rows': [a for a, x in zip(PRIMARY_ARMS, rows) if x is not None and x['label'] == 'excluded']}
        elif all(x is not None for x in rows):
            status[sid] = 'complete'
        else:
            status[sid] = 'not run'
        if ex:
            reasons[sid] = {'stages': ex, 'registered': reg is not None, 'reason': reg}
    by_tp = defaultdict(list)
    for r in belief:
        by_tp[(r['tier'], r['problem'])].append(r)
    rnd = {}
    for rs in by_tp.values():
        for j, r in enumerate(sorted(rs, key=lambda r: r['order'])):
            rnd[r['site_id']] = (int(r['tier']), j)
    rounds = defaultdict(list)
    for sid, k in rnd.items():
        rounds[k].append(sid)
    done = {k: all(status[s] != 'not run' for s in v) for k, v in rounds.items()}
    seq = [rnd[r['site_id']] for r in sorted(belief, key=lambda r: r['order'])]
    aset = sorted(s for s in status if status[s] == 'complete' and done[rnd[s]])
    info = {
        'belief_sites': len(belief),
        'status_counts': dict(Counter(status.values())),
        'rounds': len(rounds), 'rounds_complete': sum(done.values()),
        'order_is_round_robin': all(a <= b for a, b in zip(seq, seq[1:])),
        'excluded': sorted(s for s in status if status[s] == 'excluded'),
        'run_but_in_incomplete_round': sorted(s for s in status if status[s] == 'complete' and not done[rnd[s]]),
        'not_run': sum(v == 'not run' for v in status.values()),
        'excluded_in_complete_rounds': sum(status[s] == 'excluded' and done[rnd[s]] for s in status),
        'exclusion_reasons': {s: {**reasons[s], 'in_analysis_set': s in aset} for s in sorted(reasons)},
        'exclusion_rule': 'registered (design §2 a-d) reason in the stage-A or stage-B file; stage-C and '
                          "'<stage>: failed in two fresh processes' reasons are not exclusions",
        'unregistered_exclusions': unreg,
        'registered_but_judged_rows_not_excluded': sorted(stale),
    }
    return aset, info, rnd


# ---------------------------------------------------------------- contrasts
def contrast(sids, ya, yb, prob, both=True):
    d = [ya(s) - yb(s) for s in sids]
    probs, n, X, L = problem_X(d, [prob[s] for s in sids])
    G = len(probs)
    t = signflip(X, both=both)
    out = {'n_sites': len(sids), 'n_problems': G, 'L': L, 'D': t['T'] / (L * G) if G else None, **t}
    if G > 1:
        out['D_ci95'] = ci((X / L)[boot_idx(G)].mean(axis=1))
    return out


def contrast_sqrt(sids, ya, yb, prob):
    """sqrt(n) problem weights (descriptive): sum_g sqrt(n_g) dbar_g / sum_g sqrt(n_g)."""
    d = [ya(s) - yb(s) for s in sids]
    probs, n, X, L = problem_X(d, [prob[s] for s in sids])
    Xf = (X / L) * np.sqrt(n)
    t = signflip(Xf, float_stat=True)
    return {'n_sites': len(sids), 'n_problems': len(probs), 'D_sqrt_n': float(Xf.sum() / np.sqrt(n).sum()), **t}


def decomposition(sids, J, prob):
    """D(G-BAN) = -NF + (R - Bk), each term an equal-weight mean over problems of site indicators."""
    ind = {}
    for s in sids:
        g, b = J[(s, 'G')], J[(s, 'BAN')]
        flip = g['label'] != 'no_flip'
        if not flip and g['Y'] != 0:
            raise SystemExit(f'{s}: G did not flip but Y = 1')
        ind[s] = {'NF': int(not flip and b['Y'] == 1), 'R': int(flip and g['Y'] == 1 and b['Y'] == 0),
                  'Bk': int(flip and g['Y'] == 0 and b['Y'] == 1), 'd': g['Y'] - b['Y']}
    by = defaultdict(list)
    for s in sids:
        by[prob[s]].append(ind[s])
    G = len(by)
    frac = {k: sum(Fraction(sum(x[k] for x in v), len(v)) for v in by.values()) / G for k in ('NF', 'R', 'Bk', 'd')}
    if frac['d'] != -frac['NF'] + frac['R'] - frac['Bk']:
        raise SystemExit('decomposition identity failed')
    out = {k: float(v) for k, v in frac.items() if k != 'd'}
    out.update({'D': float(frac['d']), 'R_minus_Bk': float(frac['R'] - frac['Bk']),
                'identity_exact': True, 'n_sites': {k: sum(ind[s][k] for s in sids) for k in ('NF', 'R', 'Bk')}})
    for k in ('NF', 'R', 'Bk'):
        out[f'{k}_ci95'] = summ(sids, lambda s, k=k: ind[s][k], prob).get('equal_weight_ci95')
    out['R_minus_Bk_ci95'] = summ(sids, lambda s: ind[s]['R'] - ind[s]['Bk'], prob).get('equal_weight_ci95')
    out['neutralizing_rescues'] = sum(1 for s in sids if ind[s]['R'] and J[(s, 'G')].get('neutralizing'))
    return out


def landing(row):
    if row['label'] == 'no_flip':
        return 'no_flip'
    if row.get('lexeme_is_ban'):
        return 'ban_lexeme'
    if row['label'] == 'unscorable':
        return 'unscorable'
    return 'other_family' if row.get('how') == 'lexeme' else 'outside_family'


def landing_table(sids, J, arm, prob):
    rows = {s: J[(s, arm)] for s in sids if (s, arm) in J and J[(s, arm)]['label'] != 'excluded'}
    cat = {s: landing(r) for s, r in rows.items()}
    flips_ = [s for s in rows if cat[s] != 'no_flip']
    out = {'n_sites': len(rows), 'n_flips': len(flips_), 'counts': {c: sum(v == c for v in cat.values()) for c in LANDING},
           'share_of_flips_per_site': {c: (sum(cat[s] == c for s in flips_) / len(flips_) if flips_ else None)
                                       for c in LANDING[1:]},
           'fix_rate_per_site_by_landing': {c: (float(np.mean([rows[s]['Y'] for s in rows if cat[s] == c]))
                                                if any(cat[s] == c for s in rows) else None) for c in LANDING}}
    out['share_of_flips_equal_weight'] = {c: summ(flips_, lambda s, c=c: float(cat[s] == c), prob, boot=False)['equal_weight']
                                          for c in LANDING[1:]}
    return out


def rescue_break(sids, J, arm, prob):
    """arm against BAN at every site where both have a row: the 2x2 split by arm flip status."""
    ss = [s for s in sids if (s, arm) in J and J[(s, arm)]['label'] != 'excluded']
    cells = Counter()
    for s in ss:
        a, b = J[(s, arm)], J[(s, 'BAN')]
        f = 'flip' if a['label'] != 'no_flip' else 'no_flip'
        cells[f'{f}|arm{a["Y"]}|ban{b["Y"]}'] += 1
    named = {'no_flip & BAN fixes (NF)': cells['no_flip|arm0|ban1'], 'no_flip & BAN fails': cells['no_flip|arm0|ban0'],
             'flip & both fix': cells['flip|arm1|ban1'], 'flip & rescue (arm fixes, BAN fails)': cells['flip|arm1|ban0'],
             'flip & break (arm fails, BAN fixes)': cells['flip|arm0|ban1'], 'flip & both fail': cells['flip|arm0|ban0']}
    return {'n_sites': len(ss), 'per_site_counts': named,
            'neutralizing_rescues': sum(1 for s in ss if J[(s, arm)]['Y'] == 1 and J[(s, 'BAN')]['Y'] == 0
                                        and J[(s, arm)].get('neutralizing'))}


# ---------------------------------------------------------------- run outputs (descriptive)
def find_flip(run, s, arm):
    """The FLIP of an arm at a site from whichever stage file holds it (SPEC puts CONTRAST in stage A,
    the design runs it last; both are accepted), or None."""
    for st in ('A', 'B', 'C'):
        f = run[st].get(s, {}).get(arm)
        if isinstance(f, dict) and 'flipped' in f:
            return f
    return None


def kstar_stats(sids, run, arm):
    ks, dis, marg, n = [], 0, [], 0
    for s in sids:
        f = find_flip(run, s, arm)
        if f is None:
            continue
        n += 1
        dis += int(f.get('scan_replay_disagree') or 0)
        if f.get('flipped') and f.get('k_star') is not None:
            ks.append(f['k_star'])
            if f.get('margin_at_kstar') is not None:
                marg.append(f['margin_at_kstar'])
    if not n:
        return None
    ks = np.array(ks)
    return {'n_sites_with_output': n, 'n_flips': len(ks), 'flip_rate_per_site': len(ks) / n,
            'k_star_median': float(np.median(ks)) if len(ks) else None,
            'k_star_q25_q75': [float(x) for x in np.percentile(ks, [25, 75])] if len(ks) else None,
            'k_star_max': int(ks.max()) if len(ks) else None,
            'scan_replay_disagree_total': dis,
            'margin_at_kstar_median': float(np.median(marg)) if marg else None}


def numeric_summary(vals):
    v = [x for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]
    if not v:
        return None
    return {'n': len(v), 'median': float(np.median(v)), 'q25_q75': [float(x) for x in np.percentile(v, [25, 75])],
            'share_positive': float(np.mean(np.array(v) > 0))}


def dict_summary(sids, run, stage, field):
    """A per-site dict whose schema SPEC leaves to p8_2 (stage C 'collateral', stage B 'existence'),
    key by key: numbers or booleans -> share of sites > 0 and the median; non-empty lists of numbers ->
    per site the share of entries > 0, averaged over sites; anything else is listed by key only."""
    keys = defaultdict(list)
    for s in sids:
        c = run[stage].get(s, {}).get(field)
        if isinstance(c, dict):
            for k, v in c.items():
                keys[k].append(v)
    out = {}
    num = lambda v: isinstance(v, (int, float)) and math.isfinite(float(v))
    for k, vs in sorted(keys.items()):
        if all(num(v) for v in vs):
            a = np.array(vs, dtype=float)
            out[k] = {'n_sites': len(vs), 'share_sites_gt_0': float(np.mean(a > 0)), 'median': float(np.median(a))}
        elif all(isinstance(v, list) and v and all(num(x) for x in v) for v in vs):
            out[k] = {'n_sites': len(vs), 'mean_share_entries_gt_0': float(np.mean([np.mean(np.array(v, dtype=float) > 0)
                                                                                   for v in vs]))}
        else:
            out[k] = {'n_sites': len(vs), 'note': 'not numeric; not summarised'}
    return out


def collateral_sites(aset, run, prob):
    """§6 collateral as shares of sites (fixed definitions). Stage C 'collateral' = {G: argmax changes at
    positions >= h when S's best repair is teacher-forced with G's ablation, RNDk: the same per RND-k draw}.
    G: a site counts when G > 0, over sites with a collateral record (a best repair exists). RND-k: a site
    counts when more than half of its draws are > 0, over sites with >= 1 draw (RND-k needs G to flip).
    'both' = the two on the sites that have both. Per site (the §6 check) and equal-weight over problems."""
    rec = {s: run['C'][s]['collateral'] for s in aset if isinstance((run['C'].get(s) or {}).get('collateral'), dict)}
    g = {s: int(c['G'] > 0) for s, c in rec.items() if isinstance(c.get('G'), (int, float))}
    rk = {s: int(2 * sum(x > 0 for x in c['RNDk']) > len(c['RNDk'])) for s, c in rec.items()
          if isinstance(c.get('RNDk'), list) and c['RNDk']}
    both = sorted(set(g) & set(rk))

    def sh(d, ss):
        if not ss:
            return {'n_sites': 0, 'share_per_site': None, 'share_equal_weight': None}
        return {'n_sites': len(ss), 'share_per_site': float(np.mean([d[s] for s in ss])),
                'share_equal_weight': summ(ss, lambda s: d[s], prob, boot=False)['equal_weight']}
    return {'definition': collateral_sites.__doc__.split('\n', 1)[1].strip().replace('\n    ', ' '),
            'n_analysis_set': len(aset), 'n_with_collateral_record': len(rec),
            'G': sh(g, sorted(g)), 'RNDk_majority_of_draws': sh(rk, sorted(rk)),
            'on_sites_with_both': {'G': sh(g, both), 'RNDk_majority_of_draws': sh(rk, both)}}


def run_descriptives(aset, S, J, run, prob, arms):
    out = {'files': {st: len(run[st]) for st in run}}
    if not any(run.values()):
        out['note'] = 'no run outputs found'
        return out
    out['k_star'] = {a: kstar_stats(aset, run, a) for a in arms if a in STAGE_A_ARMS + STAGE_B_ARMS}
    # run/judge agreement on whether each arm flipped. p8_3 scores a flipped t* whose splice has S's own
    # lexer stream (t* in E) as no_flip; that case is counted on its own, anything else is a disagreement.
    agree = Counter()
    for (s, a), r in J.items():
        f = find_flip(run, s, a) if a in STAGE_A_ARMS + STAGE_B_ARMS else None
        if f is not None and r['label'] != 'excluded':
            rf, jf = bool(f.get('flipped')), r['label'] != 'no_flip'
            agree['agree' if rf == jf else ('run flipped, judged no_flip (t* in E)' if rf else 'disagree')] += 1
    out['run_vs_judged_flip'] = dict(agree)
    # G's first feature: layer >= 18 at position h-1. G's ranking is stage B's e_top (sorted by e, desc):
    # its first row with e > 0 and pos >= 1, at every analysis-set site whether or not G flipped. A site
    # whose e_top has no such row has no first feature and counts as outside.
    first, no_cand = [], 0
    for s in aset:
        et = run['B'].get(s, {}).get('e_top')
        if not isinstance(et, list):
            continue
        top = next((r for r in et if r[4] > 0 and r[1] >= 1), None)
        no_cand += top is None
        first.append(top is not None and top[0] >= 18 and top[1] == S[s]['h'] - 1)
    out['G_first_feature_layer18plus_at_h_minus_1'] = {
        'n_sites': len(first), 'n_analysis_set': len(aset), 'n_sites_without_a_candidate': no_cand,
        'share_per_site': float(np.mean(first)) if first else None,
        'source': "stage-B e_top: first row with e > 0 and pos >= 1 (G's ranking), every analysis-set site"}
    # margin terciles (stage A decision margin), per site
    mg = {s: run['A'][s]['decision']['margin'] for s in aset
          if isinstance(run['A'].get(s, {}).get('decision'), dict) and run['A'][s]['decision'].get('margin') is not None}
    if len(mg) >= 3:
        q1, q2 = np.quantile(list(mg.values()), [1 / 3, 2 / 3])
        terc = {s: int(v > q1) + int(v > q2) for s, v in mg.items()}
        tt = {}
        for t in range(3):
            ss = [s for s in mg if terc[s] == t]
            tt[str(t)] = {'n_sites': len(ss), **{f'Y_{a}_per_site': (float(np.mean([J[(s, a)]['Y'] for s in ss])) if ss else None)
                                                 for a in PRIMARY_ARMS},
                          'D_G_BAN_equal_weight': summ(ss, lambda s: J[(s, 'G')]['Y'] - J[(s, 'BAN')]['Y'], prob,
                                                       boot=False)['equal_weight']}
        out['margin_terciles'] = {'cuts': [float(q1), float(q2)], 'terciles': tt}
    # faithfulness: Spearman(e, measured single-ablation delta) per site
    fa = [run['B'][s]['faithfulness'] for s in aset if isinstance(run['B'].get(s, {}).get('faithfulness'), dict)]
    sp = [f['spearman'] for f in fa if f.get('spearman') is not None and math.isfinite(f['spearman'])]
    out['faithfulness'] = {'n_sites': len(fa), 'n_sites_finite': len(sp),
                           'site_median_spearman': float(np.median(sp)) if sp else None,
                           'median_qualifying_features': float(np.median([f.get('n', 0) for f in fa])) if fa else None}
    # RND-k (no flip rule, 20 draws): share of draws whose argmax left E
    rk = [np.mean([bool(x.get('changed')) for x in run['B'][s]['RNDk']]) for s in aset
          if isinstance(run['B'].get(s, {}).get('RNDk'), list) and run['B'][s]['RNDk']]
    out['RNDk_changed'] = {'n_sites': len(rk), 'mean_share_of_draws': float(np.mean(rk)) if rk else None}
    out['collateral'] = dict_summary(aset, run, 'C', 'collateral')
    out['collateral_sites'] = collateral_sites(aset, run, prob)
    out['existence_run_fields'] = dict_summary(aset, run, 'B', 'existence')
    stops = defaultdict(Counter)
    for s in aset:
        for a, g in (run['C'].get(s, {}).get('gen') or {}).items():
            if isinstance(g, dict):
                stops[a][g.get('stopped')] += 1
    out['generation_stops'] = {a: {**dict(c), 'cap_share': c['cap'] / sum(c.values())} for a, c in stops.items()}
    graph = [run['B'][s]['graph'] for s in aset if isinstance(run['B'].get(s, {}).get('graph'), dict)]
    out['graph'] = {'n_sites': len(graph),
                    'n_sel': numeric_summary([g.get('n_sel') for g in graph]),
                    'seconds': numeric_summary([g.get('seconds') for g in graph]),
                    'node0_in_logit_ids': sum(1 for g in graph if g.get('node0_id') in (g.get('logit_ids') or []))}
    out['seconds_per_site'] = {st: numeric_summary([run[st][s].get('seconds') for s in aset if s in run[st]])
                               for st in ('A', 'B')}
    return out


def hf_comparison(aset, S, J, run, tok_loader=None):
    """The committed HF table against the TL run (descriptive, §6): p8_1's 'hf_runner_up' = {known, to,
    label} of each site. Belief sites of the analysis set: HF runner-up status x TL BAN's Y, and whether
    TL BAN's lexeme (the splice of its t*) is the HF runner-up mutant's lexeme (sites whose runner-up is
    known). Slips: the HF-computed fix set (runner-up = HF argmax, known and verified) against NULL's."""
    out = {}
    cross = Counter()
    known = []
    for s in aset:
        hr = S[s].get('hf_runner_up')
        if not isinstance(hr, dict):
            continue
        hl = 'unknown' if not hr.get('known') else ('verified' if hr.get('label') == 'verified' else 'known, not verified')
        cross[f"HF runner-up {hl} | TL BAN Y={J[(s, 'BAN')]['Y']}"] += 1
        if hr.get('known'):
            known.append(s)
    out['hf_runner_up_x_TL_BAN_per_site'] = dict(sorted(cross.items()))
    lex = {'n_known': len(known)}
    if known and all('ids' in S[s] and 'judged' in S[s] for s in known):
        tok = (tok_loader or pc.tokenizer)()
        same, differ, no_ban, bad = [], [], [], []
        for s in known:
            f = find_flip(run, s, 'BAN')
            if f is None or not f.get('flipped') or f.get('t_star_id') is None:
                no_ban.append(s)
                continue
            try:
                k = json.dumps(splice_key(S[s], f['t_star_id'], tok))
            except ValueError as e:
                bad.append(f'{s}: {e}')
                continue
            hk = {ks for ks, v in S[s]['judged'].items() if v.get('to') == S[s]['hf_runner_up'].get('to')}
            (same if k in hk else differ).append(s)
        n = len(same) + len(differ)
        lex.update(same=len(same), differ=len(differ), share_differ=(len(differ) / n if n else None),
                   no_BAN_output=len(no_ban), undecodable=bad, differ_sites=sorted(differ))
    elif known:
        lex['note'] = 'sites lack ids / judged: not computed'
    out['TL_BAN_lexeme_vs_HF_runner_up'] = lex
    slips = [r['site_id'] for r in S.values() if r['kind'] == 'slip' and isinstance(r.get('hf_runner_up'), dict)]
    if slips:
        hf = {s for s in slips if S[s]['hf_runner_up'].get('known') and S[s]['hf_runner_up'].get('label') == 'verified'}
        nl = {s for s in slips if (s, 'NULL') in J and J[(s, 'NULL')]['Y'] == 1}
        out['slips'] = {'slip_sites': len(slips), 'hf_computed_fix': len(hf), 'NULL_fix': len(nl), 'both': len(hf & nl),
                        'only_HF': sorted(hf - nl), 'only_NULL': sorted(nl - hf), 'NULL_minus_HF': len(nl) - len(hf)}
    return out


# ---------------------------------------------------------------- predictions (§6)
def predictions(res):
    """Each §6 prediction against its registered range. Rates equal-weight unless 'per site'."""
    d = res['descriptive']
    rates, conf = d['arm_rates'], res['confirmatory']['sequence']
    rk = d.get('run_outputs', {}).get('k_star', {}) or {}
    lt = d.get('landing', {})

    def ew(a):
        return (rates.get(a) or {}).get('equal_weight')

    def flip_ps(a):
        return (rates.get(a) or {}).get('flip_rate_per_site')

    def add(name, value, lo=None, hi=None, note=''):
        inside = None if value is None else ((lo is None or value >= lo - 1e-12) and (hi is None or value <= hi + 1e-12))
        out.append({'prediction': name, 'value': value, 'range': [lo, hi], 'inside': inside, 'note': note})

    out = []
    g = ew('G')
    hfc = d.get('hf_comparison') or {}
    add('exclusions (belief sites)', len(d['analysis_set']['excluded']), 0, 4)
    add('BAN fix rate (TL)', ew('BAN'), 0.50, 0.70)
    add('TL BAN lexeme differs from the HF runner-up, share of sites with a known runner-up',
        (hfc.get('TL_BAN_lexeme_vs_HF_runner_up') or {}).get('share_differ'), None, 0.10)
    add('G flips, per site', flip_ps('G'), 0.90, None)
    add('G median k*', (rk.get('G') or {}).get('k_star_median'), 2, 20)
    add('G first feature at layer >= 18 and h-1, per site',
        d.get('run_outputs', {}).get('G_first_feature_layer18plus_at_h_minus_1', {}).get('share_per_site'), 0.60, None)
    add("t*_G equals BAN's lexeme, share of flips per site",
        (lt.get('G') or {}).get('share_of_flips_per_site', {}).get('ban_lexeme'), 0.60, 0.85)
    add('m of S2', conf[1]['m'], 6, 16)
    add('G fix rate', g, 0.40, 0.65)
    add('D(G-BAN)', conf[1]['D'], -0.15, 0.03)
    dec = d['decomposition_G_BAN']
    add('NF', dec['NF'], 0, 0.08)
    add('R - Bk', dec['R_minus_Bk'], -0.10, 0.05)
    add('GxA flips, per site', flip_ps('GxA'), 0.90, None)
    add('GxA fix rate', ew('GxA'), 0.40, 0.65)
    add('D(G-GxA)', conf[2]['D'], -0.06, 0.06)
    add('RND1 flips, per site', flip_ps('RND1'), 0.30, 0.90)
    add('RND1 median k* among flips', (rk.get('RND1') or {}).get('k_star_median'), 100, None)
    add('RND1 fix rate', ew('RND1'), 0.05, 0.35)
    add('D(G-RND1)', conf[0]['D'], 0.15, None)
    dec_s = [c['decision'] for c in conf]
    add('decisions: S1 rejects, S2 does not, S3 not reached', None, note=' / '.join(dec_s))
    out[-1]['inside'] = dec_s == ['reject', 'not rejected', 'not tested']
    for a, lo, hi in (('G_LP', -0.10, 0.10), ('ORC', -0.10, 0.10), ('CONTRAST', 0.10, None)):
        v = ew(a)
        add(f'{a} fix rate minus G', None if v is None or g is None else v - g, lo, hi)
    add('G1 flips, per site', flip_ps('G1'), 0.05, 0.25)
    add('G_mid flips, per site', flip_ps('G_mid'), None, 0.40)
    add('faithfulness: site median Spearman', d.get('run_outputs', {}).get('faithfulness', {}).get('site_median_spearman'), 0.3, 0.7)
    pg = ew('gen:G')
    add('program fix gen:G / site fix G', None if pg is None or not g else pg / g, 0.6, 1.0)
    pe = ew('gen:G_every')
    add('gen:G_every minus gen:G', None if pe is None or pg is None else pe - pg, None, 0)
    add('gen:NULL fix rate', ew('gen:NULL'), None, 0.05)
    ex = d.get('existence') or {}
    if ex:
        # p8_3's rows: exist:G (ORC's set and G's top features), exist:RND (matched random features)
        add('existence rate, exist:G, per site', (ex.get('exist:G') or {}).get('per_site'), 0.05, 0.25,
            note=json.dumps({k: v.get('per_site') for k, v in ex.items()}))
        add('existence rate, exist:RND (random features), per site', (ex.get('exist:RND') or {}).get('per_site'), None, 0.05)
    else:
        add('existence rate 5-25% per site vs <= 5% random', None,
            note='no existence rows in judged.jsonl; run-side fields: '
                 + json.dumps(d.get('run_outputs', {}).get('existence_run_fields') or 'none'))
    cs = d.get('run_outputs', {}).get('collateral_sites') or {}
    note = 'no collateral in run outputs' if not cs else (
        f"{cs['definition']} Equal weight: G {cs['G']['share_equal_weight']}, "
        f"RND-k {cs['RNDk_majority_of_draws']['share_equal_weight']}; on sites with both: "
        f"G {cs['on_sites_with_both']['G']['share_per_site']}, RND-k {cs['on_sites_with_both']['RNDk_majority_of_draws']['share_per_site']}")
    add('collateral: G changes an argmax, share of sites (per site)', (cs.get('G') or {}).get('share_per_site'), 0.20, 0.60, note=note)
    add('collateral: RND-k changes an argmax in most draws, share of sites (per site)',
        (cs.get('RNDk_majority_of_draws') or {}).get('share_per_site'), 0.05, 0.30)
    sl = hfc.get('slips')
    add('slips: NULL fix count minus the HF-computed fix count', None if sl is None else sl['NULL_minus_HF'], -3, 3,
        note='' if sl else 'no hf_runner_up on slip rows')
    return out


# ---------------------------------------------------------------- the analysis
def analyse(out, tok_loader=None):
    t0 = time.time()
    P = paths(out)
    S, J, run = load(P)
    alpha, alpha_info = get_alpha(out, P['sites'], P['dry_run'])
    prob = {s: r['problem'] for s, r in S.items()}
    aset, ainfo, rnd = analysis_set(S, J, run, P['kind'])
    bad = {k: ainfo[k] for k in ('unregistered_exclusions', 'registered_but_judged_rows_not_excluded') if ainfo[k]}
    if bad and not P['dry_run']:
        raise SystemExit('exclusions that design §2 does not register (a site may leave the analysis set only by an '
                         '(a)-(d) reason in its stage-A/B file; judged.jsonl must not carry a stage-C or '
                         f'failed-twice exclusion onto G/RND1/BAN/GxA): {json.dumps(bad)[:2000]}')
    if bad:
        print(f'DRY RUN: unregistered exclusions dropped from the analysis set and reported: {json.dumps(bad)[:500]}')
    if not aset:
        raise SystemExit('empty analysis set')
    probs = sorted({prob[s] for s in aset})
    pcount = Counter(prob[s] for s in aset)
    ainfo.update({'n_sites': len(aset), 'n_problems': len(probs), 'L': lcm(pcount.values()),
                  'per_problem_sites': {str(g): pcount[g] for g in probs},
                  'per_problem_sorted': sorted(pcount.values(), reverse=True),
                  'tier_counts': dict(Counter(str(S[s]['tier']) for s in aset)), 'sites': aset})

    def Y(a):
        return lambda s: J[(s, a)]['Y']

    # ---- confirmatory
    seq, stopped = [], False
    for step, a, b in pc.PRIMARY_ORDER:
        c = contrast(aset, Y(a), Y(b), prob)
        e = {'step': step, 'test': f'{a} > {b}', 'alpha': alpha, **c}
        if stopped:
            moved = {k: e.pop(k) for k in ('p', 'p_exact', 'p_mc', 'exact_count', 'exact_of', 'how', 'min_attainable_p')
                     if k in e}
            e.update(reached=False, decision='not tested', p_descriptive_not_tested=moved)
        else:
            e.update(reached=True, decision='reject' if c['p'] <= alpha else 'not rejected')
            stopped = c['p'] > alpha
        seq.append(e)
    mde = None
    if os.path.exists(P['sim_power']):
        with open(P['sim_power'], encoding='utf-8') as f:
            spw = json.load(f)
        mde = {'mde80': spw.get('mde80'), 'alpha': spw.get('alpha'), 'test_run': spw.get('test_run'),
               'sites_sha256': spw.get('sites_sha256'),
               'same_site_table': spw.get('sites_sha256') == pc.sha256(P['sites'])}
    words = []
    for e in seq:
        if e['decision'] == 'reject':
            words.append(f"{e['step']} ({e['test']}) rejects: D = {e['D']:+.3f}, m = {e['m']}, p = {e['p']:.4g} <= {alpha}")
        elif e['decision'] == 'not rejected':
            words.append(f"{e['step']} ({e['test']}) does not reject: D = {e['D']:+.3f}, m = {e['m']}, p = {e['p']:.4g} > {alpha}; "
                         f"the advantage was not detected (80% MDE: see descriptive.mde80), never read as equivalence")
        else:
            words.append(f"{e['step']} ({e['test']}) not tested (the sequence stopped)")
    res = {'created': time.strftime('%Y-%m-%d %H:%M:%S JST'), 'dry_run': P['dry_run'], 'kind': P['kind'],
           'script_sha256': pc.sha256(os.path.abspath(__file__)),
           'common_sha256': pc.sha256(os.path.join(HERE, 'p8_common.py')),
           'inputs': {P[k]: pc.sha256(P[k]) for k in ('sites', 'judged', 'alpha', 'sim_power') if os.path.exists(P[k])},
           'run_dir': P['run_dir'], 'run_files': {st: len(run[st]) for st in run},
           'alpha': alpha_info,
           'confirmatory': {'method': 'one-sided problem-level sign flip on the integer statistic; fixed sequence S1 -> S2 -> S3',
                            'sequence': seq, 'summary': words},
           'descriptive': {'note': DESCRIPTIVE, 'analysis_set': ainfo}}
    d = res['descriptive']

    # ---- arm rates
    arms = sorted({a for (_, a) in J if not a.startswith('RNDk:')},
                  key=lambda a: (0, (pc.ARMS_ORDERED + pc.ARMS_FIXED).index(a)) if a in pc.ARMS_ORDERED + pc.ARMS_FIXED else (1, a))
    rates = {}
    for a in arms:
        ss = [s for s in aset if (s, a) in J and J[(s, a)]['label'] != 'excluded']
        r = summ(ss, Y(a), prob)
        r['coverage_of_analysis_set'] = len(ss) / len(aset)
        r['flip_rate_per_site'] = (float(np.mean([J[(s, a)]['label'] != 'no_flip' for s in ss])) if ss else None)
        r['labels'] = dict(Counter(str(J[(s, a)]['label']) for s in ss))
        r['how'] = dict(Counter(str(J[(s, a)].get('how')) for s in ss))
        r['excluded_rows_in_analysis_set'] = sum(1 for s in aset if (s, a) in J and J[(s, a)]['label'] == 'excluded')
        rates[a] = r
    d['arm_rates'] = rates
    # RND-k (k*_G matched random features, no flip rule, 20 draws; rows 'RNDk:<draw>'): a site's value is
    # the mean over its draws
    rkd = defaultdict(list)
    aset_set = set(aset)
    for (s, a), r in J.items():
        if a.startswith('RNDk:') and s in aset_set and r['label'] != 'excluded':
            rkd[s].append(r)
    d['RNDk_draws_pooled'] = {**summ(sorted(rkd), lambda s: float(np.mean([r['Y'] for r in rkd[s]])), prob),
                              'draws': sum(len(v) for v in rkd.values()),
                              'changed_share_of_draws': (float(np.mean([r['label'] != 'no_flip' for v in rkd.values() for r in v]))
                                                         if rkd else None),
                              'labels': dict(Counter(r['label'] for v in rkd.values() for r in v))}

    # ---- decomposition, rescue/break, landing, P(fix | both flip)
    d['decomposition_G_BAN'] = decomposition(aset, J, prob)
    ordered = [a for a in arms if a in pc.ARMS_ORDERED]
    d['rescue_break_vs_BAN'] = {a: rescue_break(aset, J, a, prob) for a in ordered}
    d['landing'] = {a: landing_table(aset, J, a, prob) for a in ordered + ['BAN']}
    both = [s for s in aset if J[(s, 'G')]['label'] != 'no_flip' and J[(s, 'GxA')]['label'] != 'no_flip']
    d['fix_given_both_flip_G_vs_GxA'] = {'n_sites': len(both), 'G': summ(both, Y('G'), prob), 'GxA': summ(both, Y('GxA'), prob)}

    # ---- sensitivity analyses (descriptive)
    def variants():
        yield 'tier1_only', lambda s, a, b: S[s]['tier'] == 1, None
        yield 'without_rule_b', lambda s, a, b: not S[s].get('rule_b'), None
        yield 'unscorable_excluded', lambda s, a, b: 'unscorable' not in (J[(s, a)]['label'], J[(s, b)]['label']), None
        yield 'neutralizing_rescues_scored_0', lambda s, a, b: True, \
            lambda s, arm: 0 if J[(s, arm)].get('neutralizing') else J[(s, arm)]['Y']
        yield 'both_tokens_in_judged_family', lambda s, a, b: J[(s, a)].get('how') == 'lexeme' and J[(s, b)].get('how') == 'lexeme', None
    sens = {}
    for name, keep, yf in variants():
        sens[name] = {}
        for step, a, b in pc.PRIMARY_ORDER:
            ss = [s for s in aset if keep(s, a, b)]
            ya = (lambda s, a=a: yf(s, a)) if yf else Y(a)
            yb = (lambda s, b=b: yf(s, b)) if yf else Y(b)
            c = contrast(ss, ya, yb, prob, both=False) if ss else {'n_sites': 0}
            sens[name][step] = {k: c.get(k) for k in ('n_sites', 'n_problems', 'D', 'D_ci95', 'm', 'p', 'how')}
    sens['sqrt_n_weights'] = {}
    for step, a, b in pc.PRIMARY_ORDER:
        c = contrast_sqrt(aset, Y(a), Y(b), prob)
        sens['sqrt_n_weights'][step] = {k: c.get(k) for k in ('n_sites', 'n_problems', 'D_sqrt_n', 'm', 'p', 'how')}
    d['sensitivity'] = {'note': DESCRIPTIVE + ' p values here are descriptive.', **sens}

    # ---- run outputs, program fix, existence, family, slips
    d['run_outputs'] = run_descriptives(aset, S, J, run, prob, arms)
    d['program_fix'] = {a: rates[a] for a in arms if a.startswith('gen:')}
    if 'G' in rates and 'gen:G' in rates and rates['G']['equal_weight']:
        d['program_fix']['gen:G_over_site_fix_G'] = rates['gen:G']['equal_weight'] / rates['G']['equal_weight']
    d['existence'] = {a: rates[a] for a in arms if 'exist' in a.lower()}
    fam = defaultdict(list)
    for s in aset:
        fam[S[s].get('family')].append(s)
    d['family'] = {str(f): {'n_sites': len(ss), **{f'Y_{a}_per_site': float(np.mean([J[(s, a)]['Y'] for s in ss]))
                                                   for a in PRIMARY_ARMS}} for f, ss in sorted(fam.items(), key=lambda x: str(x[0]))}
    slips = [r['site_id'] for r in S.values() if r['kind'] == 'slip']
    sl = [s for s in slips if (s, 'NULL') in J and J[(s, 'NULL')]['label'] != 'excluded']
    d['slips_NULL'] = {'slip_sites': len(slips), 'with_NULL_row': len(sl), **summ(sl, Y('NULL'), prob),
                       'labels': dict(Counter(J[(s, 'NULL')]['label'] for s in sl)),
                       'verified_sites': sorted(s for s in sl if J[(s, 'NULL')]['Y'] == 1)}
    d['hf_comparison'] = hf_comparison(aset, S, J, run, tok_loader)
    d['mde80'] = mde if mde is not None else 'sim_power.json not found'
    d['predictions'] = predictions(res)
    res['seconds'] = time.time() - t0
    return res


def main():
    if MODE == 'selftest':
        selftest()
        return
    res = analyse(pc.OUT)
    path = paths(pc.OUT)['analysis']
    pc.write_json(path, res)
    print(('DRY RUN (kind %s). ' % res['kind'] if res['dry_run'] else '') + f"alpha {res['alpha']['value']} ({res['alpha']['source']})")
    a = res['descriptive']['analysis_set']
    print(f"analysis set: {a['n_sites']} sites, {a['n_problems']} problems, L {a['L']}; excluded {len(a['excluded'])}, "
          f"in incomplete rounds {len(a['run_but_in_incomplete_round'])}, not run {a['not_run']}")
    for w in res['confirmatory']['summary']:
        print(w)
    dec = res['descriptive']['decomposition_G_BAN']
    print(f"(descriptive) D(G-BAN) = -NF {dec['NF']:.3f} + (R {dec['R']:.3f} - Bk {dec['Bk']:.3f})")
    print(f"{path} sha256 {pc.sha256(path)}")


# ---------------------------------------------------------------- selftest
def _flip(arm, flipped, k=None, layer=20, pos=0):
    return {'arm': arm, 'flipped': flipped, 'k_star': k if flipped else None, 't_star_id': 100 if flipped else None,
            't_star_piece': 'x' if flipped else None, 'margin_at_kstar': 0.5 if flipped else None,
            'scan_replay_disagree': 0, 'n_candidates': 2000,
            'ablated': [[layer, pos, 1, 1.0]] * (k or 0) if flipped else [], 'completion': None}


def _synthetic(out, n_not_run=0, seed=0):
    """Selftest fixture only (np.random.default_rng(seed); never study randomness): a §2-shaped site table
    (146 belief sites in 31 problems with DESIGN_COUNTS, bugs, tiers, round-robin order, 45 slips), judged
    rows for every arm with a planted G > RND1 effect and G ~ BAN, and run outputs. The last n_not_run
    belief sites by order get no stage-B or stage-C rows (a hard stop). Returns the round of each site."""
    g = np.random.default_rng(seed)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    fams = ['comparison', 'arithmetic', 'compound', 'integer', 'range']
    sites, rnd, tasks = [], {}, list(range(100, 100 + len(DESIGN_COUNTS)))
    tier_of, bug_of = {}, {}
    per_task = {}
    for t, n in zip(tasks, DESIGN_COUNTS):
        nb = max(1, (n + 1) // 2)
        ids = [f'{t}|{smp}|{50 + smp}' for smp in range(n)]
        for j, sid in enumerate(ids):
            bug_of[sid] = f'{t}|line{j % nb}|{1000 + j % nb}'
        first = {}
        for sid in ids:
            first.setdefault(bug_of[sid], sid)
        for sid in ids:
            tier_of[sid] = 1 if first[bug_of[sid]] == sid else 2
        per_task[t] = ids
    order, pos = {}, 0
    for tier in (1, 2):
        lists = {t: [s for s in per_task[t] if tier_of[s] == tier] for t in tasks}
        for t in tasks:
            g.shuffle(lists[t])
        porder = list(g.permutation(tasks))
        j = 0
        while any(len(lists[t]) > j for t in porder):
            for t in porder:
                if len(lists[t]) > j:
                    order[lists[t][j]] = pos
                    rnd[lists[t][j]] = (tier, j)
                    pos += 1
            j += 1
    ban_bug = {b: int(g.random() < 0.6) for b in sorted(set(bug_of.values()))}
    for t in tasks:
        for sid in per_task[t]:
            kn = bool(g.random() < 0.65)
            sites.append({'site_id': sid, 'kind': 'belief', 'problem': t, 'tier': tier_of[sid], 'order': order[sid],
                          'rule_b': bool(g.random() < 0.1), 'family': fams[int(g.integers(len(fams)))],
                          'h': 60, 'bug': bug_of[sid],
                          'hf_runner_up': {'known': kn, 'to': 'x' if kn else None, 'p': 0.1 if kn else None,
                                           'label': ('verified' if ban_bug[bug_of[sid]] else 'fail') if kn else None}})
    slip_tasks = tasks[:14]
    for k in range(45):
        t = slip_tasks[k % 14]
        kn = bool(g.random() < 0.7)
        sites.append({'site_id': f'{t}|{900 + k}|7', 'kind': 'slip', 'problem': t, 'family': 'comparison', 'h': 40,
                      'bug': f'{t}|slip{k}|5', 'hf_runner_up': {'known': kn, 'to': 'y' if kn else None,
                                                                'label': ['verified', 'fail'][int(g.integers(2))] if kn else None}})
    belief = sorted((s for s in sites if s['kind'] == 'belief'), key=lambda s: s['order'])
    not_run = {s['site_id'] for s in belief[len(belief) - n_not_run:]} if n_not_run else set()
    excluded = {belief[5]['site_id']: 'a: argmax not in E', belief[40]['site_id']: 'c: attribute failed twice',
                belief[90]['site_id']: 'd: node 0 not in E'}
    judged = []
    runA, runB, runC, first_ok = {}, {}, {}, {}

    def row(sid, arm, flipped, cat, y, neutral=False):
        if cat == 'excluded':
            return {'site_id': sid, 'arm': arm, 'Y': 0, 'label': 'excluded', 'how': 'none', 'lexeme_is_ban': False,
                    'neutralizing': False}
        if not flipped:
            return {'site_id': sid, 'arm': arm, 'Y': 0, 'label': 'no_flip', 'how': 'none', 'lexeme_is_ban': False,
                    'neutralizing': False}
        if cat == 'unscorable':
            y = 0
        lab = 'verified' if y else ('unscorable' if cat == 'unscorable' else
                                    ['fail', 'lucky', 'no_compile', 'unverifiable'][int(g.integers(4))])
        return {'site_id': sid, 'arm': arm, 'Y': int(y), 'label': lab,
                'how': 'lexeme' if cat in ('ban', 'family') else 'completion', 'lexeme_is_ban': cat == 'ban',
                'neutralizing': bool(neutral)}

    spec = {  # arm: flip prob, P(ban lexeme | flip), P(fix | other family), P(fix | outside)
        'G': (0.95, 0.70, 0.6, 0.2), 'GxA': (0.95, 0.75, 0.5, 0.2), 'G_LP': (0.9, 0.7, 0.5, 0.2),
        'G1': (0.15, 0.7, 0.5, 0.2), 'G_mid': (0.3, 0.5, 0.4, 0.1), 'DE': (0.8, 0.7, 0.5, 0.2),
        'ORC': (0.95, 0.7, 0.6, 0.2), 'CONTRAST': (0.97, 0.5, 0.8, 0.3),
        **{f'RND{k}': (0.6, 0.15, 0.05, 0.02) for k in range(1, 6)}}
    for s in sites:
        sid = s['site_id']
        if s['kind'] == 'slip':
            judged.append(row(sid, 'NULL', True, 'family', int(g.random() < 0.5)))
            continue
        exc = sid in excluded
        yb = ban_bug[s['bug']]
        stage_b_run = sid not in not_run
        judged.append(row(sid, 'BAN', True, 'excluded' if exc else ('ban' if g.random() < 0.9 else 'outside'), yb))
        judged.append(row(sid, 'NULL', False, 'excluded' if exc else 'none', 0))
        flips_ = {'BAN': _flip('BAN', True, 0), 'NULL': _flip('NULL', False)}
        for arm, (pf, pb, pfam, pout) in spec.items():
            if arm in STAGE_B_ARMS and not stage_b_run:
                continue
            f = bool(g.random() < pf)
            u = g.random()
            cat = 'ban' if u < pb else ('family' if u < pb + (1 - pb) * 0.6 else ('unscorable' if g.random() < 0.1 else 'outside'))
            y = yb if cat == 'ban' else int(g.random() < (pfam if cat == 'family' else pout))
            judged.append(row(sid, arm, f, 'excluded' if exc else cat, y, neutral=g.random() < 0.05))
            flips_[arm] = _flip(arm, f, int(g.integers(1, 30 if arm[:3] != 'RND' else 900)), layer=int(g.integers(10, 26)),
                                pos=59 if g.random() < 0.7 else 30)
        why = excluded.get(sid)
        runA[sid] = {'site_id': sid, 'excluded': why if why and why.startswith('a:') else None,
                     'decision': {'top': [], 'wrong_in_E_top': True, 'margin': float(g.gamma(2.0, 2.0))},
                     **{a: flips_[a] for a in STAGE_A_ARMS if a in flips_}, 'gxa_top': [], 'seconds': 30.0}
        if stage_b_run:
            # e_top as p8_2 writes it (sorted by e): a pos-0 row (never a candidate), G's first candidate, a negative row
            fl_, fp_ = int(g.integers(10, 26)), (59 if g.random() < 0.7 else 30)
            e_top = [[25, 0, 1, 1.0, 9.0], [fl_, fp_, 2, 1.0, 5.0], [10, 59, 3, 1.0, -1.0]]
            first_ok[sid] = fl_ >= 18 and fp_ == 59
            runB[sid] = {'site_id': sid, 'excluded': why,           # p8_2 copies stage A's reason forward
                         'graph': {'n_sel': 5000, 'n_logits': 10, 'logit_ids': [1, 2], 'node0_id': 1, 'seconds': 300.0},
                         'e_top': e_top, **{a: flips_[a] for a in STAGE_B_ARMS if a in flips_},
                         'RNDk': [{'draw': k, 't_star_id': 5, 'changed': bool(g.random() < 0.3)} for k in range(20)],
                         'existence': {}, 'faithfulness': {'spearman': float(g.uniform(0.1, 0.8)), 'n': 40},
                         'seconds': 400.0}
            for arm, p in (('gen:G', 0.4), ('gen:G_every', 0.3), ('gen:BAN', 0.4), ('gen:NULL', 0.0),
                           ('exist:G', 0.15), ('exist:RND', 0.03)):
                judged.append(row(sid, arm, True, 'excluded' if exc else 'family', int(g.random() < p)))
            for k, dr in enumerate(runB[sid]['RNDk']):
                r = row(sid, f'RNDk:{k}', dr['changed'], 'excluded' if exc else ('family' if g.random() < 0.7 else 'unscorable'),
                        int(g.random() < 0.1))
                if r['label'] == 'unscorable':
                    r['how'] = 'none'
                judged.append(r)
            pr = g.random()                             # per-site RND-k change probability
            runC[sid] = {'site_id': sid, 'excluded': why,
                         'gen': {a: {'text': '', 'tokens': 10, 'stopped': 'braces' if g.random() < 0.97 else 'cap'}
                                 for a in ('G', 'G_every', 'BAN', 'NULL')},
                         'collateral': ({'G': int(g.integers(0, 3)), 'RNDk': [int(x) for x in (g.random(20) < pr) * g.integers(1, 4, 20)]
                                         if runB[sid]['G']['flipped'] else []}
                                        if g.random() < 0.9 else None)}       # None: no usable best repair
    with open(os.path.join(out, 'sites.jsonl'), 'w', encoding='utf-8') as f:
        for s in sites:
            f.write(json.dumps(s) + '\n')
    with open(os.path.join(out, 'judged.jsonl'), 'w', encoding='utf-8') as f:
        for r in judged:
            f.write(json.dumps(r) + '\n')
    for st, rr in (('A', runA), ('B', runB), ('C', runC)):
        os.makedirs(os.path.join(out, 'run', st))
        for sid, v in rr.items():
            with open(os.path.join(out, 'run', st, sid.replace('|', '_') + '.json'), 'w', encoding='utf-8') as f:
                json.dump(v, f)
    return {'round': rnd, 'not_run': not_run, 'excluded': set(excluded), 'order': order, 'first_ok': first_ok,
            'runC': runC}


def selftest():
    global KIND
    base = pc.OUT
    if os.path.abspath(base) == os.path.abspath(os.path.join(HERE, 'results', 'intervention')):
        raise SystemExit('selftest writes fixtures: set P8_OUT to a scratch directory')
    if any(ENV_PATHS.values()) or KIND != 'belief' or N_BOOT != 5000:
        raise SystemExit('selftest: unset SITES, JUDGED, RUN_DIR, ANALYSIS, KIND and N_BOOT')
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'pass' if cond else 'FAIL'}] {name} {detail}", flush=True)

    print('1. the sign-flip p')
    for x, want in (([1, 1], Fraction(1, 4)), ([1, -1], Fraction(3, 4)), ([2, 1, 1], Fraction(1, 8)),
                    ([1, 1, 1, 1], Fraction(1, 16)), ([0, 2, 0], Fraction(1, 2)), ([0, 0], Fraction(1, 1)),
                    ([3, -1, -1, -1], Fraction(9, 16))):
        X = np.array(x, dtype=np.int64)
        t = signflip(X)
        got = Fraction(t['exact_count'], t['exact_of'])
        check(f'hand case {x}', got == want, f'p = {got} (want {want})')
    g = np.random.default_rng(7)                       # selftest draws only
    worst_lit = 0
    for m in range(1, 17):
        for _ in range(3):
            x = g.integers(-6, 7, m) * 13860
            x[x == 0] = 13860
            T = int(x.sum())
            worst_lit = max(worst_lit, abs(count_ge(x, T) - count_ge_literal(x, T)))
    check('meet-in-the-middle count == literal enumeration (m = 1..16, 48 vectors)', worst_lit == 0)
    zmax = 0.0
    for m in (6, 12, 18, 24, 27, 31):
        for _ in range(3):
            x = g.integers(1, 9, 31) * 6690 * np.where(g.random(31) < 0.65, 1, -1)   # mostly positive: small p too
            x[m:] = 0
            t = signflip(x, both=True)
            se = math.sqrt(max(t['p_exact'] * (1 - t['p_exact']), 1e-6) / N_FLIPS)
            zmax = max(zmax, abs(t['p_mc'] - t['p_exact']) / se)
    check('Monte Carlo p (99,999 flips, rng(0)) agrees with exact p (m 6..31, 18 vectors)', zmax < 4.5, f'max |z| = {zmax:.2f}')
    L = lcm(DESIGN_COUNTS)
    check('L for the §2 counts is 207480', L == 207480, f'L = {L}')
    prob = [k for k, n in enumerate(DESIGN_COUNTS) for _ in range(n)]
    allint = True
    for _ in range(20):
        d = g.integers(-1, 2, len(prob))
        _, n, X, L2 = problem_X(d, prob)
        exact = sum(Fraction(int(sum(d[i] for i in range(len(prob)) if prob[i] == k)), DESIGN_COUNTS[k])
                    for k in range(len(DESIGN_COUNTS)))
        allint &= X.dtype == np.int64 and Fraction(int(X.sum()), L2) == exact
    check('integer statistic: T / L equals the exact sum of problem means (20 draws)', allint)

    print('2. end to end on a synthetic judged.jsonl (full run)')
    full = os.path.join(base, 'full')
    fx = _synthetic(full)
    res = analyse(full)
    a = res['descriptive']['analysis_set']
    check('analysis set = 146 belief sites minus 3 excluded', a['n_sites'] == 143 and len(a['excluded']) == 3,
          f"({a['n_sites']} sites, {a['n_problems']} problems)")
    check('order is round-robin', a['order_is_round_robin'])
    seq = res['confirmatory']['sequence']
    print('     ' + '\n     '.join(res['confirmatory']['summary']))
    check('S1 rejects on the planted G > RND1 effect', seq[0]['decision'] == 'reject', f"p = {seq[0]['p']:.3g}, m = {seq[0]['m']}")
    for e in seq:
        e = {**e, **e.get('p_descriptive_not_tested', {})}
        if 'p_exact' in e and 'p_mc' in e:
            se = math.sqrt(max(e['p_exact'] * (1 - e['p_exact']), 1e-6) / N_FLIPS)
            check(f"{e['step']}: exact and MC p agree", abs(e['p_exact'] - e['p_mc']) < 4.5 * se,
                  f"exact {e['p_exact']:.5f} MC {e['p_mc']:.5f} m {e['m']}")
    dec = res['descriptive']['decomposition_G_BAN']
    check('decomposition D = -NF + (R - Bk)', abs(dec['D'] - (-dec['NF'] + dec['R'] - dec['Bk'])) < 1e-12 and
          abs(dec['D'] - seq[1]['D']) < 1e-12, f"D {dec['D']:+.4f} NF {dec['NF']:.4f} R {dec['R']:.4f} Bk {dec['Bk']:.4f}")
    check('sensitivity analyses present', set(res['descriptive']['sensitivity']) >= {
        'tier1_only', 'without_rule_b', 'unscorable_excluded', 'neutralizing_rescues_scored_0',
        'both_tokens_in_judged_family', 'sqrt_n_weights'})
    check('run/judged flip agreement', res['descriptive']['run_outputs']['run_vs_judged_flip'].get('disagree', 0) == 0,
          str(res['descriptive']['run_outputs']['run_vs_judged_flip']))
    pc.write_json(os.path.join(full, 'analysis.json'), res)
    res2 = analyse(full)
    strip = lambda r: json.dumps({k: v for k, v in r.items() if k not in ('created', 'seconds', 'inputs')}, sort_keys=True,
                                 default=pc._jsonable)
    check('deterministic: a second run gives the same analysis', strip(res) == strip(res2))
    ro = res['descriptive']['run_outputs']
    ff = ro['G_first_feature_layer18plus_at_h_minus_1']
    want_ff = float(np.mean([fx['first_ok'][s] for s in a['sites']]))
    check("G's first feature from e_top at every analysis-set site (flipped or not; pos-0 row skipped)",
          ff['n_sites'] == a['n_sites'] and abs(ff['share_per_site'] - want_ff) < 1e-12,
          f"({ff['share_per_site']:.3f} over {ff['n_sites']} sites)")
    cs = ro['collateral_sites']
    recs = {s: fx['runC'][s]['collateral'] for s in a['sites'] if fx['runC'][s]['collateral'] is not None}
    wg = np.mean([recs[s]['G'] > 0 for s in recs])
    wr = [sum(x > 0 for x in recs[s]['RNDk']) > 10 for s in recs if recs[s]['RNDk']]
    pr = {p['prediction']: p for p in res['descriptive']['predictions']}
    check('collateral: share of sites with G > 0, and with RND-k > 0 in most draws (sites with draws)',
          cs['G']['n_sites'] == len(recs) and abs(cs['G']['share_per_site'] - wg) < 1e-12
          and cs['RNDk_majority_of_draws']['n_sites'] == len(wr) and abs(cs['RNDk_majority_of_draws']['share_per_site'] - np.mean(wr)) < 1e-12
          and pr['collateral: G changes an argmax, share of sites (per site)']['inside'] is not None,
          f"(G {cs['G']['share_per_site']:.3f} of {cs['G']['n_sites']}, RND-k {cs['RNDk_majority_of_draws']['share_per_site']:.3f} "
          f"of {cs['RNDk_majority_of_draws']['n_sites']})")
    check('existence predictions computed from exist:G / exist:RND rows',
          pr['existence rate, exist:G, per site']['value'] is not None
          and pr['existence rate, exist:RND (random features), per site']['value'] is not None)
    er = a['exclusion_reasons']
    check('exclusion reasons: the 3 registered (a/c/d) sites, every stage that says so, none in the set',
          sorted(er) == sorted(fx['excluded']) and all(v['registered'] and not v['in_analysis_set'] and 'C' in v['stages']
                                                     for v in er.values()) and not a['unregistered_exclusions'],
          str({s: v['reason'][:2] for s, v in er.items()}))

    print('2b. exclusions design §2 does not register (stage C, failed twice) never remove a site silently')
    keep = a['sites'][7]

    def variant(name, stage, reason, rows_excluded):
        dst = os.path.join(base, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(full, dst)
        p = os.path.join(dst, 'run', stage, keep.replace('|', '_') + '.json')
        with open(p, encoding='utf-8') as f:
            o = json.load(f)
        o['excluded'] = reason
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(o, f)
        if rows_excluded:                           # what p8_3 now does: every row of the site 'excluded'
            jr = [{**r, 'Y': 0, 'label': 'excluded', 'how': 'none'} if r['site_id'] == keep else r
                  for r in pc.read_jsonl(os.path.join(dst, 'judged.jsonl'))]
            with open(os.path.join(dst, 'judged.jsonl'), 'w', encoding='utf-8') as f:
                for r in jr:
                    f.write(json.dumps(r) + '\n')
        return dst

    for name, stage, reason in (('excl_c', 'C', 'C: failed in two fresh processes (RuntimeError: x)'),
                                ('excl_b', 'B', 'B: failed in two fresh processes (RuntimeError: x)')):
        dst = variant(name, stage, reason, True)
        try:
            analyse(dst)
            check(f'real run stops on a {stage}-stage non-(a)-(d) exclusion carried onto the primary rows', False)
        except SystemExit as e:
            check(f'real run stops on a {stage}-stage non-(a)-(d) exclusion carried onto the primary rows',
                  'does not register' in str(e) and keep in str(e))
    ENV_PATHS['ANALYSIS'] = 'analysis_dry.json'         # any env path makes a dry run (relative: under P8_OUT)
    try:
        rd = analyse(dst)
        apath = paths(dst)['analysis']
    finally:
        ENV_PATHS['ANALYSIS'] = None
    ad = rd['descriptive']['analysis_set']
    check('dry run: the site is dropped and listed with its stage and reason',
          rd['dry_run'] and keep not in ad['sites'] and keep in ad['unregistered_exclusions']
          and ad['unregistered_exclusions'][keep]['stage_reasons'] == {'B': 'B: failed in two fresh processes (RuntimeError: x)'}
          and apath == os.path.join(dst, 'analysis_dry.json'), f'(analysis -> {apath})')
    dst = variant('excl_c_kept', 'C', 'C: failed in two fresh processes (RuntimeError: x)', False)
    rk_ = analyse(dst)['descriptive']['analysis_set']
    check('a stage-C exclusion that p8_3 keeps off the primary rows: the site stays, its reason is reported',
          keep in rk_['sites'] and rk_['n_sites'] == a['n_sites'] and rk_['exclusion_reasons'][keep]['registered'] is False
          and rk_['exclusion_reasons'][keep]['in_analysis_set'] is True)

    print('3. hard stop: a partial round is dropped')
    cut = os.path.join(base, 'cut')
    n_cut = 25                                          # the fixture's tier-2 round 2 is cut part-way
    fx = _synthetic(cut, n_not_run=n_cut)
    res = analyse(cut)
    a = res['descriptive']['analysis_set']
    by_order = sorted(fx['order'], key=fx['order'].get)
    run_ids, rest = by_order[:-n_cut], by_order[-n_cut:]
    last_round = fx['round'][run_ids[-1]]
    partial = fx['round'][rest[0]] == last_round        # rounds are contiguous in the order
    want = sorted(s for s in run_ids if fx['round'][s] != last_round and s not in fx['excluded'])
    check('the cut falls inside a round', partial, f'(round {last_round})')
    check('analysis set = completed rounds minus excluded (computed from the fixture)', a['sites'] == want,
          f"({len(a['sites'])} vs {len(want)}; dropped from the partial round: {len(a['run_but_in_incomplete_round'])})")

    print('4. alpha.json handling')
    check('no alpha.json -> p8_common.ALPHA', get_alpha(full)[0] == pc.ALPHA)
    ap = os.path.join(full, 'alpha.json')
    pc.write_json(ap, {'alpha': 0.025, 'test_run': True, 'sites_sha256': pc.sha256(os.path.join(full, 'sites.jsonl'))})
    try:
        get_alpha(full)
        check('test-run alpha.json refused', False)
    except SystemExit:
        check('test-run alpha.json refused', True)
    pc.write_json(ap, {'alpha': 0.025, 'test_run': False, 'sites_sha256': 'x'})
    try:
        get_alpha(full)
        check('alpha.json from another sites.jsonl refused', False)
    except SystemExit:
        check('alpha.json from another sites.jsonl refused', True)
    pc.write_json(ap, {'alpha': 0.025, 'test_run': False, 'sites_sha256': pc.sha256(os.path.join(full, 'sites.jsonl'))})
    res = analyse(full)
    check('a valid alpha.json (0.025) is used at every step', all(e['alpha'] == 0.025 for e in res['confirmatory']['sequence']))
    os.remove(ap)

    print('5. descriptive extras on the fixture, and TL BAN vs the HF runner-up on a real tokenization')
    res = analyse(full)
    d = res['descriptive']
    check("'unverifiable' rows load and score 0", any(r['labels'].get('unverifiable') for r in d['arm_rates'].values()))
    rk = d['RNDk_draws_pooled']
    check('RND-k draws pooled per site, kept out of arm_rates', rk['n_sites'] > 0 and rk['draws'] == 20 * rk['n_sites']
          and not any(a.startswith('RNDk:') for a in d['arm_rates']), f"({rk['n_sites']} sites, {rk['draws']} draws)")
    sl = d['hf_comparison']['slips']
    check('slips: HF-computed fix set vs NULL', sl['slip_sites'] == 45 and sl['NULL_minus_HF'] == sl['NULL_fix'] - sl['hf_computed_fix'],
          f"(HF {sl['hf_computed_fix']}, NULL {sl['NULL_fix']}, both {sl['both']})")
    tok = pc.tokenizer()
    prose = 'Write a function to find the larger of two integers.'
    code = 'fn larger(a: i32, b: i32) -> i32 {\n    if a > b { a } else { b }\n}'
    enc = tok(prose + ' ' + code, return_offsets_mapping=True)
    ids, offs = enc['input_ids'], enc['offset_mapping']
    at = len(prose) + 1 + code.index(' > ') + 1
    h = next(t for t in range(1, len(offs)) if offs[t][0] <= at < offs[t][1])
    one = lambda s: tok.encode(s, add_special_tokens=False)
    key = lambda c: json.dumps(pc.lexeme_key(c))
    judged = {key(code.replace(' > ', ' < ', 1)): {'to': '<', 'label': 'verified'},
              key(code.replace(' > ', ' >= ', 1)): {'to': '>=', 'label': 'fail'}}
    plan = {'1|1|1': (one(' <'), True), '1|2|1': (one(' >='), True), '1|3|1': (one('<'), True), '1|4|1': (one(' <'), False)}
    S = {sid: {'site_id': sid, 'kind': 'belief', 'problem': 1, 'prose': prose, 'code': code, 'ids': ids, 'h': h,
               'judged': judged, 'hf_runner_up': {'known': kn, 'to': '<' if kn else None, 'label': 'verified' if kn else None}}
         for sid, (_, kn) in plan.items()}
    Jt = {(sid, 'BAN'): {'Y': int(t == one(' <'))} for sid, (t, _) in plan.items()}
    runt = {'A': {sid: {'BAN': {'flipped': True, 't_star_id': t[0]}} for sid, (t, _) in plan.items()}, 'B': {}, 'C': {}}
    hc = hf_comparison(sorted(S), S, Jt, runt, lambda: tok)['TL_BAN_lexeme_vs_HF_runner_up']
    check("' <' and '<' match the runner-up '<', ' >=' does not; the unknown runner-up is left out",
          all(len(x) == 1 for x, _ in plan.values()) and hc['n_known'] == 3 and hc['same'] == 2 and hc['differ'] == 1
          and hc['differ_sites'] == ['1|2|1'], f"({hc})")

    print('6. dry run on a pilot-shaped table (KIND=pilot, SITES/JUDGED/RUN_DIR elsewhere)')
    dry = os.path.join(base, 'dry')
    if os.path.exists(dry):
        shutil.rmtree(dry)
    os.makedirs(dry)
    with open(os.path.join(dry, 'pilot_sites.jsonl'), 'w', encoding='utf-8') as f:
        for r in pc.read_jsonl(os.path.join(full, 'sites.jsonl')):
            f.write(json.dumps({**r, 'kind': 'pilot' if r['kind'] == 'belief' else r['kind']}) + '\n')
    pc.write_json(os.path.join(dry, 'alpha.json'), {'alpha': 0.02, 'test_run': False, 'sites_sha256': 'another table'})
    ENV_PATHS.update(SITES=os.path.join(dry, 'pilot_sites.jsonl'), JUDGED=os.path.join(full, 'judged.jsonl'),
                     RUN_DIR=os.path.join(full, 'run'), ANALYSIS=os.path.join(dry, 'analysis_dry.json'))
    KIND = 'pilot'
    try:
        rd = analyse(dry)
    finally:
        ENV_PATHS.update(SITES=None, JUDGED=None, RUN_DIR=None, ANALYSIS=None)
        KIND = 'belief'
    base_res = analyse(full)
    check('dry run: marked, alpha falls back to p8_common.ALPHA, same tests as the belief run',
          rd['dry_run'] and rd['kind'] == 'pilot' and rd['alpha']['value'] == pc.ALPHA and 'dry run' in rd['alpha']['source']
          and [(e['D'], e['m'], e['decision']) for e in rd['confirmatory']['sequence']] ==
          [(e['D'], e['m'], e['decision']) for e in base_res['confirmatory']['sequence']] and not base_res['dry_run'],
          f"({rd['descriptive']['analysis_set']['n_sites']} pilot-kind sites; {rd['alpha']['source'][:60]}...)")
    print('selftest', 'PASSED' if ok else 'FAILED')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
