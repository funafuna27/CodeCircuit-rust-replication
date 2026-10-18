"""Phase 8, post hoc (decided after analysis.json was read; not in PREREGISTRATION_P8_INTERVENTION.md).
Descriptive only, CPU, labels already judged. Writes results/intervention/posthoc.json.

  single   How often does zeroing ONE feature fix the bug, as in the paper's §4.4 example? The
           registered existence rate (exist:G) skips BAN's lexeme, so a single feature that only
           hands the decision to the runner-up never counts. Here it counts: per site, among the
           replayed single-ablation flips of stage B (GxA's top 256 and G's top 128; the matched
           random features apart), (a) any flip out of E, (b) a flip to BAN's lexeme, (c) (b) with
           BAN verified, (d) a verified token other than BAN's lexeme (= exist:G).
  G_fix    G's verified sites split by where t*_G landed: BAN's lexeme or not.
  slips    NULL's fixes against the HF-computed set, split by whether the HF label was known
           before the run (the §4 prediction counted only known labels).
  gen_NULL Greedy generation without any intervention: at its verified sites, does the generated
           text begin with S's wrong piece (so the repair happens later in the line)?
"""
import collections
import json
import os

import p8_common as C

OUT = C.OUT


def ew(rows, key):
    """(count, n, per-site share, equal-weight share over problems)."""
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r['problem']].append(bool(r[key]))
    n = sum(bool(r[key]) for r in rows)
    return {'count': n, 'n_sites': len(rows), 'per_site': n / len(rows),
            'equal_weight': sum(sum(v) / len(v) for v in byp.values()) / len(byp)}


def main():
    an = json.load(open(os.path.join(OUT, 'analysis.json')))
    summ = json.load(open(os.path.join(OUT, 'judged.json')))
    sites = {r['site_id']: r for r in C.read_jsonl(os.path.join(OUT, 'sites.jsonl'))}
    J = {(r['site_id'], r['arm']): r for r in C.read_jsonl(os.path.join(OUT, 'judged.jsonl'))}
    aset = sorted(s['site_id'] if isinstance(s, dict) else s for s in an['descriptive']['analysis_set']['sites'])
    assert len(aset) == an['descriptive']['analysis_set']['n_sites']
    out = {'note': 'Post hoc, descriptive; not pre-registered. ' + __doc__.split('\n')[0],
           'inputs': {f: C.sha256(os.path.join(OUT, f)) for f in ('analysis.json', 'judged.json', 'judged.jsonl', 'sites.jsonl')}}

    et = summ['existence_tokens']
    rows = []
    for s in aset:
        g = et.get(s, {}).get('exist:G', {})
        rnd = et.get(s, {}).get('exist:RND', {})
        flips = lambda c: sum(v for k, v in c.items() if k not in ('in_E', 'repeat')) > 0
        ban_y = J[(s, 'BAN')]['Y']
        rows.append({'site': s, 'problem': sites[s]['problem'], 'flip_G': flips(g), 'flip_RND': flips(rnd),
                     'to_ban': g.get('ban_lexeme', 0) > 0, 'fix_via_ban': g.get('ban_lexeme', 0) > 0 and ban_y == 1,
                     'fix_other': J[(s, 'exist:G')]['Y'] == 1})
    for r in rows:
        r['fix_any'] = r['fix_via_ban'] or r['fix_other']
    out['single'] = {k: ew(rows, k) for k in ('flip_G', 'flip_RND', 'to_ban', 'fix_via_ban', 'fix_other', 'fix_any')}
    out['single']['token_counts_pooled'] = {
        arm: dict(sum((collections.Counter(et.get(s, {}).get(arm, {})) for s in aset), collections.Counter()))
        for arm in ('exist:G', 'exist:RND')}

    g_ver = [s for s in aset if J[(s, 'G')]['Y'] == 1]
    out['G_fix'] = {'verified': len(g_ver),
                    'at_ban_lexeme': sum(bool(J[(s, 'G')].get('lexeme_is_ban')) for s in g_ver),
                    'elsewhere': sum(not J[(s, 'G')].get('lexeme_is_ban') for s in g_ver),
                    'neutralizing_elsewhere': sum(bool(J[(s, 'G')].get('neutralizing')) and not J[(s, 'G')].get('lexeme_is_ban')
                                                  for s in g_ver)}

    slips = sorted(s for s, r in sites.items() if r['kind'] == 'slip')
    tab = collections.Counter()
    for s in slips:
        h = sites[s]['hf_runner_up']
        tab[('HF known' if h.get('known') else 'HF unknown', str(h.get('label')), J[(s, 'NULL')]['label'],
             J[(s, 'NULL')].get('how'))] += 1
    out['slips'] = {'n': len(slips), 'table': [{'hf': k[0], 'hf_label': k[1], 'NULL_label': k[2], 'NULL_how': k[3], 'sites': v}
                                               for k, v in sorted(tab.items())]}

    gv = []
    for s in aset:
        if J[(s, 'gen:NULL')]['Y'] != 1:
            continue
        cf = os.path.join(OUT, 'run', 'C', s.replace('|', '_') + '.json')
        text = json.load(open(cf))['gen']['NULL']['text']
        wp = sites[s]['wrong_piece'].replace('▁', ' ')      # SentencePiece's space mark
        gv.append({'site': s, 'starts_with_wrong_piece': text.startswith(wp),
                   'wrong_piece': sites[s]['wrong_piece'], 'gen_head': text[:60]})
    out['gen_NULL'] = {'verified': len(gv), 'starts_with_wrong_piece': sum(r['starts_with_wrong_piece'] for r in gv),
                       'sites': gv}

    C.write_json(os.path.join(OUT, 'posthoc.json'), out)
    sg = out['single']
    for k, lab in (('flip_G', 'a single top feature flips the decision out of E'),
                   ('flip_RND', 'a single matched random feature flips it'),
                   ('to_ban', "a single-feature flip lands on BAN's lexeme"),
                   ('fix_via_ban', "  ... and BAN's lexeme is verified"),
                   ('fix_other', "a single feature gives a verified token other than BAN's lexeme (exist:G)"),
                   ('fix_any', 'a single feature gives a verified token')):
        v = sg[k]
        print(f"{lab}: {v['count']}/{v['n_sites']} sites ({v['per_site']:.3f} per site, {v['equal_weight']:.3f} equal-weight)")
    print('pooled single-ablation flip tokens:', sg['token_counts_pooled'])
    print('G verified:', out['G_fix'])
    print('slips:', json.dumps(out['slips']['table']))
    print('gen:NULL verified:', out['gen_NULL']['verified'], 'start with the wrong piece:', out['gen_NULL']['starts_with_wrong_piece'])
    for r in gv:
        print('   ', r['site'], repr(r['wrong_piece']), repr(r['gen_head']))


if __name__ == '__main__':
    main()
