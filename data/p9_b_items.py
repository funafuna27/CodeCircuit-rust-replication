"""Phase 9, stage B: the items whose texts the forward pass scores. No model output is read.

Roles
  primary    violations of A2 with y a let binding and both edits before q (aligned), minus two
             exclusions fixed at gate 1: y named like x, and x's name bound again anywhere in the
             code (another let/for/parameter/closure parameter). Problems: those that hold one.
  pilot      violations with y a let binding whose y edit lies after q, in problems that hold no
             primary item: these problems can never enter the primary analysis.
  behaviour  natural mutability errors (A1, pool) whose rustc suggestion is a parameter of the
             target function (`a0` mutated without `mut`), deduplicated: S is Gemma's failing sample,
             M is S with `mut ` added before that parameter (rustc: M must not have the error).

Texts per primary/pilot item (characters; prose + ' ' + code is what the forward pass scores)
  S  the source             V  x's `mut ` moved to y (the violation)
  A  S with y's `mut ` added, x's kept (compiles; rustc unused_mut)      R  S with x's `mut ` removed
  Pz S with the `mut ` of another necessary let binding z moved to y, for every z whose own first
     violation lies after q's span (specificity control; rustc-checked here)
After both edits, a character at S offset s sits at s in V, s + L in A and Pz, s - L in R (L = len of
the moved text). Targets are stored in S offsets and are all after both edits:
  span     q_place .. q_op of the first mutation site (the registered primary span)
  reads    reads of x after the last edit and before q_place: same function as x's declaration, not
           after `.`, not inside a mutation site
  x_spans  every occurrence of x's name (excluded from the background)
  last_edit_end  the end of the later of the two edits (background tokens lie between it and q)

  OUT=results/p9_state/b_items.jsonl.gz   WORKERS=8
"""
import gzip
import json
import os
import re
import resource
import sys
import tempfile
from collections import Counter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
import p9_common as C  # noqa: E402
import p9_a1_diag as D  # noqa: E402
import p9_a2_swaps as W  # noqa: E402

OUT = os.environ.get('OUT', os.path.join(C.OUT_DIR, 'b_items.jsonl.gz'))
WORKERS = int(os.environ.get('WORKERS', 8))


def rebound(code, x):
    names = [b for b in W.bindings(code) if b['name'] == x['name']]
    closure = re.search(r'\|[^|\n]*\b' + re.escape(x['name']) + r'\b[^|\n]*\|', code)
    return len(names) > 1 or bool(closure)


def fn_extent_of(code, pos):
    """(start, end) of the top-level item (fn) holding character pos, by brace depth."""
    toks = [t for t in C.lex(code) if t[0] != 'comment']
    depth, start = 0, None
    for kind, t, s in toks:
        if kind == 'ident' and t == 'fn' and depth == 0:
            start = s
        if kind == 'punct' and t == '{':
            depth += 1
        elif kind == 'punct' and t == '}':
            depth -= 1
            if depth == 0 and start is not None and start <= pos <= s:
                return start, s + 1
    return 0, len(code)


def ident_spans(code, name):
    return [(s, s + len(t)) for kind, t, s in C.lex(code) if kind == 'ident' and t == name]


def site_spans_in_S(c, L):
    """Mutation sites of V mapped to S offsets (they lie after both edits)."""
    return [(s['s'], max(s['e'], s['s'] + 1)) for s in c['sites']]


def check(code):
    with tempfile.TemporaryDirectory(prefix='p9b_') as td:
        return D.compile_lib(code, td)


def build(job):
    """One source x violation candidate -> primary/pilot item with rustc-checked variants."""
    c, S, role, necessity = job
    L = len(c['piece'])
    x, y = c['x'], c['y']
    # in S: x's mut at x['mut_pos'], y's name at y['name_pos']; V's offsets equal S's after both edits
    A = S[:y['name_pos']] + c['piece'] + S[y['name_pos']:]
    R = S[:x['mut_pos']] + S[x['name_pos']:]
    site = c['sites'][0]
    # q span in V offsets == S offsets (both edits before q)
    q_place, q_op = c['q_char'], c['q_op_char']
    out = {'role': role, 'key': c['key'], 'x': x, 'y': y, 'piece': L, 'S': S, 'V': c['code'], 'A': A, 'R': R,
           'q_place': q_place, 'q_op': q_op, 'site_code': site['code'], 'site_label': site['label'],
           'y_other_fn': fn_extent_of(S, x['name_pos']) != fn_extent_of(S, y['name_pos']),
           'checks': {}}
    if out['V'] != (S[:x['mut_pos']] + S[x['name_pos']:y['name_pos']] + c['piece'] + S[y['name_pos']:]
                    if y['name_pos'] > x['mut_pos'] else
                    S[:y['name_pos']] + c['piece'] + S[y['name_pos']:x['mut_pos']] + S[x['name_pos']:]):
        out['checks']['V_rebuild'] = False
    rA = check(A)
    out['checks']['A_compiles'] = rA['rc'] == 0
    lo, hi = fn_extent_of(S, x['name_pos'])
    sites = site_spans_in_S(c, L)
    in_site = lambda p: any(a <= p < b for a, b in sites)
    edit_end = max(x['name_pos'], y['name_pos'] + L)
    out['last_edit_end'] = edit_end
    # x's own declaration in V: at x['mut_pos'] if y follows x, else at x['name_pos']
    x_decl_v = x['mut_pos'] if y['name_pos'] > x['mut_pos'] else x['name_pos']
    reads = []
    for a, b in ident_spans(c['code'], x['name']):
        if edit_end <= a < q_place and a != x_decl_v and lo <= a < hi and not in_site(a) \
                and c['code'][max(0, a - 1)] != '.':
            reads.append(a)
    out['reads'] = reads
    out['x_spans'] = ident_spans(c['code'], x['name'])
    # specificity placebos: other necessary let mut z, moved to the same y; legal up to q's span end
    pz = []
    for n in necessity:
        if n['class'] != 'necessary' or n['kind'] != 'let' or n['pos'] == x['name_pos']:
            continue
        z = next((b for b in W.bindings(S) if b['name_pos'] == n['pos'] and b['mut']), None)
        if z is None or z['mut_pos'] > q_place:
            continue
        Pz, piece_z = W.move_mut(S, z, y)
        if len(piece_z) != L:
            continue
        r = check(Pz)
        errs = [d for d in r['diags'] if d['level'] == 'error']
        # z's declaration in Pz (as x's in V); its first use site must lie after q's span
        z_decl = z['mut_pos'] if y['name_pos'] > z['mut_pos'] else z['name_pos']
        uses = W.use_sites(r, z_decl, z_decl + len(z['name']))
        first = uses[0]['s'] if uses else None
        # every error must be z's mutability error, and its first use site after q's span
        ok = all(C.family(d['code']) == 'mutability' and f"`{z['name']}`" in d['msg'] for d in errs) and \
            (first is None or first > q_op)
        if ok:
            pz.append({'z': z['name'], 'z_name_pos': z['name_pos'], 'z_mut_pos': z['mut_pos'], 'text': Pz,
                       'z_errors': [d['code'] for d in errs]})
    out['placebos'] = pz
    return out


def build_behaviour(job):
    key, S, decl_pos, sites = job
    M = S[:decl_pos] + 'mut ' + S[decl_pos:]
    r = check(M)
    errs = [d for d in r['diags'] if d['level'] == 'error']
    return {'role': 'behaviour', 'key': key, 'S': S, 'M': M, 'decl_pos': decl_pos, 'sites': sites,
            'M_errors': [d['code'] for d in errs], 'M_mut_errors': sum(C.family(d['code']) == 'mutability' for d in errs)}


def _init():
    resource.setrlimit(resource.RLIMIT_AS, (D.MEM_LIMIT, D.MEM_LIMIT))


def main():
    S_DIR = C.OUT_DIR
    samples = {(int(r['task_id']), int(r['sample'])): r for r in C.read_jsonl(os.path.join(S_DIR, 'pool_samples.jsonl.gz'))}
    rows = C.read_jsonl(os.path.join(S_DIR, 'a2_candidates_all.jsonl.gz'))
    jobs, excl = [], Counter()
    prim_probs = set()
    viol = []
    for r in rows:
        for c in r['cands']:
            if c['ok'] and c['type'] == 'violation' and c['y']['kind'] == 'let' and c.get('tok_q') is not None:
                viol.append((r, dict(c, key=r['key'])))
    for r, c in viol:
        if c['tok_y_mut'] < c['tok_q']:
            prim_probs.add(r['key'][0])
    for r, c in viol:
        S = samples[tuple(r['key'])]['generated_code']
        aligned = c['tok_y_mut'] < c['tok_q']
        if aligned:
            role = 'primary'
        elif r['key'][0] not in prim_probs:
            role = 'pilot'
        else:
            excl['y after q, problem in primary'] += 1
            continue
        if c['y']['name'] == c['x']['name']:
            excl[f'{role}: y named like x'] += 1
            continue
        if rebound(S, c['x']):
            excl[f'{role}: x bound again'] += 1
            continue
        jobs.append((c, S, role, r['necessity']))
    # behaviour items: natural mutability errors on a target-function parameter
    diag = {(r['task_id'], r['sample']): r for r in C.read_jsonl(os.path.join(S_DIR, 'a1_diag_pool.jsonl.gz'))}
    import p8_judge
    sigs = p8_judge.typeable()
    beh, seen = [], {}
    for k in sorted(samples):
        code = samples[k]['generated_code']
        ds = [d for d in diag[k]['lib']['diags'] if d['level'] == 'error']
        if not ds or not all(C.family(d['code']) == 'mutability' for d in ds):
            continue
        ext = C.extent(code, sigs[k[0]]['name'])
        decls = {s['s'] for d in ds for ch in d['children'] for s in ch['spans']
                 if s['w'] == 'code' and (s.get('repl') or '') == 'mut ' and C.where(s['s'], ext) == 'sig'}
        if len(decls) != 1:
            continue
        wk = C.ws_key(code)
        if wk in seen.setdefault(k[0], set()):
            continue
        seen[k[0]].add(wk)
        sites = sorted({(s['s'], s['e']) for d in ds for s in d['spans'] if s['w'] == 'code' and s['s'] > min(decls) + 2})
        beh.append((list(k), code, min(decls), sites))
    print(f'violations y=let: {len(viol)}; jobs {len(jobs)}; exclusions {dict(excl)}; behaviour sources {len(beh)}', flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = Counter()
    with Pool(WORKERS, initializer=_init) as pool, gzip.open(OUT + '.part', 'wt', encoding='utf-8') as f:
        for it in pool.imap(build, jobs, chunksize=4):
            f.write(json.dumps(it, ensure_ascii=False) + '\n')
            n[it['role']] += 1
        for it in pool.imap(build_behaviour, beh, chunksize=4):
            f.write(json.dumps(it, ensure_ascii=False) + '\n')
            n['behaviour'] += 1
    os.replace(OUT + '.part', OUT)
    print('wrote', OUT, dict(n))


if __name__ == '__main__':
    main()
