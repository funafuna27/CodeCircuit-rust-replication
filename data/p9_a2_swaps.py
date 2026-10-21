"""Phase 9, A2: enumerate `mut` relocations in compiling samples, labelled by rustc.

Source S: a compiling sample (stored label and lib-mode rustc both clean), first of its
whitespace-free duplicates within a problem, in pool order (Phase 3 samples 0-19 first).

Bindings (Phase 4's lexer): `let [mut] NAME` followed by `:`, `=` or `;`; `for [mut] NAME in`;
and `[mut] NAME :` among a function's parameters. x (the binding that loses `mut`) is a let or a
for binding. y (the one that gains it) may be any immutable binding; its kind is recorded, because
a parameter of the target function sits in the signature the prompt dictates.

For each `mut` binding x, rustc checks S with that `mut ` removed:
  necessary    errors, all in the mutability family (E0384, E0596, E0594), each naming x
  unnecessary  compiles
  other        anything else (not used)
Then x's `mut ` (the exact text from `mut` to the name) is moved to an immutable let binding y:
  violation  x necessary: S' must fail with mutability errors naming x only
  legal      x unnecessary: S' must compile
Moving the text keeps the character count; the Gemma token count and multiset are checked on
prose + ' ' + code (the text every phase graphed) and recorded, not assumed.

q, the violating token: rustc groups several mutable borrows of one binding into one diagnostic
whose primary span is the declaration ("not mutable") and whose secondary spans are the uses, so
the primary span is not q. Every span of x's mutability errors that does not cover x's declaration
is a use site; q's site is the first one. Recorded per violation: the Gemma token at the site's
start (q_place) and the first token inside the site that makes it a mutation (q_op): an
assignment operator, a method name after `.`, or `mut` in `&mut`.

Nothing generated is executed: rustc only, lib mode (--emit=metadata), 8 GiB, 60 s per call.

  POOL=all|phase3  OUT=results/p9_state/a2_candidates_<pool>.jsonl.gz  WORKERS=6  [LIMIT=n]
"""
import gzip
import json
import os
import resource
import sys
import tempfile
import time
from collections import Counter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import p9_common as C  # noqa: E402
import p9_a1_diag as D  # noqa: E402

POOL = os.environ.get('POOL', 'all')
OUT = os.environ.get('OUT', os.path.join(C.OUT_DIR, f'a2_candidates_{POOL}.jsonl.gz'))
WORKERS = int(os.environ.get('WORKERS', 6))
LIMIT = int(os.environ.get('LIMIT', 0))
ASSIGN = {'=', '+=', '-=', '*=', '/=', '%=', '^=', '&=', '|=', '<<=', '>>='}
KEYWORDS = {'mut', 'ref', '_', 'self', 'Self', 'fn', 'let', 'in', 'if', 'match', 'return'}


def bindings(code):
    """Binding sites: {'kind', 'mut', 'name', 'name_pos', 'mut_pos', 'depth'} in source order."""
    toks = C.lex(code)
    toks = [t for t in toks if t[0] != 'comment']
    out, depth = [], 0
    paren, in_sig = 0, False
    for k, (kind, t, s) in enumerate(toks):
        nxt = lambda j: toks[k + j] if k + j < len(toks) else ('', '', -1)
        if kind == 'punct' and t == '{':
            depth += 1
            in_sig = False
        elif kind == 'punct' and t == '}':
            depth -= 1
        elif kind == 'ident' and t == 'fn':
            in_sig, paren = True, 0
        elif in_sig and kind == 'punct' and t == '(':
            paren += 1
        elif in_sig and kind == 'punct' and t == ')':
            paren -= 1
        if kind == 'ident' and t in ('let', 'for'):
            j, mut = 1, False
            if nxt(1)[1] == 'mut':
                mut, j = True, 2
            name = nxt(j)
            follow = nxt(j + 1)[1]
            ok = name[0] == 'ident' and name[1] not in KEYWORDS and (
                (t == 'let' and follow in (':', '=', ';')) or (t == 'for' and follow == 'in'))
            if ok:
                out.append({'kind': t, 'mut': mut, 'name': name[1], 'name_pos': name[2],
                            'mut_pos': nxt(1)[2] if mut else None, 'depth': depth})
        elif in_sig and paren == 1 and kind == 'ident' and t not in KEYWORDS:
            prev = toks[k - 1] if k else ('', '', -1)
            if nxt(1)[1] == ':' and prev[1] in ('(', ',', 'mut'):
                out.append({'kind': 'param', 'mut': prev[1] == 'mut', 'name': t, 'name_pos': s,
                            'mut_pos': prev[2] if prev[1] == 'mut' else None, 'depth': depth})
    return out


def move_mut(code, x, y):
    """Move x's `mut ` text to just before y's name. Returns (new code, text moved)."""
    piece = code[x['mut_pos']:x['name_pos']]
    a, b = x['mut_pos'], x['name_pos']
    yp = y['name_pos']
    if yp < a:
        return code[:yp] + piece + code[yp:a] + code[b:], piece
    return code[:a] + code[b:yp] + piece + code[yp:], piece


def check(code):
    with tempfile.TemporaryDirectory(prefix='p9a2_') as td:
        return D.compile_lib(code, td)


def mut_errors_only(res, name):
    errs = [d for d in res['diags'] if d['level'] == 'error']
    return bool(errs) and all(C.family(d['code']) == 'mutability' and f'`{name}`' in d['msg'] for d in errs)


def use_sites(res, decl_pos, decl_end):
    """Spans of the mutability errors that do not cover the declaration, sorted by start."""
    sites = []
    for d in res['diags']:
        if d['level'] != 'error' or C.family(d['code']) != 'mutability':
            continue
        for s in d['spans']:
            if s['w'] == 'code' and not (s['s'] <= decl_pos < max(s['e'], s['s'] + 1)) \
                    and not (decl_pos <= s['s'] < decl_end):
                sites.append({'s': s['s'], 'e': s['e'], 'label': s.get('label'), 'code': d['code'],
                              'primary': s['p']})
    return sorted(sites, key=lambda z: (z['s'], z['e']))


def mutating_token(code, site):
    """Char offset of the token that makes the site a mutation, or None."""
    toks = [t for t in C.lex(code) if site['s'] <= t[2] < max(site['e'], site['s'] + 1)]
    for i, (kind, t, s) in enumerate(toks):
        if kind == 'punct' and t in ASSIGN:
            return s
        if kind == 'punct' and t == '.' and i + 1 < len(toks) and toks[i + 1][0] == 'ident':
            return toks[i + 1][2]
        if kind == 'ident' and t == 'mut' and i and toks[i - 1][1] == '&':
            return s
    # the site may be only the place (`x` in `x.push(..)`, `x[i] = ..`): walk on from its end over
    # fields and balanced brackets until an assignment operator or a method call
    after = [t for t in C.lex(code) if t[2] >= site['e'] and t[0] != 'comment'][:40]
    depth = 0
    for i, (kind, t, s) in enumerate(after):
        if kind == 'punct' and t in ('[', '('):
            depth += 1
        elif kind == 'punct' and t in (']', ')'):
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0 and kind == 'punct' and t in ASSIGN:
            return s
        elif depth == 0 and kind == 'punct' and t == '.' and i + 1 < len(after) and after[i + 1][0] == 'ident' \
                and i + 2 < len(after) and after[i + 2][1] in ('(', '::'):
            return after[i + 1][2]
        elif depth == 0 and kind == 'punct' and t in (';', '{', '}', ','):
            return None
    return None


def work(item):
    key, code = item
    out = {'key': key, 'necessity': [], 'cands': [], 'n_compiles': 0}
    bs = bindings(code)
    out['n_bindings'] = Counter(f"{b['kind']}{'_mut' if b['mut'] else ''}" for b in bs)
    targets = [b for b in bs if b['mut'] and b['kind'] in ('let', 'for')]
    ys = [b for b in bs if not b['mut']]          # let, for and param; the kind is recorded
    for x in targets:
        stripped = code[:x['mut_pos']] + code[x['name_pos']:]
        r = check(stripped)
        out['n_compiles'] += 1
        if r['rc'] == 0:
            nec = 'unnecessary'
        elif mut_errors_only(r, x['name']):
            nec = 'necessary'
        else:
            nec = 'other'
        out['necessity'].append({'name': x['name'], 'kind': x['kind'], 'pos': x['name_pos'], 'class': nec})
        if nec == 'other':
            continue
        for y in ys:
            new, piece = move_mut(code, x, y)
            r2 = check(new)
            out['n_compiles'] += 1
            cand = {'type': 'violation' if nec == 'necessary' else 'legal',
                    'x': {k: x[k] for k in ('kind', 'name', 'name_pos', 'mut_pos', 'depth')},
                    'y': {k: y[k] for k in ('kind', 'name', 'name_pos', 'depth')},
                    'piece': piece, 'code': new, 'rc': r2['rc'],
                    'errors': [d['code'] for d in r2['diags'] if d['level'] == 'error']}
            # the moved binding's declaration in the new code
            shift = len(piece)
            if y['name_pos'] < x['mut_pos']:
                x_decl, y_decl = x['name_pos'], y['name_pos'] + shift
            else:
                x_decl, y_decl = x['name_pos'] - shift, y['name_pos']
            cand['x_decl_new'], cand['y_decl_new'] = x_decl, y_decl
            if nec == 'necessary':
                cand['ok'] = mut_errors_only(r2, x['name'])
                sites = use_sites(r2, x_decl, x_decl + len(x['name'])) if cand['ok'] else []
                cand['sites'] = sites
                if sites:
                    cand['q_char'] = sites[0]['s']
                    cand['q_op_char'] = mutating_token(new, sites[0])
                else:
                    cand['ok'] = False
            else:
                cand['ok'] = r2['rc'] == 0
                # later uses of x in the new code, as candidate q sites for the legal control
                cand['x_uses'] = [s for kind, t, s in C.lex(new) if kind == 'ident' and t == x['name']
                                  and s > x_decl]
            out['cands'].append(cand)
    return out


def _init():
    resource.setrlimit(resource.RLIMIT_AS, (D.MEM_LIMIT, D.MEM_LIMIT))


def main():
    S = C.OUT_DIR
    if POOL == 'phase3':
        samples = C.read_jsonl(os.path.join(HERE, 'within_samples.jsonl'))
        labels = {(int(r['task_id']), int(r['sample'])): r for r in C.read_jsonl(os.path.join(HERE, 'within_labels.jsonl'))}
        diag = {(r['task_id'], r['sample']): r for r in C.read_jsonl(os.path.join(S, 'a1_diag.jsonl.gz'))}
        phase = {(int(r['task_id']), int(r['sample'])): '3' for r in samples}
    else:
        samples = C.read_jsonl(os.path.join(S, 'pool_samples.jsonl.gz'))
        labels = {(int(r['task_id']), int(r['sample'])): r for r in C.read_jsonl(os.path.join(S, 'pool_labels.jsonl.gz'))}
        diag = {(r['task_id'], r['sample']): r for r in C.read_jsonl(os.path.join(S, 'a1_diag_pool.jsonl.gz'))}
        phase = {(int(r['task_id']), int(r['sample'])): r['phase'] for r in samples}
    by_key = {(int(r['task_id']), int(r['sample'])): r for r in samples}
    seen, items, n_comp = {}, [], 0
    for k in sorted(by_key):
        r = by_key[k]
        if not C.truthy(labels[k]['compiles']) or diag[k]['lib']['rc'] != 0:
            continue
        n_comp += 1
        wk = C.ws_key(r['generated_code'])
        if wk in seen.setdefault(k[0], set()):
            continue
        seen[k[0]].add(wk)
        items.append((k, r['generated_code']))
    if LIMIT:
        items = items[:LIMIT]
    print(f'pool {POOL}: {len(by_key)} samples, {n_comp} compile (stored and lib), {len(items)} unique sources',
          flush=True)

    tok = C.tokenizer()
    t0, n = time.time(), 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with Pool(WORKERS, initializer=_init) as pool, gzip.open(OUT + '.part', 'wt', encoding='utf-8') as f:
        for res in pool.imap(work, items, chunksize=2):
            k = tuple(res['key'])
            r = by_key[k]
            prefix = r['text'] + ' '
            enc_s = tok(prefix + r['generated_code'], return_offsets_mapping=True)
            res['phase'] = phase[k]
            res['passes'] = C.truthy(labels[k]['passes'])
            res['n_tok_src'] = len(enc_s['input_ids'])
            for c in res['cands']:
                enc = tok(prefix + c['code'], return_offsets_mapping=True)
                ids = enc['input_ids']
                c['n_tok'] = len(ids)
                c['same_len'] = len(ids) == len(enc_s['input_ids'])
                c['same_multiset'] = Counter(ids) == Counter(enc_s['input_ids'])
                off = enc['offset_mapping']

                def tok_at(ch):
                    if ch is None:
                        return None
                    pos = len(prefix) + ch
                    return next((i for i, (a, b) in enumerate(off) if a <= pos < b), None)
                c['tok_x_decl'] = tok_at(c['x_decl_new'])
                c['tok_y_decl'] = tok_at(c['y_decl_new'])
                c['tok_y_mut'] = tok_at(c['y_decl_new'] - len(c['piece']))
                if c['type'] == 'violation' and c.get('q_char') is not None:
                    c['tok_q'] = tok_at(c['q_char'])
                    c['tok_q_op'] = tok_at(c.get('q_op_char'))
                    for name in ('tok_q', 'tok_q_op'):
                        if c[name] is not None:
                            c[name + '_piece'] = tok.convert_ids_to_tokens(ids[c[name]])
                if c['type'] == 'legal':
                    c['tok_x_uses'] = [tok_at(u) for u in c['x_uses']]
                c['tok_mut_piece'] = tok.convert_ids_to_tokens(ids[c['tok_y_mut']]) if c['tok_y_mut'] is not None else None
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
            n += 1
            if n % 500 == 0 or n == len(items):
                print(f'  {n}/{len(items)} sources  {time.time() - t0:.0f} s', flush=True)
    os.replace(OUT + '.part', OUT)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
