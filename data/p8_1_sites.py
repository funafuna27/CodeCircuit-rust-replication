"""Phase 8, step 1: the site table and the pilot pool-A table. CPU only (tokenizer, lexer, rustc);
no model, no GPU. Design §2-§3, §5, §9; SPEC 'sites'.

 1 Re-judge. Phase 6's verified-repair sites are the (task, sample, pos) with a repair that
   ownfix_verify.json calls verified: 210 sites in 35 problems. Every mutant Phase 6 judged there
   (759) is judged again twice, by the study's judge unchanged ('old') and by the finite-aware judge
   (p8_judge, 'new'), each with Phase 6's protocol: one judging, two more if it passes, then for a
   stable pass the calibrated extra inputs stored in ownfix_verify.json. Labels: verified / lucky /
   unverifiable / fail / no_compile. The old labels must equal the stored ones at all 759 (and the
   first verdict field by field). Where old and new differ in anything (a case index, a label), every
   case that passes the old judge and fails the new one must return inf/-inf/NaN (p8_judge.probe_values).
   A site stays if a finite-aware verified repair remains at it.
 2 Kind. belief: S's token was the HF argmax at the site (ownfix_logprob s_top); slip: it was not.
 3 Tokens. ids = Gemma's ids of prose + ' ' + code (BOS first); h = the token whose characters cover
   the edit's first character (p6_2's rule); decision at h-1; wrong_id = ids[h].
   The code a token sequence stands for is decode(ids[1:]) (special tokens kept as text, no clean-up)
   after prose + ' '. E = every vocabulary id v such that the code of ids with ids[h] := v has S's
   lexeme_key. judged maps json.dumps(lexeme_key(mutant code)) to its 'to' and finite-aware label;
   verified_ids = ids whose splice has a verified mutant's key; neutralizing maps the same keys to the
   patterns the edit introduces (neutral_counts).
   Shortcut: only candidate ids are lexed: those whose decoded piece without whitespace equals the
   whitespace-free text token h must have for S's or a judged mutant's key (targets), plus every id
   whose piece contains '/', a quote or U+FFFD and every byte token. Why this is complete: without
   '/' or a quote a piece can neither open or close a comment or literal nor change how the comments
   and literals of the rest lex, so the splice's non-comment text is S's outside token h; equal lexer
   streams have equal concatenated token texts, hence equal text without whitespace, which pins the
   piece's whitespace-free text. Checked against the full vocabulary at EXHAUSTIVE sites: the first
   belief site (by site_id) of each edit family and of the pilot table.
   rustc filter: 'whitespace' in lexeme_key is m1.lex's (Python str.isspace), which also skips
   U+2009, U+3000, U+00A0, U+001C-U+001F etc. that rustc rejects ('unknown start of token'). An id whose
   decoded piece holds such a character (c.isspace() and c not in RUST_WS, rustc's
   Pattern_White_Space) is left out of E and verified_ids: its splice is not a Rust program. Dropped
   ids are listed per site in sites.json (E.rustc_filter), and the first site of each dropped piece is
   spliced and judged (must be no_compile). Deviation from SPEC's E (lexeme_key equality alone).
   Scoring a t* whose piece holds such a character must go to the compiler, not the judged map
   (p8_3's concern; the judged map's keys are lexeme_key's).
 4 bug = task|S's edited line without whitespace|wrong_id.
 5 Tiers and order (belief sites): tier 1 = one site per bug drawn with rng(6) (bugs in sorted order,
   a bug's sites in (sample, pos) order, one integers(n) draw each); tier 2 = the rest. Problems are
   ordered by rng(9).permutation of the sorted problem list; within a problem rng(8, task) permutes
   its tier-1 sites, then (same generator) its tier-2 sites. Global order: tier 1 round-robin over
   problems (round r takes the r-th site of each problem, problems in their order), then tier 2 the
   same way. Slips (NULL only, descriptive) are built the same way with keys rng(6, 1), rng(8, task,
   1), rng(9, 1) and follow every belief site in the order.
 6 Calibration. p6_1b's rebuild (p6_1b_verify.inputs_for, reference_values; the stored labels'
   passing samples, first CALIB whitespace-distinct; the old judge) must reproduce the stored extra
   inputs of all 35 problems. rule_b: a verified repair at the site whose code without whitespace is
   one of its problem's calibrating samples (ownfix_verify 'calib'), i.e. its extra inputs were
   calibrated on itself. Its verdict is recomputed with that sample left out (the same rebuild without
   it; the repair judged on the remaining inputs by the finite-aware judge). Descriptive: labels are
   not changed (the flag drives the 'without the rule-B sites' sensitivity analysis).
   Also descriptive, for the f64+isclose site problems: a calibration finite-aware throughout (the
   passing samples re-judged by the finite-aware judge, the survivors calibrate with it) and whether
   each kept site there keeps a verified repair on those inputs (sites.json 'finite_calibration').
   The site table itself uses the stored inputs (SPEC).
 7 neutralizing (design §5): x-x, x%x, x/x, *=1, /=1, %=1, %1, +0, -0, *1, /1 on the lexer stream,
   counted per program; a mutant is flagged with each pattern that occurs more often in it than in S.
 8 HF BAN (descriptive, design §6): the HF runner-up at a belief site is known when the best
   log-prob-scored alternative beats all unscored mass (as the scouting a1_sites.py); its label.
 9 Pilot pool A: problems outside the 35 whose stable repairs are all unverifiable (38); their
   belief sites among the sites with a stable repair. rng(3) chooses 30 of the problems that have
   one (choice without replacement over the sorted list), then one site per chosen problem in draw
   order (integers(n) over its belief sites in (sample, pos) order).
   Same schema, kind 'pilot', tier 1, order = draw order; their judged mutants are re-judged as in 1.

Writes P8_OUT (default data/results/intervention/): sites.jsonl (belief rows in run order, then
slips), pilot_sites.jsonl, sites.json (counts, per-problem counts, bugs, tiers, hashes, rustc
version, the re-judge agreement table, every check), rejudge.jsonl (one row per re-judged mutant:
stored, old and finite-aware verdicts) and sites_judge_cache.jsonl (every judging, keyed by the
judge tag and its inputs; a restart reuses only lines of the current judge tag = sha256 of p8_judge.py,
1_rust_correct.py, mbpp_rust.py, ownfix_verify.json, rustc --version, the timeouts and the typeable
table, exactly as p8_3_judge.judge_tag;
sites.json records the tag, the cache's sha256 and how many lines were reused / ignored / run; delete
it to judge from scratch. Not p8_3's judge_cache.jsonl).

Usage (cwd data/):  python p8_1_sites.py [--test]
  --test   also runs the judge canaries; exit 1 if any check fails (SPEC tests A). Without it a
           failed check gives exit 2 after writing.
Env: WORKERS=8 (judge processes, 8 GiB RLIMIT_AS each)  EXHAUSTIVE=1 (0 skips the full-vocabulary
     check)  P8_OUT
data/1_code_correct.py is never opened, imported or run.
"""
import ast
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
import p8_common as C  # noqa: E402
import p8_judge as J  # noqa: E402

WORKERS = int(os.environ.get('WORKERS', 8))
EXHAUSTIVE = os.environ.get('EXHAUSTIVE', '1') == '1'
CACHE = os.path.join(C.OUT, 'sites_judge_cache.jsonl')
N_PILOT = 30
RISKY_CHARS = '/"\'�'
BYTE_RE = re.compile(r'<0x[0-9A-F]{2}>')
LABELS = ('verified', 'lucky', 'unverifiable', 'fail', 'no_compile', 'judge_error')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def nows(s):
    return re.sub(r'\s+', '', s)


RUST_WS = set('\t\n\x0b\x0c\r \x85‎‏  ')   # rustc's Pattern_White_Space


def nonrust_ws(s):
    """s holds a character m1.lex skips as whitespace but rustc does not accept as whitespace."""
    return any(c.isspace() and c not in RUST_WS for c in s)


def key_of(code):
    return json.dumps(C.lexeme_key(code))


def mkey(m):
    return (m['task_id'], m['sample'], m['pos'], m['to'])


def sid(t, s, p):
    return f'{t}|{s}|{p}'


# ---------------------------------------------------------------- neutralizing patterns
KEYWORDS = {'as', 'break', 'const', 'continue', 'crate', 'else', 'enum', 'extern', 'fn', 'for', 'if',
            'impl', 'in', 'let', 'loop', 'match', 'mod', 'move', 'mut', 'pub', 'ref', 'return',
            'static', 'struct', 'trait', 'type', 'unsafe', 'use', 'where', 'while', 'dyn'}
PREC = {'*': 11, '/': 11, '%': 11, '+': 10, '-': 10, '<<': 9, '>>': 9, '&': 8, '^': 7, '|': 6,
        '==': 5, '!=': 5, '<': 5, '>': 5, '<=': 5, '>=': 5, '&&': 4, '||': 3, '..': 2, '..=': 2}
_SUF = r'(?:_?(?:[iu](?:8|16|32|64|128|size)|f32|f64))?'
ONE = re.compile(r'1(?:\.0*)?' + _SUF)
ZERO = re.compile(r'0(?:\.0*)?' + _SUF)


def _is_operand_end(tk):
    k, t = tk
    return k in ('int', 'num', 'str') or (k == 'ident' and t not in KEYWORDS) or t in (')', ']', '?')


def _close(T, i):
    o = T[i][1]
    c = {'(': ')', '[': ']'}[o]
    d = 0
    for j in range(i, len(T)):
        if T[j][0] == 'punct' and T[j][1] == o:
            d += 1
        elif T[j][0] == 'punct' and T[j][1] == c:
            d -= 1
            if d == 0:
                return j
    return None


def _chain_end(T, i):
    """End (exclusive) of the operand chain starting at T[i]: an atom (identifier, literal or a
    parenthesised group) followed by field/path/call/index/?/macro postfixes; None if none starts."""
    n = len(T)
    if i >= n:
        return None
    k, t = T[i]
    if k in ('int', 'num', 'str') or (k == 'ident' and t not in KEYWORDS):
        i += 1
    elif k == 'punct' and t == '(':
        j = _close(T, i)
        if j is None:
            return None
        i = j + 1
    else:
        return None
    while i < n:
        t = T[i][1]
        if t in ('.', '::') and i + 1 < n and T[i + 1][0] in ('ident', 'int'):
            i += 2
        elif t in ('(', '['):
            j = _close(T, i)
            if j is None:
                return None
            i = j + 1
        elif t == '!' and i + 1 < n and T[i + 1][1] in ('(', '['):
            j = _close(T, i + 1)
            if j is None:
                return None
            i = j + 1
        elif t == '?':
            i += 1
        else:
            break
    return i


def _binary_at(T, k):
    return k >= 1 and _is_operand_end(T[k - 1])


def _rhs_is_literal(T, k, pat):
    """T[k+1] is the literal and nothing after it binds tighter than a binary operator would."""
    return (k + 1 < len(T) and T[k + 1][0] in ('int', 'num') and pat.fullmatch(T[k + 1][1]) is not None
            and (k + 2 >= len(T) or T[k + 2][1] not in ('.', '(', '[', 'as', '::')))


def neutral_counts(code):
    """Occurrences of each neutralizing pattern in code's lexer stream (comments ignored).
      x-x x%x x/x  a binary -, % or / whose left and right operand chains are token-identical and are
                   its true operands: the operator before the left chain is not unary and does not
                   bind at least as tightly (left-assoc), the operator after the right chain does not
                   bind more tightly, and neither side is cast with `as`
      *1 /1 %1     a binary *, / or % whose right operand is the literal 1 (1, 1.0, 1i64, ...)
      *=1 /=1 %=1  a compound *=, /=, %= whose right-hand side is the literal 1 alone
      +0 -0        a binary + or - whose right operand is the literal 0, or += / -= with 0 alone"""
    T = [(k, t) for k, t, _ in C.lex(code) if k != 'comment']
    out = Counter()
    n = len(T)
    for k in range(n):
        kind, t = T[k]
        if kind != 'punct':
            continue
        nxt_end = k + 2 >= n or (T[k + 2][1] not in PREC and T[k + 2][1] not in ('.', '(', '[', 'as', '::'))
        if t in ('*=', '/=', '%=') and _rhs_is_literal(T, k, ONE) and nxt_end:
            out[t + '1'] += 1
        if t in ('+=', '-=') and _rhs_is_literal(T, k, ZERO) and nxt_end:
            out[t[0] + '0'] += 1
        if not _binary_at(T, k):
            continue
        if t in ('*', '/', '%') and _rhs_is_literal(T, k, ONE):
            out[t + '1'] += 1
        if t in ('+', '-') and _rhs_is_literal(T, k, ZERO):
            out[t + '0'] += 1
        if t in ('-', '%', '/'):
            e = _chain_end(T, k + 1)
            if e is None:
                continue
            s = next((s for s in range(max(0, k - 80), k) if _chain_end(T, s) == k), None)
            if s is None or T[s:k] != T[k + 1:e]:
                continue
            P = PREC[t]
            if s >= 1:
                b = T[s - 1][1]
                if b == 'as' or b in ('.', '::'):
                    continue
                if b in PREC or b == '!':
                    unary = not _binary_at(T, s - 1)
                    if unary or PREC.get(b, 99) >= P:
                        continue
            if e < n and (T[e][1] == 'as' or PREC.get(T[e][1], 0) > P):
                continue
            out['x' + t + 'x'] += 1
    return out


def neutralizing(code_s, code_m):
    cs, cm = neutral_counts(code_s), neutral_counts(code_m)
    bad = sorted(p for p in cm if cm[p] > cs.get(p, 0))
    assert all(p in C.NEUTRALIZING for p in bad), bad
    return bad


# ---------------------------------------------------------------- tokens
class Vocab:
    def __init__(self, tok):
        self.tok = tok
        self.n = len(tok)
        self.dec = [tok.decode([i], clean_up_tokenization_spaces=False) for i in range(self.n)]
        pieces = tok.convert_ids_to_tokens(list(range(self.n)))
        self.byte = {i for i, p in enumerate(pieces) if BYTE_RE.fullmatch(p)}
        self.risky = sorted(self.byte | {i for i, d in enumerate(self.dec) if any(c in d for c in RISKY_CHARS)})
        self.by_nows = defaultdict(list)
        for i, d in enumerate(self.dec):
            self.by_nows[nows(d)].append(i)


def decode_code(tok, ids, prose):
    """The code a token sequence stands for: decode(ids[1:]) after prose + ' ', or None."""
    text = tok.decode(ids[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    head = prose + ' '
    return text[len(head):] if text.startswith(head) else None


def splice_key(tok, ids, h, v, prose):
    s = list(ids)
    s[h] = v
    code = decode_code(tok, s, prose)
    return None if code is None else key_of(code)


def _comment_free(code):
    mask = bytearray(len(code))
    for k, t, s in C.lex(code):
        if k == 'comment':
            mask[s:s + len(t)] = b'\x01' * len(t)
    return mask


def targets(code, c0, c1, edits):
    """Whitespace-free text token h must decode to (comments aside) for the splice to have S's key or
    the key of code with each (pos, frm, to) edit."""
    mS = _comment_free(code)
    za = nows(''.join(ch for i, ch in enumerate(code[:c0]) if not mS[i]))
    zb = nows(''.join(ch for i, ch in enumerate(code[c1:], start=c1) if not mS[i]))
    out = {nows(code[c0:c1])}
    for pos, frm, to in edits:
        m = code[:pos] + to + code[pos + len(frm):]
        mm = _comment_free(m)
        zm = nows(''.join(ch for i, ch in enumerate(m) if not mm[i]))
        if len(zm) >= len(za) + len(zb) and zm.startswith(za) and zm.endswith(zb):
            out.add(zm[len(za):len(zm) - len(zb)])
    return out


def tokenize_site(tok, prose, code, pos):
    text = prose + ' ' + code
    enc = tok(text, return_offsets_mapping=True)
    ids, offs = enc['input_ids'], enc['offset_mapping']
    at = len(prose) + 1 + pos
    h = next(t for t in range(1, len(offs)) if offs[t][0] <= at < offs[t][1])
    code_toks = [t for t in range(1, len(offs)) if offs[t][0] >= len(prose)]
    return ids, h, len(code_toks) - 1 - code_toks.index(h)


def candidate_map(V, ids, h, prose, code, edits):
    """{key: sorted ids} over the shortcut's candidates, and the candidate count."""
    head = len(prose) + 1
    c0 = len(V.tok.decode(ids[1:h], clean_up_tokenization_spaces=False)) - head
    c1 = len(V.tok.decode(ids[1:h + 1], clean_up_tokenization_spaces=False)) - head
    assert 0 <= c0 <= edits[0][0] < c1, (c0, c1, edits[0])
    cand = set(V.risky)
    for tg in targets(code, c0, c1, edits):
        cand.update(V.by_nows.get(tg, ()))
    out = defaultdict(list)
    for v in sorted(cand):
        k = splice_key(V.tok, ids, h, v, prose)
        if k is not None:
            out[k].append(v)
    return out, len(cand)


# exhaustive check workers (no rlimit; tokenizer per process)
_EX_TOK = None


def _ex_init():
    global _EX_TOK
    _EX_TOK = C.tokenizer()


def _ex_chunk(args):
    prose, ids, h, lo, hi, keys = args
    found, bad = defaultdict(list), 0
    for v in range(lo, hi):
        k = splice_key(_EX_TOK, ids, h, v, prose)
        if k is None:
            bad += 1
        elif k in keys:
            found[k].append(v)
    return dict(found), bad


# ---------------------------------------------------------------- judging jobs
def judge_tag():
    """Hash of everything a verdict depends on besides the program, computed exactly as
    p8_3_judge.judge_tag (so the two tags are comparable): the judge's source files, the verification
    data, the rustc version, the compile/run timeouts (env COMPILE_TIMEOUT / RUN_TIMEOUT) and the
    typeable-signature table."""
    h = hashlib.sha256()
    for p in (J.__file__, J.JUDGE_FILE, os.path.join(HERE, 'mbpp_rust.py'), J.VERIFY_FILE):
        h.update(f"{os.path.basename(p)} {C.sha256(p)}\n".encode())
    h.update(J.rustc_version().encode())
    h.update(f"\nCOMPILE_TIMEOUT {J._NEW.COMPILE_TIMEOUT} RUN_TIMEOUT {J._NEW.RUN_TIMEOUT} "
             f"OLD {J._OLD.COMPILE_TIMEOUT} {J._OLD.RUN_TIMEOUT}\n".encode())
    h.update(json.dumps(J.typeable(), sort_keys=True, default=repr).encode())
    return h.hexdigest()


def _jkey(tag, kind, args):
    return hashlib.sha256(json.dumps([tag, kind, args], sort_keys=True, default=repr).encode()).hexdigest()[:40]


def _job(job):
    key, kind, args = job
    try:
        if kind == 'label':
            mode, task, code = args
            r = J.label(task, code, finite=(mode == 'new'))
        elif kind == 'cases':
            mode, task, code, cases = args
            f = J.judge_cases(task, code, ast.literal_eval(cases), finite=(mode == 'new'))
            r = None if f is None else sorted(f)
        elif kind == 'judge1':
            mode, task, code = args
            r = J.judge_once(task, code, finite=(mode == 'new'))
        elif kind == 'values':
            task, code, cases = args
            r = J.probe_values(task, code, None if cases is None else ast.literal_eval(cases))
        else:
            raise ValueError(kind)
        return key, {'ok': True, 'r': r}
    except Exception as e:                        # surfaced in the checks, never a silent label
        return key, {'ok': False, 'r': f'{type(e).__name__}: {e}'[:300]}


class Jobs:
    """Judgings keyed by the judge tag and their inputs; finished ones are appended to CACHE as they
    arrive. A line written under another judge tag (another judge, harness, verify file or rustc) is
    never reused: its key cannot match, and it is skipped (and counted) on load."""

    def __init__(self, pool, tag):
        self.pool, self.tag = pool, tag
        self.done = {}
        self.stats = Counter()
        if os.path.exists(CACHE):
            with open(CACHE, encoding='utf-8') as f:
                for ln in f:
                    if ln.strip():
                        try:
                            d = json.loads(ln)
                        except json.JSONDecodeError:  # a line cut by a crash
                            self.stats['cut lines'] += 1
                            continue
                        if d.get('tag') != tag:
                            self.stats['lines of another judge tag (ignored)'] += 1
                        elif d['v']['ok']:
                            self.done[d['k']] = d['v']
                            self.stats['lines reused'] += 1
        self.stats['judgings run now'] = 0
        log(f'judge cache {CACHE}: tag {tag[:16]}; {dict(self.stats)}')

    def run(self, items, what):
        """items: [(kind, args)] with cases passed as repr strings. Returns [result] in order."""
        keys = [_jkey(self.tag, k, a) for k, a in items]
        todo, seen = [], set()
        for key, (k, a) in zip(keys, items):
            if key not in self.done and key not in seen:
                seen.add(key)
                todo.append((key, k, a))
        uniq = len(set(keys))
        log(f'{what}: {len(items)} judgings ({len(items) - uniq} repeat an identical one), '
            f'{uniq - len(todo)} cached, {len(todo)} to run')
        t0 = time.time()
        with open(CACHE, 'a', encoding='utf-8') as f:
            for i, (key, v) in enumerate(self.pool.imap_unordered(_job, todo, chunksize=1), 1):
                self.done[key] = v
                f.write(json.dumps({'k': key, 'tag': self.tag, 'v': v}) + '\n')
                f.flush()
                self.stats['judgings run now'] += 1
                if i % 200 == 0:
                    log(f'  {i}/{len(todo)} ({time.time() - t0:.0f} s)')
        errs = [self.done[k]['r'] for k in keys if not self.done[k]['ok']]
        if errs:
            raise SystemExit(f'{what}: {len(errs)} judgings raised, e.g. {errs[0]}')
        return [self.done[k]['r'] for k in keys]


# ---------------------------------------------------------------- labels
def stored_label(m, ver):
    if m['compiles'] is None:
        return 'judge_error'
    if m['compiles'] is False:
        return 'no_compile'
    if not m['is_r']:
        return 'fail'                           # fails, or passed once but not in 3 judgings
    return {True: 'verified', False: 'lucky', None: 'unverifiable'}[ver[mkey(m)]]


VFIELDS = ('compiles', 'passes', 'n_pass', 'stage', 'error')


def fail_idx(v):
    return set() if v['passes'] else {int(i) for i in re.findall(r'(?:FAIL|PANIC) (\d+)', v.get('error') or '')}


def table(pairs):
    t = defaultdict(Counter)
    for a, b in pairs:
        t[a][b] += 1
    return {a: dict(sorted(c.items())) for a, c in sorted(t.items())}


# ---------------------------------------------------------------- order
def assign_order(rows, salt, start):
    """Tiers (rng(6, *salt)) and round-robin order (rng(8, task, *salt), rng(9, *salt)); see the
    docstring, item 5. Returns the next free order index."""
    by_bug = defaultdict(list)
    for r in rows:
        by_bug[r['bug']].append(r)
    g6 = C.rng(6, *salt)
    tier1 = set()
    for b in sorted(by_bug):
        ss = sorted(by_bug[b], key=lambda r: (r['sample'], r['pos']))
        tier1.add(ss[int(g6.integers(len(ss)))]['site_id'])
    for r in rows:
        r['tier'] = 1 if r['site_id'] in tier1 else 2
    problems = sorted({r['problem'] for r in rows})
    porder = [problems[i] for i in C.rng(9, *salt).permutation(len(problems))]
    lists = {}
    for t in problems:
        g8 = C.rng(8, t, *salt)
        for tier in (1, 2):
            ss = sorted((r for r in rows if r['problem'] == t and r['tier'] == tier),
                        key=lambda r: (r['sample'], r['pos']))
            lists[(t, tier)] = [ss[i] for i in g8.permutation(len(ss))]
    seq = []
    for tier in (1, 2):
        rnd = 0
        while True:
            got = [lists[(t, tier)][rnd] for t in porder if rnd < len(lists[(t, tier)])]
            if not got:
                break
            seq += got
            rnd += 1
    assert len(seq) == len(rows)
    for i, r in enumerate(seq):
        r['order'] = start + i
    return start + len(seq), porder


# ---------------------------------------------------------------- main
def main(test):
    t_start = time.time()
    checks = {}

    def check(name, ok, detail=None):
        checks[name] = {'ok': bool(ok), **({'detail': detail} if detail is not None else {})}
        log(('PASS ' if ok else 'FAIL ') + name + ('' if detail is None else f': {detail}'))

    os.makedirs(C.OUT, exist_ok=True)
    if test:
        ok, rows = J.canaries()
        check('judge canaries', ok, [f"{r['canary']}: finite {r.get('finite')} old {r.get('old')}" for r in rows])
    rustc = J.rustc_version()

    # ------------------------------------------------ data
    sources = {(s['task_id'], s['sample']): s for s in C.read_jsonl(os.path.join(C.OWNFIX, 'ownfix_sources.jsonl.gz'))}
    muts = C.read_jsonl(os.path.join(C.OWNFIX, 'ownfix_mutants.jsonl.gz'))
    lps = defaultdict(list)
    for r in C.read_jsonl(os.path.join(C.OWNFIX, 'ownfix_logprob.jsonl.gz')):
        lps[(r['task_id'], r['sample'], r['pos'])].append(r)
    vraw = J.verify_data()
    ver = {}
    for k, x in vraw['mutants'].items():
        t, s, p, to = k.split('|', 3)
        ver[(int(t), int(s), int(p), to)] = x
    by_site = defaultdict(list)
    for m in muts:
        by_site[(m['task_id'], m['sample'], m['pos'])].append(m)
    sites210 = sorted({k[:3] for k, x in ver.items() if x is True})
    P35 = sorted({k[0] for k in sites210})
    judged210 = [m for k in sites210 for m in sorted(by_site[k], key=lambda m: m['to'])]
    check('210 verified-repair sites in 35 problems, 759 judged mutants',
          (len(sites210), len(P35), len(judged210)) == (210, 35, 759),
          f'{len(sites210)} sites, {len(P35)} problems, {len(judged210)} mutants')
    typ = J.typeable()
    close = {t for t, s in typ.items() if s['ret'] == 'f64' and any(c[2] for c in s['cases'])}

    # pool A (label-free draw): problems outside the 35 whose stable repairs are all unverifiable
    rep_by_problem = defaultdict(list)
    for m in muts:
        if m['is_r']:
            rep_by_problem[m['task_id']].append(m)
    poolA = sorted(t for t, rs in rep_by_problem.items() if t not in P35 and all(ver[mkey(m)] is None for m in rs))
    poolA_sites = sorted({(m['task_id'], m['sample'], m['pos']) for t in poolA for m in rep_by_problem[t]})

    def s_top_of(k):
        vals = {r['s_top'] for r in lps[k]}
        assert len(vals) == 1, (k, vals)
        return vals.pop()

    beliefA = defaultdict(list)
    for k in poolA_sites:
        if s_top_of(k):
            beliefA[k[0]].append(k)
    probsA = sorted(beliefA)
    g3 = C.rng(3)
    pick = g3.choice(len(probsA), N_PILOT, replace=False)
    pilot_keys = []
    for i in pick:
        ss = sorted(beliefA[probsA[int(i)]], key=lambda k: (k[1], k[2]))
        pilot_keys.append(ss[int(g3.integers(len(ss)))])
    judgedP = [m for k in pilot_keys for m in sorted(by_site[k], key=lambda m: m['to'])]
    check('pilot pool A: 38 problems, disjoint from the 35; 30 of those with belief sites drawn, one site each',
          len(poolA) == 38 and not set(poolA) & set(P35) and len(probsA) >= N_PILOT
          and len({k[0] for k in pilot_keys}) == N_PILOT,
          f'pool A {len(poolA)} problems ({len(poolA_sites)} repair sites), {len(probsA)} with belief sites '
          f'({sum(map(len, beliefA.values()))}); {len(judgedP)} judged mutants at the 30 pilot sites')
    check('pool A has no f64+isclose problem (old and finite judges write identical programs there)',
          not set(poolA) & close, sorted(set(poolA) & close))

    # ------------------------------------------------ re-judge (old and finite-aware)
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(WORKERS, initializer=J.init_worker)
    jtag = judge_tag()
    jobs = Jobs(pool, jtag)
    allm = judged210 + judgedP
    res = jobs.run([('label', [mode, m['task_id'], m['code']]) for m in allm for mode in ('old', 'new')],
                   're-judge (old, finite)')
    for i, m in enumerate(allm):
        m['_old'], m['_new'] = res[2 * i], res[2 * i + 1]
        m['_stored'] = stored_label(m, ver)

    def agreement(ms, name):
        lab_ok = all(m['_old']['label'] == m['_stored'] for m in ms)
        v_ok = all(all(m['_old']['verdicts'][0][f] == m[f] for f in VFIELDS) for m in ms)
        stab_ok = all(m['_old']['stable_pass'] == bool(m['is_r']) for m in ms)
        bad = [f"{sid(*mkey(m)[:3])} {m['to']}: stored {m['_stored']} old {m['_old']['label']}"
               for m in ms if m['_old']['label'] != m['_stored']][:10]
        check(f'{name}: old judge reproduces the stored labels ({len(ms)} mutants)', lab_ok and v_ok and stab_ok,
              {'labels': lab_ok, 'first verdict fields': v_ok, 'stability': stab_ok, 'disagreements': bad})

    agreement(judged210, 'sites')
    agreement(judgedP, 'pilot')

    # where old and finite-aware differ, the values must be non-finite
    diff = []
    for m in allm:
        o, n = m['_old'], m['_new']
        d_mbpp = set()
        for vo, vn in zip(o['verdicts'], n['verdicts']):
            d_mbpp |= fail_idx(vn) - fail_idx(vo)
        d_extra = set(n['extra_bad'] or []) - set(o['extra_bad'] or [])
        if o['label'] != n['label'] or d_mbpp or d_extra or len(o['verdicts']) != len(n['verdicts']) \
                or (o['extra'] != n['extra']):
            diff.append((m, sorted(d_mbpp), sorted(d_extra)))
    probes = jobs.run([('values', [m['task_id'], m['code'], None]) for m, _, _ in diff]
                      + [('values', [m['task_id'], m['code'], repr(J.stored_cases(m['task_id']))])
                         for m, _, de in diff if de], 'value probes where the judges differ')
    it_m, it_e = iter(probes[:len(diff)]), iter(probes[len(diff):])
    unexplained = []
    for m, dm, de in diff:
        vals_m = next(it_m)
        vals_e = next(it_e) if de else None
        m['_probe'] = {'mbpp': vals_m, 'extra': vals_e, 'mbpp_cases_changed': dm, 'extra_cases_changed': de}
        ok = vals_m is not None and all(J.nonfinite(vals_m[i]) for i in dm) and \
            (not de or (vals_e is not None and all(J.nonfinite(vals_e[i]) for i in de))) and \
            m['task_id'] in close and (dm or de)
        if not ok:
            unexplained.append(f"{sid(*mkey(m)[:3])} {m['to']}: old {m['_old']['label']} new {m['_new']['label']} "
                               f"mbpp {dm} {vals_m} extra {de}")
    check('finite-aware judge differs only where an output is non-finite', not unexplained,
          {'mutants that differ': len(diff), 'unexplained': unexplained[:10]})

    # ------------------------------------------------ sites after the fix
    keep = [k for k in sites210 if any(m['_new']['label'] == 'verified' for m in by_site[k])]
    dropped = [k for k in sites210 if k not in set(keep)]

    # calibration: p6_1b's rebuild (old judge, stored passing labels) must reproduce the stored inputs
    log('rebuilding calibrations')
    import p6_1b_verify as V6                    # noqa: E402  (loads m1_make_mutants, p6_1_repairs)
    from p6_1_repairs import load_samples        # noqa: E402
    from datasets import load_dataset            # noqa: E402
    ds = load_dataset('mbpp', 'sanitized')
    mbpp = {int(t['task_id']): t for s in ('train', 'test', 'validation', 'prompt') for t in ds[s]}
    samples = load_samples()
    passing = {}
    for (t, smp), (s, lab) in sorted(samples.items()):
        if t in P35 and lab['passes'] and not s['truncated']:
            passing.setdefault(t, {}).setdefault(nows(s['generated_code']), s['generated_code'])
    calib, cand = {}, {}
    for t in P35:
        calib[t] = list(passing.get(t, {}).values())[:V6.CALIB]
        assert [nows(c) for c in calib[t]] == vraw['problems'][str(t)]['calib'], f'calib keys differ for {t}'
        sig = typ[t]
        is_close = any(c[2] for c in sig['cases'])
        cand[t] = [(a, v, is_close) for a, v in V6.reference_values(mbpp[t], sig, V6.inputs_for(sig))]
    res = jobs.run([('cases', ['old', t, c, repr(cand[t])]) for t in P35 for c in calib[t]],
                   'calibration rebuild (35 problems, old judge)')
    it = iter(res)
    calbad = {(t, j): next(it) for t in P35 for j in range(len(calib[t]))}

    def kept_cases(cands, bads):
        """p6_1b: the candidate inputs every calibrating sample that compiles and finishes passes
        (none if no sample does)."""
        fails = [b for b in bads if b is not None]
        bad = set().union(*map(set, fails)) if fails else set()
        return [c for i, c in enumerate(cands) if i not in bad] if fails else []

    cal_report = {}
    for t in P35:
        st = vraw['problems'][str(t)]['cases']
        co = kept_cases(cand[t], [calbad[(t, j)] for j in range(len(calib[t]))])
        cal_report[t] = {'n_calib': len(calib[t]), 'n_candidates': len(cand[t]),
                         'stored_inputs': len(ast.literal_eval(st)), 'old_reproduces': repr(co) == st}
    check('calibration rebuild with the old judge reproduces the stored extra inputs (35 problems)',
          all(r['old_reproduces'] for r in cal_report.values()),
          sorted(t for t, r in cal_report.items() if not r['old_reproduces']))

    # rule B: verified (finite) repairs at kept sites that equal a calibrating sample. Leave-one-out:
    # p6_1b's calibration without that sample (old judge, as the stored inputs), the repair judged on
    # the inputs that remain by the finite-aware judge (as its Phase 8 label)
    rb = []
    for k in keep:
        t = k[0]
        ckeys = vraw['problems'][str(t)]['calib']
        for m in by_site[k]:
            if m['_new']['label'] == 'verified' and nows(m['code']) in ckeys:
                j = ckeys.index(nows(m['code']))
                rb.append((k, m, j, kept_cases(cand[t], [calbad[(t, i)] for i in range(len(calib[t])) if i != j])))
    rb_run = [x for x in rb if len(x[3]) >= J.MIN_INPUTS]   # else the leave-out verdict is None
    res = jobs.run([('cases', ['new', k[0], m['code'], repr(cs)]) for k, m, j, cs in rb_run],
                   'rule-B repairs on leave-one-out inputs')
    lo_verdict = {mkey(m): (f is not None and not f) for (k, m, j, cs), f in zip(rb_run, res)}
    rule_b = defaultdict(list)
    for k, m, j, cs in rb:
        rule_b[k].append({'to': m['to'], 'key': key_of(m['code']), 'calib_index': j,
                          'n_inputs_leave_out': len(cs), 'verdict_leave_out': lo_verdict.get(mkey(m))})

    # descriptive: a calibration that is finite-aware throughout, for the f64+isclose site problems
    # (elsewhere both judges write identical programs). Every sample the stored labels call passing is
    # judged once by the finite-aware judge; the first CALIB whitespace-distinct ones that still pass
    # calibrate the candidates with the finite-aware judge; each verified repair at a kept site there
    # is judged on the inputs that remain. Labels in the table are NOT changed by this.
    closeP = [t for t in P35 if t in close]
    pas = [(t, s['generated_code']) for (t, smp), (s, lab) in sorted(samples.items())
           if t in closeP and lab['passes'] and not s['truncated']]
    res = jobs.run([('judge1', ['new', t, c]) for t, c in pas], 'finite re-judge of passing samples (f64+isclose)')
    passing_f = defaultdict(dict)
    for (t, c), v in zip(pas, res):
        if v['compiles'] and v['passes']:
            passing_f[t].setdefault(nows(c), c)
    calib_f = {t: list(passing_f[t].values())[:V6.CALIB] for t in closeP}
    res = jobs.run([('cases', ['new', t, c, repr(cand[t])]) for t in closeP for c in calib_f[t]],
                   'finite calibration (f64+isclose)')
    it = iter(res)
    inputs_f = {t: kept_cases(cand[t], [next(it) for _ in calib_f[t]]) for t in closeP}
    fr = [(k, m) for k in keep if k[0] in closeP for m in by_site[k] if m['_new']['label'] == 'verified']
    fr_run = [(k, m) for k, m in fr if len(inputs_f[k[0]]) >= J.MIN_INPUTS]
    res = jobs.run([('cases', ['new', k[0], m['code'], repr(inputs_f[k[0]])]) for k, m in fr_run],
                   'repairs on finite calibration (f64+isclose)')
    fr_verdict = {mkey(m): (f is not None and not f) for (k, m), f in zip(fr_run, res)}
    finite_cal = {'problems': {}, 'sites': {}}
    for t in closeP:
        finite_cal['problems'][str(t)] = {
            'passing_stored': len({nows(c) for tt, c in pas if tt == t}), 'passing_finite': len(passing_f[t]),
            'calib_stored': len(calib[t]), 'calib_finite': len(calib_f[t]),
            'calib_same': [nows(c) for c in calib_f[t]] == vraw['problems'][str(t)]['calib'],
            'inputs_stored': cal_report[t]['stored_inputs'], 'inputs_finite': len(inputs_f[t]),
            'inputs_same': repr(inputs_f[t]) == vraw['problems'][str(t)]['cases']}
    for k, m in fr:
        e = finite_cal['sites'].setdefault(sid(*k), {'kind': 'belief' if s_top_of(k) else 'slip', 'repairs': {}})
        e['repairs'][m['to']] = fr_verdict.get(mkey(m))
    for e in finite_cal['sites'].values():
        e['still_verified'] = any(v is True for v in e['repairs'].values())
    finite_cal['sites_not_verified'] = sorted(s_ for s_, e in finite_cal['sites'].items() if not e['still_verified'])
    log(f"finite-aware calibration (descriptive): {finite_cal['problems']}; kept sites that would lose "
        f"every verified repair: {finite_cal['sites_not_verified']}")
    pool.close()
    pool.join()

    # ------------------------------------------------ rows
    log('tokenizing, E, judged maps')
    tok = C.tokenizer()
    Vc = Vocab(tok)
    splice_stats = Counter()
    splice_of, splice_bad = {}, []
    ws_dropped = {}

    def build_row(k, kind):
        t, smp, pos = k
        s = sources[(t, smp)]
        prose, code = s['prose'], s['code']
        ms = sorted(by_site[k], key=lambda m: m['to'])
        frm, fam, line = ms[0]['from'], ms[0]['family'], ms[0]['line']
        assert all((m['from'], m['family'], m['line']) == (frm, fam, line) for m in ms)
        assert code[pos:pos + len(frm)] == frm and line == code.count('\n', 0, pos)
        ids, h, tte = tokenize_site(tok, prose, code, pos)
        assert ids[0] == tok.bos_token_id and len(ids) == s['tokens'] and h >= 2
        s_code = decode_code(tok, ids, prose)
        splice_stats['S decodes to its code'] += s_code == code
        cmap, n_cand = candidate_map(Vc, ids, h, prose, code, [(pos, frm, m['to']) for m in ms])
        skey = key_of(code)
        judged, neut, vkeys = {}, {}, set()
        for m in ms:
            mk = key_of(m['code'])
            assert mk not in judged and mk != skey
            neut[mk] = neutralizing(code, m['code'])
            judged[mk] = {'to': m['to'], 'label': m['_new']['label'], 'neutralizing': bool(neut[mk])}
            if m['_new']['label'] == 'verified':
                vkeys.add(mk)
            mids = tok(prose + ' ' + m['code'])['input_ids']
            dif = [i for i in range(min(len(ids), len(mids))) if ids[i] != mids[i]]
            if len(mids) == len(ids) and dif == [h]:
                sp = list(ids)
                sp[h] = mids[h]
                good = decode_code(tok, sp, prose) == m['code']
                splice_of[mkey(m)] = ('single', good, mids[h] in cmap.get(mk, ()))
                if not good:
                    splice_bad.append(f"{sid(*k)} {m['to']}")
            else:
                splice_of[mkey(m)] = ('not single at h', None, None)
        rows_lp = lps[k]
        p_wrong = {math.exp(r['orig_logprob']) for r in rows_lp}
        assert len(p_wrong) == 1
        p_wrong = p_wrong.pop()
        s_top = s_top_of(k)
        # HF runner-up (scouting a1_sites.py's rule)
        scored = {r['to']: math.exp(r['edit_logprob']) for r in rows_lp if r['edit_logprob'] is not None}
        rem = max(0.0, 1.0 - p_wrong - sum(scored.values()))
        top = max(scored.items(), key=lambda x: x[1]) if scored else (None, 0.0)
        known = top[0] is not None and top[1] > rem and (s_top or top[1] > p_wrong)
        lab_of = {m['to']: m['_new']['label'] for m in ms}
        line_text = code.split('\n')[line]
        # rustc filter: an id whose piece holds a character m1.lex skips but rustc rejects is not in E
        # or verified_ids (its splice does not compile); recorded per site in sites.json
        e_all = sorted(cmap.get(skey, []))
        v_all = sorted(v for kk in vkeys for v in cmap.get(kk, []))
        E = [v for v in e_all if not nonrust_ws(Vc.dec[v])]
        v_ids = [v for v in v_all if not nonrust_ws(Vc.dec[v])]
        ws_dropped[sid(*k)] = {'E': [v for v in e_all if v not in E],
                               'verified_ids': [v for v in v_all if v not in v_ids]}
        return {
            'site_id': sid(*k), 'task': t, 'sample': smp, 'pos': pos, 'line': line, 'kind': kind,
            'prose': prose, 'code': code, 'ids': ids, 'h': h, 'wrong_id': ids[h],
            'wrong_piece': tok.convert_ids_to_tokens(ids[h]), 'E': E,
            'family': fam, 'frm': frm, 'judged': judged,
            'verified_ids': v_ids,
            'bug': f'{t}|{nows(line_text)}|{ids[h]}', 'tier': None, 'order': None, 'problem': t,
            'tokens_to_end': tte, 'rule_b': bool(rule_b.get(k)), 'rule_b_detail': rule_b.get(k, []),
            'neutralizing': neut, 'p_wrong_hf': p_wrong, 's_top': bool(s_top),
            'hf_runner_up': {'known': bool(known), 'to': top[0] if known else None,
                             'label': lab_of.get(top[0]) if known else None, 'p': top[1] if known else None},
            'n_candidates': n_cand,
        }, cmap

    rows, cmaps = [], {}
    for k in keep:
        r, cm = build_row(k, 'belief' if s_top_of(k) else 'slip')
        rows.append(r)
        cmaps[r['site_id']] = cm
    prow = []
    for k in pilot_keys:
        r, cm = build_row(k, 'pilot')
        prow.append(r)
        cmaps[r['site_id']] = cm
    for k in dropped:                            # built only for the splice check over all 759
        build_row(k, 'dropped')
    for name, ms in (('759 at the 210 sites', judged210), ('pilot', judgedP)):
        sp = [splice_of[mkey(m)] for m in ms]
        splice_stats[f'{name}: mutants'] = len(sp)
        splice_stats[f'{name}: single-token replacement at h'] = sum(x[0] == 'single' for x in sp)
        splice_stats[f'{name}: its splice decodes to the stored code'] = sum(x[1] is True for x in sp)
        splice_stats[f'{name}: its id reaches the key through the shortcut'] = sum(x[2] is True for x in sp)
    belief = [r for r in rows if r['kind'] == 'belief']
    slips = [r for r in rows if r['kind'] == 'slip']
    nxt, porder_b = assign_order(belief, (), 0)
    _, porder_s = assign_order(slips, (1,), nxt)
    for i, r in enumerate(prow):
        r['tier'], r['order'] = 1, i
    rows.sort(key=lambda r: r['order'])

    # ------------------------------------------------ checks on the table
    nb_p = len({r['problem'] for r in belief})
    nb_b = len({r['bug'] for r in belief})
    check('counts after the fix: 146 belief sites / 31 problems / 94 bugs; 45 slips',
          (len(belief), nb_p, nb_b, len(slips)) == (146, 31, 94, 45),
          f'{len(belief)} belief / {nb_p} problems / {nb_b} bugs; {len(slips)} slips '
          f'({len({r["problem"] for r in slips})} problems, {len({r["bug"] for r in slips})} bugs); '
          f'{len(dropped)} sites dropped')
    check('tier 1 = one site per bug (belief and slips)',
          sum(r['tier'] == 1 for r in belief) == nb_b
          and sum(r['tier'] == 1 for r in slips) == len({r['bug'] for r in slips}),
          f"belief tier 1 {sum(r['tier'] == 1 for r in belief)}, tier 2 {sum(r['tier'] == 2 for r in belief)}")
    check('every source decodes back to its text (210 sites and pilot)',
          splice_stats['S decodes to its code'] == len(rows) + len(prow) + len(dropped))
    n_single = sum(x[0] == 'single' for x in splice_of.values())
    check('decoded single-token splices equal the stored mutant code (759 and pilot)',
          not splice_bad and n_single > 0 and len(splice_of) == len(allm)
          and sum(x[2] is True for x in splice_of.values()) == n_single,
          {**dict(splice_stats), 'bad': splice_bad[:10]})
    check("every judged label is one of SPEC's five (no judge_error)",
          all(j['label'] in LABELS[:5] for r in rows + prow for j in r['judged'].values()),
          dict(Counter(j['label'] for r in rows + prow for j in r['judged'].values())))
    check('wrong_id in E at every site (sites and pilot)', all(r['wrong_id'] in r['E'] for r in rows + prow),
          f"E size: {dict(sorted(Counter(len(r['E']) for r in rows + prow).items()))}")
    check('no id in E or verified_ids holds a character rustc rejects as whitespace',
          not any(nonrust_ws(Vc.dec[v]) for r in rows + prow for v in r['E'] + r['verified_ids']),
          f"E ids dropped at {sum(bool(ws_dropped[r['site_id']]['E']) for r in rows)} sites, "
          f"{sum(bool(ws_dropped[r['site_id']]['E']) for r in prow)} pilot sites")
    # evidence for the filter: the first dropped id of each distinct piece, its splice judged (old judge)
    ws_probe = {}
    for r in sorted(rows + prow, key=lambda r: r['site_id']):
        for v in ws_dropped[r['site_id']]['E'] + ws_dropped[r['site_id']]['verified_ids']:
            if Vc.dec[v] not in ws_probe:
                sp = list(r['ids'])
                sp[r['h']] = v
                ws_probe[Vc.dec[v]] = (r['site_id'], v, r['task'], decode_code(tok, sp, r['prose']))
    with ctx.Pool(WORKERS, initializer=J.init_worker) as pool2:
        jobs.pool = pool2
        res = jobs.run([('label', ['old', t, code]) for _, _, t, code in ws_probe.values()],
                       'splices of the dropped pieces (rustc filter evidence)')
    ws_compile = {piece: {'site_id': s_, 'id': v, 'label': x['label'],
                          'error': str(x['verdicts'][0].get('error'))[:120]}
                  for (piece, (s_, v, _, _)), x in zip(ws_probe.items(), res)}
    check('every dropped piece, spliced at its first site, does not compile',
          all(x['label'] == 'no_compile' for x in ws_compile.values()), ws_compile)
    check('pilot rows: belief (s_top) and one per problem', all(r['s_top'] for r in prow)
          and len({r['problem'] for r in prow}) == N_PILOT)

    # exhaustive check of the shortcut (full vocabulary)
    ex_report = {}
    if EXHAUSTIVE:
        fams = {}
        for r in sorted(belief, key=lambda r: r['site_id']):
            fams.setdefault(r['family'], r)
        ex_sites = list(fams.values()) + [sorted(prow, key=lambda r: r['site_id'])[0]]
        log(f'exhaustive E check on {len(ex_sites)} sites: {[r["site_id"] for r in ex_sites]}')
        step = 4096
        with ctx.Pool(WORKERS, initializer=_ex_init) as xp:
            for r in ex_sites:
                keys = {key_of(r['code'])} | set(r['judged'])
                parts = xp.map(_ex_chunk, [(r['prose'], r['ids'], r['h'], lo, min(lo + step, Vc.n), keys)
                                           for lo in range(0, Vc.n, step)])
                full, nbad = defaultdict(list), 0
                for f, b in parts:
                    nbad += b
                    for kk, vs in f.items():
                        full[kk] += vs
                short = cmaps[r['site_id']]
                same = all(sorted(full.get(kk, [])) == sorted(short.get(kk, [])) for kk in keys)
                ex_report[r['site_id']] = {'family': r['family'], 'same': same, 'E': len(full.get(key_of(r['code']), [])),
                                           'ids_reaching_judged_keys': sum(len(full.get(kk, [])) for kk in r['judged']),
                                           'decode_not_prefixed': nbad, 'candidates_lexed': r['n_candidates']}
                log(f"  {r['site_id']} {r['family']}: same {same}, E {ex_report[r['site_id']]['E']}")
        check('shortcut E / judged-key ids equal the full-vocabulary scan', all(x['same'] for x in ex_report.values())
              and len(ex_report) >= 5, ex_report)

    # ------------------------------------------------ write
    def dump(path, rr):
        tmp = path + '.partial'
        with open(tmp, 'w', encoding='utf-8') as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    p_sites, p_pilot = os.path.join(C.OUT, 'sites.jsonl'), os.path.join(C.OUT, 'pilot_sites.jsonl')
    dump(p_sites, rows)
    dump(p_pilot, prow)
    rj = []
    for m in allm:
        rj.append({'set': 'pilot' if (m['task_id'], m['sample'], m['pos']) in set(pilot_keys) else 'sites',
                   'site_id': sid(m['task_id'], m['sample'], m['pos']), 'to': m['to'], 'key': key_of(m['code']),
                   'stored': {**{f: m[f] for f in VFIELDS}, 'is_r': m['is_r'], 'verify': ver.get(mkey(m)),
                              'label': m['_stored']},
                   'old': m['_old'], 'new': m['_new'], 'probe': m.get('_probe')})
    dump(os.path.join(C.OUT, 'rejudge.jsonl'), rj)

    def per_problem(rr):
        return dict(sorted(Counter(r['problem'] for r in rr).items(), key=lambda x: (-x[1], x[0])))

    def bugs(rr):
        b = defaultdict(list)
        for r in sorted(rr, key=lambda r: r['order']):
            b[r['bug']].append(r['site_id'])
        return dict(sorted(b.items()))

    hf = Counter()
    for r in belief:
        hf['unknown' if not r['hf_runner_up']['known'] else r['hf_runner_up']['label']] += 1
    changes = [{'site_id': sid(*mkey(m)[:3]), 'to': m['to'], 'stored': m['_stored'], 'new': m['_new']['label'],
                'probe_mbpp': (m.get('_probe') or {}).get('mbpp')}
               for m in judged210 if m['_new']['label'] != m['_stored']]
    neut = Counter(p for r in rows for ps in r['neutralizing'].values() for p in ps)
    neut_ver = Counter(p for r in rows for kk, ps in r['neutralizing'].items() if r['judged'][kk]['label'] == 'verified'
                       for p in ps)
    files = ['p8_1_sites.py', 'p8_judge.py', 'p8_common.py', 'm1_make_mutants.py', '1_rust_correct.py',
             'mbpp_rust.py', 'p6_1b_verify.py', 'p6_1_repairs.py']
    summary = {
        'script': 'p8_1_sites.py', 'rustc': rustc,
        'code_sha256': {f: C.sha256(os.path.join(HERE, f)) for f in files},
        'inputs_sha256': {f: C.sha256(os.path.join(C.OWNFIX, f)) for f in
                          ('ownfix_sources.jsonl.gz', 'ownfix_mutants.jsonl.gz', 'ownfix_logprob.jsonl.gz')}
        | {'ownfix_verify.json': C.sha256(J.VERIFY_FILE)},
        'outputs_sha256': {'sites.jsonl': C.sha256(p_sites), 'pilot_sites.jsonl': C.sha256(p_pilot)},
        'streams': {'3': 'pilot draw', '6': 'tier 1 (belief: rng(6); slips: rng(6, 1))',
                    '8': 'within-problem shuffle (belief: rng(8, task); slips: rng(8, task, 1))',
                    '9': 'problem order (belief: rng(9); slips: rng(9, 1))'},
        'counts': {'sites_before_fix': len(sites210), 'problems_before_fix': len(P35), 'judged_mutants': len(judged210),
                   'sites': len(rows), 'belief': len(belief), 'belief_problems': nb_p, 'belief_bugs': nb_b,
                   'belief_sources': len({(r['task'], r['sample']) for r in belief}),
                   'slip': len(slips), 'slip_problems': len({r['problem'] for r in slips}),
                   'slip_bugs': len({r['bug'] for r in slips}), 'dropped': len(dropped),
                   'pilot': len(prow), 'pool_A_problems': len(poolA), 'pool_A_belief_problems': len(probsA),
                   'pool_A_belief_sites': sum(map(len, beliefA.values()))},
        'per_problem': {'belief': per_problem(belief), 'slip': per_problem(slips)},
        'per_problem_sorted': {'belief': sorted(Counter(r['problem'] for r in belief).values(), reverse=True),
                               'slip': sorted(Counter(r['problem'] for r in slips).values(), reverse=True)},
        'problem_order': {'belief': porder_b, 'slip': porder_s},
        'tiers': {'belief': dict(Counter(r['tier'] for r in belief)), 'slip': dict(Counter(r['tier'] for r in slips))},
        'bugs': {'belief': bugs(belief), 'slip': bugs(slips)},
        'dropped_sites': [{'site_id': sid(*k), 'kind': 'belief' if s_top_of(k) else 'slip',
                           'repairs': {m['to']: [m['_stored'], m['_new']['label']] for m in by_site[k] if m['is_r']}}
                          for k in dropped],
        'rejudge': {'n': len(judged210),
                    'stored_x_old': table((m['_stored'], m['_old']['label']) for m in judged210),
                    'stored_x_finite': table((m['_stored'], m['_new']['label']) for m in judged210),
                    'label_changes': changes,
                    'pilot_stored_x_old': table((m['_stored'], m['_old']['label']) for m in judgedP),
                    'pilot_stored_x_finite': table((m['_stored'], m['_new']['label']) for m in judgedP)},
        'calibration': {str(t): r for t, r in cal_report.items()},
        'finite_calibration': finite_cal,
        'rule_b': {r['site_id']: r['rule_b_detail'] for r in rows if r['rule_b']},
        'hf_runner_up_belief': dict(hf),
        'neutralizing': {'judged_mutants_flagged': dict(neut), 'verified_flagged': dict(neut_ver)},
        'E': {'size': dict(sorted(Counter(len(r['E']) for r in rows).items())),
              'sites_where_another_id_splices_back': sum(len(r['E']) > 1 for r in rows),
              'belief_sites_where_another_id_splices_back': sum(len(r['E']) > 1 for r in belief),
              'rustc_filter': {
                  'rule': "ids whose decoded piece holds a character c with c.isspace() and c not in "
                          "rustc's Pattern_White_Space are left out of E and verified_ids (rustc rejects c)",
                  'rust_whitespace': sorted(f'U+{ord(c):04X}' for c in RUST_WS),
                  'sites_with_E_dropped': sum(bool(ws_dropped[r['site_id']]['E']) for r in rows),
                  'pilot_with_E_dropped': sum(bool(ws_dropped[r['site_id']]['E']) for r in prow),
                  'sites_with_verified_ids_dropped': sum(bool(ws_dropped[r['site_id']]['verified_ids'])
                                                         for r in rows + prow),
                  'pieces': sorted({Vc.dec[v] for r in rows + prow for x in ws_dropped[r['site_id']].values()
                                    for v in x}),
                  'compile_check': ws_compile,
                  'dropped': {r['site_id']: ws_dropped[r['site_id']] for r in rows + prow
                              if any(ws_dropped[r['site_id']].values())}},
              'exhaustive': ex_report},
        'judge_cache': {'path': os.path.basename(CACHE), 'judge_tag': jtag,
                        'sha256': C.sha256(CACHE), 'stats': dict(jobs.stats)},
        'splices': dict(splice_stats),
        'pilot': {'problems_drawn': [r['problem'] for r in prow], 'site_ids': [r['site_id'] for r in prow],
                  'hf_runner_up': dict(Counter('unknown' if not r['hf_runner_up']['known'] else r['hf_runner_up']['label']
                                               for r in prow))},
        'checks': checks, 'checks_ok': all(c['ok'] for c in checks.values()),
    }
    C.write_json(os.path.join(C.OUT, 'sites.json'), summary)
    log(f"wrote {p_sites} ({len(rows)} rows), {p_pilot} ({len(prow)}), sites.json, rejudge.jsonl "
        f"in {time.time() - t_start:.0f} s")
    log(f"belief {len(belief)} / {nb_p} problems / {nb_b} bugs; slips {len(slips)}; per problem "
        f"{summary['per_problem_sorted']['belief']}; HF runner-up {dict(hf)}; rule B sites "
        f"{len(summary['rule_b'])}; neutralizing {dict(neut)}")
    failed = [k for k, c in checks.items() if not c['ok']]
    log('ALL CHECKS PASS' if not failed else f'CHECKS FAILED: {failed}')
    return not failed


if __name__ == '__main__':
    ok = main('--test' in sys.argv)
    sys.exit(0 if ok else (1 if '--test' in sys.argv else 2))
