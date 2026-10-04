"""Phase 4, step 1: one-token mutants of passing solutions, judged and paired.
PREREGISTRATION_MUTANT.md §2.

Phase 3's pairs were two different solutions, and length explained 31% of the
feature differences within a pair although the members differed by a median of
two tokens. Here both members of a pair are the same passing solution with one
token replaced, and the token count is held exactly equal: one replacement
leaves the solution passing every assert (the mutant "survives"), the other
makes it fail at least one (it is "killed"). Both carry one edit, so the edit
itself cannot tell them apart.

  source   Phase 3's samples (within_samples.jsonl, within_labels.jsonl). Per
           problem, its eligible passing samples in index order: passes, not
           truncated, prose + ' ' + code <= 140 Gemma tokens.
  mutant   one token inside a function body -- brace depth >= 1, so never a
           signature; never inside a string, char or comment; never a generic
           angle bracket -- replaced by another member of its family
           (FAMILIES). Kept only if prose + ' ' + code has exactly the source's
           Gemma token count.
  judge    1_rust_correct.judge, unchanged. Compiles and passes every assert:
           survives. Compiles and fails one: killed. Does not compile, or the
           judge fails: dropped. Each worker runs under an 8 GiB address-space
           limit, inherited by rustc and the binary, so a mutant that allocates
           without bound dies instead of taking the machine with it.
  source   the first eligible passing sample with >= 1 survivor and >= 1 killed.
  pair     a survivor drawn uniformly, then a killed mutant drawn uniformly from
           the survivor's family if that family has one, else from all killed
           mutants of the source. rng = default_rng(SEED + task_id).
  cap      more than 150 pairs: 150 drawn uniformly, default_rng(SEED).
  gate     >= 70 pairs, else exit code 2.

Writes every judged mutant (MUTANTS), the graph pipeline's DATA_FILE (OUT_ROWS:
one row per member, id task_id * 100 + 0 for the survivor and + 1 for the killed
mutant, survivor first) and the pair list (OUT_PAIRS).

  SAMPLES=within_samples.jsonl  LABELS=within_labels.jsonl  WORKERS=6
  MUTANTS=mutants_judged.jsonl  OUT_ROWS=mutant_rows.jsonl  OUT_PAIRS=mutant_pairs.json
"""
import importlib.util
import json
import os
import re
import resource
import sys
import tempfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mbpp_rust import load_typeable  # noqa: E402

_spec = importlib.util.spec_from_file_location('judge_mod', os.path.join(HERE, '1_rust_correct.py'))
judge_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judge_mod)

SAMPLES = os.environ.get('SAMPLES', 'within_samples.jsonl')
LABELS = os.environ.get('LABELS', 'within_labels.jsonl')
MUTANTS = os.environ.get('MUTANTS', 'mutants_judged.jsonl')
OUT_ROWS = os.environ.get('OUT_ROWS', 'mutant_rows.jsonl')
OUT_PAIRS = os.environ.get('OUT_PAIRS', 'mutant_pairs.json')
WORKERS = int(os.environ.get('WORKERS', 6))
SEED = 20261005
TOKEN_CAP, MIN_PAIRS, MAX_PAIRS = 140, 70, 150
MEM_LIMIT = 8 << 30
MODEL_NAME = 'google/gemma-2-2b-it'

FAMILIES = {
    'comparison': ['<', '<=', '>', '>=', '==', '!='],
    'arithmetic': ['+', '-', '*', '/', '%'],
    'compound':   ['+=', '-=', '*=', '/=', '%='],
    'logical':    ['&&', '||'],
    'boolean':    ['true', 'false'],
    'integer':    ['0', '1', '2'],
    'range':      ['..', '..='],
}
FAMILY_OF = {t: f for f, ts in FAMILIES.items() for t in ts}

PUNCT = sorted(['..=', '...', '<<=', '>>=', '::', '->', '=>', '==', '!=', '<=', '>=',
                '&&', '||', '+=', '-=', '*=', '/=', '%=', '^=', '&=', '|=', '<<', '>>',
                '..'], key=len, reverse=True)
RAW_STR = re.compile(r'b?r(#*)"(?:.|\n)*?"\1')
CHAR = re.compile(r"b?'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^\\'\n])'")
LIFETIME = re.compile(r"'[A-Za-z_]\w*")
IDENT = re.compile(r'[A-Za-z_]\w*')
NUMBER = re.compile(r'\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?(?:[iu](?:8|16|32|64|128|size)|f32|f64)?')


def lex(code):
    """(kind, text, start) for every token of a Rust source."""
    out, i, n = [], 0, len(code)
    while i < n:
        c = code[i]
        if c.isspace():
            i += 1
            continue
        if code.startswith('//', i):
            j = code.find('\n', i)
            j = n if j == -1 else j
            out.append(('comment', code[i:j], i))
            i = j
            continue
        if code.startswith('/*', i):
            j = code.find('*/', i + 2)
            j = n if j == -1 else j + 2
            out.append(('comment', code[i:j], i))
            i = j
            continue
        m = RAW_STR.match(code, i) or CHAR.match(code, i)
        if m:
            out.append(('str', m.group(0), i))
            i = m.end()
            continue
        if c == '"' or code.startswith('b"', i):
            j = i + (2 if c == 'b' else 1)
            while j < n and code[j] != '"':
                j += 2 if code[j] == '\\' else 1
            j = min(j + 1, n)
            out.append(('str', code[i:j], i))
            i = j
            continue
        m = LIFETIME.match(code, i) if c == "'" else None
        if m:
            out.append(('life', m.group(0), i))
            i = m.end()
            continue
        m = IDENT.match(code, i)
        if m:
            out.append(('ident', m.group(0), i))
            i = m.end()
            continue
        m = NUMBER.match(code, i)
        if m:
            t = m.group(0)
            out.append(('int' if t.isdigit() else 'num', t, i))
            i = m.end()
            continue
        p = next((p for p in PUNCT if code.startswith(p, i)), c)
        out.append(('punct', p, i))
        i += len(p)
    return out


def sites(code):
    """(start, old, family) for every token that may be replaced."""
    toks = lex(code)
    out, depth, generic = [], 0, 0
    for k, (kind, t, start) in enumerate(toks):
        prev = toks[k - 1] if k else ('', '', -1)
        if kind == 'punct':
            if t == '{':
                depth += 1
                continue
            if t == '}':
                depth -= 1
                continue
            if t == '<' and (prev[1] == '::' or (prev[0] == 'ident' and prev[1][:1].isupper())):
                generic += 1                      # Vec<, HashMap<, ::<
                continue
            if t == '>' and generic:
                generic -= 1
                continue
            if t == '>>' and generic:
                generic = max(0, generic - 2)
                continue
        if depth < 1:
            continue
        if kind == 'punct' and t in FAMILY_OF and FAMILY_OF[t] not in ('boolean', 'integer'):
            out.append((start, t, FAMILY_OF[t]))
        elif kind == 'ident' and t in FAMILIES['boolean']:
            out.append((start, t, 'boolean'))
        elif kind == 'int' and t in FAMILIES['integer'] and prev[1] != '.':
            out.append((start, t, 'integer'))       # not a tuple field, x.0
    return out


def mutants(code):
    for start, old, fam in sites(code):
        for new in FAMILIES[fam]:
            if new != old:
                yield {'pos': start, 'line': code.count('\n', 0, start), 'from': old,
                       'to': new, 'family': fam,
                       'code': code[:start] + new + code[start + len(old):]}


_typeable = None


def _init():
    global _typeable
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT, MEM_LIMIT))
    _typeable, _ = load_typeable()


def _judge(args):
    tid, code = args
    try:
        with tempfile.TemporaryDirectory(prefix='mjudge_') as td:
            return judge_mod.judge(code, _typeable[tid], Path(td))
    except Exception as e:                        # e.g. MemoryError reading output
        return {'compiles': None, 'passes': None, 'n_pass': 0, 'stage': 'judge',
                'error': f'{type(e).__name__}'[:160]}


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    if not Path(judge_mod.RUSTC).exists():
        sys.exit(f"rustc not found at {judge_mod.RUSTC}")
    samples = {(int(r['task_id']), int(r['sample'])): r for r in read_jsonl(SAMPLES)}
    labels = {(int(r['task_id']), int(r['sample'])): r for r in read_jsonl(LABELS)}
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ntok = lambda text: len(tok(text)['input_ids'])

    queue = {}                                    # task -> eligible passing samples, in order
    for key in sorted(samples):
        s, lab = samples[key], labels[key]
        if s['truncated'] or not lab['passes']:
            continue
        n = ntok(s['text'] + ' ' + s['generated_code'])
        if n <= TOKEN_CAP:
            queue.setdefault(key[0], []).append((key[1], n))
    print(f"{len(queue)} problems with an eligible passing sample "
          f"({sum(len(v) for v in queue.values())} samples)")

    chosen, judged_all, rnd = {}, [], 0
    with Pool(WORKERS, initializer=_init) as pool:
        while True:
            todo = [(t, q.pop(0)) for t, q in queue.items() if t not in chosen and q]
            if not todo:
                break
            rnd += 1
            batch = []
            for tid, (smp, n) in todo:
                s = samples[(tid, smp)]
                for m in mutants(s['generated_code']):
                    if ntok(s['text'] + ' ' + m['code']) == n:
                        batch.append({'task_id': tid, 'sample': smp, 'tokens': n, **m})
            verdicts = pool.map(_judge, [(b['task_id'], b['code']) for b in batch], chunksize=4)
            by_src = {}
            for b, v in zip(batch, verdicts):
                b.update(v)
                judged_all.append(b)
                by_src.setdefault((b['task_id'], b['sample']), []).append(b)
            for tid, (smp, n) in todo:
                ms = by_src.get((tid, smp), [])
                surv = [m for m in ms if m['compiles'] and m['passes']]
                kill = [m for m in ms if m['compiles'] and m['passes'] is False]
                if surv and kill:
                    chosen[tid] = (smp, surv, kill)
            print(f"  round {rnd}: {len(todo)} sources, {len(batch)} mutants judged, "
                  f"{len(chosen)} problems with a source so far", flush=True)

    with open(MUTANTS, 'w', encoding='utf-8') as f:
        for b in judged_all:
            f.write(json.dumps(b, ensure_ascii=False) + '\n')

    pairs = []
    for tid in sorted(chosen):
        smp, surv, kill = chosen[tid]
        rng = np.random.default_rng(SEED + tid)
        sv = surv[rng.integers(len(surv))]
        same = [m for m in kill if m['family'] == sv['family']]
        pool_k = same or kill
        kl = pool_k[rng.integers(len(pool_k))]
        pairs.append({'problem': tid, 'sample': smp, 'survivor': sv, 'killed': kl,
                      'same_family': bool(same)})
    n_found = len(pairs)
    if n_found > MAX_PAIRS:
        keep = np.random.default_rng(SEED).choice(n_found, MAX_PAIRS, replace=False)
        pairs = [pairs[i] for i in sorted(keep)]

    comp = [b for b in judged_all if b['compiles']]
    print(f"\n{len(judged_all)} mutants judged in {rnd} rounds: {len(comp)} compile, "
          f"{sum(1 for b in comp if b['passes'])} survive, "
          f"{sum(1 for b in comp if b['passes'] is False)} killed, "
          f"{sum(1 for b in judged_all if b['compiles'] is None)} judge failures")
    print("  where killed mutants stop: " + ", ".join(
        f"{k} {v}" for k, v in Counter((b['error'] or '')[:14] for b in comp
                                       if b['passes'] is False).most_common(4)))
    print(f"  problems with a usable source: {n_found}; pairs kept: {len(pairs)}"
          + (f" (capped at {MAX_PAIRS})" if n_found > MAX_PAIRS else ""))
    if pairs:
        print(f"  same family in the pair: {sum(p['same_family'] for p in pairs)} of {len(pairs)}")
        print("  survivor families: " + ", ".join(
            f"{k} {v}" for k, v in Counter(p['survivor']['family'] for p in pairs).most_common()))
        print("  killed families:   " + ", ".join(
            f"{k} {v}" for k, v in Counter(p['killed']['family'] for p in pairs).most_common()))
        dch = np.array([len(p['survivor']['code']) - len(p['killed']['code']) for p in pairs])
        print(f"  chars survivor - killed: mean {dch.mean():+.2f}, |d| max {np.abs(dch).max()}")
        tp = np.array([p['survivor']['tokens'] for p in pairs])
        print(f"  tokens per member: median {np.median(tp):.0f}, range {tp.min()}..{tp.max()}")

    with open(OUT_PAIRS, 'w', encoding='utf-8') as f:
        json.dump([{'problem': p['problem'], 'source_sample': p['sample'],
                    'pass_id': p['problem'] * 100, 'fail_id': p['problem'] * 100 + 1,
                    'pass_tokens': p['survivor']['tokens'], 'fail_tokens': p['killed']['tokens'],
                    'survivor_edit': {k: p['survivor'][k] for k in ('line', 'from', 'to', 'family')},
                    'killed_edit': {k: p['killed'][k] for k in ('line', 'from', 'to', 'family')},
                    'killed_error': p['killed']['error'], 'same_family': p['same_family']}
                   for p in pairs], f, indent=1)
    with open(OUT_ROWS, 'w', encoding='utf-8') as f:
        for p in pairs:
            prose = samples[(p['problem'], p['sample'])]['text']
            for m, lab, member in ((p['survivor'], 1, 0), (p['killed'], 0, 1)):
                f.write(json.dumps({
                    'task_id': p['problem'] * 100 + member, 'text': prose + ' ' + m['code'],
                    'prose': prose, 'generated_code': m['code'], 'label': lab,
                    'problem': p['problem'], 'sample': p['sample'], 'tokens': m['tokens'],
                    'edit': {k: m[k] for k in ('pos', 'line', 'from', 'to', 'family')},
                }, ensure_ascii=False) + '\n')
    print(f"wrote {MUTANTS}, {OUT_PAIRS} and {OUT_ROWS} ({2 * len(pairs)} rows)")

    if len(pairs) < MIN_PAIRS:
        print(f"GATE NOT MET: {len(pairs)} < {MIN_PAIRS} pairs")
        sys.exit(2)
    print(f"GATE MET: {len(pairs)} >= {MIN_PAIRS} pairs")


if __name__ == '__main__':
    main()
