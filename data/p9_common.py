"""Phase 9 shared helpers: rustc error families, the target function's extent, sample pools.

  family(code, level)     the error family of a diagnostic code
  extent(code, name)      character extents of the target function: signature and body
  where(pos, ext)         'sig' | 'body' | 'other' for a character offset in the generated code
  ws_key(code)            the study's duplicate key: the code with all whitespace removed
  lex(code)               Phase 4's Rust lexer (m1_make_mutants.lex), unchanged
"""
import gzip
import importlib.util
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'results', 'p9_state')

# Borrow-check families: rustc reports these only for bodies that type-check.
MUTABILITY = {'E0384', 'E0596', 'E0594'}
MOVE = {'E0382', 'E0505', 'E0507', 'E0508', 'E0509'}
BORROW = {'E0499', 'E0502', 'E0503', 'E0506', 'E0501'}
LIFETIME = {'E0106', 'E0597', 'E0515', 'E0716', 'E0373', 'E0621', 'E0700', 'E0521', 'E0712', 'E0713',
            'E0495', 'E0759', 'E0311', 'E0310', 'E0309'}
RESOLVE = {'E0412', 'E0425', 'E0433', 'E0432', 'E0423', 'E0424', 'E0428', 'E0405', 'E0407', 'E0408',
           'E0416', 'E0422', 'E0426', 'E0430', 'E0431', 'E0434', 'E0435', 'E0437', 'E0438', 'E0451',
           'E0603', 'E0624', 'E0658', 'E0404', 'E0574', 'E0576', 'E0583', 'E0601'}
PATTERN = {'E0004', 'E0005', 'E0023', 'E0026', 'E0027', 'E0029', 'E0164', 'E0532'}
BORROWCK = MUTABILITY | MOVE | BORROW | LIFETIME - {'E0106'}
FAMILIES = ('syntax', 'resolve', 'type', 'pattern', 'lifetime', 'mutability', 'move', 'borrow',
            'deny_lint', 'other')


def family(code, level='error'):
    if code is None:
        return 'syntax'
    if not code.startswith('E'):
        return 'deny_lint' if level == 'error' else 'lint'
    for name, s in (('mutability', MUTABILITY), ('move', MOVE), ('borrow', BORROW),
                    ('lifetime', LIFETIME), ('resolve', RESOLVE), ('pattern', PATTERN)):
        if code in s:
            return name
    return 'type'


def ws_key(code):
    return re.sub(r'\s+', '', code)


_M1 = None


def m1():
    """m1_make_mutants, loaded once (it imports transformers)."""
    global _M1
    if _M1 is None:
        spec = importlib.util.spec_from_file_location('m1_make_mutants', os.path.join(HERE, 'm1_make_mutants.py'))
        _M1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_M1)
    return _M1


def lex(code):
    return m1().lex(code)


def extent(code, name):
    """{'fn': start of `fn`, 'open': the body's `{`, 'close': its `}` + 1} for the first top-level
    `fn <name>`; None if absent or unbalanced."""
    toks = lex(code)
    depth = 0
    for k, (kind, t, start) in enumerate(toks):
        if kind == 'punct' and t == '{':
            depth += 1
        elif kind == 'punct' and t == '}':
            depth -= 1
        elif depth == 0 and kind == 'ident' and t == 'fn' and k + 1 < len(toks) and toks[k + 1][1] == name:
            d, open_ = 0, None
            for kind2, t2, s2 in toks[k + 2:]:
                if kind2 != 'punct':
                    continue
                if t2 == '{':
                    if d == 0 and open_ is None:
                        open_ = s2
                    d += 1
                elif t2 == '}':
                    d -= 1
                    if d == 0 and open_ is not None:
                        return {'fn': start, 'open': open_, 'close': s2 + 1}
                elif t2 == ';' and d == 0 and open_ is None:
                    return None
            return None
    return None


def where(pos, ext):
    if ext is None:
        return 'other'
    if ext['fn'] <= pos < ext['open']:
        return 'sig'
    if ext['open'] <= pos < ext['close']:
        return 'body'
    return 'other'


def tokenizer():
    """Gemma's tokenizer from the local snapshot (as p8_common.tokenizer)."""
    import glob
    from transformers import AutoTokenizer
    snaps = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/*/'))
    if len(snaps) != 1:
        raise SystemExit(f'expected one tokenizer snapshot, found {snaps}')
    return AutoTokenizer.from_pretrained(snaps[0])


def read_jsonl(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def truthy(v):
    return v in (True, 'True', 'true', 1)
