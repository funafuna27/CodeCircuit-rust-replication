"""Phase 8 (graph-guided interventions at Gemma's own errors): shared constants and helpers.
PREREGISTRATION_INTERVENTION.md. Every Phase 8 script imports this module; nothing here
reads a label except load_sites, whose rows carry the verdicts the judge step needs."""
import glob
import hashlib
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# ---------------------------------------------------------------- randomness
ROOT = 20261015
STREAMS = {0: 'sign-flip tests', 1: 'bootstrap intervals (descriptive)', 2: 'RND orderings (site, draw)',
           3: 'pilot site draw (pool A)', 4: 'size simulation', 5: 'power simulation',
           6: 'tier-1 site per bug', 8: 'within-problem site shuffle (task)', 9: 'problem order',
           10: 'existence: random single features (site)', 11: 'RND-k draws (site, draw)'}
NEVER = {0, 77, 20261001, 20261005, 20261008, 20261012, 20261014}
assert ROOT not in NEVER


def rng(s, *k):
    if s not in STREAMS:
        raise ValueError(f"stream {s} is not registered")
    return np.random.default_rng([ROOT, s, *k])


# ---------------------------------------------------------------- paths and fixed parameters
OUT = os.environ.get('P8_OUT', os.path.join(HERE, 'results', 'intervention'))
WORK = os.environ.get('P8_WORK', '/home/user1/p8_work')          # graphs live here only while a site runs
OWNFIX = os.path.join(HERE, 'results', 'ownfix', 'data')
MODEL_NAME, TRANSCODER_NAME = 'google/gemma-2-2b-it', 'gemma'
GRAPH_KW = dict(max_n_logits=10, desired_logit_prob=0.95, batch_size=64, max_feature_nodes=8192, offload='cpu')
K_MAX = 1024                       # flip-rule cap, every ordered arm
SCAN_BATCH = 64                    # screening batch; a flip is confirmed by batch-1 replay
SOFTCAP = 30.0                     # Gemma-2 final logit soft-cap
TOP_C = 64                         # decision logits: bf16 top-64 union E(wrong)
GEN_LINE_CAP = 48                  # line completion cap (tokens)
GEN_EXTRA = 64                     # program generation cap beyond S's tokens_to_end
HARD_STOP_GPU_H = 28.0             # whole Phase 8 (P0 + P1 + main)
ALPHA = 0.03                       # each step of S1 -> S2 -> S3 (may only fall, by the §5 rule)
MID_LAYERS = range(8, 18)          # G-mid
ARMS_ORDERED = ('G', 'GxA', 'RND1', 'RND2', 'RND3', 'RND4', 'RND5', 'G_LP', 'G1', 'G_mid', 'DE', 'ORC', 'CONTRAST')
ARMS_FIXED = ('BAN', 'NULL')
PRIMARY_ORDER = (('S1', 'G', 'RND1'), ('S2', 'G', 'BAN'), ('S3', 'G', 'GxA'))
NEUTRALIZING = ('x-x', 'x%x', 'x/x', '*=1', '/=1', '%=1', '%1', '+0', '-0', '*1', '/1')


# ---------------------------------------------------------------- io
def path(name):
    return name if os.path.isabs(name) else os.path.join(HERE, name)


def read_jsonl(name):
    import gzip
    p = path(name)
    op = gzip.open if p.endswith('.gz') else open
    with op(p, 'rt', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def _jsonable(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


def write_json(p, obj):
    """Atomic: write .partial, fsync, rename."""
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    tmp = p + '.partial'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, default=_jsonable)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def tokenizer():
    """Gemma's tokenizer from the local snapshot (as p7_common)."""
    from transformers import AutoTokenizer
    snaps = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/*/'))
    if len(snaps) != 1:
        raise SystemExit(f"expected one tokenizer snapshot, found {snaps}")
    return AutoTokenizer.from_pretrained(snaps[0])


# ---------------------------------------------------------------- Rust lexing (m1_make_mutants.lex, unchanged)
_m1 = None


def _load_m1():
    global _m1
    if _m1 is None:
        spec = importlib.util.spec_from_file_location('m1_make_mutants', os.path.join(HERE, 'm1_make_mutants.py'))
        _m1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_m1)
    return _m1


def lex(code):
    return _load_m1().lex(code)


# rustc's Pattern_White_Space. m1.lex skips anything str.isspace() accepts, which also covers
# characters rustc rejects (U+00A0, U+2009, U+3000, ...); a program with one of those between
# tokens does not compile, so it must not share a key with the program without it.
RUST_WS = set('\t\n\x0b\x0c\r \x85‎‏  ')


def lexeme_key(code):
    """The lexer stream without positions and comments: two programs with the same key differ
    only in whitespace outside literals (and comments). Whitespace rustc does not accept adds a
    final ('nonrust_ws', '') token, so such a program never matches a compiling one."""
    toks = lex(code)
    key = tuple((k, t) for k, t, _ in toks if k != 'comment')
    end = 0
    for _, t, s in toks + [('', '', len(code))]:
        if any(c not in RUST_WS for c in code[end:s]):
            return key + (('nonrust_ws', ''),)
        end = max(end, s + len(t))
    return key


# ---------------------------------------------------------------- sites
def load_sites(name='results/intervention/sites.jsonl'):
    return read_jsonl(os.path.join(OUT, os.path.basename(name)) if name.startswith('results/intervention') else name)
