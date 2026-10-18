"""Phase 8, step 3: blind scoring of every arm's t* (final_design.md §5; SPEC.md 'judged outcomes').

CPU only, after the GPU run and never beside it: refuses to start, and stops mid-way (its cache
kept), while another process holds the GPU lock $P8_WORK/gpu.lock (p8_2_run.Engine and so
p8_0_checks flock it; tested non-blocking and released at once), or a p8_2_run / run_p8.sh /
p8_0_checks process is alive that is not a CPU selftest ('--test' argument or MODE=selftest in its
environment); its own parents excepted.

  input    SITES (the site table: prose, code, ids, h, judged) and RUN_DIR/<stage>/<site>.json for
           stages A, B, C (SPEC 'run outputs'). Labels enter only through the site table's
           'judged' map and the blind judge.
  rule     one rule for every arm (design §5):
           1 lexeme      S with ids[h] := t*, decoded. If its lexer stream (p8_common.lexeme_key)
                         equals a judged mutant's at the site: that mutant's stored (finite-aware)
                         label.
           2 completion  otherwise S's text from token h to the end of its edited line is replaced
                         by t*'s piece + the recorded completion (the text generated after t*), cut
                         at the first newline in them; S's text from the end of the edited line on
                         is kept; the program is judged blind. No newline (the cap was hit) or no
                         completion recorded: 'unscorable'. A completion that p8_2_run ended at
                         '<eos>'/'<end_of_turn>' before any newline is also 'unscorable' (design §5
                         does not define the line then), counted apart from the cap case: anomaly
                         'completion_eos' vs 'no_newline', and the summary's 'unscorable_by_reason'.
           A flipped t* whose splice has S's own lexer stream (t* in E) counts as 'no_flip'.
           gen:<arm>     S's prefix (ids[1:h]) + the generated text (from token h on), judged blind
                         as a whole program; stopped at the cap: 'unscorable'. A trailing '<eos>' or
                         '<end_of_turn>' string is dropped.
           RNDk:<draw>   lexeme rule only (no completion is recorded): 'unscorable' otherwise.
           exist:G, exist:RND  (design §5 existence rate; stage B 'existence': 'flips' = the
                         replayed single-ablation flips among ORC's set and G's top features,
                         'random_flips' = the matched random features). Lexeme rule only (no
                         completion is recorded for single ablations): each distinct lexeme of the
                         flip tokens, skipping S's own (t in E) and BAN's lexeme, gets its judged
                         label or 'unscorable'; the row's label is the best of them (verified >
                         lucky > unverifiable > fail > no_compile > unscorable), 'no_flip' if none
                         is left; so Y = 1 iff some single ablation yields a verified token other
                         than BAN's lexeme. neutralizing: every best-label lexeme is neutralizing.
           excluded      a non-null 'excluded' in the stage A or B file of a site makes every row of
                         that site 'excluded' (site-level, design §2 (a)-(d): all label-free). A
                         stage-C 'excluded' (stage C, which uses labels, failed twice) does not
                         exclude the site: only its stage-C rows (gen:* and, unless another stage
                         file holds it, CONTRAST) are 'excluded'; listed as 'stage_C_excluded'.
                         A stage-C file with excluded=None and a 'failed' field (p8_2_run's
                         failed_record, accepted by a person) is treated the same way.
           missing       a primary arm (G, BAN, RND1, GxA) expected but absent from a stage file that
                         exists: nothing is judged or written, exit 5. Other missing arms: a warning
                         and the summary's 'missing_arms'.
           undefined     an arm whose stage-file value is null and that no other stage file of the
                         site defines (stage C's CONTRAST when the site has no usable repair) gets
                         no row; it is listed in the summary ('undefined_arms').
           A completion whose program has S's lexer stream (the line rebuilt the bug) is judged
           like any other program; such rows are counted ('completion_is_S').
  judge    blind: the work list holds (key, task, code) only, key = sha256 of [task, lexer
           stream]; one judging per key. Work order = ascending key: a fixed pseudo-random
           permutation that interleaves the arms and needs no rng stream (p8_common registers no
           stream for it, and a seeded shuffle would add one). The code judged for a
           key is the smallest of its texts, so it does not depend on which arm produced it.
           p8_judge.label (judge3, then verify_extra for a stable pass; the function the site
           table's re-judge uses) gives verified / lucky / unverifiable / fail / no_compile /
           judge_error; a judge_error is judged once more, and if it stays it is scored 'fail'
           (counted in the summary). Pool(initializer=p8_judge.init_worker); every worker's
           RLIMIT_AS is checked to be <= 8 GiB.
  cache    CACHE: append-only jsonl, one fsynced line per key, tagged with a hash of the judge's
           inputs (p8_judge.py, 1_rust_correct.py, mbpp_rust.py, ownfix_verify.json, rustc
           version, COMPILE_TIMEOUT / RUN_TIMEOUT, the typeable-signature table); entries with
           another tag are ignored. A partial last line left by a crash is
           cut off. Restart = rerun; nothing is judged twice.
  Y        1 iff label == 'verified'.
  flags    lexeme_is_ban: the arm's single-token splice has BAN's lexer stream (gen rows: the whole
           program has it). neutralizing: the scored program has more occurrences than S of a
           pre-registered pattern (x-x, x%x, x/x, *=1, /=1, %=1, %1, +0, -0, *1, /1), counted by
           p8_1_sites.neutral_counts, the rule the site table's flags come from; False for rows
           without a scored program.
  writes   JUDGED: one row per (site, arm), SPEC schema plus 'stage' and 'key' (the cache key, or
           null). SUMMARY: counts, cache use, audits (blind vs stored label where a judged program
           equals a judged mutant; our neutralizing flag vs the site table's), anomalies, missing
           and undefined arms, the existence token counts per site, hashes.
           Nothing final is written if any judging raised (exit 3; rerun).

Usage (cwd data/):  python p8_3_judge.py            (exit 0 done, 3 judgings raised, 4 GPU job seen,
                                                     5 a primary arm missing)
                    python p8_3_judge.py --test     (or MODE=selftest)
Env: P8_OUT=data/results/intervention  SITES=sites.jsonl  RUN_DIR=$P8_OUT/run
     CACHE=$P8_OUT/blind_judge_cache.jsonl  JUDGED=$P8_OUT/judged.jsonl  SUMMARY=$P8_OUT/judged.json
     WORKERS=6 (judge processes, 8 GiB RLIMIT_AS each); relative names are under P8_OUT.
     TEST_DIR=/home/user1/p8design/impl/D/selftest (selftest: SPEC test D on synthetic FLIPs built
     from Phase 6 judged mutants; writes nothing under data/).
data/1_code_correct.py is never opened, imported or run.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''           # this step never touches the GPU
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from multiprocessing import Pool  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p8_common as C  # noqa: E402
import p8_judge as J  # noqa: E402
from p8_1_sites import neutral_counts  # noqa: E402  (the site table's neutralizing rule)


def _in_out(name, default):
    v = os.environ.get(name, default)
    return v if os.path.isabs(v) else os.path.join(C.OUT, v)


SITES = _in_out('SITES', 'sites.jsonl')
RUN_DIR = _in_out('RUN_DIR', 'run')
CACHE = _in_out('CACHE', 'blind_judge_cache.jsonl')   # not judge_cache.jsonl: p8_1_sites owns that
JUDGED = _in_out('JUDGED', 'judged.jsonl')
SUMMARY = _in_out('SUMMARY', 'judged.json')
WORKERS = int(os.environ.get('WORKERS', 6))
MEM = 8 << 30
GPU_MARKERS = ('p8_2_run', 'run_p8.sh', 'p8_0_checks')
GPU_LOCK = os.path.join(C.WORK, 'gpu.lock')           # p8_2_run.gpu_lock() flocks this file
SITE_STAGES = ('A', 'B')       # a stage-A/B 'excluded' excludes the site (design §2 (a)-(d), label-free)
PRIMARY = ('G', 'BAN', 'RND1', 'GxA')                # missing from a present stage file: exit 5
C_ROWS = ('CONTRAST',) + tuple('gen:' + g for g in ('G', 'G_every', 'BAN', 'NULL'))  # belief stage-C rows
STORED = ('verified', 'lucky', 'fail', 'unverifiable', 'no_compile', 'judge_error')
SPECIAL_TAILS = ('<eos>', '<end_of_turn>')
FLIP_A = ('BAN', 'NULL', 'GxA', 'DE')
FLIP_B = ('G', 'G_LP', 'G1', 'G_mid', 'RND1', 'RND2', 'RND3', 'RND4', 'RND5', 'ORC')
GEN = ('G', 'G_every', 'BAN', 'NULL')
EXPECTED = {('A', 'belief'): FLIP_A, ('A', 'slip'): ('NULL',), ('B', 'belief'): FLIP_B}
ARM_ORDER = ('NULL', 'BAN', 'GxA', 'DE', 'G', 'G_LP', 'G1', 'G_mid', 'RND1', 'RND2', 'RND3', 'RND4',
             'RND5', 'ORC', 'CONTRAST')
EXIST = (('exist:G', 'flips'), ('exist:RND', 'random_flips'))
EXIST_RANK = ('verified', 'lucky', 'unverifiable', 'fail', 'no_compile', 'unscorable')


def arm_rank(arm):
    if arm in ARM_ORDER:
        return (0, ARM_ORDER.index(arm), 0, '')
    if arm.startswith('RNDk:'):
        return (1, 0, int(arm[5:]), '')
    if arm in dict(EXIST):
        return (2, [a for a, _ in EXIST].index(arm), 0, '')
    if arm.startswith('gen:') and arm[4:] in GEN:
        return (3, GEN.index(arm[4:]), 0, '')
    return (4, 0, 0, arm)


def out_label(raw):
    """judged.jsonl label of a judge label: a persistent judge_error scores as a failure."""
    return 'fail' if raw == 'judge_error' else raw


# ---------------------------------------------------------------- no judging beside a GPU job
def _ancestors():
    out, pid = set(), os.getpid()
    while pid > 1 and pid not in out:
        out.add(pid)
        try:
            with open(f'/proc/{pid}/status') as f:
                pid = int(next(l for l in f if l.startswith('PPid:')).split()[1])
        except (OSError, StopIteration, ValueError):
            break
    return out


def gpu_lock_held(path=None):
    """True iff another process holds the GPU lock that p8_2_run.Engine (and so p8_0_checks) takes
    with flock. Tested with LOCK_EX|LOCK_NB and released at once: judging never blocks a GPU start."""
    path = path or GPU_LOCK
    if not os.path.exists(path):
        return False
    with open(path, 'a') as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fh, fcntl.LOCK_UN)
    return False


def _is_selftest(pid, args):
    if '--test' in args:
        return True
    try:
        with open(f'/proc/{pid}/environ', 'rb') as f:
            return b'MODE=selftest' in f.read().split(b'\0')
    except OSError:
        return False


def gpu_jobs():
    """[(pid, what)] of Phase 8 GPU work: the GPU lock held by another process, or a process whose
    command line names p8_2_run / run_p8.sh / p8_0_checks, unless it is a CPU selftest ('--test'
    argument or MODE=selftest in its environment); this process's own parents excepted."""
    skip, found = _ancestors(), []
    if gpu_lock_held():
        found.append((0, f"{GPU_LOCK} is locked"))
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) in skip:
            continue
        try:
            with open(f'/proc/{d}/cmdline', 'rb') as f:
                args = [a.decode('utf-8', 'replace') for a in f.read().split(b'\0') if a]
        except OSError:
            continue
        cmd = ' '.join(args)
        if any(m in cmd for m in GPU_MARKERS) and not _is_selftest(d, args):
            found.append((int(d), cmd[:160]))
    return found


def refuse_if_gpu():
    jobs = gpu_jobs()
    if jobs:
        print("REFUSED: a Phase 8 GPU job is running; judging waits until it is over: "
              + '; '.join(f"pid {p}: {c}" for p, c in jobs), flush=True)
        sys.exit(4)


# ---------------------------------------------------------------- the judge (owner A's module)
def judge_tag():
    """Hash of everything a verdict depends on besides the program: the judge's source files, the
    verification data, the rustc version, the compile/run timeouts and the typeable-signature table."""
    h = hashlib.sha256()
    for p in (J.__file__, J.JUDGE_FILE, os.path.join(HERE, 'mbpp_rust.py'), J.VERIFY_FILE):
        h.update(f"{os.path.basename(p)} {C.sha256(p)}\n".encode())
    h.update(J.rustc_version().encode())
    h.update(f"\nCOMPILE_TIMEOUT {J._NEW.COMPILE_TIMEOUT} RUN_TIMEOUT {J._NEW.RUN_TIMEOUT} "
             f"OLD {J._OLD.COMPILE_TIMEOUT} {J._OLD.RUN_TIMEOUT}\n".encode())
    h.update(json.dumps(J.typeable(), sort_keys=True, default=repr).encode())
    return h.hexdigest()


def _init():
    J.init_worker()
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if soft == resource.RLIM_INFINITY or soft > MEM:        # init_worker should have set it
        resource.setrlimit(resource.RLIMIT_AS, (MEM, MEM))


def _brief(r):
    return {'label': r['label'], 'extra': r['extra'], 'extra_bad': r['extra_bad'],
            'verdicts': [[v.get('compiles'), v.get('passes'), v.get('n_pass'), v.get('stage'),
                          (v.get('error') or '')[:160]] for v in r['verdicts']]}


def _work(item):
    """item = (key, task, code): nothing else reaches the judge."""
    key, task, code = item
    rl = resource.getrlimit(resource.RLIMIT_AS)[0]
    try:
        r = J.label(task, code)
        tries = 1
        if r['label'] == 'judge_error':                 # the judge itself raised: once more
            r, tries = J.label(task, code), 2
        if r['label'] not in STORED:
            raise ValueError(f"p8_judge.label gave {r['label']!r}")
        return key, dict(_brief(r), tries=tries), None, rl
    except Exception as e:                       # not cached: a rerun judges it again
        return key, None, f"{type(e).__name__}: {e}"[:300], rl


# ---------------------------------------------------------------- neutralizing (syntactic, design §5)
def neutral_new(counts_s, code):
    """The pre-registered patterns code has more often than S (p8_1_sites.neutralizing's rule)."""
    cm = neutral_counts(code)
    return sorted(p for p in cm if cm[p] > counts_s.get(p, 0))


# ---------------------------------------------------------------- programs from t*
def cache_key(task, lk):
    return hashlib.sha256(json.dumps([int(task), [list(x) for x in lk]]).encode('utf-8')).hexdigest()


class Programs:
    """Site texts and the programs an arm's t* stands for (tokenizer only)."""

    def __init__(self, tok):
        self.tok = tok
        self.nvocab = len(tok)

    def dec(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def prepare(self, rows):
        out = {}
        for i, r in enumerate(rows):
            sid = r['site_id']
            for f in ('site_id', 'kind', 'prose', 'code', 'ids', 'h', 'wrong_id', 'judged'):
                if f not in r:
                    raise SystemExit(f"site {sid}: no '{f}' (SPEC.md site schema)")
            if sid in out:
                raise SystemExit(f"site {sid} appears twice")
            ids, h = [int(x) for x in r['ids']], int(r['h'])
            if ids[0] != self.tok.bos_token_id or not 1 <= h < len(ids) or ids[h] != int(r['wrong_id']):
                raise SystemExit(f"site {sid}: ids/h/wrong_id inconsistent")
            text = self.dec(ids[1:])
            if text != r['prose'] + ' ' + r['code']:
                raise SystemExit(f"site {sid}: ids do not decode to prose + ' ' + code")
            pre = self.dec(ids[1:h])
            if not text.startswith(pre) or len(pre) < len(r['prose']) + 1:
                raise SystemExit(f"site {sid}: the prefix ids[1:h] does not decode to a prefix of the code")
            eol = text.find('\n', len(pre))
            for k, v in r['judged'].items():
                if v.get('label') not in STORED:
                    raise SystemExit(f"site {sid}: judged label {v.get('label')!r} not in {STORED}")
            out[sid] = dict(row=r, sid=sid, order=(r.get('order', i), i),
                            task=int(r.get('task', sid.split('|')[0])),
                            kind='slip' if r['kind'] == 'slip' else 'belief',
                            ids=ids, h=h, prose=r['prose'], code=r['code'], text=text, pre=pre,
                            eol=len(text) if eol < 0 else eol, lk=C.lexeme_key(r['code']),
                            ncount=neutral_counts(r['code']), judged=r['judged'])
        return out

    def _strip(self, s, text):
        p = s['prose'] + ' '
        return text[len(p):] if text.startswith(p) else None

    def check_id(self, s, t):
        if not isinstance(t, int) or not 0 <= t < self.nvocab:
            raise SystemExit(f"site {s['sid']}: t* {t!r} is not a vocabulary id")

    def splice(self, s, t):
        """S with ids[h] := t, decoded (the single-token splice), or None if it loses the prose."""
        return self._strip(s, self.dec(s['ids'][1:s['h']] + [t] + s['ids'][s['h'] + 1:]))

    def line(self, s, t, completion):
        """(code, None) or (None, why): t*'s piece + completion up to the first newline in place of
        S's text from token h to the end of the edited line; S's text after it kept."""
        head = self.dec(s['ids'][1:s['h']] + [t]) + completion
        if not head.startswith(s['pre']):
            return None, 'prefix_merge'
        nl = head.find('\n', len(s['pre']))
        if nl < 0:                  # p8_2_run stops at a newline, <eos>/<end_of_turn>, or the cap
            return None, 'completion_eos' if completion.endswith(SPECIAL_TAILS) else 'no_newline'
        return self._strip(s, head[:nl] + s['text'][s['eol']:]), None

    def gen(self, s, text):
        return self._strip(s, s['pre'] + text)


# ---------------------------------------------------------------- run outputs
def load_runs(run_dir, sites):
    runs, counts, ignored = defaultdict(dict), Counter(), []
    if not os.path.isdir(run_dir):
        raise SystemExit(f"no run outputs at {run_dir}")
    for stage in sorted(os.listdir(run_dir)):
        d = os.path.join(run_dir, stage)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.json'):
                continue                                   # .partial and other files
            with open(os.path.join(d, fn), encoding='utf-8') as f:
                obj = json.load(f)
            sid = obj.get('site_id') if isinstance(obj, dict) else None
            if sid is None:
                ignored.append(f"{stage}/{fn}")
                continue
            if fn != sid.replace('|', '_') + '.json':
                raise SystemExit(f"{stage}/{fn} holds site {sid}")
            if sid not in sites:
                raise SystemExit(f"{stage}/{fn}: site {sid} is not in the site table (wrong SITES?)")
            runs[sid][stage] = obj
            counts[stage] += 1
    return dict(runs), dict(counts), ignored


def _is_flip(v):
    return isinstance(v, dict) and 'flipped' in v and 't_star_id' in v


def collect(sites, runs, P):
    """Rows (label final, or pending on a cache key), the programs to judge {key: (task, code)},
    anomalies, missing arms and info {'undefined_arms', 'existence'}. Uses no label except the
    site table's 'judged' map."""
    rows, need, anomalies, missing = [], {}, Counter(), []
    info = {'undefined_arms': [], 'existence': {}, 'stage_C_excluded': [], 'unscorable': defaultdict(list)}
    special = set(P.tok.all_special_ids)

    def add_prog(s, code, lk):
        key = cache_key(s['task'], lk)
        old = need.get(key)
        if old is None or code < old[1]:               # smallest text: arm-free representative
            need[key] = (s['task'], code)
        return key

    for sid in sorted(runs, key=lambda x: sites[x]['order']):
        s, files = sites[sid], runs[sid]
        # site-level exclusion: stage A/B only (design §2 (a)-(d), all label-free); a stage-C
        # 'excluded' (stage C failed twice: label-using work) excludes the stage-C rows only
        excl = {st: files[st]['excluded'] for st in SITE_STAGES if (files.get(st) or {}).get('excluded')}
        fc = files.get('C') or {}
        # p8_2_run writes a failed stage C (accepted by a person) as excluded=None, failed='...':
        # treated like a stage-C 'excluded' (its gen:* and CONTRAST rows 'excluded', site kept)
        c_excl = None if excl else (fc.get('excluded') or (f"C failed: {fc['failed']}" if fc.get('failed') else None))
        if c_excl:
            info['stage_C_excluded'].append([sid, c_excl])

        def excluded_at(stage):
            return bool(excl) or (stage == 'C' and bool(c_excl))
        ban =next((o['BAN'] for o in files.values() if _is_flip(o.get('BAN'))), None)
        ban_lk = None
        if ban and ban['flipped'] and ban['t_star_id'] is not None:
            bcode = P.splice(s, int(ban['t_star_id']))
            ban_lk = None if bcode is None else C.lexeme_key(bcode)
        seen = set()

        def row(arm, stage, label, how, key=None, is_ban=False, neut=(), stored=None, why=None):
            if arm in seen:
                raise SystemExit(f"site {sid}: arm {arm} appears in two stage files")
            seen.add(arm)
            if label == 'unscorable':
                info['unscorable'][why or 'other'].append([sid, arm])
            rows.append(dict(site_id=sid, arm=arm, stage=stage, label=label, how=how, key=key,
                             lexeme_is_ban=bool(is_ban), neutralizing=bool(neut), _neut=list(neut),
                             _stored=stored, _rank=(s['order'], arm_rank(arm))))

        def single(arm, stage, t):
            """(code, lk) of t*'s single-token splice, or None (row written as unscorable)."""
            P.check_id(s, t)
            if t in special:
                anomalies['special_token_t_star'] += 1
            code = P.splice(s, t)
            if code is None:
                anomalies['splice_lost_prose'] += 1
                row(arm, stage, 'unscorable', 'none', why='splice_lost_prose')
                return None
            return code, C.lexeme_key(code)

        def flip_row(arm, stage, rec):
            if excluded_at(stage):
                return row(arm, stage, 'excluded', 'none')
            if rec is None:
                missing.append([sid, stage, arm])
                return None
            if rec.get('arm') not in (None, arm):
                raise SystemExit(f"site {sid}: FLIP under '{arm}' says arm '{rec.get('arm')}'")
            if not rec['flipped']:
                return row(arm, stage, 'no_flip', 'none')
            if rec['t_star_id'] is None:
                raise SystemExit(f"site {sid} {arm}: flipped without a t*")
            t = int(rec['t_star_id'])
            sp = single(arm, stage, t)
            if sp is None:
                return None
            code, lk = sp
            if lk == s['lk']:                              # t* in E: the program is S
                anomalies['t_star_in_E'] += 1
                return row(arm, stage, 'no_flip', 'none')
            is_ban = ban_lk is not None and lk == ban_lk
            ks = json.dumps(lk)
            if ks in s['judged']:
                return row(arm, stage, out_label(s['judged'][ks]['label']), 'lexeme', None, is_ban,
                           neutral_new(s['ncount'], code))
            comp = rec.get('completion')
            if comp is None:
                anomalies['completion_missing'] += 1
                return row(arm, stage, 'unscorable', 'completion', None, is_ban, why='completion_missing')
            pcode, why = P.line(s, t, comp)
            if pcode is None:
                why = why or 'completion_lost_prose'
                anomalies[why] += 1
                return row(arm, stage, 'unscorable', 'completion', None, is_ban, why=why)
            plk = C.lexeme_key(pcode)
            if plk == s['lk']:
                anomalies['completion_is_S'] += 1
            j = s['judged'].get(json.dumps(plk))
            return row(arm, stage, None, 'completion', add_prog(s, pcode, plk), is_ban,
                       neutral_new(s['ncount'], pcode), j['label'] if j else None)

        for stage in sorted(files):
            o = files[stage]
            exp = EXPECTED.get((stage, s['kind']), ())
            if stage == 'C' and c_excl and s['kind'] == 'belief' and not any(
                    'CONTRAST' in files[st] for st in files if st != 'C'):
                exp = ('CONTRAST',)                  # written as 'excluded' (C failed twice)
            for arm in list(dict.fromkeys(list(exp) + sorted((k for k, v in o.items() if _is_flip(v)),
                                                             key=arm_rank))):
                if s['kind'] == 'slip' and arm != 'NULL' and o.get(arm) is None:
                    continue
                flip_row(arm, stage, o.get(arm))
            for d in o.get('RNDk') or []:
                arm = f"RNDk:{int(d['draw'])}"
                if excluded_at(stage):
                    row(arm, stage, 'excluded', 'none')
                    continue
                t = d.get('t_star_id')
                if not d.get('changed') or t is None:
                    row(arm, stage, 'no_flip', 'none')
                    continue
                sp = single(arm, stage, int(t))
                if sp is None:
                    continue
                code, lk = sp
                is_ban = ban_lk is not None and lk == ban_lk
                j = s['judged'].get(json.dumps(lk))
                if lk == s['lk']:
                    row(arm, stage, 'no_flip', 'none')
                elif j:
                    row(arm, stage, out_label(j['label']), 'lexeme', None, is_ban, neutral_new(s['ncount'], code))
                else:
                    row(arm, stage, 'unscorable', 'none', None, is_ban, why='lexeme_not_judged')
            gens = o.get('gen') or {}
            gexp = GEN if stage == 'C' and s['kind'] == 'belief' else ()
            for g in list(dict.fromkeys(list(gexp) + sorted(gens, key=lambda a: arm_rank('gen:' + a)))):
                arm, rec = 'gen:' + g, gens.get(g)
                if excluded_at(stage):
                    row(arm, stage, 'excluded', 'none')
                    continue
                if rec is None:
                    missing.append([sid, stage, arm])
                    continue
                if rec.get('stopped') == 'cap':
                    row(arm, stage, 'unscorable', 'completion', why='gen_cap')
                    continue
                text = rec['text']
                for tail in SPECIAL_TAILS:
                    if text.endswith(tail):
                        text = text[:-len(tail)]
                        anomalies['gen_special_tail_dropped'] += 1
                code_pre = s['pre'][len(s['prose']) + 1:]
                if len(code_pre) >= 10 and text.startswith(code_pre[:40]):
                    anomalies['gen_text_repeats_prefix'] += 1      # GEN text must start at token h
                code = P.gen(s, text)
                plk = C.lexeme_key(code)
                if plk == s['lk']:
                    anomalies['gen_is_S'] += 1
                j = s['judged'].get(json.dumps(plk))
                row(arm, stage, None, 'completion', add_prog(s, code, plk),
                    ban_lk is not None and plk == ban_lk, neutral_new(s['ncount'], code), j['label'] if j else None)
        ex = (files.get('B') or {}).get('existence')
        if isinstance(ex, dict):                      # existence rate (design §5): lexeme rule only
            for arm, field in EXIST:
                if excl:
                    row(arm, 'B', 'excluded', 'none')
                    continue
                best, seen_lk, counts = {}, set(), Counter()   # label -> [neutralizing patterns per lexeme]
                for f in ex.get(field) or []:
                    if f[3] is None:
                        continue
                    t = int(f[3])
                    P.check_id(s, t)
                    code = P.splice(s, t)
                    if code is None:
                        anomalies['splice_lost_prose'] += 1
                        counts['unscorable'] += 1
                        best.setdefault('unscorable', []).append([])
                        continue
                    lk = C.lexeme_key(code)
                    if lk == s['lk'] or lk == ban_lk or lk in seen_lk:
                        counts['in_E' if lk == s['lk'] else 'ban_lexeme' if lk == ban_lk else 'repeat'] += 1
                        continue
                    seen_lk.add(lk)
                    j = s['judged'].get(json.dumps(lk))
                    lab = out_label(j['label']) if j else 'unscorable'
                    counts[lab] += 1
                    best.setdefault(lab, []).append(neutral_new(s['ncount'], code) if j else [])
                info['existence'].setdefault(sid, {})[arm] = dict(counts)
                if not best:
                    row(arm, 'B', 'no_flip', 'none')
                    continue
                lab = min(best, key=EXIST_RANK.index)
                neut = sorted({p for ps in best[lab] for p in ps}) if all(best[lab]) else []
                row(arm, 'B', lab, 'none' if lab == 'unscorable' else 'lexeme', None, False, neut,
                    why='lexeme_not_judged')
        for stage in sorted(files) if not excl else ():
            if excluded_at(stage):
                continue
            for arm, v in files[stage].items():
                if v is None and arm in ARM_ORDER and arm not in seen and (s['kind'] != 'slip' or arm == 'NULL'):
                    info['undefined_arms'].append([sid, stage, arm, files[stage].get(arm.lower() + '_note')])
                    seen.add(arm)
    rows.sort(key=lambda r: r['_rank'])
    return rows, need, anomalies, missing, info


# ---------------------------------------------------------------- cache
def load_cache(path, tag):
    good, stale, dup = {}, 0, 0
    if not os.path.exists(path):
        return good, stale, dup
    with open(path, 'rb') as f:
        data = f.read()
    end = data.rfind(b'\n') + 1
    if end < len(data):
        print(f"cache: cutting a partial last line ({len(data) - end} bytes) left by an interrupted run")
        with open(path, 'r+b') as f:
            f.truncate(end)
            f.flush()
            os.fsync(f.fileno())
    for line in data[:end].decode('utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r['judge'] != tag:
            stale += 1
        elif r['key'] in good:
            dup += 1
        else:
            good[r['key']] = r
    return good, stale, dup


def judge_all(items, cache_f, cache, tag):
    """Judge every (key, task, code) in the given order; append each verdict to the cache."""
    if not items:
        return {}, 0
    code_of = {k: (t, c) for k, t, c in items}
    errors, rl_max, t0, last = {}, 0, time.time(), time.time()
    print(f"judging {len(items)} programs with {WORKERS} workers", flush=True)
    with Pool(WORKERS, initializer=_init) as pool:
        for n, (key, res, err, rl) in enumerate(pool.imap_unordered(_work, items, chunksize=1), 1):
            if rl == resource.RLIM_INFINITY or rl > MEM:
                pool.terminate()
                raise SystemExit(f"a judge worker runs without the 8 GiB RLIMIT_AS (soft limit {rl})")
            rl_max = max(rl_max, rl)
            if err:
                errors[key] = err
            else:
                task, code = code_of[key]
                rec = dict(key=key, task=task, judge=tag, **res, code=code)
                cache_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                cache_f.flush()
                os.fsync(cache_f.fileno())
                cache[key] = rec
            if time.time() - last > 30:
                last = time.time()
                if gpu_jobs():
                    pool.terminate()
                    print("STOPPED: a Phase 8 GPU job started; the cache keeps what was judged; "
                          "rerun when the GPU is idle", flush=True)
                    sys.exit(4)
            if n % 50 == 0 or n == len(items):
                print(f"  {n}/{len(items)} judged, {time.time() - t0:.0f} s", flush=True)
    return errors, rl_max


# ---------------------------------------------------------------- scoring
def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    tmp = p + '.partial'
    with open(tmp, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def score(sites_path, run_dir, cache_path, judged_path, summary_path, P=None):
    refuse_if_gpu()
    P = P or Programs(C.tokenizer())
    srows = C.read_jsonl(sites_path)
    sites = P.prepare(srows)
    runs, run_counts, ignored = load_runs(run_dir, sites)
    rows, need, anomalies, missing, info = collect(sites, runs, P)
    print(f"{len(sites)} sites, {len(runs)} with run outputs {run_counts}; {len(rows)} rows, "
          f"{len(need)} distinct programs for the blind judge", flush=True)
    miss_p = [m for m in missing if m[2] in PRIMARY]
    if miss_p:
        print(f"MISSING: {len(miss_p)} primary arm(s) {PRIMARY} absent from a stage file that exists "
              f"(a broken run output; p8_4 would drop the site silently); nothing judged or written: "
              + '; '.join(' '.join(m) for m in miss_p[:20]), flush=True)
        sys.exit(5)
    if missing:
        print(f"WARNING: {len(missing)} non-primary arm(s) missing from present stage files: "
              + '; '.join(' '.join(m) for m in missing[:20]), flush=True)
    tag = judge_tag()
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    with open(cache_path, 'a', encoding='utf-8') as cache_f:
        try:
            fcntl.flock(cache_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(f"another p8_3_judge run holds {cache_path}")
        cache, stale, dup = load_cache(cache_path, tag)
        todo = sorted(k for k in need if k not in cache)          # ascending key: arm-free order
        items = [(k, need[k][0], need[k][1]) for k in todo]
        errors, rl_max = judge_all(items, cache_f, cache, tag)
    if errors:
        for k, e in sorted(errors.items())[:20]:
            print(f"  JUDGE ERROR {k[:12]} (task {need[k][0]}): {e}")
        print(f"{len(errors)} judgings raised; nothing final written; rerun to retry them")
        sys.exit(3)

    out, audit, label_by_arm, jerr = [], [], defaultdict(Counter), []
    for r in rows:
        if r['label'] is None:
            raw = cache[r['key']]['label']
            if raw == 'judge_error':
                jerr.append([r['site_id'], r['arm'], r['key']])
            r['label'] = out_label(raw)
            if r['_stored'] is not None:
                audit.append((r['site_id'], r['arm'], raw, r['_stored']))
        y = int(r['label'] == 'verified')
        label_by_arm[r['arm']][r['label']] += 1
        out.append({'site_id': r['site_id'], 'arm': r['arm'], 'Y': y, 'label': r['label'], 'how': r['how'],
                    'lexeme_is_ban': r['lexeme_is_ban'], 'neutralizing': r['neutralizing'],
                    'stage': r['stage'], 'key': r['key']})
    write_jsonl(judged_path, out)

    # audit: our neutralizing rule on each judged mutant against the site table's stored flags
    # (the mutant's code is rebuilt as code[:pos] + to + code[pos + len(frm):] and must have its key)
    nf, nf_dis = Counter(), []
    for s in sites.values():
        stored, r = s['row'].get('neutralizing'), s['row']
        if not isinstance(stored, dict) or not isinstance(r.get('pos'), int) or not isinstance(r.get('frm'), str):
            continue
        for ks, j in s['judged'].items():
            if ks not in stored:
                continue
            mcode = r['code'][:r['pos']] + j['to'] + r['code'][r['pos'] + len(r['frm']):]
            if json.dumps(C.lexeme_key(mcode)) != ks:
                nf['rebuild_mismatch'] += 1
                continue
            mine = neutral_new(s['ncount'], mcode)
            nf['n'] += 1
            nf['agree'] += mine == sorted(stored[ks])
            if mine != sorted(stored[ks]) and len(nf_dis) < 20:
                nf_dis.append([s['sid'], j.get('to'), stored[ks], mine])
    comp = [r for r in out if r['how'] == 'completion' and not r['arm'].startswith('gen:')]
    unsc = sum(r['label'] == 'unscorable' for r in comp)
    if comp and unsc > len(comp) / 2:
        print(f"WARNING: {unsc} of {len(comp)} completion-path rows are unscorable: check that FLIP "
              f"completions keep their newline (SPEC: text up to the first newline)")
    summary = {
        'sites_file': sites_path, 'sites_sha256': C.sha256(sites_path), 'n_sites': len(sites),
        'run_dir': run_dir, 'run_files': run_counts, 'sites_with_runs': len(runs), 'ignored_files': ignored,
        'judge_tag': tag, 'judge_module_sha256': C.sha256(J.__file__), 'rustc': J.rustc_version(),
        'script_sha256': C.sha256(os.path.abspath(__file__)), 'rows': len(out),
        'labels_by_arm': {a: dict(c) for a, c in sorted(label_by_arm.items(), key=lambda x: arm_rank(x[0]))},
        'Y_by_arm': {a: sum(r['Y'] for r in out if r['arm'] == a) for a in
                     sorted(label_by_arm, key=arm_rank)},
        'how': dict(Counter(r['how'] for r in out)),
        'neutralizing_rows': [[r['site_id'], r['arm'], r['label'], rr['_neut']]
                              for r, rr in zip(out, rows) if r['neutralizing']][:200],
        'cache': {'file': cache_path, 'entries_used': len(cache) - len(items), 'stale_ignored': stale,
                  'duplicates_ignored': dup, 'programs_needed': len(need), 'judged_now': len(items),
                  'work_order': 'ascending cache key (sha256 of [task, lexer stream])'},
        'worker_rlimit_as': rl_max if items else None,
        'judge_error_scored_fail': jerr,
        'audit_blind_vs_stored': {'n': len(audit), 'agree': sum(out_label(a[2]) == out_label(a[3]) for a in audit),
                                  'disagree': [list(a) for a in audit if out_label(a[2]) != out_label(a[3])][:50]},
        'audit_neutralizing': {'n_with_site_flag': nf['n'], 'agree': nf['agree'], 'disagree': nf_dis,
                               'rebuild_mismatch': nf['rebuild_mismatch']},
        'anomalies': dict(anomalies), 'missing_arms': missing[:200], 'n_missing_arms': len(missing),
        'undefined_arms': info['undefined_arms'],
        'stage_C_excluded': info['stage_C_excluded'],
        'unscorable_by_reason': {k: {'n': len(v), 'rows': v[:200]} for k, v in sorted(info['unscorable'].items())},
        'existence_tokens': info['existence'],
        'completion_path': {'rows': len(comp), 'unscorable': unsc},
    }
    C.write_json(summary_path, summary)
    print(f"wrote {judged_path} ({len(out)} rows) and {summary_path}")
    for a, c in summary['labels_by_arm'].items():
        print(f"  {a:12s} " + ', '.join(f"{k} {v}" for k, v in sorted(c.items())))
    print(f"  cache: {len(items)} judged now, {len(cache) - len(items)} reused, {stale} stale ignored; "
          f"audit blind vs stored {summary['audit_blind_vs_stored']['agree']}/{len(audit)}; "
          f"judge_error scored fail {len(jerr)}; anomalies {dict(anomalies)}; missing arms {len(missing)}; "
          f"undefined arms {len(info['undefined_arms'])}; stage-C-only exclusions {len(info['stage_C_excluded'])}; "
          f"unscorable by reason { {k: len(v) for k, v in sorted(info['unscorable'].items())} }")
    return out, summary


# ---------------------------------------------------------------- selftest (SPEC test D)
def selftest():
    import shutil
    td = os.environ.get('TEST_DIR', '/home/user1/p8design/impl/D/selftest')
    if os.path.realpath(td).startswith(os.path.realpath(os.path.dirname(HERE)) + os.sep):
        raise SystemExit("TEST_DIR must lie outside the repository")
    fails, checks = [], 0

    def check(cond, what):
        nonlocal checks
        checks += 1
        print(('  ok    ' if cond else '  FAIL  ') + what, flush=True)
        if not cond:
            fails.append(what)

    refuse_if_gpu()
    print(f"selftest in {td}; judge module {J.__file__}")
    for f in ('test_sites.jsonl', 'blind_judge_cache.jsonl', 'judged.jsonl', 'judged.json'):
        if os.path.exists(os.path.join(td, f)):
            os.remove(os.path.join(td, f))
    shutil.rmtree(os.path.join(td, 'run'), ignore_errors=True)
    os.makedirs(td, exist_ok=True)

    print("1. neutralizing patterns (p8_1_sites.neutral_counts)")
    base = neutral_counts('fn f() { z }')
    for code, want in (('x - x', True), ('a.x - x', False), ('x - x.len()', False), ('y + x - x', False),
                       ('x * 1', True), ('x * 10', False), ('x *= 1;', True), ('x *= 1 + y;', False),
                       ('x + 0', True), ('let a = -0;', False), ('x % 1', True), ('x / 1.0', True),
                       ('x % x', True), ('a[i] * 1', True), ('1.max(x) * 2', False)):
        check(bool(neutral_new(base, 'fn f() { ' + code + ' }')) == want, f"neutralizing({code!r}) == {want}")

    print("2. synthetic sites from Phase 6 judged mutants (CPU)")
    tok = C.tokenizer()
    P = Programs(tok)
    od = C.OWNFIX
    src = {(s['task_id'], s['sample']): s for s in C.read_jsonl(os.path.join(od, 'ownfix_sources.jsonl.gz'))}
    ver = {}
    with open(os.path.join(od, 'ownfix_verify.json'), encoding='utf-8') as f:
        vj = json.load(f)
    for k, v in vj['mutants'].items():
        t, smp, pos, to = k.split('|', 3)
        ver[(int(t), int(smp), int(pos), to)] = v
    # V: a belief site (problem 443) with verified, lucky and fail mutants; PA: a pool-A site
    # (problem 20: stable repairs, all unverifiable); L: a slip site; X: a site the run excludes
    want_sites = {(443, 48, 99): 'belief', (20, 1, 42): 'pilot', (58, 166, 53): 'slip', (6, 17, 93): 'belief'}
    by = defaultdict(list)
    for m in C.read_jsonl(os.path.join(od, 'ownfix_mutants.jsonl.gz')):
        if (m['task_id'], m['sample'], m['pos']) in want_sites:
            by[(m['task_id'], m['sample'], m['pos'])].append(m)

    def p6_label(m, v):
        if m['compiles'] is False:
            return 'no_compile'
        if m['compiles'] and m.get('is_r'):
            return {True: 'verified', False: 'lucky', None: 'unverifiable'}[v]
        return 'fail'

    rows, info = [], {}
    for order, (key, kind) in enumerate(want_sites.items()):
        s = src[key[:2]]
        prose, code = s['prose'], s['code']
        e = tok(prose + ' ' + code, return_offsets_mapping=True)
        ids, offs = e['input_ids'], e['offset_mapping']
        at = len(prose) + 1 + key[2]
        h = next(t for t in range(1, len(offs)) if offs[t][0] <= at < offs[t][1])
        judged, tid, mcode = {}, {}, {}
        for m in by[key]:
            judged[json.dumps(C.lexeme_key(m['code']))] = {'to': m['to'], 'label': p6_label(m, ver.get(key + (m['to'],)))}
            mids = tok(prose + ' ' + m['code'])['input_ids']
            diff = [i for i in range(min(len(ids), len(mids))) if ids[i] != mids[i]]
            tid[m['to']] = mids[h] if len(mids) == len(ids) and diff == [h] else None
            mcode[m['to']] = m['code']
        sid = f"{key[0]}|{key[1]}|{key[2]}"
        rows.append({'site_id': sid, 'task': key[0], 'sample': key[1], 'pos': key[2], 'line': by[key][0]['line'],
                     'kind': kind, 'prose': prose, 'code': code, 'ids': ids, 'h': h, 'wrong_id': ids[h],
                     'wrong_piece': tok.convert_ids_to_tokens(ids[h]), 'family': by[key][0]['family'],
                     'frm': by[key][0]['from'], 'judged': judged, 'problem': key[0], 'order': order,
                     'neutralizing': {k: [] for k in judged}})
        info[sid] = (tid, mcode, {j['to']: j['label'] for j in judged.values()})
    V, PA, L, X = (r['site_id'] for r in rows)
    print(f"  sites {V} (belief), {PA} (pool A), {L} (slip), {X} (to be excluded); stored labels: "
          + '; '.join(f"{sid}: {info[sid][2]}" for sid in (V, PA, L)))
    # a planted wrong neutralizing flag in the site table: the audit must report it, the rows must not use it
    rows[0]['neutralizing'][json.dumps(C.lexeme_key(info[V][1]['!=']))] = ['x-x']
    sites_path = os.path.join(td, 'test_sites.jsonl')
    write_jsonl(sites_path, rows)
    sites = P.prepare(rows)
    n_spl = [(sid, to) for sid in (V, PA, L, X) for to in info[sid][0] if info[sid][0][to] is not None]
    check(len(n_spl) >= 15 and all(P.splice(sites[sid], info[sid][0][to]) == info[sid][1][to] for sid, to in n_spl),
          f"decoded single-token splices equal the stored mutant code ({len(n_spl)} mutants)")
    check(not any(v is True for k, v in ver.items() if k[0] == 20)
          and any(v is None for k, v in ver.items() if k[0] == 20),
          "problem 20 is pool-A-like: stable repairs, none verified")

    sp = lambda name: tok.convert_tokens_to_ids(name)  # noqa: E731
    SPACE, EOS = sp('▁'), tok.eos_token_id
    tV, cV, labV = info[V]
    tP, cP, labP = info[PA]
    sV, sP = sites[V], sites[PA]

    def region(s, code):
        text = s['prose'] + ' ' + code
        eol = text.find('\n', len(s['pre']))
        return text[len(s['pre']):eol if eol >= 0 else len(text)]

    def tail(s, code):
        return (s['prose'] + ' ' + code)[len(s['pre']):]

    def F(arm, t=None, completion=None, flipped=True):
        return {'arm': arm, 'flipped': flipped, 'k_star': 1 if flipped else None, 't_star_id': t,
                't_star_piece': tok.convert_ids_to_tokens(t) if t is not None else None,
                'margin_at_kstar': 0.5 if flipped else None, 'scan_replay_disagree': 0, 'n_candidates': 9,
                'ablated': [], 'completion': completion}

    nospace = lambda t: sp(tok.convert_ids_to_tokens(t).lstrip('▁'))  # noqa: E731
    rV = region(sV, cV['<='])
    check(rV.startswith(' ') and region(sV, sV['code']).startswith(' ')
          and nospace(tV['<']) not in (tV['<'], tok.unk_token_id),
          "test site V: edited token starts with a space; its no-space variant is another id")
    runs = {
        ('A', V): {'site_id': V, 'excluded': None, 'BAN': F('BAN', tV['<']), 'NULL': F('NULL', flipped=False),
                   'GxA': F('GxA', tV['!=']), 'DE': F('DE', EOS, ' largest {\n'),
                   'CONTRAST': F('CONTRAST', nospace(tV['<']))},
        ('B', V): {'site_id': V, 'excluded': None,
                   'G': F('G', SPACE, rV[1:] + '\n        junk after the newline'),
                   'G_LP': F('G_LP', SPACE, rV[1:].replace(' ', '   ') + '\n'),
                   'G1': F('G1', SPACE, '@@ largest {\n'),
                   'G_mid': F('G_mid', SPACE, rV[1:]),
                   'RND1': F('RND1', nospace(tV['>='])),
                   'RND2': F('RND2', SPACE, '< largest * 1 {\n'),
                   'RND3': F('RND3', SPACE, region(sV, sV['code'])[1:] + '\n'),   # the line rebuilds S
                   'RND4': F('RND4', SPACE, None),
                   'RND5': F('RND5', nospace(sV['ids'][sV['h']])),
                   'ORC': F('ORC', SPACE, region(sV, cV['>='])[1:] + '\n'),
                   'RNDk': [{'draw': 1, 't_star_id': tV['<'], 'changed': True},
                            {'draw': 2, 't_star_id': sV['ids'][sV['h']], 'changed': False},
                            {'draw': 3, 't_star_id': SPACE, 'changed': True}],
                   # existence: G-side flips = BAN's lexeme, lucky '!=', verified '<=' (no-space id),
                   # a whitespace twin of the wrong token; random flips = BAN's lexeme, an unjudged
                   # token, fail '>=' twice (one distinct lexeme)
                   'existence': {'n_single': 384, 'flips': [[25, 40, 7, tV['<']], [24, 40, 9, tV['!=']],
                                                            [23, 41, 3, nospace(tV['<='])],
                                                            [20, 39, 1, nospace(sV['ids'][sV['h']])]],
                                 'random_n': 128, 'random_flips': [[3, 40, 5, tV['<']], [4, 40, 6, SPACE],
                                                                   [5, 40, 7, tV['>=']], [6, 41, 8, tV['>=']]]}},
        ('C', V): {'site_id': V, 'gen': {'G': {'text': tail(sV, cV['<=']), 'tokens': 30, 'stopped': 'braces'},
                                         'G_every': {'text': tail(sV, cV['!=']) + '<eos>', 'tokens': 31, 'stopped': 'eos'},
                                         'BAN': {'text': tail(sV, cV['<'])[:20], 'tokens': 99, 'stopped': 'cap'},
                                         'NULL': {'text': tail(sV, sV['code']), 'tokens': 30, 'stopped': 'braces'}},
                   'collateral': {}},
        ('A', PA): {'site_id': PA, 'excluded': None, 'BAN': F('BAN', tP['>']), 'NULL': F('NULL', flipped=False),
                    'GxA': F('GxA', tP['<']), 'DE': F('DE', SPACE, region(sP, cP['!='])[1:] + '\n'),
                    'CONTRAST': None, 'contrast_note': 'no stable-passing judged repair'},
        ('A', L): {'site_id': L, 'excluded': None, 'NULL': F('NULL', info[L][0]['<'])},
        ('A', X): {'site_id': X, 'excluded': 'a: argmax not in E(wrong)', 'decision': None,
                   'BAN': None, 'NULL': None, 'GxA': None, 'DE': None},
        ('B', X): {'site_id': X, 'excluded': None, 'G': F('G', SPACE, 'x\n')},
    }
    run_dir = os.path.join(td, 'run')
    for (stage, sid), obj in runs.items():
        C.write_json(os.path.join(run_dir, stage, sid.replace('|', '_') + '.json'), obj)
    expect = {
        (V, 'BAN'): ('verified', 1, 'lexeme', True), (V, 'NULL'): ('no_flip', 0, 'none', False),
        (V, 'GxA'): ('lucky', 0, 'lexeme', False), (V, 'DE'): ('no_compile', 0, 'completion', False),
        (V, 'CONTRAST'): ('verified', 1, 'lexeme', True),
        (V, 'G'): ('verified', 1, 'completion', False), (V, 'G_LP'): ('verified', 1, 'completion', False),
        (V, 'G1'): ('no_compile', 0, 'completion', False), (V, 'G_mid'): ('unscorable', 0, 'completion', False),
        (V, 'RND1'): ('fail', 0, 'lexeme', False), (V, 'RND2'): ('verified', 1, 'completion', False),
        (V, 'RND3'): ('fail', 0, 'completion', False), (V, 'RND4'): ('unscorable', 0, 'completion', False),
        (V, 'exist:G'): ('verified', 1, 'lexeme', False), (V, 'exist:RND'): ('fail', 0, 'lexeme', False),
        (V, 'RND5'): ('no_flip', 0, 'none', False), (V, 'ORC'): ('fail', 0, 'completion', False),
        (V, 'RNDk:1'): ('verified', 1, 'lexeme', True), (V, 'RNDk:2'): ('no_flip', 0, 'none', False),
        (V, 'RNDk:3'): ('unscorable', 0, 'none', False),
        (V, 'gen:G'): ('verified', 1, 'completion', False), (V, 'gen:G_every'): ('lucky', 0, 'completion', False),
        (V, 'gen:BAN'): ('unscorable', 0, 'completion', False), (V, 'gen:NULL'): ('fail', 0, 'completion', False),
        (PA, 'BAN'): ('unverifiable', 0, 'lexeme', True), (PA, 'NULL'): ('no_flip', 0, 'none', False),
        (PA, 'GxA'): ('fail', 0, 'lexeme', False), (PA, 'DE'): ('unverifiable', 0, 'completion', False),
        (L, 'NULL'): ('verified', 1, 'lexeme', False),
    }
    for a in FLIP_A + FLIP_B:
        expect[(X, a)] = ('excluded', 0, 'none', False)
    check(labV.get('<') == 'verified' and labV.get('!=') == 'lucky' and labV.get('>=') == 'fail'
          and labP.get('>') == 'unverifiable' and labP.get('<') == 'fail',
          "test sites carry verified, lucky, fail and unverifiable judged mutants")

    print("3. scoring, first run (blind judge through the cache)")
    paths = [sites_path, run_dir, os.path.join(td, 'blind_judge_cache.jsonl'), os.path.join(td, 'judged.jsonl'),
             os.path.join(td, 'judged.json')]
    out, summ = score(*paths, P=P)
    got = {(r['site_id'], r['arm']): r for r in out}
    check(set(got) == set(expect), f"one row per (site, arm): {len(got)} rows, expected {len(expect)}")
    for k, (lab, y, how, isban) in expect.items():
        r = got.get(k)
        check(r is not None and (r['label'], r['Y'], r['how'], r['lexeme_is_ban']) == (lab, y, how, isban),
              f"{k[0]} {k[1]:12s} -> {lab}, Y {y}, {how}, ban {isban}"
              + ('' if r is None else f"   [got {r['label']}, {r['Y']}, {r['how']}, {r['lexeme_is_ban']}]"))
    check(all(r['Y'] == int(r['label'] == 'verified') for r in out), "Y = 1 iff verified, every row")
    check([k for k, r in got.items() if r['neutralizing']] == [(V, 'RND2')],
          "only the '* 1' completion is flagged neutralizing (a neutralizing rescue)")
    an = summ['audit_neutralizing']
    check(an['n_with_site_flag'] == sum(len(r['judged']) for r in rows) and an['agree'] == an['n_with_site_flag'] - 1
          and len(an['disagree']) == 1 and an['disagree'][0][1] == '!=',
          f"neutralizing audit: {an['agree']}/{an['n_with_site_flag']} agree, the planted flag reported")
    check(got[(V, 'G')]['key'] == got[(V, 'G_LP')]['key'] == got[(V, 'gen:G')]['key'] is not None,
          "a whitespace variant and the same program from another arm share one cache key")
    check(summ['cache']['judged_now'] == summ['cache']['programs_needed'] == 8,
          f"8 distinct programs judged once each (judged {summ['cache']['judged_now']})")
    check(summ['audit_blind_vs_stored']['n'] == 6 and summ['audit_blind_vs_stored']['agree'] == 6,
          f"blind judge agrees with stored labels where a program equals a judged mutant "
          f"({summ['audit_blind_vs_stored']['agree']}/{summ['audit_blind_vs_stored']['n']})")
    check(summ['worker_rlimit_as'] == MEM, f"judge workers ran under RLIMIT_AS = 8 GiB ({summ['worker_rlimit_as']})")
    an = summ['anomalies']
    check(an == {'completion_missing': 1, 'no_newline': 1, 't_star_in_E': 1, 'special_token_t_star': 1,
                 'gen_special_tail_dropped': 1, 'completion_is_S': 1, 'gen_is_S': 1},
          f"anomalies counted {an}")
    check(summ['undefined_arms'] == [[PA, 'A', 'CONTRAST', 'no stable-passing judged repair']]
          and (PA, 'CONTRAST') not in got, f"a null CONTRAST gets no row and is listed ({summ['undefined_arms']})")
    et = summ['existence_tokens'].get(V, {})
    check(et == {'exist:G': {'ban_lexeme': 1, 'lucky': 1, 'verified': 1, 'in_E': 1},
                 'exist:RND': {'ban_lexeme': 1, 'unscorable': 1, 'fail': 1, 'repeat': 1}}
          and set(summ['existence_tokens']) == {V},
          f"existence: BAN's lexeme and S's own lexeme skipped, best label kept ({et})")

    print("4. the arm never reaches the judge")
    _, need4, _, _, _ = collect(sites, load_runs(run_dir, sites)[0], P)
    items = sorted((k, t, c) for k, (t, c) in need4.items())
    check(all(len(it) == 3 and it[0] == cache_key(it[1], C.lexeme_key(it[2])) for it in items),
          "work items are (key, task, code) with key = sha256([task, lexer stream])")
    import inspect
    check(list(inspect.signature(_work).parameters) == ['item'], "the worker takes the work item only")
    with open(paths[2], encoding='utf-8') as f:
        crec = [json.loads(l) for l in f if l.strip()]
    check(all(set(r) == {'key', 'task', 'judge', 'label', 'extra', 'extra_bad', 'verdicts', 'tries', 'code'}
              for r in crec), "cache entries hold no arm or site field")
    perm = {'G': 'ORC', 'ORC': 'G', 'G1': 'G_LP', 'G_LP': 'G1', 'DE': 'GxA', 'GxA': 'DE'}
    pruns = load_runs(run_dir, sites)[0]
    for sid, files in pruns.items():
        for st, o in files.items():
            moved = {perm.get(k, k): (dict(v, arm=perm.get(k, k)) if _is_flip(v) else v) for k, v in o.items()}
            if 'gen' in moved:
                moved['gen'] = {('NULL' if g == 'G' else 'G' if g == 'NULL' else g): v for g, v in moved['gen'].items()}
            files[st] = moved
    _, need5, _, _, _ = collect(sites, pruns, P)
    check(sorted((k, t, c) for k, (t, c) in need5.items()) == items,
          "permuting the arm names leaves the judge's work list unchanged")

    print("5. restart: rerun, and a crash with a partial cache line")
    with open(paths[3], 'rb') as f:
        b1 = f.read()
    out2, summ2 = score(*paths, P=P)
    with open(paths[3], 'rb') as f:
        same2 = f.read() == b1
    check(summ2['cache']['judged_now'] == 0 and same2, "rerun judges nothing and writes identical judged.jsonl")
    with open(paths[2], 'rb') as f:
        lines = f.read().splitlines(keepends=True)
    with open(paths[2], 'wb') as f:
        f.writelines(lines[:-3])
        f.write(b'{"key": "partial line left by a cra')
    out3, summ3 = score(*paths, P=P)
    with open(paths[3], 'rb') as f:
        same3 = f.read() == b1
    check(summ3['cache']['judged_now'] == 3 and same3,
          f"after a crash (3 lines lost, partial line) only those 3 are judged ({summ3['cache']['judged_now']}); "
          f"identical judged.jsonl")

    print("6. refuses beside a GPU job")
    global GPU_LOCK
    real_lock, GPU_LOCK = GPU_LOCK, os.path.join(td, 'gpu.lock')   # never touch the real lock here
    sleeper = 'import sys, time; time.sleep(60)'

    def refused_now():
        try:
            refuse_if_gpu()
        except SystemExit as e:
            return e.code == 4
        return False
    try:
        dummy = subprocess.Popen([sys.executable, '-c', sleeper, 'p8_2_run.py'])
        try:
            time.sleep(1.0)
            check(refused_now() and any(p == dummy.pid for p, _ in gpu_jobs()),
                  "refuses (exit 4) while a p8_2_run process is alive")
        finally:
            dummy.kill()
            dummy.wait()
        check(not gpu_jobs(), "runs again once it is gone")
        cpu = [subprocess.Popen([sys.executable, '-c', sleeper, 'p8_2_run.py', '--test']),
               subprocess.Popen([sys.executable, '-c', sleeper, 'p8_0_checks.py'], env=dict(os.environ, MODE='selftest'))]
        try:
            time.sleep(1.0)
            check(not gpu_jobs() and not refused_now(),
                  "a CPU selftest (p8_2_run.py --test, MODE=selftest p8_0_checks.py) does not block judging")
        finally:
            for p in cpu:
                p.kill()
                p.wait()
        holder = subprocess.Popen([sys.executable, '-c', 'import fcntl, sys, time; f = open(sys.argv[1], "w"); '
                                   'fcntl.flock(f, fcntl.LOCK_EX); print("held", flush=True); time.sleep(60)',
                                   GPU_LOCK], stdout=subprocess.PIPE, text=True)
        try:
            holder.stdout.readline()
            check(gpu_lock_held() and refused_now() and any(p == 0 for p, _ in gpu_jobs()),
                  "refuses (exit 4) while another process holds the GPU lock (no matching command line)")
        finally:
            holder.kill()
            holder.wait()
        check(not gpu_lock_held() and not gpu_jobs(), "the lock released: runs again; the test left no lock held")
    finally:
        GPU_LOCK = real_lock

    print("6b. stage-C-only exclusion, <eos> completions, missing primary arms, the cache tag")
    import copy
    base_rows = {(r['site_id'], r['arm']): (r['label'], r['key']) for r in collect(sites, load_runs(run_dir, sites)[0], P)[0]}
    r8 = copy.deepcopy(load_runs(run_dir, sites)[0])
    r8[PA]['C'] = {'site_id': PA, 'excluded': 'C: failed in two fresh processes (RuntimeError: x)', 'seconds': 0.0}
    r8[V]['C'] = {'site_id': V, 'excluded': 'C: failed in two fresh processes (RuntimeError: y)', 'seconds': 0.0}
    del r8[V]['A']['CONTRAST']                        # V's CONTRAST now belongs to the failed stage C
    r8[V]['B']['G_mid'] = dict(r8[V]['B']['G_mid'], completion=rV[1:] + '<eos>')
    rows8, _, an8, miss8, info8 = collect(sites, r8, P)
    g8 = {(r['site_id'], r['arm']): (r['label'], r['key']) for r in rows8}
    c_arms = {(sid, a) for sid in (V, PA) for a in C_ROWS} - {(PA, 'CONTRAST')}
    check(all(g8.get(k, (None,))[0] == 'excluded' for k in c_arms)
          and all(r['label'] != 'excluded' for r in rows8 if r['site_id'] in (V, PA) and r['stage'] != 'C'),
          "a stage-C-only exclusion marks only the stage-C rows (gen:*, CONTRAST) 'excluded'")
    check(all(g8[k] == base_rows[k] for k in base_rows if k[0] in (V, PA) and k not in c_arms and k != (V, 'G_mid'))
          and all(g8[(V, a)][0] not in ('excluded', None) or g8[(V, a)][1] for a in PRIMARY),
          "... and leaves every stage A and B row (G, BAN, RND1, GxA included) scored as before")
    check(sorted(x[0] for x in info8['stage_C_excluded']) == sorted([V, PA]) and (PA, 'CONTRAST') not in g8
          and [PA, 'A', 'CONTRAST', 'no stable-passing judged repair'] in info8['undefined_arms'],
          "stage-C exclusions are listed; PA's null CONTRAST (stage A) stays undefined")
    check(g8[(V, 'G_mid')] == ('unscorable', None) and an8.get('completion_eos') == 1 and an8.get('no_newline', 0) == 0
          and info8['unscorable']['completion_eos'] == [[V, 'G_mid']],
          f"a completion ended at <eos> before a newline: unscorable, counted as completion_eos, not no_newline ({dict(an8)})")
    rf = copy.deepcopy(r8)                            # p8_2_run.failed_record's stage-C form
    rf[V]['C'] = {'site_id': V, 'excluded': None, 'failed': 'RuntimeError: y', 'gen': None, 'gen_cap': None,
                  'collateral': None, 'CONTRAST': None, 'contrast_note': 'stage C failed (RuntimeError: y)'}
    rowsf, _, _, missf, infof = collect(sites, rf, P)
    gf = {(r['site_id'], r['arm']): (r['label'], r['key']) for r in rowsf}
    check(all(gf[(V, a)][0] == 'excluded' for a in C_ROWS) and not missf
          and all(gf[k] == g8[k] for k in g8 if k[0] == V and k not in {(V, a) for a in C_ROWS})
          and V in [x[0] for x in infof['stage_C_excluded']]
          and not any(u[0] == V and u[1] == 'C' for u in infof['undefined_arms']),
          "a stage C written as failed (excluded=None, 'failed') is handled like a stage-C exclusion")
    rb = copy.deepcopy(load_runs(run_dir, sites)[0])
    rb[V]['B']['excluded'] = 'c: attribute() failed in two fresh processes (x)'
    gb = {(r['site_id'], r['arm']): r['label'] for r in collect(sites, rb, P)[0] if r['site_id'] == V}
    check(gb and all(v == 'excluded' for v in gb.values()) and (V, 'BAN') in gb,
          f"a stage-B exclusion still excludes every row of the site ({len(gb)} rows)")
    mdir = os.path.join(td, 'run_missing')
    shutil.rmtree(mdir, ignore_errors=True)
    shutil.copytree(run_dir, mdir)
    mf = os.path.join(mdir, 'A', V.replace('|', '_') + '.json')
    with open(mf, encoding='utf-8') as f:
        mo = json.load(f)
    del mo['GxA']
    C.write_json(mf, mo)
    mpaths = [sites_path, mdir] + [os.path.join(td, 'missing_' + n) for n in ('cache.jsonl', 'judged.jsonl', 'judged.json')]
    code5 = None
    try:
        score(*mpaths, P=P)
    except SystemExit as e:
        code5 = e.code
    check(code5 == 5 and not any(os.path.exists(p) for p in mpaths[2:]),
          f"a primary arm (GxA) missing from a present stage file: exit 5, nothing judged or written (exit {code5})")
    shutil.rmtree(mdir, ignore_errors=True)
    t1 = judge_tag()
    J._NEW.COMPILE_TIMEOUT += 1
    try:
        t2 = judge_tag()
    finally:
        J._NEW.COMPILE_TIMEOUT -= 1
    check(t1 != t2 and judge_tag() == t1 == summ['judge_tag'], "the cache tag changes with COMPILE_TIMEOUT")

    print("7. the real site tables (p8_1_sites output), if present: splices on CPU")
    for name in ('sites.jsonl', 'pilot_sites.jsonl'):
        p = os.path.join(C.OUT, name)
        if not os.path.exists(p):
            print(f"  skip  {p} does not exist")
            continue
        real = P.prepare(C.read_jsonl(p))             # schema, ids decode to prose + ' ' + code, labels
        bad_e = [(sid, v) for sid, s in real.items() for v in [s['ids'][s['h']]] + list(s['row']['E'])
                 if C.lexeme_key(P.splice(s, int(v)) or '') != s['lk']]
        n_e = sum(1 + len(s['row']['E']) for s in real.values())
        bad_v, n_v = [], 0
        for sid, s in real.items():
            for v in s['row'].get('verified_ids') or []:
                n_v += 1
                j = s['judged'].get(json.dumps(C.lexeme_key(P.splice(s, int(v)) or '')))
                if j is None or j['label'] != 'verified':
                    bad_v.append((sid, v))
        check(not bad_e and not bad_v,
              f"{name}: {len(real)} sites prepared; wrong_id and E ids splice to S ({n_e - len(bad_e)}/{n_e}); "
              f"verified_ids land on 'verified' judged mutants ({n_v - len(bad_v)}/{n_v})")

    print(f"\nselftest: {checks - len(fails)} of {checks} checks passed (judge module {J.__file__})")
    if fails:
        print("FAILED: " + '; '.join(fails))
        sys.exit(1)
    print("SELFTEST PASSED")


def main():
    if os.environ.get('MODE') == 'selftest' or '--test' in sys.argv[1:]:
        return selftest()
    score(SITES, RUN_DIR, CACHE, JUDGED, SUMMARY)


if __name__ == '__main__':
    main()
