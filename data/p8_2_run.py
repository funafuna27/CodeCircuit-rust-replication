"""Phase 8, step 2: the GPU run (final_design.md §2-§4, §9-§10; SPEC.md 'run outputs').

Stages, in this order (run_p8.sh), each over every site of SITES in the table's run order:
  A  graph-free and label-free. Belief/pilot sites: exclusion (b) the prefix ids, (a) the fp32
     decision at h-1, (d) GxA's logit node 0; the decision record; NULL, BAN, GxA, DE.
     Slips: the decision record and NULL only.
  B  graph, label-free. attribute() (the study call: p8_common.GRAPH_KW); e = signed influence on
     logit node 0 (and on the log-prob target, G_LP) in float64 on the CPU; G, G_LP, G1, G_mid,
     RND1-RND5, RNDk (20 draws) = the primary part, checkpointed (RUN_DIR/B/<site>.primary.ckpt)
     before the descriptive part: the single ablations (ORC, existence, faithfulness), ORC.
     The graph is never saved and is deleted as soon as e and the RND pool are extracted.
  C  greedy programs for G, G_every, BAN, NULL (lexer brace stop rule); then the label-using
     work: collateral (teacher-forced best repair, G against the RNDk draws) and CONTRAST.
Outputs: RUN_DIR/<stage>/<site_id with | -> _>.json (SPEC shapes, plus the descriptive fields
listed below), RUN_DIR/ledger.jsonl (one fsynced line per event), P8_CLOCK (default
data/results/intervention/gpu_clock.jsonl whatever P8_OUT is: every GPU process's wall time from
process start, beaten every 30 s; the hard stop reads it).

Common machinery (final_design.md §3)
  model     ReplacementModel.from_pretrained(gemma-2-2b-it, 'gemma', bf16) from the INSTALLED
            circuit_tracer (the one that built the study's graphs; this module never puts the
            repository's circuit_tracer/ on sys.path). Decoders stay lazy (read from the
            safetensors files on demand), so the card holds ~8 GiB and forwards do not page.
  numerics  torch.use_deterministic_algorithms(True, warn_only=True), CUBLAS_WORKSPACE_CONFIG
            :4096:8, TF32 off: index_add_ (feature_intervention, our hook, attribute()) then sums
            in a fixed order, so a site redone in a fresh process gives the same bytes. Any op
            without a deterministic kernel is recorded ('det_warnings' in the ledger).
  ablation  our hook = feature_intervention(freeze_attention=False, constrained_layers=None) with
            value 0, batched: at blocks.l.ln2.hook_normalized the current input is re-encoded
            (transcoders.encode_layer on whole rows, ENC_CHUNK rows at a time; a batch-1 call is
            the identical call feature_intervention makes); at blocks.l.hook_mlp_out.hook_out_grad
            the delta zeros().index_add_(-a * W_dec) is added, entries in (row, pos, feat) order,
            zero activations dropped, position 0 never ablated (model.zero_positions).
  decision  fp32: ln_final.hook_normalized (fp32) * ln_final.w @ W_U[:, C] + b_U[C], soft-cap 30,
            C = the bf16 top-64 (ties by id) | E, sorted by id; argmax = the first maximum.
            margin = best E logit - best non-E logit. A flip = the argmax over C is not in E.
  flip_scan one function for every ordered arm: k = 1..min(K_MAX, candidates), cumulative; screen
            SCAN_BATCH rows per forward; every row whose screen flips or whose screened margin is
            below the guard is replayed at batch 1 in k order (P0 (b): a re-encoded feature at its
            JumpReLU threshold moves batch-64 rows; guard = max(2 delta, 2 x P0's largest batch-64
            margin deviation), replay_guard(); first P0 0.110, one-per-problem P0 0.376); k* = the
            smallest replayed flip, t* its argmax; if none, scanning continues.
            scan_replay_disagree = replayed rows (up to k*) whose screen flip status differs. completion: greedy (full-vocabulary fp32 argmax)
            from ids[:h] + [t*] under the arm's ablation (prefix positions only) until a newline,
            <eos>/<end_of_turn> or GEN_LINE_CAP tokens; the text after t*'s piece.
  singles   one feature ablated at a time, 63 per forward plus one clean row; effects against
            that in-batch clean row: drop (margin), dtarget (node-0 target = hook_normalized * w .
            demeaned W_U column, fp32), dtarget_fs (the same with ln_final's scale held clean); a
            screened flip or a screened margin below the guard is replayed at batch 1. ORC orders
            GxA's top ORC_N by drop; existence = replayed flips among ORC's set and G's top EXIST_G,
            and among EXIST_RND random features matched to G's top 128 (rng 10); faithfulness =
            Spearman(e, dtarget) over G's top 128 with |e| > 3 delta, those singles at batch 1
            (FAITH_BATCH, as P0 (a)); the screen-batch value is kept as spearman_screen.
  DE        a * W_dec . (ln_final.w * u / clean scale) for the active features at h-1, u = the wrong
            token's demeaned W_U column.
  RND       rng(2, task, sample, pos, draw); RNDk rng(11, ..., draw); existence rng(10, ...).
            p8_2_run.Pool.match: per rank of G, one of the 5 eligible features nearest in rank on
            a*||W_dec|| at the same (layer, pos), else same pos nearest layer, else same layer nearest
            pos; eligible excludes G's and GxA's top 64, earlier draws and the entry itself.
Descriptive fields beyond SPEC: FLIP.K (positions scanned), FLIP.fullvocab_t (full-vocabulary
argmax at k*), FLIP.kind (match kinds for RND), A.decision.C/tl_argmax/fullvocab_argmax/scale,
A.gxa, A.de_n_candidates, B.graph.e_iterations/n_active/logit_p, B.rnd_match, B.singles
([l, p, f, drop, dtarget, dtarget_fs, screened argmax, replayed flip t]), RNDk[].ablated/margin
(ablated = [l, p, f, activation] for every arm), B.faithfulness.sign_agree/batch/spearman_screen/
spearman_final_scale_frozen, B.existence.screen_batch/replay_guard, B.settings (run_settings()),
B.descriptive_failed/orc_note (the descriptive part failed twice: ORC, existence, faithfulness,
singles null, primary arms kept), C.excluded_in (an A/B exclusion copied), C.failed (a stage-C
failure a person accepted: gen, collateral, CONTRAST null), C.gen_cap, C.gen.G.first_matches_t_star,
C.collateral {repair_to, repair_label, positions, G (argmax changes at positions >= h), G_k, RNDk
[per draw], teacher_agree_clean}, C.contrast_repair.
CONTRAST is written to the stage-C file (design: it runs after every label-free arm; SPEC lists
it under A). Labels are read only in stage C (best repair: the 'verified' judged mutant with the
highest clean logit; kind 'pilot' rows only: fallback 'unverifiable' then 'lucky') and never in A
or B. Stage C never writes an exclusion of its own (it only copies one from A or B, excluded_in).

Guards before any GPU work: SITES must be set (no default); kinds belief/slip/pilot only, never
mixed; a pilot-only table may not write to $P8_OUT/run; belief/slip rows (the 35 site problems)
need every PREREGISTRATION_P8*.md at the repository root tracked by git and unmodified against
HEAD (hard rule: no GPU on them before the pre-registration is committed; its sha256 and commit
go into proc_start); p0.json must say pass (ALLOW_P0_FAIL=1: tests on pilot rows only).

Process protocol (final_design.md 'Implementation plan'): one process per chunk; restart, never
retry. A process takes the GPU lock ($P8_WORK/gpu.lock) before touching RUN_DIR, refuses to run
if an earlier proc_start of this RUN_DIR had another site table, p0.json, delta, run_settings()
or pre-registration marker, starts its GPU clock, refuses to start below MIN_FREE_GIB of free
device memory (another GPU job), loads the model, checks the canary of p0.json (clean and with 8
features ablated; within delta, expected bitwise), then handles at most CHUNK sites whose stage
file is missing: ledger 'start', compute, atomic write (.partial, fsync, rename), ledger 'done'
with the site's seconds. A raised error is logged ('error', with the stage-B phase) and the
process exits 1; the next process tries the site again. Attempts (site_attempts): an error is a
start whose process logged an error; a death is a start with neither error nor done in the current
boot (usage limit, kill, host OOM); starts from an earlier boot (reboot) are not counted.
Resolution: stage B, attribute() raised in 2 processes -> excluded, design §2 (c) (the only
exclusion a failure produces); stage B, the descriptive part raised in 2 processes -> the
checkpointed primary part with descriptive_failed; any other site with 2 errors or MAX_DEATHS (3)
deaths -> ledger 'needs_person' and exit 6: the driver stops and nothing is excluded. A person
fixes the cause and reruns, or sets ACCEPT_FAILED='<stage>:<site_id>[,...]' to record it
(failed_record: A -> excluded 'failed: ...' (label-free, not a registered reason, so reported
separately), B -> primary part if checkpointed else excluded 'failed: ...', C -> never excluded).
Hard stop, before each site: summed GPU wall time of every Phase 8 GPU process (P8_CLOCK; P0, P1
and main) >= 28 h (HARD_STOP_GPU_H) minus the reserve. Stage B's reserve leaves stage C its time
(reserve_hours: RESERVE_H if set, else (finished non-excluded B sites + 1) x C seconds per site x
1.5; C seconds = median of this RUN_DIR's stage-C sites once >= 5, else C_SEC_PER_SITE, default
60); stages A and C stop at exactly 28 h. To be stated in PREREGISTRATION_P8.
Exit codes: 0 stage complete, 10 chunk done (more to do), 3 hard stop, 4 configuration / canary /
P0 problem, 5 GPU busy, 6 a site needs a person, 1 crash.

Usage (cwd data/):
  STAGE=A python p8_2_run.py                 one chunk of stage A (run_p8.sh loops)
  python p8_2_run.py --test  (or MODE=selftest)   CPU self-tests: the stop rule teacher-forced on
                                             every Phase 6 source with a stable repair, e on
                                             random DAGs, RND matching on a synthetic pool
Env: STAGE  SITES (required: pilot_sites.jsonl or sites.jsonl)  RUN_DIR=run (P1: pilot/run;
     relative names are under P8_OUT)  CHUNK=10  P0_JSON=$P8_OUT/p0.json  RESERVE_H (unset: the
     dynamic reserve)  C_SEC_PER_SITE=60  P8_OUT  P8_WORK  P8_CLOCK  MIN_FREE_GIB=9.5
     ORC_N=256  EXIST_G=128 (P1 may fix 128/64; equal in every process of a RUN_DIR)
     ACCEPT_FAILED (a person's decision, above)
     ALLOW_P0_FAIL=1 (tests only, pilot rows only: run although p0.json says pass false; logged)
     FAULT=<stage>:<site_id>:<point>[:always][:raise]  (tests only: os._exit(97) - or with ':raise'
     a RuntimeError - at <point> on the site's first attempt (every attempt with ':always');
     points: start, attribute (inside attribute()'s error wrapper), after_graph, mid_scan,
     mid_singles, before_write)
data/1_code_correct.py is never opened, imported or run.
"""
import os
import time

T_START = time.time()        # process start: the GPU clock's t0 (model load included)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import fcntl  # noqa: E402
import gc  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from functools import partial  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p8_common as C  # noqa: E402

ENC_CHUNK = 16            # rows per transcoder re-encode (memory); a batch-1 call is feature_intervention's call
ORC_N = int(os.environ.get('ORC_N', 256))        # GxA's top ORC_N, measured singly (pilot may set 128)
EXIST_G = int(os.environ.get('EXIST_G', 128))    # G's top EXIST_G, measured singly (pilot may set 64)
EXIST_RND = 128           # matched random single features (existence)
FAITH_N = 128             # faithfulness: G's top 128 with |e| > 3 delta
FAITH_BATCH = 1           # faithfulness singles at batch 1 (as P0 (a) measured them)
RND_DRAWS, RNDK_DRAWS = 5, 20
RND_NEAR = 5              # draw among the 5 eligible features nearest in rank
RND_EXCL = 64             # G's and GxA's top 64 are never drawn
TOP_OUT = 2000            # e_top / gxa_top rows
EOS_IDS = (1, 107)        # <eos>, <end_of_turn>
MAX_ITER = 64             # e: power-series cap (a 26-layer DAG needs <= 27)
STAGES = ('A', 'B', 'C')
KINDS = ('belief', 'slip', 'pilot')
MAX_DEATHS = 3            # a site whose process dies (no error logged) this often in one boot stops the driver
TIMING_KEYS = ('seconds',)
TEST_STREAM = 100         # self-test synthetic data only: default_rng([ROOT, 100, k]); never a study quantity
assert TEST_STREAM not in C.STREAMS and TEST_STREAM not in C.NEVER


def test_rng(k):
    return np.random.default_rng([C.ROOT, TEST_STREAM, k])


def _out(name, default):
    v = os.environ.get(name, default)
    return v if os.path.isabs(v) else os.path.join(C.OUT, v)


def run_settings():
    """Settings that change outputs (and may be set by env); recorded in proc_start and stage B and
    required equal across every process of a RUN_DIR."""
    return {'ORC_N': ORC_N, 'EXIST_G': EXIST_G, 'EXIST_RND': EXIST_RND, 'FAITH_N': FAITH_N,
            'FAITH_BATCH': FAITH_BATCH, 'ENC_CHUNK': ENC_CHUNK, 'RND_DRAWS': RND_DRAWS, 'RNDK_DRAWS': RNDK_DRAWS,
            'RND_NEAR': RND_NEAR, 'RND_EXCL': RND_EXCL, 'K_MAX': C.K_MAX, 'SCAN_BATCH': C.SCAN_BATCH,
            'TOP_C': C.TOP_C, 'GEN_LINE_CAP': C.GEN_LINE_CAP, 'GEN_EXTRA': C.GEN_EXTRA, 'GRAPH_KW': C.GRAPH_KW}


# ---------------------------------------------------------------- small io
def append_line(p, obj):
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    with open(p, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, default=C._jsonable) + '\n')
        f.flush()
        os.fsync(f.fileno())


def read_lines(p):
    """jsonl rows; a torn last line (a crash mid-write) is skipped."""
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def fname(sid):
    return sid.replace('|', '_') + '.json'


def site_key(sid):
    t, s, p = sid.split('|')
    return int(t), int(s), int(p)


def strip_timing(o):
    if isinstance(o, dict):
        return {k: strip_timing(v) for k, v in o.items() if k not in TIMING_KEYS}
    if isinstance(o, list):
        return [strip_timing(v) for v in o]
    return o


def boot_id():
    try:
        with open('/proc/sys/kernel/random/boot_id') as f:
            return f.read().strip()
    except OSError:
        return 'unknown'


# ---------------------------------------------------------------- GPU clock (hard stop)
# One clock for the whole of Phase 8 (P0 + P1 + main), whatever P8_OUT is: P8_CLOCK, default
# data/results/intervention/gpu_clock.jsonl (fixed; tests that must not count set P8_CLOCK).
CLOCK_PATH = os.environ.get('P8_CLOCK', os.path.join(HERE, 'results', 'intervention', 'gpu_clock.jsonl'))
BEAT_S = 30.0


class Clock:
    """Every GPU process appends {pid, boot, t0, t} lines; its GPU time is max(t) - t0. Created
    before the model is loaded (t0 = process start) and beats every BEAT_S seconds from a daemon
    thread, so a process killed from outside loses at most BEAT_S seconds."""

    def __init__(self, script, t0=None):
        import threading
        self.path = CLOCK_PATH
        self.key = {'script': script, 'pid': os.getpid(), 'boot': boot_id(), 't0': t0 or time.time()}
        self._lk = threading.Lock()
        self.beat(event='start')
        th = threading.Thread(target=self._loop, daemon=True)
        th.start()

    def _loop(self):
        while True:
            time.sleep(BEAT_S)
            try:
                self.beat()
            except OSError:
                pass

    def beat(self, **kw):
        with self._lk:
            append_line(self.path, {**self.key, 't': time.time(), **kw})


def used_hours(extra_paths=()):
    per = {}
    for p in [CLOCK_PATH, *extra_paths]:
        for r in read_lines(p):
            k = (r.get('boot'), r.get('pid'), r.get('t0'))
            per[k] = max(per.get(k, 0.0), float(r['t']) - float(r['t0']))
    return sum(per.values()) / 3600.0


# ---------------------------------------------------------------- lexer stop rule (CPU)
def depth0_closings(code):
    """'}' tokens of the lexer stream (comments and literals excluded) that bring the brace depth
    back to 0."""
    d = n = 0
    for k, t, _ in C.lex(code):
        if k != 'punct':
            continue
        if t == '{':
            d += 1
        elif t == '}':
            d -= 1
            if d == 0:
                n += 1
    return n


# ---------------------------------------------------------------- e: signed influence (CPU, float64)
def compute_e(A_ff, L_rows, act, w, max_iter=MAX_ITER, first_only=False):
    """e = (sum_{m>=1} w L~ (A~)^(m-1)) * a on the feature block, float64. A_ff[i, j] = edge from
    selected feature j to feature i, L_rows[k, j] = edge from j to logit node k, act = the selected
    features' activations; A~ = columns divided by act. Returns (e, iterations), iterations as
    circuit_tracer.graph.compute_influence counts them (multiplications after the first term).
    first_only: the m = 1 term only (times a), iterations 0 (self-test of design §9 (a))."""
    act = np.asarray(act, dtype=np.float64)
    d = (np.asarray(w, dtype=np.float64) @ np.asarray(L_rows, dtype=np.float64)) / act
    if first_only:
        return d * act, 0
    At = np.asarray(A_ff, dtype=np.float64) / act[None, :]
    t = d.copy()
    cur = d
    it = 0
    while cur.any():
        if it >= max_iter:
            raise RuntimeError(f"e did not converge in {max_iter} iterations")
        cur = cur @ At
        t += cur
        it += 1
    e = t * act
    if not np.isfinite(e).all():
        raise RuntimeError("e has non-finite values")
    return e, it


def graph_e(graph):
    """e for node 0 and for the log-prob target, from a circuit_tracer Graph (CPU)."""
    import torch
    adj = graph.adjacency_matrix.cpu()
    n_sel = len(graph.selected_features)
    p = graph.logit_probabilities.detach().cpu().double().numpy()
    n_log = len(p)
    N = adj.shape[0]
    A_ff = adj[:n_sel, :n_sel].double().numpy()
    L = adj[N - n_log:, :n_sel].double().numpy()
    sel = graph.selected_features.cpu()
    act = graph.activation_values.cpu()[sel].float().double().numpy()
    w0 = np.zeros(n_log)
    w0[0] = 1.0
    e0, it0 = compute_e(A_ff, L, act, w0)
    wlp = -p.copy()
    wlp[0] = 1.0 - p[0]
    elp, it1 = compute_e(A_ff, L, act, wlp)
    del A_ff, L
    feats = graph.active_features.cpu()[sel].long().numpy()                      # [n_sel, 3]
    _ = torch
    return feats, act, e0, elp, max(it0, it1)


def ordered(feats, act, score, pos_min=1, layers=None):
    """Candidates with score > 0 at positions >= pos_min, descending (ties: lower index first):
    [(layer, pos, feat, act, score)]."""
    idx = np.nonzero((score > 0) & (feats[:, 1] >= pos_min))[0]
    if layers is not None:
        idx = idx[np.isin(feats[idx, 0], list(layers))]
    idx = idx[np.argsort(-score[idx], kind='stable')]
    return [(int(feats[i, 0]), int(feats[i, 1]), int(feats[i, 2]), float(act[i]), float(score[i])) for i in idx]


def top_rows(feats, act, score, n=TOP_OUT):
    idx = np.argsort(-score, kind='stable')[:n]
    return [[int(feats[i, 0]), int(feats[i, 1]), int(feats[i, 2]), float(act[i]), float(score[i])] for i in idx]


# ---------------------------------------------------------------- RND matching (CPU)
class Pool:
    """Active features by (layer, pos), each cell sorted by s = a * ||W_dec|| (desc, then feat)."""

    def __init__(self, feats, act, dec_norm, pos_min=1):
        self.cells = defaultdict(list)
        self.s, self.act = {}, {}
        for (l, p, f), a in zip(feats.tolist(), act.tolist()):
            if p < pos_min:
                continue
            s = float(a) * float(dec_norm[l][f])
            self.cells[(l, p)].append((-s, f))
            self.s[(l, p, f)] = s
            self.act[(l, p, f)] = float(a)
        for k in self.cells:
            self.cells[k].sort()
        self.layers = sorted({l for l, _ in self.cells})
        self.positions = sorted({p for _, p in self.cells})

    def _near(self, cell, s_ref, own, used):
        lst = self.cells.get(cell, [])
        if own is not None:
            r = next(i for i, (_, f) in enumerate(lst) if f == own)
        else:
            r = sum(1 for ns, _ in lst if -ns > s_ref)
        el = [(abs(i - r), i, f) for i, (_, f) in enumerate(lst)
              if (cell[0], cell[1], f) not in used and f != own]
        el.sort()
        return [f for _, _, f in el[:RND_NEAR]]

    def match(self, order, excluded, g):
        """One drawn feature per entry of order [(l, p, f, ...)]: from the RND_NEAR eligible features
        nearest in rank on s at the same (l, p); else the same pos at the nearest layer; else the
        same layer at the nearest pos. Eligible = not excluded, not drawn before, not the entry
        itself. Returns ([(l, p, f, a, s)], Counter of match kinds): a = the activation, s =
        a * ||W_dec|| (the matching quantity)."""
        used = set(excluded)
        out, kinds = [], Counter()
        for ent in order:
            l, p, f = ent[0], ent[1], ent[2]
            s_ref = self.s.get((l, p, f), 0.0)
            cand, kind, cell = self._near((l, p), s_ref, f, used), 'exact', (l, p)
            if not cand:
                for l2 in sorted(self.layers, key=lambda x: (abs(x - l), x)):
                    if l2 != l and (l2, p) in self.cells:
                        cand = self._near((l2, p), s_ref, None, used | {(l, p, f)})
                        if cand:
                            kind, cell = 'layer', (l2, p)
                            break
            if not cand:
                for p2 in sorted(self.positions, key=lambda x: (abs(x - p), x)):
                    if p2 != p and (l, p2) in self.cells:
                        cand = self._near((l, p2), s_ref, None, used | {(l, p, f)})
                        if cand:
                            kind, cell = 'pos', (l, p2)
                            break
            if not cand:
                kinds['none'] += 1
                continue
            f2 = cand[int(g.integers(len(cand)))]
            key = (cell[0], cell[1], f2)
            used.add(key)
            kinds[kind] += 1
            out.append((cell[0], cell[1], int(f2), self.act[key], self.s[key]))
        return out, kinds


# ---------------------------------------------------------------- the GPU engine
class EveryPos:
    """Ablate (layer, feat) pairs at every position >= 1 of whatever sequence is run (G_every)."""

    def __init__(self, pairs):
        self.pairs = sorted({(int(l), int(f)) for l, f in pairs})


_DET_WARN = set()


def _showwarning(message, category, filename, lineno, file=None, line=None):
    s = f"{category.__name__}: {str(message)[:200]}"
    if s not in _DET_WARN:
        _DET_WARN.add(s)
        print('WARNING', s, flush=True)


def gpu_lock():
    """Exclusive, non-blocking; exit 5 if another Phase 8 GPU process holds it."""
    os.makedirs(C.WORK, exist_ok=True)
    fh = open(os.path.join(C.WORK, 'gpu.lock'), 'w')
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('another Phase 8 GPU process holds the lock', flush=True)
        sys.exit(5)
    return fh


class Engine:
    def __init__(self, lock=None):
        import torch
        self.torch = torch
        self._lock = lock if lock is not None else gpu_lock()
        self.guard = None          # flip_scan / singles replay guard (replay_guard); set by main_run
        self.note = {}             # per-site notes for the ledger (e.g. from_checkpoint)
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
        need = float(os.environ.get('MIN_FREE_GIB', 9.5))
        if free < need:
            print(f'only {free:.2f} GiB free on the GPU (< {need}): another job is using it', flush=True)
            sys.exit(5)
        warnings.showwarning = _showwarning
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        import circuit_tracer
        if os.path.dirname(os.path.dirname(os.path.abspath(circuit_tracer.__file__))) == C.REPO:
            raise SystemExit("the repository's circuit_tracer/ was imported; run from data/ (the installed one built the graphs)")
        from circuit_tracer import ReplacementModel
        self.ct_version = getattr(circuit_tracer, '__version__', None) or _dist_version()
        self.model = ReplacementModel.from_pretrained(C.MODEL_NAME, C.TRANSCODER_NAME, dtype=torch.bfloat16).to('cuda')
        gc.collect()
        torch.cuda.empty_cache()
        m = self.model
        self.tok = m.tokenizer
        self.L, self.d, self.V = m.cfg.n_layers, m.cfg.d_model, m.cfg.d_vocab
        self.fin, self.fout = m.feature_input_hook, m.feature_output_hook
        zp = m.zero_positions
        self.zero_pos = set(range(zp.start or 0, zp.stop))
        self.WU = m.unembed.W_U                                  # [d, V] bf16
        bU = m.unembed.b_U
        self.bU = bU.float() if bU is not None else torch.zeros(self.V, device='cuda')
        self.w_ln = m.ln_final.w.float()
        self.WU_mean = self.WU.mean(dim=-1, keepdim=True)        # bf16, as AttributionTargets
        with torch.no_grad():
            self.dec_norm = np.stack([m.transcoders[l].W_dec.float().norm(dim=1).cpu().numpy()
                                      for l in range(self.L)])   # [L, d_tc] (lazy decoders read once)
        gc.collect()
        torch.cuda.empty_cache()
        self._dec = {}
        self.t_load = time.time()

    # -------------------------------------------------------- helpers
    def dec(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def u_vec(self, t):
        """The demeaned W_U column of token t (bf16, exactly as attribution's logit vectors)."""
        return (self.WU[:, t:t + 1] - self.WU_mean)[:, 0]

    def reset_site(self):
        self._dec = {}

    def dec_rows(self, layer, feats):
        """W_dec rows (bf16, cuda) for the given feature ids, via transcoders._get_decoder_vectors
        (the reader feature_intervention uses), cached per site."""
        torch = self.torch
        feats = feats.cpu().contiguous()
        need = torch.unique(feats)
        ent = self._dec.get(layer)
        missing = need if ent is None else need[~torch.isin(need, ent[0])]
        if len(missing):
            rows = self.model.transcoders._get_decoder_vectors(layer, missing)
            if ent is None:
                ids_all, rows_all = missing, rows
            else:
                ids_all, rows_all = torch.cat([ent[0], missing]), torch.cat([ent[1], rows])
            o = torch.argsort(ids_all)
            ent = (ids_all[o], rows_all[o.to(rows_all.device)])
            self._dec[layer] = ent
        idx = torch.searchsorted(ent[0], feats)
        return ent[1][idx.to(ent[1].device)]

    def device_check(self):
        bad = [n for n, p in self.model.named_parameters() if p.device.type != 'cuda']
        if bad:
            raise RuntimeError(f"parameters off the GPU: {bad[:5]} ({len(bad)})")

    # -------------------------------------------------------- the ablation hook
    def _plan(self, rows, n):
        torch = self.torch
        per = defaultdict(list)
        for b, abl in enumerate(rows):
            if isinstance(abl, EveryPos):
                for l, f in abl.pairs:
                    per[l].extend((b, p, f) for p in range(1, n))
            else:
                for e in abl:
                    per[int(e[0])].append((b, int(e[1]), int(e[2])))
        plan = {}
        for l in sorted(per):
            ent = sorted(set(per[l]))
            t = torch.tensor(ent, dtype=torch.long)
            bs = t[:, 0].numpy()
            ranges = [(int(np.searchsorted(bs, s, 'left')), int(np.searchsorted(bs, s + ENC_CHUNK, 'left')))
                      for s in range(0, len(rows), ENC_CHUNK)]
            zero = torch.tensor([p in self.zero_pos for p in t[:, 1].tolist()], dtype=torch.bool)
            plan[l] = {'t': t.cuda(), 'dec': self.dec_rows(l, t[:, 2]), 'ranges': ranges, 'zero': zero.cuda()}
        return plan

    def _enc_hook(self, acts, hook, P, store, l):
        torch = self.torch
        t = P['t']
        vals = torch.zeros(len(t), dtype=self.model.cfg.dtype, device=acts.device)
        B = acts.shape[0]
        for ci, s in enumerate(range(0, B, ENC_CHUNK)):
            lo, hi = P['ranges'][ci]
            if hi == lo:
                continue
            enc = self.model.transcoders.encode_layer(acts[s:s + ENC_CHUNK], l)
            tt = t[lo:hi]
            vals[lo:hi] = enc[tt[:, 0] - s, tt[:, 1], tt[:, 2]]
            del enc
        vals[P['zero']] = 0
        store[l] = vals
        return acts

    def _out_hook(self, acts, hook, P, store, l, B, n):
        torch = self.torch
        a = store.pop(l)
        keep = (a != 0).nonzero().squeeze(1)
        delta = torch.zeros(B * n, self.d, dtype=self.model.cfg.dtype, device=acts.device)
        if len(keep):
            t = P['t'][keep]
            vecs = P['dec'][keep] * (0.0 - a[keep]).unsqueeze(1)
            delta.index_add_(0, t[:, 0] * n + t[:, 1], vecs)
        return acts + delta.view(B, n, self.d)

    def run(self, ids, rows, positions=None, want_scale=False, null_layers=()):
        """Forward ids (list) once per row of rows (each a list of (layer, pos, feat, ...) to zero,
        or an EveryPos). Returns {'n': ln_final.hook_normalized at the last position [B, d] fp32
        (or at `positions`, [B, m, d]), 's': hook_scale at the last position}."""
        torch = self.torch
        n, B = len(ids), len(rows)
        tokens = torch.tensor(ids, dtype=torch.long, device='cuda').unsqueeze(0).expand(B, -1)
        plan = self._plan(rows, n)
        for l in null_layers:                                   # P0 (c): hooks that add zeros
            if l not in plan:
                plan[l] = {'t': torch.zeros((0, 3), dtype=torch.long, device='cuda'),
                           'dec': torch.zeros((0, self.d), dtype=self.model.cfg.dtype, device='cuda'),
                           'ranges': [(0, 0)] * len(range(0, B, ENC_CHUNK)),
                           'zero': torch.zeros(0, dtype=torch.bool, device='cuda')}
        store, cap = {}, {}
        hooks = []
        for l, P in plan.items():
            hooks.append((f'blocks.{l}.{self.fin}', partial(self._enc_hook, P=P, store=store, l=l)))
            hooks.append((f'blocks.{l}.{self.fout}', partial(self._out_hook, P=P, store=store, l=l, B=B, n=n)))

        def cap_n(a, hook):
            cap['n'] = (a[:, -1] if positions is None else a[:, positions]).clone()

        def cap_s(a, hook):
            cap['s'] = a[:, -1, 0].clone()

        hooks.append(('ln_final.hook_normalized', cap_n))
        if want_scale:
            hooks.append(('ln_final.hook_scale', cap_s))
        with torch.no_grad(), self.model.hooks(fwd_hooks=hooks):
            resid = self.model(tokens, stop_at_layer=self.L)
            self.model.ln_final(resid)
        del resid
        return cap

    # -------------------------------------------------------- logits
    def full_argmax(self, norm):
        """fp32 soft-capped logits over the whole vocabulary, chunked; the first maximum. norm [..., d]."""
        torch = self.torch
        x = (norm * self.w_ln).reshape(-1, self.d)
        best_v = torch.full((x.shape[0],), -float('inf'), device=x.device)
        best_i = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for s in range(0, self.V, 32768):
            z = x @ self.WU[:, s:s + 32768].float() + self.bU[s:s + 32768]
            z = C.SOFTCAP * torch.tanh(z / C.SOFTCAP)
            i = z.argmax(dim=1)
            v = z.gather(1, i[:, None])[:, 0]
            better = v > best_v
            best_v = torch.where(better, v, best_v)
            best_i = torch.where(better, i + s, best_i)
        return best_i.reshape(norm.shape[:-1])

    def site(self, prefix, Eset):
        """Clean pass at the decision position: C, the fp32 decision, the full-vocabulary argmax."""
        torch = self.torch
        S = Site()
        S.prefix, S.E = [int(x) for x in prefix], set(int(x) for x in Eset)
        self.reset_site()
        with torch.no_grad():
            lg = self.model(torch.tensor(S.prefix, device='cuda')[None])[0, -1].float().cpu().numpy()
        S.tl_argmax = int(np.argmax(lg))
        top = np.lexsort((np.arange(len(lg)), -lg))[:C.TOP_C]
        S.C = sorted(set(int(x) for x in top) | S.E)
        S.Ct = torch.tensor(S.C, device='cuda')
        S.inE = torch.tensor([c in S.E for c in S.C], device='cuda')
        S.WU_C = self.WU[:, S.Ct].float()
        S.bU_C = self.bU[S.Ct]
        S.u0 = None
        cap = self.run(S.prefix, [[]], want_scale=True)
        S.norm0, S.scale = cap['n'][0], float(cap['s'][0])
        S.clean = self.decide(S, [[]], cap=cap)
        S.full_argmax = int(self.full_argmax(cap['n'])[0])
        return S

    def decide(self, S, rows, cap=None):
        torch = self.torch
        if cap is None:
            cap = self.run(S.prefix, rows)
        x = cap['n'] * self.w_ln
        z = C.SOFTCAP * torch.tanh((x @ S.WU_C + S.bU_C) / C.SOFTCAP)
        am = z.argmax(dim=1)
        ninf = torch.tensor(-float('inf'), device=z.device)
        margin = torch.where(S.inE, z, ninf).max(1).values - torch.where(S.inE, ninf, z).max(1).values
        out = {'z': z.cpu().numpy(), 'am': S.Ct[am].tolist(), 'flip': (~S.inE[am]).tolist(),
               'margin': margin.tolist(), 'norm': cap['n']}
        if S.u0 is not None:
            out['target'] = (x @ S.u0).tolist()
            if 's' in cap:                      # the same target with ln_final's scale held at its clean value
                out['target_fs'] = ((x * (cap['s'] / S.scale)[:, None]) @ S.u0).tolist()
        return out

    # -------------------------------------------------------- arms
    def _guard(self):
        if self.guard is None:
            raise RuntimeError('Engine.guard (replay_guard) is not set')
        return self.guard

    def flip_scan(self, S, arm, order, K=None, fault=None):
        """The flip rule for every ordered arm; order = [(l, p, f, a, ...)]. Each screened row whose
        screen says flip, or whose screened margin is below the guard (replay_guard: at least twice
        P0 (b)'s largest batch-64 margin deviation), is replayed at batch 1 in k order; the first
        replayed flip is k*. Rows above the guard are taken as no flip without replay."""
        K = min(C.K_MAX if K is None else K, len(order))
        guard = self._guard()
        disagree = 0
        for k0 in range(0, K, C.SCAN_BATCH):
            ks = list(range(k0 + 1, min(k0 + C.SCAN_BATCH, K) + 1))
            r = self.decide(S, [order[:k] for k in ks])
            if fault and k0 == 0:
                fault('mid_scan')
            near = [i for i in range(len(ks)) if r['flip'][i] or r['margin'][i] < guard]
            if not near:
                continue
            first = None
            for i in near:
                k = ks[i]
                r1 = r if len(ks) == 1 else self.decide(S, [order[:k]])
                if r1['flip'][0] != r['flip'][i]:
                    disagree += 1
                if r1['flip'][0]:
                    first = (k, r1)
                    break
            if first is not None:
                k, r1 = first
                t = int(r1['am'][0])
                abl = order[:k]
                return self.flip_rec(arm, True, k, t, r1['margin'][0], disagree, len(order), K, abl,
                                     self.completion(S, abl, t), int(self.full_argmax(r1['norm'])[0]))
        return self.flip_rec(arm, False, None, None, None, disagree, len(order), K, [], None, None)

    def fixed_flip(self, S, arm):
        """NULL (no intervention) and BAN (argmax over C minus E), k_star 0."""
        z = S.clean['z'][0]
        if arm == 'NULL':
            t = S.clean['am'][0]
            flipped = t not in S.E
        else:
            mask = np.array([c not in S.E for c in S.C])
            t = S.C[int(np.argmax(np.where(mask, z, -np.inf)))]
            flipped = True
        comp = self.completion(S, [], t) if flipped else None
        return self.flip_rec(arm, flipped, 0, int(t) if flipped else None, S.clean['margin'][0], 0, 0, 0, [],
                             comp, S.full_argmax if arm == 'NULL' else None)

    def flip_rec(self, arm, flipped, k, t, margin, disagree, n_cand, K, abl, completion, full_t):
        return {'arm': arm, 'flipped': bool(flipped), 'k_star': k, 't_star_id': t,
                't_star_piece': self.dec([t]) if t is not None else None,
                'margin_at_kstar': None if margin is None else float(margin),
                'scan_replay_disagree': int(disagree), 'n_candidates': int(n_cand), 'K': int(K),
                'ablated': [[int(e[0]), int(e[1]), int(e[2]), float(e[3])] for e in abl],
                'completion': completion, 'fullvocab_t': full_t}

    def completion(self, S, abl, t):
        """Greedy text after t* (full-vocabulary fp32 argmax) under abl at the prefix positions,
        until a newline, <eos>/<end_of_turn>, or GEN_LINE_CAP tokens."""
        head = self.dec(S.prefix[1:] + [t])
        seq, gen = S.prefix + [t], []
        if '\n' in head[len(self.dec(S.prefix[1:])):]:
            return ''
        comp = ''
        for _ in range(C.GEN_LINE_CAP):
            nxt = int(self.full_argmax(self.run(seq, [abl])['n'])[0])
            seq.append(nxt)
            gen.append(nxt)
            full = self.dec(S.prefix[1:] + [t] + gen)
            comp = full[len(head):] if full.startswith(head) else self.dec(gen)
            if '\n' in comp or nxt in EOS_IDS:
                break
        return comp

    def singles(self, S, feats, batch=C.SCAN_BATCH):
        """Each feature ablated alone. batch > 1: forwards of one clean row + (batch - 1) single
        ablations, effects measured against that in-batch clean row (the batch offset cancels);
        batch 1: each at batch 1 against the batch-1 clean pass. Effects: drop = margin decrease,
        dtarget = node-0 target decrease, dtarget_fs = the same with ln_final's scale held clean.
        A screened flip, or (batch > 1) a screened margin below the guard, is replayed at batch 1
        (flip, t; flip stays None for rows not replayed)."""
        guard = self._guard() if batch > 1 else None
        out = []
        per = max(1, batch - 1) if batch > 1 else 1
        ref1 = None
        if batch == 1:
            ref1 = self.decide(S, [[]], cap=self.run(S.prefix, [[]], want_scale=True))
        for s in range(0, len(feats), per):
            chunk = feats[s:s + per]
            if batch > 1:
                r = self.decide(S, [[]] + [[f] for f in chunk], cap=self.run(S.prefix, [[]] + [[f] for f in chunk],
                                                                             want_scale=True))
                ref, off = 0, 1
            else:
                r = self.decide(S, [[chunk[0]]], cap=self.run(S.prefix, [[chunk[0]]], want_scale=True))
                ref, off = None, 0
            for i, f in enumerate(chunk):
                j = i + off
                base = r if ref is not None else ref1
                b = ref if ref is not None else 0
                d = {'f': [int(f[0]), int(f[1]), int(f[2])],
                     'drop': float(base['margin'][b] - r['margin'][j]),
                     'dtarget': float(base['target'][b] - r['target'][j]) if 'target' in r else None,
                     'dtarget_fs': float(base['target_fs'][b] - r['target_fs'][j]) if 'target_fs' in r else None,
                     'screen_t': int(r['am'][j]), 'flip': None, 't': None}
                if r['flip'][j] or (guard is not None and r['margin'][j] < guard):
                    r1 = r if batch == 1 else self.decide(S, [[f]])
                    d['flip'] = bool(r1['flip'][0])
                    d['t'] = int(r1['am'][0]) if r1['flip'][0] else None
                out.append(d)
        return out

    def logit_rows(self, prefix, inject=None, de_g=None):
        """Mirror of attribute()'s phases 0-3 (installed attribute_transformerlens): setup_attribution,
        the batch-GRAPH_KW['batch_size'] forward under ctx.install_hooks with the same offloading,
        AttributionTargets, then ONE compute_batch over the logit vectors (or `inject` [r, d]).
        Returns the active features, their activations, the rows (fp32, CPU; feature part, error
        and token parts), the logit targets and, if de_g is given, the direct effect
        decoder_vec . de_g of the active features at the last position."""
        torch = self.torch
        from circuit_tracer.attribution.targets import AttributionTargets
        from circuit_tracer.utils.disk_offload import offload_modules
        m = self.model
        kw = C.GRAPH_KW
        handles = []
        torch.cuda.empty_cache()
        try:
            input_ids = m.ensure_tokenized(list(prefix))
            ctx = m.setup_attribution(input_ids)
            am = ctx.activation_matrix
            if kw['offload']:
                handles += offload_modules(m.transcoders, kw['offload'])
            with ctx.install_hooks(m):
                residual = m.forward(input_ids.expand(kw['batch_size'], -1), stop_at_layer=m.cfg.n_layers)
                ctx._resid_activations[-1] = m.ln_final(residual)
            if kw['offload']:
                handles += offload_modules([block.mlp for block in m.blocks], kw['offload'])
            n_layers, n_pos, _ = am.shape
            targets = AttributionTargets(attribution_targets=None, logits=ctx.logits[0, -1],
                                         unembed_proj=m.unembed.W_U, tokenizer=m.tokenizer,
                                         max_n_logits=kw['max_n_logits'], desired_logit_prob=kw['desired_logit_prob'])
            if kw['offload']:
                handles += offload_modules([m.unembed, m.embed], kw['offload'])
            vecs = targets.logit_vectors if inject is None else inject
            rows = ctx.compute_batch(layers=torch.full((vecs.shape[0],), n_layers),
                                     positions=torch.full((vecs.shape[0],), n_pos - 1),
                                     inject_values=vecs).float().cpu()
            idx = am.indices().T.cpu().numpy()
            vals = am.values().float().cpu().numpy()
            de = None
            if de_g is not None:
                last = np.nonzero(idx[:, 1] == n_pos - 1)[0]
                de = np.zeros(len(vals))
                if len(last):
                    dv = ctx.decoder_vecs[torch.tensor(last, device=ctx.decoder_vecs.device)].float()
                    de[last] = (dv @ de_g).double().cpu().numpy()
            out = {'feats': idx, 'acts': vals, 'rows': rows, 'nnz': len(vals),
                   'logit_ids': [int(t.vocab_idx) for t in targets.logit_targets],
                   'logit_p': targets.logit_probabilities.float().cpu().tolist(), 'de': de}
            del ctx, residual, am, targets, rows
        finally:
            for h in handles:
                h()
        gc.collect()
        torch.cuda.empty_cache()
        return out

    def generate(self, S, abl, first, cap, depth_S, prose):
        """Greedy program from ids[:h] + [first]; stop: depth-0 '}' count of the code reaches S's,
        <eos>/<end_of_turn>, or `cap` generated tokens."""
        pre = self.dec(S.prefix[1:])
        head = prose + ' '
        seq, gen = S.prefix + [first], [first]
        while True:
            text = self.dec(seq[1:])
            if gen[-1] in EOS_IDS:
                stopped = 'eos'
                break
            if text.startswith(head) and depth0_closings(text[len(head):]) >= depth_S:
                stopped = 'braces'
                break
            if len(gen) >= cap:
                stopped = 'cap'
                break
            nxt = int(self.full_argmax(self.run(seq, [abl])['n'])[0])
            seq.append(nxt)
            gen.append(nxt)
        full = self.dec(S.prefix[1:] + gen)
        return {'text': full[len(pre):] if full.startswith(pre) else self.dec(gen), 'tokens': len(gen),
                'stopped': stopped}


class Site:
    pass


def _dist_version():
    try:
        from importlib.metadata import version
        return version('circuit-tracer')
    except Exception:
        return None


# ---------------------------------------------------------------- stages
def decision_record(S):
    z = S.clean['z'][0]
    o = np.lexsort((np.array(S.C), -z))[:10]
    return {'top': [[int(S.C[i]), float(z[i])] for i in o], 'wrong_in_E_top': S.clean['am'][0] in S.E,
            'argmax': int(S.clean['am'][0]), 'margin': float(S.clean['margin'][0]), 'C': S.C,
            'tl_argmax': S.tl_argmax, 'fullvocab_argmax': S.full_argmax, 'scale': S.scale}


def stage_A(E, s, delta, fault):
    sid, kind = s['site_id'], s['kind']
    out = {'site_id': sid, 'kind': kind, 'excluded': None}
    ids, h = [int(x) for x in s['ids']], int(s['h'])
    Eset = set(int(x) for x in s['E'])
    if E.tok(s['prose'] + ' ' + s['code']).input_ids != ids or ids[h] != int(s['wrong_id']) \
            or int(s['wrong_id']) not in Eset:
        out['excluded'] = 'b: the prefix ids differ from the site table (or wrong_id / E inconsistent)'
        return out
    S = E.site(ids[:h], Eset)
    out['decision'] = decision_record(S)
    if kind == 'slip':
        out['NULL'] = E.fixed_flip(S, 'NULL')
        return out
    if S.clean['am'][0] not in S.E:
        out['excluded'] = 'a: the fp32 argmax at h-1 is not in E(wrong)'
        return out
    if S.clean['margin'][0] < 2 * delta:
        out['excluded'] = f"a: best E logit leads the best non-E logit by {S.clean['margin'][0]:.6g} < 2 delta"
        return out
    u = E.u_vec(int(s['wrong_id'])).float()
    gx = E.logit_rows(S.prefix, de_g=E.w_ln * u / S.scale)
    node0 = gx['logit_ids'][0]
    out['gxa'] = {'node0_id': node0, 'logit_ids': gx['logit_ids'], 'logit_p': gx['logit_p'], 'n_active': gx['nnz']}
    if node0 not in S.E:
        out['excluded'] = 'd: logit node 0 is not in E(wrong)'
        return out
    S.u0 = E.u_vec(node0).float()
    feats, acts = gx['feats'], gx['acts']
    score = gx['rows'][0, :gx['nnz']].double().numpy()
    out['NULL'] = E.fixed_flip(S, 'NULL')
    out['BAN'] = E.fixed_flip(S, 'BAN')
    out['GxA'] = E.flip_scan(S, 'GxA', ordered(feats, acts, score), fault=fault)
    out['DE'] = E.flip_scan(S, 'DE', ordered(feats, acts, gx['de']))
    out['gxa_top'] = top_rows(feats, acts, score)
    out['de_n_candidates'] = int((gx['de'] > 0).sum())
    return out


def stage_B(E, s, A, delta, fault, ckpt):
    """Two parts. Primary (graph, e, G, G_LP, G1, G_mid, RND1-RND5, RNDk) is computed once and saved
    to `ckpt` (atomic) before the descriptive part (singles, ORC, existence, faithfulness) starts; a
    restarted process reuses a checkpoint instead of rebuilding the graph (the primary part is
    deterministic, so its bytes are the same either way). If the descriptive part fails in two
    fresh processes, main_run writes the primary part with 'descriptive_failed' (stage_B_primary_only)."""
    sid = s['site_id']
    ids, h = [int(x) for x in s['ids']], int(s['h'])
    S = E.site(ids[:h], set(int(x) for x in s['E']))
    if S.C != A['decision']['C'] or S.clean['am'][0] != A['decision']['argmax']:
        raise RuntimeError('the clean decision differs from stage A (non-determinism)')
    if os.path.exists(ckpt):
        with open(ckpt, encoding='utf-8') as f:
            out = json.load(f)
        E.note['from_checkpoint'] = True
    else:
        out = stage_B_primary(E, s, A, S, fault)
        C.write_json(ckpt, out)
        E.note['from_checkpoint'] = False
    if out.get('excluded'):
        out.pop('_work', None)
        return out
    if fault:
        fault('mid_singles')
    S.u0 = E.u_vec(int(out['graph']['node0_id'])).float()
    S.clean = E.decide(S, [[]])
    w = out.pop('_work')
    G = [tuple(x) for x in w['G_top']]
    gxa = [tuple(r) for r in A['gxa_top'] if r[4] > 0 and r[1] >= 1]
    rnd_single = [tuple(x) for x in w['rnd_single']]
    out.update(stage_B_descriptive(E, S, G, gxa, rnd_single, w['rnd_single_kind'], delta))
    return out


def stage_B_primary(E, s, A, S, fault):
    torch = E.torch
    sid = s['site_id']
    out = {'site_id': sid, 'excluded': None}
    from circuit_tracer import attribute
    torch.cuda.empty_cache()
    t0 = time.time()
    try:
        if fault:
            fault('attribute')
        graph = attribute(prompt=S.prefix, model=E.model, **C.GRAPH_KW)
    except Exception as exc:
        raise RuntimeError(f'attribute() failed: {type(exc).__name__}: {exc}') from exc
    t_graph = time.time() - t0
    gc.collect()
    torch.cuda.empty_cache()
    E.device_check()
    if graph.input_tokens.tolist() != S.prefix:
        raise RuntimeError('graph input tokens differ from the prefix')
    logit_ids = graph.logit_token_ids.tolist()
    feats_sel, act_sel, e0, elp, iters = graph_e(graph)
    pool = Pool(graph.active_features.cpu().long().numpy(), graph.activation_values.cpu().float().numpy(), E.dec_norm)
    out['graph'] = {'n_sel': int(len(graph.selected_features)), 'n_logits': len(logit_ids), 'logit_ids': logit_ids,
                    'logit_p': graph.logit_probabilities.detach().cpu().float().tolist(), 'node0_id': int(logit_ids[0]),
                    'n_active': int(len(graph.activation_values)), 'e_iterations': int(iters), 'seconds': t_graph}
    del graph
    gc.collect()
    torch.cuda.empty_cache()
    if fault:
        fault('after_graph')
    if logit_ids[0] not in S.E:
        out['excluded'] = 'd: logit node 0 of the graph is not in E(wrong)'
        return out
    S.u0 = E.u_vec(int(logit_ids[0])).float()
    S.clean = E.decide(S, [[]])
    out['e_top'] = top_rows(feats_sel, act_sel, e0)
    G = ordered(feats_sel, act_sel, e0)
    out['G'] = E.flip_scan(S, 'G', G, fault=fault)
    out['G_LP'] = E.flip_scan(S, 'G_LP', ordered(feats_sel, act_sel, elp))
    out['G1'] = E.flip_scan(S, 'G1', G, K=1)
    out['G_mid'] = E.flip_scan(S, 'G_mid', [x for x in G if x[0] in C.MID_LAYERS])
    gxa = [tuple(r) for r in A['gxa_top'] if r[4] > 0 and r[1] >= 1]
    excl = {tuple(x[:3]) for x in G[:RND_EXCL]} | {tuple(x[:3]) for x in gxa[:RND_EXCL]}
    key = site_key(sid)
    match = {}
    for dr in range(1, RND_DRAWS + 1):
        order, kinds = pool.match(G[:C.K_MAX], excl, C.rng(2, *key, dr))
        match[f'RND{dr}'] = dict(kinds)
        rec = E.flip_scan(S, f'RND{dr}', order)
        rec['kind'] = dict(kinds)
        out[f'RND{dr}'] = rec
    rndk = []
    if out['G']['flipped']:
        kG = out['G']['k_star']
        for dr in range(1, RNDK_DRAWS + 1):
            order, kinds = pool.match(G[:kG], excl, C.rng(11, *key, dr))
            r = E.decide(S, [order])
            rndk.append({'draw': dr, 't_star_id': int(r['am'][0]), 'changed': bool(r['flip'][0]),
                         'margin': float(r['margin'][0]), 'n': len(order), 'kind': dict(kinds),
                         'ablated': [[int(x[0]), int(x[1]), int(x[2]), float(x[3])] for x in order]})
    out['RNDk'] = rndk
    # what the descriptive part needs (the graph is gone by then); dropped from the stage file
    rnd_single, rkinds = pool.match(G[:EXIST_RND], excl, C.rng(10, *key))
    match['existence_random'] = dict(rkinds)
    out['rnd_match'] = match
    out['_work'] = {'G_top': [list(x) for x in G[:max(EXIST_G, FAITH_N, EXIST_RND)]],
                    'rnd_single': [list(x) for x in rnd_single], 'rnd_single_kind': dict(rkinds)}
    return out


def stage_B_descriptive(E, S, G, gxa, rnd_single, rkinds, delta):
    """Single ablations: GxA's top ORC_N and G's top EXIST_G (screen batch SCAN_BATCH, rows below
    the guard replayed at batch 1), EXIST_RND matched random features; ORC; existence;
    faithfulness = Spearman(e, dtarget) over G's top FAITH_N with |e| > 3 delta, measured at batch 1
    (FAITH_BATCH, as P0 (a) measured it; the batch-SCAN_BATCH value is kept as spearman_screen)."""
    out = {}
    orc_set = gxa[:ORC_N]
    todo, seen = [], set()
    for x in orc_set + G[:max(EXIST_G, FAITH_N)]:
        if tuple(x[:3]) not in seen:
            seen.add(tuple(x[:3]))
            todo.append(x)
    t1 = time.time()
    meas = {tuple(d['f']): d for d in E.singles(S, todo)}
    rmeas = E.singles(S, rnd_single)
    t_single = time.time() - t1
    orc_order = sorted(range(len(orc_set)), key=lambda i: (-meas[tuple(orc_set[i][:3])]['drop'], i))
    out['ORC'] = E.flip_scan(S, 'ORC', [orc_set[i] for i in orc_order])
    ex_pool = [meas[tuple(x[:3])] for x in orc_set] + [meas[tuple(x[:3])] for x in G[:EXIST_G]]
    ex_flips = sorted({tuple(d['f']) + (d['t'],) for d in ex_pool if d['flip']})
    out['existence'] = {'n_single': len({tuple(d['f']) for d in ex_pool}), 'flips': [list(x) for x in ex_flips],
                        'random_n': len(rmeas), 'random_flips': [d['f'] + [d['t']] for d in rmeas if d['flip']],
                        'random_kind': dict(rkinds), 'seconds_singles': t_single, 'screen_batch': C.SCAN_BATCH,
                        'replay_guard': E.guard}
    from scipy.stats import spearmanr
    fsel = [x for x in G[:FAITH_N] if abs(x[4]) > 3 * delta]
    t2 = time.time()
    fmeas = {tuple(d['f']): d for d in E.singles(S, fsel, batch=FAITH_BATCH)} if fsel else {}
    t_faith = time.time() - t2

    def rho_of(pairs):
        return float(spearmanr([a for a, _ in pairs], [b for _, b in pairs])[0]) if len(pairs) >= 3 else None

    fe = [(x[4], fmeas[tuple(x[:3])]['dtarget']) for x in fsel]
    fe2 = [(x[4], fmeas[tuple(x[:3])]['dtarget_fs']) for x in fsel]
    fes = [(x[4], meas[tuple(x[:3])]['dtarget']) for x in fsel]
    out['faithfulness'] = {'spearman': rho_of(fe), 'n': len(fe), 'batch': FAITH_BATCH,
                           'spearman_final_scale_frozen': rho_of(fe2), 'spearman_screen': rho_of(fes),
                           'sign_agree': (sum(1 for a, b in fe if np.sign(a) == np.sign(b)) / len(fe)) if fe else None,
                           'seconds': t_faith}
    out['singles'] = [[*d['f'], d['drop'], d['dtarget'], d['dtarget_fs'], d['screen_t'], d['t']]
                      for d in list(meas.values()) + rmeas]       # l, p, f, drop, dtarget, dtarget_fs, screen t, replayed flip t
    out['settings'] = run_settings()
    return out


def stage_B_primary_only(ckpt, why):
    """The stage-B file when the descriptive part failed in two fresh processes: the primary arms
    stand; ORC, existence, faithfulness and singles are null (p8_3: ORC undefined)."""
    with open(ckpt, encoding='utf-8') as f:
        out = json.load(f)
    out.pop('_work', None)
    out.update({'ORC': None, 'existence': None, 'faithfulness': None, 'singles': None,
                'orc_note': f'descriptive part failed in two fresh processes ({why[:300]})',
                'descriptive_failed': why[:300], 'from_checkpoint': True, 'settings': run_settings()})
    return out


def best_repair(E, s, S):
    """Label-using (stage C only): the judged mutant with label 'verified' whose token at h has the
    highest clean fp32 logit; ties by token id. Only for kind 'pilot' rows (no verified mutants)
    the fallback 'unverifiable', then 'lucky' (stable passes) applies; a belief site without a usable
    verified repair gets None (CONTRAST and collateral null). Returns dict or None."""
    torch = E.torch
    t, smp, pos = site_key(s['site_id'])
    frm = s['frm']
    code = s['code']
    if code[pos:pos + len(frm)] != frm:
        return None
    tiers = (('verified',), ('unverifiable', 'lucky')) if s['kind'] == 'pilot' else (('verified',),)
    for labels in tiers:
        cands = []
        for k, v in sorted(s.get('judged', {}).items()):
            if v.get('label') not in labels:
                continue
            code_r = code[:pos] + v['to'] + code[pos + len(frm):]
            ids_r = E.tok(s['prose'] + ' ' + code_r).input_ids
            h = int(s['h'])
            if ids_r[:h] != S.prefix or len(ids_r) <= h:
                continue
            tid = ids_r[h]
            z = float(C.SOFTCAP * torch.tanh(((S.norm0 * E.w_ln) @ E.WU[:, tid].float() + E.bU[tid]) / C.SOFTCAP))
            cands.append((-z, tid, v['to'], ids_r, v['label']))
        if cands:
            cands.sort(key=lambda c: (c[0], c[1], c[2]))
            z, tid, to, ids_r, lab = cands[0]
            return {'to': to, 'token': int(tid), 'logit': -z, 'ids': ids_r, 'label': lab, 'n_candidates': len(cands)}
    return None


def stage_C(E, s, A, B, fault):
    torch = E.torch
    sid = s['site_id']
    out = {'site_id': sid, 'excluded': None}
    ids, h = [int(x) for x in s['ids']], int(s['h'])
    S = E.site(ids[:h], set(int(x) for x in s['E']))
    if S.C != A['decision']['C']:
        raise RuntimeError('the clean decision differs from stage A (non-determinism)')
    S.u0 = E.u_vec(int(B['graph']['node0_id'])).float()
    code = s['code']
    depth_S = depth0_closings(code)
    tte = s.get('tokens_to_end')
    if tte is None:
        tte = len(ids) - 1 - h
    cap = int(tte) + C.GEN_EXTRA
    # G's ablation: k* features if G flipped, else the K features it scanned
    if B['G']['flipped']:
        g_abl = [tuple(x) for x in B['G']['ablated']]
    else:
        g_abl = [(r[0], r[1], r[2], r[3]) for r in B['e_top'] if r[4] > 0 and r[1] >= 1][:B['G']['K']]
    every = EveryPos([(x[0], x[2]) for x in g_abl])
    gen = {}
    r = E.decide(S, [g_abl])
    first_G = int(r['am'][0])
    gen['G'] = E.generate(S, g_abl, first_G, cap, depth_S, s['prose'])
    gen['G']['first_matches_t_star'] = (first_G == B['G']['t_star_id']) if B['G']['flipped'] else None
    if fault:
        fault('mid_scan')
    r = E.run(S.prefix, [every])
    first_E = int(E.decide(S, None, cap=r)['am'][0])
    gen['G_every'] = E.generate(S, every, first_E, cap, depth_S, s['prose'])
    gen['BAN'] = E.generate(S, [], int(A['BAN']['t_star_id']), cap, depth_S, s['prose'])
    gen['NULL'] = E.generate(S, [], int(S.clean['am'][0]), cap, depth_S, s['prose'])
    out['gen'] = gen
    out['gen_cap'] = cap
    # ---- label-using work (after every label-free arm)
    rep = best_repair(E, s, S)
    if rep is None:
        out['collateral'] = None
        out['CONTRAST'] = None
        out['contrast_note'] = 'no usable repair (verified; pilot rows: else unverifiable or lucky) whose prefix tokens equal ids[:h]'
        return out
    rows = [[], g_abl] + [[tuple(x) for x in d['ablated']] for d in B['RNDk']]
    ids_r = rep['ids']
    cap_r = E.run(ids_r, rows, positions=slice(h, None))
    am = E.full_argmax(cap_r['n']).cpu().numpy()                 # [rows, positions]
    ch = (am != am[0:1]).sum(axis=1)
    out['collateral'] = {'repair_to': rep['to'], 'repair_label': rep['label'], 'repair_token': rep['token'],
                         'positions': int(am.shape[1]), 'G': int(ch[1]), 'G_k': len(g_abl),
                         'RNDk': [int(x) for x in ch[2:]], 'teacher_agree_clean': int((am[0][:-1] == np.array(ids_r[h + 1:])).sum())}
    v = (E.WU[:, int(s['wrong_id'])].float() - E.WU[:, rep['token']].float()).to(E.model.cfg.dtype)[None]
    gx = E.logit_rows(S.prefix, inject=v)
    score = gx['rows'][0, :gx['nnz']].double().numpy()
    out['CONTRAST'] = E.flip_scan(S, 'CONTRAST', ordered(gx['feats'], gx['acts'], score))
    out['contrast_repair'] = {'to': rep['to'], 'token': rep['token'], 'label': rep['label'],
                              'n_candidates': rep['n_candidates']}
    del torch
    return out


# ---------------------------------------------------------------- driver
def stop(msg, code=4):
    """Exit for a person: 4 = configuration / P0 / canary problem, 6 = a site needs a decision."""
    print(msg, flush=True)
    sys.exit(code)


def load_site_table():
    """SITES must be named explicitly (no default: a forgotten SITES must not reach the 35 site
    problems). Kinds must be belief, slip or pilot."""
    if not os.environ.get('SITES'):
        stop('SITES is not set (no default): SITES=pilot_sites.jsonl for P1, SITES=sites.jsonl for the main run')
    p = _out('SITES', '')
    rows = C.read_jsonl(p)
    bad = sorted({str(r.get('kind')) for r in rows} - set(KINDS))
    if bad:
        stop(f'site table {p}: unknown kinds {bad}')
    order = sorted(range(len(rows)), key=lambda i: (rows[i].get('order', i), i))
    return [rows[i] for i in order], p


def _git(*args):
    import subprocess
    return subprocess.run(['git', '-C', C.REPO, *args], capture_output=True, text=True)


def prereg_marker(sites):
    """Hard rule: no GPU on the 35 site problems before PREREGISTRATION_P8 is committed. Rows of
    kind belief/slip run only if every PREREGISTRATION_P8*.md at the repository root is tracked by
    git, unmodified against HEAD and has a commit. Pilot-only tables need no marker (returns None)."""
    if {s['kind'] for s in sites} <= {'pilot'}:
        return None
    files = sorted(glob.glob(os.path.join(C.REPO, 'PREREGISTRATION_P8*.md')))
    if not files:
        stop('belief/slip sites need a committed PREREGISTRATION_P8*.md at the repository root; none exists')
    out = []
    for f in files:
        rel = os.path.relpath(f, C.REPO)
        tracked = _git('ls-files', '--error-unmatch', rel).returncode == 0
        clean = _git('diff', '--quiet', 'HEAD', '--', rel).returncode == 0
        commit = _git('log', '-1', '--format=%H', '--', rel).stdout.strip()
        if not (tracked and clean and commit):
            stop(f'{rel}: not committed or modified since its commit (tracked {tracked}, clean {clean}); '
                 'belief/slip sites refused')
        out.append({'file': rel, 'sha256': C.sha256(f), 'commit': commit})
    return out


def site_attempts(led, stage, boot):
    """Per site of this stage: {'errors': [ledger error rows], 'deaths': n, 'reboots': n}. An
    attempt = a 'start' row. It is an error if its process logged an 'error' for the site; a death
    if the process left neither 'error' nor 'done' and started in this boot (killed from outside
    or crashed hard: usage limit, kill, host OOM); a reboot if it started in an earlier boot
    (never counted). Starts without a boot field (old ledgers) count as deaths."""
    ended = {}
    for r in led:
        if r.get('stage') == stage and r.get('event') in ('error', 'done') and 'site_id' in r:
            ended.setdefault((r.get('boot'), r.get('pid'), r['site_id']), r)
    out = defaultdict(lambda: {'errors': [], 'deaths': 0, 'reboots': 0})
    for r in led:
        if r.get('stage') != stage or r.get('event') != 'start':
            continue
        sid = r['site_id']
        e = ended.get((r.get('boot'), r.get('pid'), sid))
        if e is None and r.get('boot') is None:
            e = ended.get((None, r.get('pid'), sid))
        if e is not None:
            if e['event'] == 'error':
                out[sid]['errors'].append(e)
        elif r.get('boot') in (None, boot):
            out[sid]['deaths'] += 1
        else:
            out[sid]['reboots'] += 1
    return out


def reserve_hours(stage, sites, run_dir, led):
    """Stage B stops early enough to leave stage C its time: RESERVE_H if set, else (stage-B sites
    finished and not excluded + 1) x C seconds per site x 1.5 / 3600. C seconds per site = the
    median of this RUN_DIR's finished stage-C sites once there are >= 5, else C_SEC_PER_SITE
    (default 60; P1 measured 18-53 s). Stages A and C: 0 (C stops at exactly HARD_STOP_GPU_H)."""
    if stage != 'B':
        return 0.0, None
    if os.environ.get('RESERVE_H'):
        return float(os.environ['RESERVE_H']), 'RESERVE_H'
    secs = [float(r['seconds']) for r in led if r.get('stage') == 'C' and r.get('event') == 'done'
            and not r.get('excluded') and not r.get('recovered') and r.get('seconds')]
    c_sec = float(np.median(secs)) if len(secs) >= 5 else float(os.environ.get('C_SEC_PER_SITE', 60))
    in_table = {s['site_id'] for s in sites if s['kind'] != 'slip'}
    finished = {r['site_id'] for r in led if r.get('stage') == 'B' and r.get('event') == 'done'
                and not r.get('excluded') and r.get('site_id') in in_table}
    n = len(finished) + 1
    return n * c_sec * 1.5 / 3600.0, {'n_sites': n, 'c_sec': c_sec, 'from_ledger': len(secs) >= 5}


def check_consistent(led, cur):
    """Every proc_start of a RUN_DIR must agree on the site table, p0.json, delta, the settings and
    the pre-registration marker; otherwise earlier stage files would be reused under other inputs."""
    for r in led:
        if r.get('event') != 'proc_start':
            continue
        for k in ('sites_sha256', 'p0_sha256', 'delta', 'settings', 'prereg', 'guard'):
            if k in r and r[k] != cur[k]:
                stop(f"{k} differs from an earlier process of this RUN_DIR (pid {r.get('pid')}): "
                     f"{json.dumps(r[k], default=C._jsonable)[:200]} != {json.dumps(cur[k], default=C._jsonable)[:200]}")


def replay_guard(p0, delta):
    """The batch-1 replay guard of flip_scan and singles: max(2 delta, 2 x the largest batch-64 vs
    batch-1 margin deviation P0 (b) measured), so a screened row that batch 1 would flip is replayed."""
    dev = ((p0.get('b') or {}).get('descriptive_decision_level') or {}).get('margin_dev_max')
    return max(2.0 * delta, 2.0 * float(dev)) if dev is not None else 2.0 * delta


def accepted(stage, sid):
    """ACCEPT_FAILED='B:12|3|45,C:...': a person's decision to record a site that failed (see
    failed_record) instead of stopping."""
    return f'{stage}:{sid}' in [x.strip() for x in os.environ.get('ACCEPT_FAILED', '').split(',') if x.strip()]


def failed_record(stage, sid, why, ckpt):
    """The stage file written for a failed site that a person accepted. A: excluded with a
    non-registered reason ('failed: ...'). B: the primary part if its checkpoint exists, else
    excluded ('failed: ...'). C: never excluded (stage C uses labels): gen, collateral and CONTRAST
    null with the failure in 'failed'."""
    if stage == 'B' and os.path.exists(ckpt):
        return stage_B_primary_only(ckpt, why)
    if stage == 'C':
        return {'site_id': sid, 'excluded': None, 'failed': why[:300], 'gen': None, 'gen_cap': None,
                'collateral': None, 'CONTRAST': None, 'contrast_note': f'stage C failed ({why[:200]})'}
    return {'site_id': sid, 'excluded': f'failed: {stage} failed in fresh processes, accepted by a person ({why[:300]})'}


def canary_z(E, cn):
    """The canary's fp32 decision logits over its stored C: one batch of 2 rows (clean; the stored
    8 features ablated). P0 stores them; every run process recomputes them."""
    S = Site()
    S.prefix, S.E = [int(x) for x in cn['ids']], set()
    S.C = [int(x) for x in cn['C']]
    S.Ct = E.torch.tensor(S.C, device='cuda')
    S.inE = E.torch.tensor([False] * len(S.C), device='cuda')
    S.WU_C = E.WU[:, S.Ct].float()
    S.bU_C = E.bU[S.Ct]
    S.u0 = None
    E.reset_site()
    return E.decide(S, [[], [tuple(x) for x in cn['ablate']]])['z']


def canary(E, p0):
    """P0's stored canary: clean and with 8 features ablated; within delta (expected bitwise)."""
    cn = p0['canary']
    z = canary_z(E, cn)
    ref = np.array([cn['clean'], cn['ablated']])
    dev = float(np.abs(z - ref).max())
    return dev, bool((z == ref).all())


def main_run():
    stage = os.environ.get('STAGE')
    if stage not in STAGES:
        stop(f"STAGE must be one of {STAGES}")
    run_dir = _out('RUN_DIR', 'run')
    sites, sites_path = load_site_table()
    kinds = {s['kind'] for s in sites}
    if kinds <= {'pilot'} and os.path.realpath(run_dir) == os.path.realpath(os.path.join(C.OUT, 'run')):
        stop('a pilot site table must not write to the main RUN_DIR ($P8_OUT/run): use RUN_DIR=pilot/run')
    if kinds & {'belief', 'slip'} and 'pilot' in kinds:
        stop('the site table mixes pilot and belief/slip rows')
    p0_path = _out('P0_JSON', 'p0.json')
    if not os.path.exists(p0_path):
        stop(f'no P0 result at {p0_path}')
    with open(p0_path, encoding='utf-8') as f:
        p0 = json.load(f)
    allow_fail = os.environ.get('ALLOW_P0_FAIL') == '1'          # implementation tests only; logged
    if not p0.get('pass') and not allow_fail:
        stop('P0 did not pass; refusing to run')
    if allow_fail and kinds & {'belief', 'slip'}:
        stop('ALLOW_P0_FAIL is for tests on pilot rows only')
    prereg = prereg_marker(sites)
    delta = float(p0['delta'])
    lock = gpu_lock()                          # before any cleanup or ledger write (exit 5 if held)
    sdir = os.path.join(run_dir, stage)
    os.makedirs(sdir, exist_ok=True)
    ledger = os.path.join(run_dir, 'ledger.jsonl')
    chunk = int(os.environ.get('CHUNK', 10))
    boot = boot_id()
    led = read_lines(ledger)
    cur = {'sites_sha256': C.sha256(sites_path), 'p0_sha256': C.sha256(p0_path), 'delta': delta,
           'settings': run_settings(), 'prereg': prereg, 'guard': replay_guard(p0, delta)}
    check_consistent(led, cur)
    for p in glob_partial(sdir):
        os.remove(p)
    for p in glob.glob(os.path.join(sdir, '*.primary.ckpt')):      # orphan: killed between write and cleanup
        if os.path.exists(p[:-len('.primary.ckpt')] + '.json'):
            os.remove(p)

    def applies(s):
        return stage == 'A' or s['kind'] != 'slip'

    done = {r['site_id'] for r in led if r.get('stage') == stage and r.get('event') == 'done'}
    todo = []
    for s in sites:
        if not applies(s):
            continue
        p = os.path.join(sdir, fname(s['site_id']))
        if os.path.exists(p):
            if s['site_id'] not in done:
                append_line(ledger, {'event': 'done', 'stage': stage, 'site_id': s['site_id'], 'recovered': True,
                                     't': time.time(), 'pid': os.getpid(), 'boot': boot})
            continue
        if s['site_id'] in done:
            stop(f"ledger says {stage} {s['site_id']} is done but its file is missing")
        if stage in ('B', 'C'):
            pa = os.path.join(run_dir, 'A', fname(s['site_id']))
            if not os.path.exists(pa):
                stop(f"stage A has no file for {s['site_id']}")
            if stage == 'C' and not os.path.exists(os.path.join(run_dir, 'B', fname(s['site_id']))):
                continue                       # not reached by stage B (hard stop): not in the analysis set
        todo.append(s)
    if not todo:
        print(f'stage {stage}: nothing to do', flush=True)
        sys.exit(0)
    reserve, rinfo = reserve_hours(stage, sites, run_dir, led)
    if used_hours() >= C.HARD_STOP_GPU_H - reserve:
        print(f'hard stop: {used_hours():.2f} GPU-h used (stage {stage} reserve {reserve:.2f} h)', flush=True)
        sys.exit(3)
    clock = Clock('p8_2_run', t0=T_START)       # before the model load: load time is GPU time
    E = Engine(lock=lock)
    E.guard = cur['guard']
    append_line(ledger, {'event': 'proc_start', 'stage': stage, 'pid': os.getpid(), 'boot': boot, 't': time.time(),
                         'sites': os.path.abspath(sites_path), **cur, 'p0_pass': bool(p0.get('pass')),
                         'allow_p0_fail': allow_fail, 'circuit_tracer': E.ct_version, 'chunk': chunk,
                         'reserve_h': reserve, 'reserve_info': rinfo, 'clock': CLOCK_PATH,
                         'accept_failed': os.environ.get('ACCEPT_FAILED', ''),
                         'load_seconds': time.time() - T_START})
    dev, exact = canary(E, p0)
    append_line(ledger, {'event': 'canary', 'stage': stage, 'pid': os.getpid(), 'boot': boot, 't': time.time(),
                         'max_abs_dev': dev, 'bitwise': exact, 'delta': delta})
    if dev > delta:
        stop(f'canary failed: max |dev| {dev} > delta {delta}')
    fault_spec = os.environ.get('FAULT', '')
    att = site_attempts(led, stage, boot)
    n_done = 0
    for s in todo:
        sid = s['site_id']
        if n_done >= chunk:
            break
        reserve, rinfo = reserve_hours(stage, sites, run_dir, read_lines(ledger))
        if used_hours() >= C.HARD_STOP_GPU_H - reserve:
            append_line(ledger, {'event': 'hard_stop', 'stage': stage, 'pid': os.getpid(), 'boot': boot,
                                 't': time.time(), 'gpu_h': used_hours(), 'reserve_h': reserve, 'reserve_info': rinfo})
            clock.beat(event='hard_stop')
            sys.exit(3)
        out_path = os.path.join(sdir, fname(sid))
        ckpt = os.path.join(sdir, fname(sid)[:-5] + '.primary.ckpt')
        a = att[sid]
        errs = a['errors']
        t0 = time.time()
        obj = None
        if errs or a['deaths']:
            why = errs[-1]['error'] if errs else 'the process died (no error logged)'
            n_attr = sum(1 for e in errs if e['error'].startswith('RuntimeError: attribute() failed'))
            n_desc = sum(1 for e in errs if e.get('phase') == 'descriptive')
            if stage == 'B' and n_attr >= 2:
                obj = {'site_id': sid, 'excluded': f'c: attribute() failed in two fresh processes ({why[:300]})'}
            elif stage == 'B' and n_desc >= 2 and os.path.exists(ckpt):
                obj = stage_B_primary_only(ckpt, why)
            elif len(errs) >= 2 or a['deaths'] >= MAX_DEATHS:
                if accepted(stage, sid):
                    obj = failed_record(stage, sid, why, ckpt)
                else:
                    append_line(ledger, {'event': 'needs_person', 'stage': stage, 'site_id': sid, 'boot': boot,
                                         'errors': len(errs), 'deaths': a['deaths'], 'last': why[:300],
                                         't': time.time(), 'pid': os.getpid()})
                    clock.beat(event='needs_person')
                    stop(f"{stage} {sid}: {len(errs)} errors, {a['deaths']} deaths in fresh processes (last: {why[:200]}). "
                         f"Fix the cause and rerun, or record it with ACCEPT_FAILED='{stage}:{sid}' (see failed_record)", 6)
        if obj is not None:
            obj['seconds'] = 0.0
            C.write_json(out_path, obj)
            append_line(ledger, {'event': 'done', 'stage': stage, 'site_id': sid, 'excluded': obj.get('excluded'),
                                 'resolved': True, 'seconds': 0.0, 't': time.time(), 'pid': os.getpid(), 'boot': boot})
            print(f"{stage} {sid} resolved without a run: excluded {obj.get('excluded')}", flush=True)
            continue
        attempt = len(errs) + a['deaths'] + 1
        append_line(ledger, {'event': 'start', 'stage': stage, 'site_id': sid, 'attempt': attempt,
                             't': t0, 'pid': os.getpid(), 'boot': boot})

        def fault(point, _sid=sid, _first=(attempt == 1)):
            parts = fault_spec.split(':')
            if len(parts) < 3 or parts[0] != stage or parts[1] != _sid or parts[2] != point:
                return
            if not (_first or 'always' in parts[3:]):
                return
            print(f'FAULT injected at {stage} {_sid} {point} ({parts[3:]})', flush=True)
            if 'raise' in parts[3:]:
                raise RuntimeError(f'injected fault at {point}')
            os._exit(97)

        E.note = {'phase': 'primary' if stage == 'B' else None}
        try:
            fault('start')
            prev = {}
            for st in STAGES[:STAGES.index(stage)]:
                with open(os.path.join(run_dir, st, fname(sid)), encoding='utf-8') as f:
                    prev[st] = json.load(f)
            src = next((st for st in prev if prev[st].get('excluded')), None)
            if src:                                     # stage C only copies an A/B exclusion
                obj = {'site_id': sid, 'excluded': prev[src]['excluded'], 'excluded_in': src}
            elif stage == 'A':
                obj = stage_A(E, s, delta, fault)
            elif stage == 'B':
                obj = stage_B(E, s, prev['A'], delta, fault, ckpt)
            else:
                obj = stage_C(E, s, prev['A'], prev['B'], fault)
            E.torch.cuda.synchronize()
            gc.collect()
            E.torch.cuda.empty_cache()
            E.device_check()
            fault('before_write')
        except Exception as exc:
            if stage == 'B' and os.path.exists(ckpt):
                E.note['phase'] = 'descriptive'
            append_line(ledger, {'event': 'error', 'stage': stage, 'site_id': sid, 'attempt': attempt,
                                 'phase': E.note.get('phase'), 'error': f'{type(exc).__name__}: {exc}',
                                 't': time.time(), 'pid': os.getpid(), 'boot': boot, 'seconds': time.time() - t0})
            clock.beat(event='error')
            traceback.print_exc()
            sys.exit(1)
        obj['seconds'] = time.time() - t0
        C.write_json(out_path, obj)
        append_line(ledger, {'event': 'done', 'stage': stage, 'site_id': sid, 'excluded': obj.get('excluded'),
                             'seconds': obj['seconds'], 'from_checkpoint': E.note.get('from_checkpoint'),
                             't': time.time(), 'pid': os.getpid(), 'boot': boot})
        if os.path.exists(ckpt):
            os.remove(ckpt)
        clock.beat(event='site', site_id=sid)
        print(f"{time.strftime('%H:%M:%S')} {stage} {sid} {obj['seconds']:.1f}s"
              + (f" excluded: {obj['excluded']}" if obj.get('excluded') else ''), flush=True)
        n_done += 1
    left = sum(1 for s in todo if not os.path.exists(os.path.join(sdir, fname(s['site_id']))))
    append_line(ledger, {'event': 'proc_end', 'stage': stage, 'pid': os.getpid(), 'boot': boot, 't': time.time(),
                         'sites_done': n_done, 'left': left, 'det_warnings': sorted(_DET_WARN)})
    clock.beat(event='end')
    sys.exit(0 if left == 0 else 10)


def glob_partial(d):
    return [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.partial')]


# ---------------------------------------------------------------- CPU self-tests
def random_dag(g, n_layers=8, n_pos=6, per_layer=12, n_logits=3):
    feats = [(l, int(g.integers(1, n_pos))) for l in range(n_layers) for _ in range(per_layer)]
    n_sel, n_err, n_tok = len(feats), n_layers * n_pos, n_pos
    N = n_sel + n_err + n_tok + n_logits
    A = np.zeros((N, N))
    for i, (li, pi) in enumerate(feats):
        for j, (lj, pj) in enumerate(feats):
            if lj < li and pj <= pi and g.random() < 0.4:
                A[i, j] = g.normal()
        A[i, n_sel:n_sel + n_err + n_tok] = g.normal(size=n_err + n_tok) * (g.random(n_err + n_tok) < 0.3)
    for k in range(n_logits):
        A[N - n_logits + k, :n_sel + n_err + n_tok] = g.normal(size=n_sel + n_err + n_tok) * (g.random(n_sel + n_err + n_tok) < 0.5)
    act = g.random(n_sel) * 3 + 0.05
    return A, act, n_sel, n_logits, n_layers


def e_unit_tests(n_dags=20):
    """final_design.md §9 P0 (a), first part: on random DAGs e equals ((I - A~)^-1 - I)^T w * a,
    circuit_tracer's compute_influence, the logit row at first order, the logit-row entry for
    features with no outgoing feature edge; <= n_layers + 1 iterations; finite."""
    import torch
    from circuit_tracer.graph import compute_influence
    g = test_rng(0)                     # synthetic test data only (no study quantity)
    res = {'dags': n_dags, 'dense_max_rel': 0.0, 'lib_max_rel': 0.0, 'first_order_max_rel': 0.0,
           'leaf_max_rel': 0.0, 'max_iterations': 0, 'iteration_bound_ok': True, 'finite': True}
    for _ in range(n_dags):
        A, act, n_sel, n_log, n_layers = random_dag(g)
        N = A.shape[0]
        At = A.copy()
        At[:, :n_sel] /= act[None, :]
        w = np.zeros(N)
        w[N - n_log] = 1.0
        e, it = compute_e(A[:n_sel, :n_sel], A[N - n_log:, :n_sel], act, np.eye(n_log)[0])
        dense = (w @ (np.linalg.inv(np.eye(N) - At) - np.eye(N)))[:n_sel] * act
        lib = compute_influence(torch.tensor(At), torch.tensor(w)).numpy()[:n_sel] * act
        sc = np.abs(dense).max()
        res['dense_max_rel'] = max(res['dense_max_rel'], float(np.abs(e - dense).max() / sc))
        res['lib_max_rel'] = max(res['lib_max_rel'], float(np.abs(e - lib).max() / sc))
        d1, _ = compute_e(A[:n_sel, :n_sel], A[N - n_log:, :n_sel], act, np.eye(n_log)[0], first_only=True)
        dense1 = (w @ At)[:n_sel] * act                    # first-order term on the full N x N matrix
        res['first_order_max_rel'] = max(res['first_order_max_rel'],
                                         float(np.abs(d1 - dense1).max() / np.abs(dense1).max()),
                                         float(np.abs(d1 - A[N - n_log, :n_sel]).max() / np.abs(A[N - n_log]).max()))
        leaf = ~(A[:n_sel, :n_sel] != 0).any(axis=0)
        if leaf.any():
            res['leaf_max_rel'] = max(res['leaf_max_rel'], float(np.abs(e[leaf] - A[N - n_log, :n_sel][leaf]).max() / sc))
        res['max_iterations'] = max(res['max_iterations'], it)
        res['iteration_bound_ok'] &= it <= n_layers + 1
        res['finite'] &= bool(np.isfinite(e).all())
    res['pass'] = bool(res['dense_max_rel'] < 1e-9 and res['lib_max_rel'] < 1e-9 and res['first_order_max_rel'] < 1e-12
                       and res['leaf_max_rel'] < 1e-9 and res['iteration_bound_ok'] and res['finite'])
    return res


def first_order_check(n_dags=5):
    """compute_e's first-order term (first_only) for a mixed target w over every logit row equals
    w @ adjacency[N - n_logits:, :n_sel] taken from the full N x N matrix (absolute error)."""
    g = test_rng(1)
    worst = 0.0
    for _ in range(n_dags):
        A, act, n_sel, n_log, _ = random_dag(g)
        N = A.shape[0]
        wl = g.normal(size=n_log)
        wf = np.zeros(N)
        wf[N - n_log:] = wl
        d, _ = compute_e(A[:n_sel, :n_sel], A[N - n_log:, :n_sel], act, wl, first_only=True)
        worst = max(worst, float(np.abs(d - (wf @ A)[:n_sel]).max()))
    return worst


def stop_rule_test(tok):
    """Teacher-forced stop rule on every Phase 6 source with a stable repair: feeding S's own tokens
    from its first code token on, the rule must fire at a token after which only whitespace is left."""
    srcs = [s for s in C.read_jsonl(os.path.join(C.OWNFIX, 'ownfix_sources.jsonl.gz')) if s['stable'] and s['n_repairs']]
    bad, n = [], 0
    for s in srcs:
        text = s['prose'] + ' ' + s['code']
        enc = tok(text, return_offsets_mapping=True)
        ids, offs = enc['input_ids'], enc['offset_mapping']
        head = s['prose'] + ' '
        first = next(t for t in range(1, len(offs)) if offs[t][0] >= len(s['prose']))
        target = depth0_closings(s['code'])
        fired = None
        for t in range(first, len(ids)):
            cur = tok.decode(ids[1:t + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if cur.startswith(head) and depth0_closings(cur[len(head):]) >= target:
                fired = t
                break
        n += 1
        rest = None if fired is None else text[offs[fired][1]:]
        if fired is None or rest.strip():
            bad.append({'task': s['task_id'], 'sample': s['sample'], 'fired': fired, 'last': len(ids) - 1,
                        'rest': None if rest is None else rest[:60]})
    return {'sources': n, 'bad': len(bad), 'examples': bad[:10], 'pass': not bad}


def matching_test():
    g = test_rng(2)
    feats = np.array([[l, p, f] for l in range(4) for p in range(1, 4) for f in range(6)])
    act = g.random(len(feats)) + 0.1
    dn = g.random((4, 6)) + 0.5
    P = Pool(feats, act, dn)
    order = [tuple(x) for x in feats[:12]]
    out, kinds = P.match(order, {tuple(x) for x in feats[:3]}, test_rng(3))
    out2, _ = P.match(order, {tuple(x) for x in feats[:3]}, test_rng(3))
    keys = [x[:3] for x in out]
    ok = (out == out2 and len(set(keys)) == len(keys) and not (set(keys) & {tuple(x) for x in feats[:3]})
          and sum(kinds.values()) == len(order))
    # fallbacks: (0,1,0) has no eligible neighbour (its only one is excluded) -> layer 1 at pos 1;
    # (3,2,0) is alone and no other layer has pos 2 -> layer 3 at pos 3
    P2 = Pool(np.array([[0, 1, 0], [0, 1, 1], [1, 1, 0], [3, 2, 0], [3, 3, 5]]), np.ones(5), np.ones((4, 6)))
    out3, kinds3 = P2.match([(0, 1, 0), (0, 1, 1), (3, 2, 0)], {(0, 1, 1)}, test_rng(4))
    want = [(1, 1, 0), (0, 1, 0), (3, 3, 5)]
    return {'deterministic_unique_eligible': bool(ok), 'kinds': dict(kinds), 'fallback_kinds': dict(kinds3),
            'pass': bool(ok and [x[:3] for x in out3] == want and kinds3 == Counter(exact=1, layer=1, pos=1))}


def selftest():
    tok = C.tokenizer()
    res = {'e_unit': e_unit_tests(), 'first_order_abs': first_order_check(), 'stop_rule': stop_rule_test(tok),
           'matching': matching_test()}
    res['pass'] = bool(res['e_unit']['pass'] and res['stop_rule']['pass'] and res['matching']['pass']
                       and res['first_order_abs'] < 1e-12)
    print(json.dumps(res, indent=1, default=C._jsonable))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    if '--test' in sys.argv or os.environ.get('MODE') == 'selftest':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        selftest()
    main_run()
