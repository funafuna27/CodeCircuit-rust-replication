# Phase 8 implementation spec (binding for every implementer)

The design is `/home/user1/p8design/final_design.md` (read it first; this spec resolves its open
decisions and fixes the interfaces). Shared module: `data/p8_common.py` (already written; use it,
do not fork it; if you need a helper there, add it in your own file and say so in your report).

## Resolved decisions
1. The inf judge bug is confirmed and handled by an erratum (separate work). Phase 8 uses the
   finite-aware judge everywhere (tolerant f64 branch requires both values finite; `v == e` unchanged;
   NaN never matches).
2. Primary target: logit node 0 (the wrong token, demeaned W_U column). G-LP is descriptive.
3. alpha 0.03 at each step S1 -> S2 -> S3 on the full analysis set; tier 1 is a sensitivity analysis.
   alpha may only fall to 0.025 or 0.02 by the post-pilot size rule (design §5).
4. Full scope with the whole-phase hard stop of 28 GPU-h (`p8_common.HARD_STOP_GPU_H`).
5. K_MAX = 1024 for every ordered arm (fixed, not tunable).

## Hard rules
- Never open, run or import `data/1_code_correct.py`. The Rust judge is `data/1_rust_correct.py`
  (import it via importlib like `m1_make_mutants.py` does).
- Before the pre-registration is committed, NO GPU computation may touch the 35 site problems
  (the problems of the 210 verified-repair sites). GPU tests use pool A (problems with stable repairs
  that are all unverifiable; disjoint from the 35) or the retained Phase 5 graphs (read-only).
- Only one GPU process at a time on this machine (the card runs near full VRAM when building graphs).
- Generated code runs only in subprocesses with timeouts, judge workers under an 8 GiB RLIMIT_AS
  (`Pool(initializer=...)` that calls `resource.setrlimit`).
- Never call `graph.to_pt`. Never keep a graph after its site is done. Nothing under ~/matched_graph_keep
  or ~/mutant_graph_keep is ever written or deleted.
- Never `model.reset_hooks(including_permanent=True)` (the permanent hooks carry the transcoder outputs).
- Do not edit existing repo files. New files only, in `data/` with the names below. Commit nothing.

## Files and owners
| file | owner | role |
|---|---|---|
| `data/p8_common.py` | (given) | constants, rng streams, io, tokenizer, lexer |
| `data/p8_judge.py` | A | finite-aware judge: `judge(code, sig, workdir)` same return dict as `1_rust_correct.judge`; `judge3(task, code)` (3 stable judgings); `verify_extra(task, code)` (calibrated extra inputs of `ownfix_verify.json['problems'][str(task)]`, same rule as `p6_1b_verify`); `init_worker()` (rlimit + typeable); canaries |
| `data/p8_1_sites.py` | A | site table + pilot pool A table |
| `data/p8_sim.py` | B | size and power simulation (design §5, §8), the alpha rule |
| `data/p8_4_stats.py` | B | analysis (design §5) from judged outcomes |
| `data/p8_0_checks.py` | C | P0 (design §9) on retained Phase 5 graphs |
| `data/p8_2_run.py`, `data/run_p8.sh` | C | GPU driver, stages A/B/C, ledger, restart |
| `data/p8_3_judge.py` | D | blind scoring of every arm's t* (design §5 scoring rule) |

## Interfaces
### sites (`results/intervention/sites.jsonl`, one row per site; `pilot_sites.jsonl` same schema)
```
site_id        str  "task|sample|pos"           task, sample, pos (char offset in code), line
kind           "belief" | "slip"                 (pilot rows: "pilot")
prose, code    str  S's text; model input text = prose + ' ' + code (no chat template)
ids            list[int] HF token ids of the full text (BOS first)
h              int  index in ids of the token holding the edit's first char; decision at h-1
wrong_id       int  ids[h]; wrong_piece str
E              list[int] every vocab id v with lexeme_key(S with ids[h]:=v decoded) == lexeme_key(S)
family, frm    str
judged         dict lexeme_key-string -> {"to": str, "label": "verified"|"lucky"|"fail"|"unverifiable"|"no_compile"}
               for every judged mutant at this site, labels after the finite-aware re-judge
               (key = json.dumps(lexeme_key(code))); used by p8_3 only
verified_ids   list[int] token ids whose single-token splice gives a verified program (descriptive)
bug            str  "task|edited line without whitespace|wrong_id"
tier           1 | 2 ; order int (global run order, round-robin over problems)
problem        int ; tokens_to_end int ; rule_b bool ; neutralizing flags per judged mutant
p_wrong_hf, s_top bool (Phase 6 HF numbers, descriptive)
```
`results/intervention/sites.json`: counts, per-problem site counts, bugs, tiers, hashes of
sites.jsonl / pilot_sites.jsonl, rustc version, the 759-mutant re-judge agreement table.

### run outputs (`results/intervention/run/<stage>/<site_id with | replaced by _>.json`)
Stage A (graph-free): `{"site_id", "excluded": null|reason, "decision": {"top": [[id, logit_fp32], ...10],
"wrong_in_E_top": bool, "margin": float}, "BAN": FLIP, "NULL": FLIP, "GxA": FLIP, "DE": FLIP,
"CONTRAST": FLIP, "gxa_top": [[layer,pos,feat,act,score], ...2000], "seconds": float}`
Stage B (graph): `{"site_id", "excluded", "graph": {"n_sel", "n_logits", "logit_ids", "node0_id",
"seconds"}, "e_top": [[layer,pos,feat,act,e], ...2000], "G": FLIP, "G_LP": FLIP, "G1": FLIP,
"G_mid": FLIP, "RND1".."RND5": FLIP, "RNDk": [ {draw, t_star_id, changed} x20 ], "ORC": FLIP,
"existence": {...}, "faithfulness": {"spearman", "n"}, "seconds"}`
Stage C: `{"site_id", "gen": {"G": GEN, "G_every": GEN, "BAN": GEN, "NULL": GEN}, "collateral": {...}}`

FLIP = `{"arm", "flipped": bool, "k_star": int|null, "t_star_id": int|null, "t_star_piece": str|null,
"margin_at_kstar": float|null, "scan_replay_disagree": int, "n_candidates": int,
"ablated": [[layer,pos,feat,act], ... first k_star], "completion": str|null}`
(`completion`: greedy text from ids[:h]+[t*] to the first newline under the same arm (cap 48 tokens),
recorded at run time for every flipped arm, used by p8_3 only when the splice is not a judged lexeme).
BAN and NULL use the same FLIP shape (k_star 0).
GEN = `{"text": str, "tokens": int, "stopped": "braces"|"eos"|"cap"}`.

### judged outcomes (`results/intervention/judged.jsonl`)
`{"site_id", "arm", "Y": 0|1, "label": "verified"|"lucky"|"fail"|"no_compile"|"unscorable"|"no_flip"|"excluded",
"how": "lexeme"|"completion"|"none", "lexeme_is_ban": bool, "neutralizing": bool}` — one row per (site, arm),
including slips (NULL only) and generation rows (`arm` = "gen:G" etc.).
### analysis (`results/intervention/analysis.json`) — p8_4.

## Tests each owner must pass before reporting (write them as `MODE=selftest` or a `--test` flag)
A: canaries (INFINITY -> fail, NaN -> fail, exact -> pass, 1.0005*e for isclose -> pass); old-path re-judge of
   the 759 judged mutants at the 210 sites agrees 100% with stored labels; finite path differs only where an
   output is non-finite; counts 146 belief / 31 problems / 94 bugs / 45 slips (or explain any difference);
   decoded single-token splices equal the stored mutant code; wrong_id in E; pilot pool A has 38 problems.
B: exact enumeration and Monte Carlo sign-flip p agree; integer statistic; size table reproduces
   final/g3_sim.log's worst cells within MC error; analysis runs end to end on synthetic judged.jsonl.
C: P0 (a)-(e) as in the design, on retained graphs; run stages A/B/C end to end on <= 3 pool-A sites,
   including an injected fault + restart giving identical outputs; GPU-seconds per site logged.
D: scoring on synthetic FLIPs built from the judged mutants of pilot/pool-A sites (verified -> 1, lucky -> 0,
   fail -> 0, a whitespace variant maps to the same label, an out-of-family completion is judged blind
   through the cache); arms are hidden from the judge (cache key = lexeme stream only).
