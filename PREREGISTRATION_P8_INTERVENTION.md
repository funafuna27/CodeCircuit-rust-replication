# Pre-registration — graph-ranked feature ablation at the model's own one-token bugs (Phase 8)

Written 2026-10-15 (Japan time). No GPU run has touched any of the 35 problems that hold the
sites (§2) before this commit, and `p8_2_run.py` refuses those sites until this file is
committed. Several things existed before this commit, all listed in §7:
- the site table;
- the P0 checks on 20 retained Phase 5 graphs, all outside the 35 problems;
- the P1 pilot on 30 sites from 30 other problems ("pool A"), whose repairs are all unverifiable,
  so every pilot outcome is 0 by construction;
- the size and power simulations on the real site structure.

The full design, as written on 10-14 before any GPU work, is `data/intervention_design/final_design.md`.
The interface specification is `data/intervention_design/SPEC.md`. This document fixes what the
design leaves open and records every change made since (§8). Where they differ, this document
governs.

## 1. Why this experiment

**The paper's second claim.** In §4.4 of arXiv:2602.07080, zeroing a transcoder feature picked from
the attribution graph "fixes erroneous logic": `high = mid` becomes `high = mid - 1`. The paper
shows one case, with no rate and no baseline.

**What Phases 1–7 left.** As a predictor of correctness, the graph never beat cheaper signals:
- Phase 5: the output distribution and the edit's identity match or beat it;
- Phase 7: it adds about 0 to them, and a probe from the same forward pass does as well.

The intervention claim is the paper's remaining untested one. It also needs labels that Phase 6
already built: Gemma's own wrong programs, each with a one-token repair verified beyond MBPP's
asserts.

**The question.** At a site where Gemma confidently writes a wrong token, zero the features the graph
says drive that token, in graph order, until the decision leaves the token. Is the result a
verified repair more often than with:
- (i) the same procedure on a matched random ordering of features (S1);
- (ii) a ban on the wrong token, which gives the model's second choice (S2);
- (iii) the same procedure ranked by a graph-free one-backward attribution (S3)?

Support for §4.4 as a *graph* procedure needs all three. Even then it covers only one-token
substitutions at known bug sites, and only the target used here (§2).

## 2. Design — fixed

### Sites (`p8_1_sites.py`; `data/results/intervention/sites.jsonl`)

**Judge.** Every label comes from the finite-aware judge (`p8_judge.py`). Its tolerant f64 branch
requires both values to be finite (readme, "Erratum"). Re-judging all 759 stored mutants with the
unchanged judge reproduced every stored label. With the finite-aware judge, 33 verified repairs
become failures, all in problems 137 and 248.

**Belief sites.** These are Phase 6's verified repair sites where Gemma's wrong token was the HF argmax:
- 146 sites, 31 problems, 94 distinct bugs. A bug is (problem, edited line without whitespace,
  wrong id).
- Sites per problem: [19, 13, 13, 12, 10, 8, 8, 8, 7, 7, 5, 5, 4, 4, 3, 2, 2, 2, 2, then twelve 1s].
- Estimand: the average problem.

**Slips.** These are 45 sites (14 problems) where the wrong token was not the argmax. They get the NULL arm only and are
descriptive.

**E(wrong).** The vocabulary ids whose single-token splice has the same lexer stream as S. Ids whose
piece holds whitespace rustc does not accept (U+00A0, U+2009, U+3000, …) are left out (§8). The
same rule makes `p8_common.lexeme_key` mark such a program, so it never matches a compiling one.

**Run order** (label-free, in the table):
- Tier 1 is one site per bug (94); tier 2 is the remaining 52.
- Within a tier, sites are shuffled per problem and the run goes round-robin over problems.
- Slips come after all belief sites.

**Exclusions** (label-free, before scoring, reported, never replaced):
- (a) Using the fp32 decision logits, the argmax at h−1 is not in E(wrong), or the best E logit
  leads the best non-E logit by less than 2δ.
- (b) The prefix ids differ from the table.
- (c) `attribute()` fails in two fresh processes.
- (d) Logit node 0 is not in E(wrong).

δ = 0.190, the noise floor measured in P0 (§7).

### Intervention (`p8_2_run.py`)

- **Input.** The 1-D `ids[:h]` with BOS. The decision is at h−1.
- **Decision logits.** fp32 `ln_final.hook_normalized` × `ln_final.w` @ `W_U[:, C]` + `b_U[C]`,
  soft-capped at 30. C = the bf16 top-64 ∪ E. Every argmax uses these logits.
- **Graph.** The study's unchanged call: `max_n_logits` 10, `desired_logit_prob` 0.95,
  `batch_size` 64, `max_feature_nodes` 8192, offload cpu. No graph is kept; `to_pt` is never
  called.
- **Target.** Logit node 0, the demeaned `W_U` column of the wrong token.
- **G ranking.** The signed linearised effect e = compute_influence(Ã, w)·a, in float64 on CPU. Ã is
  the adjacency with each feature column divided by its activation. Candidates are the selected
  features at positions 1..h−1 with e > 0, in descending order. The declared fallback was not
  needed: P0's e checks pass (§7).
- **Ablation.** Value 0. Our hook at `blocks.l.hook_mlp_out` re-encodes the current
  `ln2.hook_normalized` and adds the bf16 delta exactly as `feature_intervention` builds it. At
  batch 1 this equals `feature_intervention(freeze_attention=False, constrained_layers=None)`
  bitwise (P0 (b)).

**Flip rule** (one function for every ordered arm):
- Ablate the top k cumulatively for k = 1..K_MAX. **K_MAX = 1024**, or the candidate list if shorter.
- Screen 64 rows per forward.
- Every row whose screen flips, or whose screened margin is below the replay guard, is replayed at
  batch 1 in k order.
- The guard is max(2δ, 2 × P0's largest batch-64 margin deviation) = 0.753.
- k* is the smallest replayed k whose argmax is not in E; t* is that argmax.
- No confirmed flip by K_MAX means no flip (Y = 0).
- Decisions are taken only from batch-1 rows (§8).

### Arms

| arm | role | definition |
|---|---|---|
| G | primary | §2 ranking, flip rule |
| RND1 | S1 | per rank j of G's list, one of the 5 eligible features at the same (layer, pos) nearest in rank on a·‖W_dec‖ (eligible: not in G's or GxA's top 64, not drawn before; fallbacks nearest layer, then nearest pos); flip rule |
| BAN | S2 | argmax over C \ E |
| GxA | S3 | the `compute_batch` row for node 0 (one backward through frozen attention, all active features at 1..h−1), positive entries descending; flip rule |
| RND2–5, RND-k (20 draws), G_LP, G1, G_mid (layers 8–17), DE, ORC (GxA's top 256 measured singly, plus G's top 128 for existence), CONTRAST (label-using, run last), NULL | descriptive | as in the design §4 |
| gen:G, gen:G_every, gen:BAN, gen:NULL | descriptive | greedy program generation, lexer brace stop, cap tokens-to-end + 64 |

### Scoring (`p8_3_judge.py`, blind; the GPU is idle)

One rule for every arm:
1. If the lexer stream of the single-token splice equals one of the site's judged mutants, use that
   mutant's finite-aware label.
2. Otherwise, take the line completion recorded at t*. This is a greedy continuation under the same
   arm (prefix hooks kept) up to a newline, capped at 48 tokens. Splice it in place of the edited
   line, keep S's text after it, and judge it blind with the finite-aware judge. The judge cache is
   keyed only on the lexer stream, so the arm never reaches the judge.
3. No newline within the cap, or a completion that ends in `<eos>`, is unscorable and scores 0.

**Outcome.** Y = 1 iff verified: the program passes the asserts in 3 of 3 judgings and passes the
problem's calibrated extra inputs. Lucky, fail, no_compile, unverifiable, unscorable and no_flip
all score 0.

**Neutralizing flag.** A syntactic flag, as in the design §5, reported with every rescue.

## 3. Analysis — fixed (`p8_4_stats.py`)

**Analysis set.** The belief sites in completed round-robin rounds, minus exclusions.

**Contrasts.** For arms X and Z, d̄_g is problem g's mean of Y_X − Y_Z, and D is the equal-weight
mean over problems.

**Fixed sequence.** Each step runs only if the previous one rejected:
- S1: G > RND1;
- S2: G > BAN;
- S3: G > GxA.

**Test.** Each step is a one-sided problem-level sign-flip test:
- The statistic is the integer d̄_g × LCM of the site counts (207480 for the full set).
- The p-value is exact by enumeration over the m discordant problems when m ≤ 24. Otherwise it
  uses 99,999 flips plus the identity (stream 0).
- H0: the d̄_g are sign-symmetric and problems are independent. This is a superpopulation
  assumption, because the arms are deterministic.
- **α = 0.03 at each step.** The post-pilot rule kept it (§7).
- m is always reported. S2 cannot reject unless m ≥ 6.

**Decomposition** (pre-registered, descriptive): D(G−BAN) = −NF + (R − Bk). NF covers sites where
G does not flip but BAN fixes; R covers rescues among G's flips; Bk covers breaks among G's flips.

**Descriptive only.** Nothing below can change S1–S3:
- per-arm rates (equal weight and per site) with bootstrap intervals (stream 1, 5000 draws over
  problems);
- flip rate and k*;
- the landing table (BAN's lexeme / other family member / outside the family / unscorable);
- the rescue/break table;
- P(fix | both flip) for G against GxA;
- the existence rate;
- faithfulness (site-median Spearman of e against measured single ablations);
- program fix by generation;
- collateral;
- margin terciles, family, slips;
- the sensitivity set: tier 1 only, without rule-B sites, unscorable excluded, neutralizing
  rescues scored 0, both tokens in the judged family, √n weights.

A non-rejection always reads "the advantage was not detected; the 80% MDE is X" (§6), never as
equivalence.

## 4. Predictions

**Written on 10-14, before any GPU data** (design §6, unchanged). Rates are equal-weight over
problems unless marked per site.
- Exclusions 0–4.
- BAN fix rate 0.50–0.70. The TL BAN lexeme differs from the HF runner-up at ≤ 10% of sites.
- G:
  - flips at ≥ 90% of sites (per site), with median k* 2–20;
  - its first feature is at layer ≥ 18 and position h−1 at ≥ 60% of sites;
  - t*_G equals BAN's lexeme at 60–85% of flips;
  - m is 6–16;
  - fix rate 0.40–0.65.
- D(G−BAN) in [−0.15, +0.03]; NF in [0, 0.08]; R−Bk in [−0.10, +0.05].
- GxA flips at ≥ 90%; fix rate 0.40–0.65; D(G−GxA) within ±0.06.
- RND1 flips at 30–90%, median k* ≥ 100; fix rate 0.05–0.35; D(G−RND1) ≥ +0.15.
- Decisions: S1 rejects, S2 does not, S3 is not reached.
- Secondary arms:
  - G_LP and ORC within ±0.10 of G; CONTRAST ≥ G + 0.10;
  - G1 flips at 5–25%; G_mid at ≤ 40%.
- Spearman 0.3–0.7.
- Existence 5–25%, against ≤ 5% for random features.
- Program fix: gen:G is 0.6–1.0 × G's site rate; gen:G_every ≤ gen:G; gen:NULL ≤ 0.05.
- Collateral: 20–60% of sites for G, 5–30% for RND-k.
- Slips: NULL fixes the HF-computed set ±3.

**Revised on 10-15 after the pilot** (pool A, label-free quantities only, §7). These are added, not
substituted. Both sets are scored.
- G flips at 0.65–0.88 (per site), median k* 20–150.
- G's first feature is at layer ≥ 18 and h−1 at 0.30–0.65.
- t*_G equals BAN's lexeme at 0.30–0.60 of flips.
- RND1 flips at ≤ 0.15.
- Faithfulness 0.05–0.30.
- Collateral for G 0.45–0.85.
- G fix rate 0.25–0.55. D(G−BAN) in [−0.30, 0.00].
- Decisions: S1 rejects, S2 does not, S3 is not reached (unchanged).

## 5. What each outcome means

- **S1 does not reject.** Audit the plumbing first. If it is clean, graph-ranked flips land on
  verified repairs no more often than matched random flips. In the pilot, random orderings almost
  never flip, so S1 mostly asks whether the graph's flips are ever repairs.
- **S1 only.** The ranking matters, but no advantage over simply banning the wrong token was
  detected. If t*_G mostly equals BAN's lexeme, the "fix" is token suppression.
- **S1 and S2, not S3.** No advantage over a one-backward attribution was detected. The graph's
  multi-hop paths and pruning are not shown to matter.
- **All three reject.** This is the first quantified support for §4.4, limited to one-token
  substitutions at known sites and the node-0 target. Calling it "targeted" also needs low
  collateral.
- **D(G−BAN) < 0.** The decomposition says whether G fails to flip (NF) or flips to the wrong place
  (Bk).

Together with Phase 7, anything short of all three keeps the study's reading: the graph is not
shown to do more than cheap methods.

## 6. Power (`p8_sim.py MODE=power`, real structure, α 0.03; `sim_power.json`)

The model is the design's g4: rescues only at BAN-fail sites, breaks only at BAN-fix sites (rate
b), outcomes shared within a bug, unknown BAN outcomes imputed at 0.5.

80% MDE of S2:

| b | problem SD 0 | problem SD 1 |
|---|---|---|
| 0 | 0.136 | 0.140 |
| 0.05 | 0.188 | 0.214 |
| 0.10 | 0.224 | 0.244 |

- **S1.** Power is about 1 whenever G has any rescue rate above RND1's.
- **Joint S2 and S3** at D(G−BAN) ≈ 0.23 (b 0.10, r 0.80, SD 0): 0.63 if GxA agrees with G at 30%
  of bugs, 0.43 at 50%, 0.06 at 80%.
- **Reading.** "All three reject" needs G to lead GxA by about 0.12 or more.

**Size** (`MODE=size`, 132 null cells with within-bug sharing, `sim_size.json`): the worst size is
0.046 at α 0.03.

## 7. What existed before this commit

**P0** (`p8_0_checks.py`, `results/intervention/p0.json`). It used 20 retained Phase 5 graphs, one
per problem, none among the 35, read-only. All gates pass:
- **(0) Noise floor.** δ = 0.190. Batch-1 runs are bitwise repeatable.
- **(a) e checks.**
  - The unit tests on 20 random DAGs pass: maximum relative error 1.3e-12, at most 8 iterations,
    all values finite.
  - On the graphs, at most 26 iterations.
  - The median Spearman of e against measured single ablations is 0.30, exactly at the gate, over
    2–56 qualifying features (median 5.5); 6 of 19 graphs are negative.
  - Signs agree only once ln_final's scale is held fixed, as the attribution assumes.
- **(b) Hook.** Bitwise equal to `feature_intervention` at batch 1 (0 ulp). Batch-64 rows deviate
  by up to 1.06 logits (§8).
- **(c), (d), (e)** pass.

**P1** (`results/intervention/pilot/`). 30 pool-A sites, one per problem, drawn with stream 3. Two
were excluded by rule (a) (margins 0.10 and 0.15 < 2δ). Every outcome is 0 by construction (pool A
has no verified repair). Label-free quantities:

| arm | flips (of 28) | median k* | t* = BAN's lexeme |
|---|---|---|---|
| G | 21 | 57 [13, 193] | 9 of 21 |
| GxA | 21 | 24 | 13 |
| G_LP | 21 | 12 | 15 |
| ORC | 25 | 5 | 24 |
| CONTRAST | 26 | 11.5 | 21 |
| G_mid | 13 | 28 | 11 |
| DE | 11 | 20 | 9 |
| G1 | 1 | 1 | 1 |
| RND1–5 | 1–2 | 57–998 | all |

- **Other quantities.**
  - Faithfulness: site median 0.16.
  - G's first feature at layer ≥ 18 and h−1: 46%.
  - Collateral for G: 68% of sites.
  - RND-k changes the argmax in 1.2% of draws.
  - TL BAN equals the HF runner-up at 23 of 23 sites.
  - Generation stops on braces at 27 of 28 sites.
- **Restart test.** An injected fault and restart gave identical outputs.
- **Futility rule.** G flipped at 21 of 30 (≥ 15), so the main run goes ahead.
- **Post-pilot α rule.** q = 0.679 (share of flips where G's lexeme ≠ BAN's). Grid q ± 0.1. The
  worst size at 0.03 is 0.045 ≤ 0.05, so **α stays 0.03** (`alpha.json`).
- **ORC size.** Single ablations take 4.6 s per site (≤ 1 min), so ORC stays at 256 + 128.
- **Cost.** Stage A 13 s per site; stage B 275 s (graph 163 s); stage C 23 s (medians). The GPU clock
  stands at 3.6 GPU-h. The main run is expected to take about 13 GPU-h, inside the 28 GPU-h hard
  stop for the whole phase.

## 8. Changes from the design (all made before any GPU run on the 35 problems)

1. **P0 gate (b) was restated.**
   - As designed, it also asked batch-64 rows to match batch 1 within δ. They do not: up to 1.06
     logits, with margin deviations up to 0.38 and 18 flip-status disagreements over 1280 rows.
   - The cause is a re-encoded feature sitting at its JumpReLU threshold, which moves whole rows.
     It is not a kernel artefact that could be debugged away.
   - The design says a failing gate is debugged, never loosened. Rather than loosen it, no decision
     is now taken from a batch-64 row: the screen only selects rows, and every flip and every margin
     below the guard (0.753) is decided at batch 1.
   - The gate is now the batch-1 equality, which passes bitwise. The batch-64 deviation is reported
     and sets the guard.
   - Any screened row above the guard that would flip at batch 1 is a miss of the screen. Its size
     is bounded only by P0's measurement. Scan/replay disagreements are reported per arm (8 for G
     in the pilot).
2. **P0 graph choice.** No retained graph has top probability ≥ 0.8. P0 used the highest-probability
   eligible graph of each of 20 problems outside the 35 (p 0.62–0.79).
3. **E(wrong) leaves out pieces with whitespace rustc rejects.** This applies at 71 of 191 sites.
   Such a token now counts as a flip and scores no_compile, consistently in `p8_1`, `p8_3`,
   `p8_4` and `p8_sim`.
4. **The pilot's sampling frame.** 33 of pool A's 38 problems have a belief site; the 30 were drawn
   from those 33.
5. **Completions ending in `<eos>`** are unscorable (0), like a completion with no newline.
6. **Runtime settings** (recorded in every stage file):
   - ORC_N 256, EXIST_G 128, EXIST_RND 128;
   - MAX_DEATHS 3: a site whose process dies 3 times in one boot stops the driver for a person;
   - the stage-B reserve for stage C is 1.5 × the measured stage-C time.
7. **Phase 1–7 judge.** The inf erratum (readme) was found while designing this phase. Phase 8 uses
   the finite-aware judge throughout.

## 9. Execution

```
cd data && SITES=sites.jsonl bash run_p8.sh        # stages A, B, C; hard stop 28 GPU-h for the phase
SITES=sites.jsonl RUN_DIR=run python p8_3_judge.py  # blind, GPU idle
python p8_4_stats.py                               # analysis.json
```

- **Hard stop.** The analysis set is the completed rounds at the hard stop.
- **Disk.** The run keeps only per-site JSON (under 0.5 GB). Retained graphs are never written.
- **Toolchain.** The rustc version is frozen: no `rustup update` during Phase 8.

## 10. Hashes and versions (sha256, first 16 hex)

- rustc: `rustc 1.98.1 (48a229cea 2026-09-01)`; circuit_tracer: 0.5.3.dev1+g6018ed8d3 | torch 2.11.0+cu128

| file | sha256 |
|---|---|
| `data/results/intervention/sites.jsonl` | `3fe80119aaf4bde0` |
| `data/results/intervention/pilot_sites.jsonl` | `48d8a84596d3d193` |
| `data/results/intervention/sites.json` | `3c102a07557507ac` |
| `data/results/intervention/p0.json` | `c5c5b1de301ff2ca` |
| `data/results/intervention/alpha.json` | `5336db604d50903f` |
| `data/results/intervention/sim_power.json` | `8c7af3640dffe74d` |
| `data/results/intervention/sim_size.json` | `bb0714dfaa946c78` |
| `data/p8_common.py` | `0452d67c1eb48c13` |
| `data/p8_judge.py` | `2db06ad8f027fc31` |
| `data/p8_1_sites.py` | `5cfb64648278422b` |
| `data/p8_0_checks.py` | `8b0777c56ac0542e` |
| `data/p8_2_run.py` | `6d9886f5425c09ab` |
| `data/run_p8.sh` | `54188900aac7bea8` |
| `data/p8_3_judge.py` | `c868033e184fe7b2` |
| `data/p8_4_stats.py` | `adb63a0290ead358` |
| `data/p8_sim.py` | `a46e1e32f46333bb` |
| `data/intervention_design/final_design.md` | `8f10ccd4315e3a36` |
| `data/intervention_design/SPEC.md` | `eafd72f75422002a` |
