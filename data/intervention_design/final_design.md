# Phase 8 design (revised): graph-ranked ablation at Gemma's own one-token bugs

The checks behind this revision were run on CPU only. No GPU was used and no intervention data exist. Scripts and logs are in `/home/user1/p8design/final/`:
- `g1_inf.py`, `.log`, `.json`: non-finite repairs;
- `g2_bugs.py`, `.log`: corrected structure;
- `g3_sim.py`, `.log`: size checks with bug sharing;
- `g4_pow03.py`, `.log`: power at α 0.03;
- `g5_cost.py`, `.log`: graph time;
- `g6_oc.py`: operating characteristics of the futility rule.

## 1. Question and claim
- **The paper's claim (§4.4):** zeroing transcoder features chosen from the attribution graph "fixes erroneous logic". The paper shows one case and no baseline.
- **What is tested here.** At Gemma's own confident one-token bugs, we zero the features that the graph says drive the wrong token, in graph order, until the decision leaves the wrong token. We then ask whether the result is a verified repair more often than with:
  - (i) the same flip procedure applied to a matched random ordering of features;
  - (ii) a ban on the wrong token, which gives the model's second choice;
  - (iii) the same procedure ranked by a graph-free one-backward attribution (GxA).
- **What counts as support.** Support for §4.4 as a graph procedure needs all three comparisons. Even then it covers only one-token substitutions at known bug sites, and only for the target used here (§3).
- **Blindness.** The repair's identity never enters feature selection.

## 2. Units
- **The judge is corrected first. This was the fatal finding.**
  - The tolerant f64 check in `data/1_rust_correct.py` accepts ±inf: `(v-e).abs() <= 1e-3*v.abs().max(e.abs())` becomes `inf <= inf`, which is true.
  - I compiled every verified repair in the only two f64+isclose site problems, 137 and 248. 33 of the 304 verified repairs return `inf` on every MBPP and extra input.
  - Phase 8 uses a new finite-aware wrapper and leaves the repo unchanged:
    - the tolerant branch requires both values to be finite;
    - `v == e` still matches equal infinities;
    - NaN never matches, which is Python's `isclose` rule.
- **Population after the correction.**
  - 146 belief sites, 120 sources, 31 problems.
  - 94 distinct bugs, where a bug is (problem, edited line without whitespace, wrong id).
  - Sites per problem: [19,13,13,12,10,8,8,8,7,7,5,5,4,4,3,2,2,2,2, then twelve 1s].
  - Dropped: 11 of the 16 sites in problem 137, and all 4 in problem 248.
  - The formal re-judge (Implementation) must reproduce these numbers.
  - Estimand: the average problem.
- **Run order** (label-free, committed in advance):
  - Tier 1: one RNG-chosen site per bug (94 sites). Tier 2: the remaining 52.
  - Within a tier, sites are shuffled within each problem with `default_rng([D,8,task])` and problems are ordered with `default_rng([D,9])`. The run goes round-robin across problems.
  - **Hard stop: 28 GPU-h for the whole of Phase 8 (P0 + P1 + main).** The analysis set is the completed rounds.
- **Exclusions** (label-free, applied before scoring, reported, never replaced):
  - (a) Using the fp32 decision logits (§3): the argmax at h−1 is not in E(wrong), or the best E logit leads the best non-E logit by less than 2δ (δ = noise floor measured in P0).
  - (b) The prefix ids differ from the site table.
  - (c) `attribute()` fails in two fresh processes.
  - (d) Logit node 0 is not in E(wrong).
- **Slips (45 sites, 14 problems):** NULL only (the greedy token at h−1, scored by the §5 rule), descriptive. The other arms are undefined there, because the wrong token is not the argmax.
- **Sensitivity analyses.** They are descriptive and cannot change S1–S3:
  - tier 1 only;
  - without the rule-B sites;
  - unjudgeable sites excluded rather than scored 0;
  - neutralizing rescues scored 0;
  - only sites where both arms' token falls in the judged family;
  - √n weights.

## 3. Intervention
- **Input:** the 1-D `ids[:h]` with BOS kept. The decision is at h−1.
- **Token equivalence.**
  - E(wrong) is every vocabulary id v such that S with id h replaced by v has the same Rust lexer stream as S (`m1_make_mutants.lex`, whitespace outside literals ignored).
  - It is computed on CPU and committed with the site table.
  - The same lexer map ties whitespace variants of family tokens to their stored labels.
  - At 108 of the original 161 belief sites, some other id splices back to S.
- **Decision logits in fp32.**
  - Take `ln_final.hook_normalized` (fp32), multiply by `ln_final.w` and `W_U[:,C].float()`, where C = the bf16 top-64 ∪ E, then apply the soft-cap of 30.
  - Every argmax uses these logits: exclusions, BAN, the flip test and t*.
- **Graph:** the unchanged study call (max_n_logits 10, desired_logit_prob 0.95, batch_size 64, max_feature_nodes 8192, offload cpu).
  - Set `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8` before importing torch.
  - Call `empty_cache()` before each `attribute()`.
- **Target:** logit node 0, the demeaned W_U column of the wrong token; w is one-hot on it.
- **G ranking:**
  - e = compute_influence(Ã, w)·a, computed in float64 on CPU.
  - Ã is the adjacency with each feature column divided by `activation_values[selected_features]`.
  - Candidates are the selected features at positions 1..h−1 with e > 0, in descending order.
- **Declared fallback** (only if the P0 e-check still fails after debugging): circuit-tracer's native unsigned influence on node 0, restricted to features with e > 0.
- **Ablation:**
  - Value 0.
  - Our own hook at `blocks.l.hook_mlp_out` re-encodes the current `ln2.hook_normalized` at (l, pos). It adds a bf16 delta built exactly as `feature_intervention` builds it: zeros, then `index_add_` of −a·W_dec, then add.
  - This equals `feature_intervention(freeze_attention=False, constrained_layers=None)`.
  - The permanent hooks are never reset.
- **Flip rule and K_max:**
  - Ablate the top k cumulatively for k = 1..K_max.
  - **K_max = 1024, fixed now** (or the full candidate list if it is shorter). It is the same for every scanned arm and is not tunable.
  - Screen in batches of 64 rows. When a batch first shows a flip, replay every k in that batch at batch size 1.
  - k* is the smallest replayed k whose argmax is not in E; t* is that argmax.
  - If no confirmed flip occurs by K_max, Y = 0.
  - Report scan/replay disagreements and the fp32 margin at k*.

## 4. Arms
Every arm starts from the same TL prefix and uses the same flip rule, K_max and scoring.
- **G:** §3.
- **BAN:** the argmax over C \ E.
- **GxA:**
  - The `compute_batch` row: `setup_attribution` plus one backward on the node-0 vector.
  - It covers all active features at positions 1..h−1, through frozen attention; positive entries in descending order.
  - G−GxA therefore isolates the multi-hop paths plus pruning.
- **RND (ordered):**
  - For each rank j of G's list, draw one feature uniformly from the 5 eligible features at the same (layer, pos) that are nearest in rank on a·‖W_dec‖.
  - Eligible excludes G's and GxA's top 64 and any earlier draw.
  - Fallback: the same pos at the nearest layer, then the same layer at the nearest pos. Report the share of exact matches.
  - The sequence is ablated cumulatively under the flip rule.
  - Draw 1 (`default_rng([D,2,site,1])`) enters S1; draws 2–5 are descriptive.
  - Under H0 the two orderings are exchangeable.
- **Descriptive arms:**
  - **G-LP:** the same graph with w = (1−p0)·e0 − Σ_{j≥1} p_j·e_j. This is approximately ∇log p(wrong); the demeaning cancels to about 0.05.
  - **RND-k:** k*_G matched features with no flip rule, 20 draws.
  - **G1:** k = 1 only.
  - **G-mid:** layers 8–17.
  - **DE:** direct effect.
  - **ORC:** GxA's top 256 re-ranked by measured single-feature ablations.
  - **CONTRAST:** label-using, v = W_U[wrong] − W_U[best repair]. It runs after all label-free arms.
  - **NULL:** no intervention.

## 5. Outcomes and test
- **Scoring t\*:** one rule for every arm.
  1. If the lexer stream of the single-token splice equals one of the site's judged mutants, use that mutant's corrected label.
  2. Otherwise (this includes the first piece of a multi-token edit), complete the line:
     - continue greedily from `ids[:h]+[t*]` under the same arm (prefix hooks kept, no intervention at new positions) until a newline, with a cap of 48 tokens;
     - splice in S's text after the edited line;
     - judge blind with the finite-aware judge, through a cache keyed on the lexer stream;
     - if no newline appears within the cap, the site is unscorable and scores 0.
- **Primary outcome:** Y = 1 iff the program is verified: it passes the asserts stably in 3 of 3 judgings and passes the calibrated extra inputs. Everything else scores 0.
- **Neutralizing flag** (syntactic, pre-registered): the edit introduces `x-x`, `x%x`, `x/x`, `*=1`, `/=1`, `%=1`, `%1`, `±0`, `*1` or `/1`. Flagged rescues are marked.
- **Primary contrast:** D(G−BAN), the equal-weight mean over problems of each problem's mean Y_G − Y_BAN.
- **Pre-registered decomposition:** D = −NF + (R − Bk), with each term equal-weighted. Here
  - NF covers sites where G does not flip but BAN fixes;
  - R covers rescues among G's flips;
  - Bk covers breaks among G's flips.
- **Test:** a fixed sequence that stops at the first test that does not reject:
  - S1: G > RND (draw 1);
  - S2: G > BAN;
  - S3: G > GxA.
- **Test details:**
  - Each step is a one-sided problem-level sign-flip.
  - H0: the d̄_g are sign-symmetric and problems are independent. This is a superpopulation assumption, because the arms are deterministic.
  - The statistic is computed as integers: d̄_g × the LCM of the analysis set's site counts, which is 207480 for the full set.
  - The p-value is exact by enumeration over the m discordant problems when m ≤ 24. Otherwise use 99,999 flips plus the identity, drawn with `default_rng([D,0])`.
- **α = 0.03 at each step.**
  - g3 ran 132 null cells on the corrected structure: unconstrained hom/skew± and BAN-structured symmetric/asymmetric, with within-bug sharing ρ ∈ {0, .5, .9, 1}.
  - Worst size was 0.071 at α 0.05, 0.060 at 0.04 and 0.045 at 0.03.
- **α rule after the pilot:**
  - Rerun g3 at q ± 0.1, where q is the pilot's label-free share of sites where G's lexeme ≠ BAN's.
  - If the worst size at 0.03 exceeds 0.05, α becomes the largest value in {0.025, 0.02} with size ≤ 0.05.
  - This is fixed before any main graph is built.
- **m must be reported.** S2 cannot reject unless m ≥ 6.
- **Withdrawn:** the non-superiority test. Intervals are descriptive.
- **Secondaries (descriptive):**
  - flip rate and k* for each arm;
  - where t* lands, on lexemes: BAN's lexeme / other family member / outside the family / unscorable;
  - the rescue/break table against BAN;
  - P(fix | both flip) for G against GxA;
  - **existence rate:** the share of sites where some single ablation among ORC's 256 and G's top 128 yields a verified token other than BAN's lexeme, against 128 matched random single features;
  - **faithfulness:** Spearman(e, measured single-ablation Δ) over G's top 128 with |e| > 3δ, as a site median, with the count of qualifying features;
  - **program fix** by greedy generation for G, G-everywhere, BAN and NULL. Generation stops when the depth-0 `}` count on the lexer stream equals S's, or at `<eos>`/`<end_of_turn>`, capped at tokens_to_end + 64;
  - **collateral:** teacher-force S with its best repair and count argmax changes at positions ≥ h, G against RND-k;
  - margin terciles, family, slips.

## 6. Predictions
Rates are equal-weight over problems unless marked "per site".
- **Exclusions:** 0–4 sites.
- **BAN:**
  - Computed, not predicted: the HF runner-up is verified at 79 sites, fails at 17 and is unknown at 50. This table is committed.
  - Predicted TL rate: 0.50–0.70. The TL BAN lexeme differs from the HF runner-up at ≤ 10% of sites.
- **G:**
  - It flips at ≥ 90% of sites per site, with median k* of 2–20.
  - Its first feature is at layer ≥ 18 and position h−1 at ≥ 60% of sites per site.
  - t*_G equals BAN's lexeme at 60–85% of flips per site.
  - m is 6–16.
  - Fix rate 0.40–0.65.
  - D(G−BAN) lies in [−0.15, +0.03]; NF in [0, 0.08]; R−Bk in [−0.10, +0.05].
- **GxA:** flips at ≥ 90% of sites; fix rate 0.40–0.65; D(G−GxA) within ±0.06.
- **RND draw 1:**
  - It flips at 30–90% of sites per site, with median k* ≥ 100 among flips.
  - Fix rate 0.05–0.35; D(G−RND) ≥ +0.15.
- **Decisions:** S1 rejects; S2 does not; S3 is not reached.
- **Secondary arms:**
  - G-LP: fix rate within ±0.10 of G.
  - G1 flips at 5–25% of sites per site; G-mid flips at ≤ 40%.
  - ORC is within ±0.10 of G; CONTRAST is ≥ G + 0.10.
- **Spearman:** 0.3–0.7.
- **Existence rate:** 5–25% of sites per site, against ≤ 5% for random features.
- **Program fix:** G is 0.6–1.0 × its site fix rate; G-everywhere ≤ G; NULL ≤ 0.05.
- **Collateral:** 20–60% of sites for G, against 5–30% for RND-k.
- **Slips:** NULL fixes the HF-computed set ±3.

## 7. What each result would mean
A non-rejection always reads "the advantage was not detected; the 80% MDE is X" (§8). It never reads as equivalence. Mechanism claims come from the decomposition and the landing table, not from the tests.
- **S1 fails:** first audit the plumbing. If it is clean, graph-ranked flips land on repairs no more often than matched random flips.
- **S1 only:** the ranking matters, but an advantage over BAN was not detected. If t*_G equals BAN's lexeme at most flips, the "fix" is token suppression (descriptive). Together with Phase 7, the graph is not shown to be usable beyond cheap methods.
- **S1 and S2, not S3:** an advantage over the one-backward attribution was not detected. S3's power depends on κ (§8).
- **All three reject:** the first quantified support for §4.4. It is limited to one-token substitutions at known sites and to the node-0 target. Calling it "targeted" also needs low collateral.
- **D < 0:** the decomposition says whether the cause is G failing to flip or G flipping to the wrong place.
- **G-LP lands differently from G:** the S2 reading depends on the target.
- **Futility stop fires:** "the procedure did not flip in pool A; §4.4 was not tested on the verified set."

## 8. Power
Source: g4 on the corrected structure.
- Model: rescues only at BAN-fail sites, breaks only at BAN-fix sites (rate b), outcomes shared within a bug, unknown BAN outcomes imputed at 0.5, problem SD 0–1, α 0.03.
- D is the realised effect.

S2 power:

| b | D≈0.10 | D≈0.14–0.19 | D≈0.20–0.27 |
|---|---|---|---|
| 0 | 0.52–0.59 | 0.92–0.94 (D 0.15) | ≥ 0.997 |
| 0.05 | 0.35–0.46 (D 0.11) | 0.69–0.83 (D 0.18–0.19) | 0.89–0.96 (D 0.24–0.27) |
| 0.10 | — | 0.40–0.55 (D 0.14–0.16) | 0.64–0.84 (D 0.20–0.23) |

- **80% MDE for S2:**
  - about 0.13 when G never breaks a BAN fix;
  - about 0.19–0.22 at b = 0.05;
  - about 0.22–0.26 at b = 0.10.
  - The largest attainable D is about 0.37.
- **Joint S2 and S3,** where κ is the share of bugs at which GxA's outcome equals G's. At D(G−BAN) ≈ 0.22 and b = 0:

  | κ | D(G−GxA) | joint power |
  |---|---|---|
  | 0.3 | ≈ 0.15 | 0.92–0.95 |
  | 0.5 | ≈ 0.11 | 0.65–0.72 |
  | 0.8 | ≈ 0.045 | 0.05–0.07 |

- **"All three reject" therefore needs G to lead GxA by about 0.12 or more.** S1's power is about 1 when RND fixes at ≤ 0.15.

## 9. Pilot
The pilot never touches the 35 site problems.
- **P0:** no new graphs, about 1 GPU-h. It uses 20 retained graphs from `~/matched_graph_keep`, read-only, whose top logit has p ≥ 0.8.
  - **(0) Noise floor δ:** run each prefix at batch size 1 twice and as one row of a batch of 64. Record the maximum |Δ fp32 logit| and how often the top-2 order flips.
  - **(a) Unit tests for e:**
    - on a random DAG, e equals ((I−Ã)⁻¹−I)ᵀw·a in float64;
    - the first-order term equals `adjacency[N−n_logits, :n_sel]`;
    - features with no outgoing feature edges equal their logit-row entry;
    - at most 27 iterations; every value finite.
    - Then compare e with measured single ablations over the top 200 features with |e| > 3δ: signs should agree, and the median Spearman must be ≥ 0.3.
  - **(b)** At batch size 1 with k ∈ {1, 8, 64}, our hook must match `feature_intervention` bitwise or within 1 ulp. Batch-64 rows must match batch 1 within δ.
  - **(c)** A null hook must reproduce the clean logits bitwise.
  - **(d)** The `compute_batch` row must match the stored logit row.
  - **(e)** The TL argmax must match the stored top logit, and the fp32 argmax over C must match a chunked full-vocabulary fp32 argmax.
  - Any failure means debugging, never loosening the gate.
- **P1:**
  - 30 belief sites from pool A (38 problems with only unverifiable repairs), one per problem, drawn with `default_rng([D,3])`.
  - Run the full pipeline, including an injected fault and a restart. The resumed outputs must be identical.
- **The pilot may fix:**
  - code bugs;
  - the declared e fallback;
  - the ORC size: 256 + 128 if it takes ≤ 1 min per site, otherwise 128 + 64;
  - the generation cap, if truncation above 5% comes from a bug.
- **Futility rule.** If G flips fewer than 15 of the 30 pilot sites, build no main graphs; run BAN, GxA and NULL descriptively. P(stop) is 0.43 if the true flip rate is 0.5, 0.10 at 0.6 and 0.006 at 0.7.
- **The pilot may not change:** the population, order, target, ablation, propagation settings, K_max, arms, scoring, test or layer band. α changes only by the §5 rule.

## 10. Cost

| Step | GPU-h |
|---|---|
| P0 | ~1 |
| P1 (30 graphs plus arms) | ~3 |
| Stage A: graph-free arms (BAN, GxA, NULL, DE, slips), all sites | ~0.8 |
| Stage B: 146 graphs (empirical length curve) | 13.5 at q50 / 15.7 at q75 / 17.8 at q90 |
| Graph arms, generation, collateral (~1.5 min per site) | ~3.7 |
| **Total** | **~22–26** |

- **Coverage.** All 146 sites are expected within 28 h. If contention doubles graph time, round-robin ends after about 30 of the 52 tier-2 sites. Tier 1 alone takes 8.7–11.4 h of graphs.
- **Disk:**
  - Never call `to_pt`. Each site extracts at most 2 MB, under 0.5 GB in total.
  - After each site: delete the graph, call `empty_cache()`, and check that the transcoders, blocks, embed and unembed are all on cuda.
- **CPU judging:** 1–3 h, never while the GPU run is going.

## 11. Risks
- **e is unvalidated:** P0 checks it, and the declared fallback exists.
- **K_max = 1024 allows heavy perturbation:** k* and collateral are reported, and RND has the same cap.
- **Linearised selection versus the real model:** ORC and the faithfulness check measure the gap.
- **The target choice:** G-LP shows whether it matters.
- **Substitutions only:** insertions appear only through line completion and generation.
- **TL versus HF numerics:** handled by fp32 decision logits, the replay and exclusion (a).
- **Resampled bugs:** handled by the bug-shared simulation, tier 1 and equal weights.
- **The judge bug:** handled by the finite-aware judge and an audit.
- **Validity rests on simulation:** hence α 0.03 and descriptive intervals.
- **Amplification is not tested,** because its target needs the repair identity.

## Open decisions for the user
1. **A study-wide audit for the inf bug in Phases 1–7.** The same check sits in every phase's judge and covers the 11 typeable f64+isclose tasks. Possible effects:
   - Gemma outputs returning inf were labelled correct in Phases 1–3;
   - inf-returning mutants counted as survivors in Phases 4–5;
   - the repair labels used in Phases 6–7 contain the 33 false repairs.

   **Recommendation:** run the CPU audit before the Phase 8 pre-registration. Count label changes per phase, add an erratum to readme/report v11 without rewriting history, and recompute only the headline numbers whose labels change.
2. **The primary target.** Node 0 is faithful to the paper's graphs; the log-prob target better matches "the decision". **Recommendation:** keep node 0 primary and G-LP secondary.
3. **α 0.03 on the full set, or tier 1 only at 0.04 as the primary.** **Recommendation:** 0.03 on the full set, with tier 1 as a sensitivity analysis.
4. **Scope.** Full (about 22–26 GPU-h) or tier 1 only (about 15 h). Phase 7 makes S3 power ≤ 0.3 likely. **Recommendation:** full, with the hard stop. The comparisons against BAN, RND and GxA and the decomposition are the §4.4 result even if it is null.

## Implementation plan
**Scripts** (all new, no repo file edited):
- **`p8_judge.py`:**
  - In the worker initializer, patch `build_case` in every loaded copy of `1_rust_correct`.
  - Canaries returning INFINITY, NaN and an exact value must judge fail, fail, pass.
- **`p8_1_sites.py`:**
  - Re-judge all 759 mutants. The old path must agree 100% with the stored labels. The finite-aware path may differ only where an output is non-finite.
  - Rebuild the site table.
  - Build E and the lexeme map, the bugs, tiers and order, the HF BAN table, the neutralizing and rule-B flags, and the hashes.
- **`p8_sim.py`:** g3/g4 generalised.
- **`p8_0_checks.py`:** P0.
- **`p8_2_run.py` + `run_p8.sh`:**
  - One process per chunk, with a per-site ledger.
  - Atomic writes: write `.partial`, fsync, then rename.
  - Restart rather than retry; exclude a site only after 2 fresh-process failures.
  - A canary prefix at each process start, compared with a stored reference within δ.
  - RNG streams keyed per site; seconds per site logged.
  - Stages: A (graph-free), B (graphs plus graph arms), C (generation, collateral, CONTRAST). It records t*, k* and line completions, never labels.
- **`p8_3_judge.py`:** judges only inside `Pool(initializer=m1._init)` after the GPU run; cache keyed on the lexer stream; blind to the arm.
- **`p8_4_stats.py`:** integer or exact sign-flip, the decomposition, the descriptives.

**Tests before the pre-registration:**
- the judge canaries and the 759-mutant agreement;
- decoded splices equal the stored mutant code at all 759;
- E contains `wrong_id`;
- the stop rule, teacher-forced, ends exactly at S's last character for every source;
- the e unit tests and P0;
- the P1 restart test;
- a dry run of `p8_4` on the P1 outputs;
- `rustc --version` recorded, and no rustup update during Phase 8.

**Order:**
1. CPU work: judge, audit if chosen, site table.
2. P0.
3. P1.
4. Rerun the simulation and fix α.
5. Dry run.
6. Commit PREREGISTRATION_P8: hashes of the site list, E, order and code; rustc version; α; K_max; the predictions.
7. Stage A.
8. Stage B, up to the hard stop.
9. Stage C.
10. Blind judging, with the GPU idle.
11. Join the labels.
12. Analysis.

**Budget:** 22–26 GPU-h (capped at 28), under 0.5 GB of new disk, 2–4 CPU-h.

## Appendix: how each critic issue was handled

| Issue | How it was handled |
|---|---|
| Adv-1 (fatal): inf passes the judge | Confirmed (33 repairs; 146 sites / 31 problems / 94 bugs). Finite-aware judge, re-judge, audit offered as decision 1. |
| Adv-2: id-level BAN and out-of-family scoring | E(wrong) masked by BAN and used by the flip rule; one rule with lexeme map plus line completion; sensitivity on in-family landings. |
| Adv-3: the no-flip penalty depends on K_max | K_max fixed at 1024; decomposition pre-registered. |
| Adv-4: the target is not neutral | G-LP secondary; conclusion stated as conditional on the target; decision 2. |
| Adv-5: bf16 flip point | fp32 decision logits, batch-1 replay, δ-based exclusion. |
| Adv-6: S1 nearly guaranteed | RND ordered with the flip rule; reworded. |
| Adv-7: independent-site simulations | g3 with bug sharing on the corrected counts; α lowered to 0.03. |
| Adv-8: neutralizing repairs | Syntactic flag plus a sensitivity analysis. |
| Stats-1: RND not exchangeable | Ordered RND with the same flip rule and K_max; RND-k kept descriptive. |
| Stats-2: power | §8 rebuilt (structured, bug-shared, realised D, joint power). |
| Stats-3: §7 read as equivalence | Rewritten as MDE readings; decomposition. |
| Stats-4: null wording and ties | H0 restated; integer statistic; exact enumeration. |
| Stats-5: the size rerun has no consequence | α rule pre-registered. |
| Stats-6: BAN prediction not blind | HF table committed; range labelled as computed. |
| Stats-7: incomplete predictions | Every predicted quantity given a numeric range and weighting; sensitivity analyses cannot change the decision. |
| Stats-8: futility | Reworded; operating characteristics stated. |
| Feas-1: P0 gates tighter than bf16 resolution | Noise floor measured; bitwise tests at batch 1; |e| > 3δ restriction. |
| Feas-2: in-process retries | Chunked driver, ledger, restart, canary, fault test. |
| Feas-3: cost estimate | Empirical length curve; whole-phase hard stop; graph-free arms first; allocator setting. |
| Feas-4: token-id variants | E set and lexeme map. |
| Feas-5: brace-based stop rule | Lexer-based stop plus the teacher-forced check. |
| Feas-6: judging environment | Judging only in the separate step; toolchain frozen; re-judge agreement required. |
| Feas-7: indexing traps in e | Unit tests; float64 on CPU. |
| Feas-8: RND pool can run empty | Exclusions limited to the top 64 of G and GxA; fallback chain; exact-match share reported. |
| Feas-9: slip arms undefined | NULL only, with the reason stated. |
