# Pre-registration — the closing comparison on the retained pairs (Phase 7)

Written 2026-10-14 (Japan time). Two things existed before this document was
committed, and both are listed in §2 under "Before this commit":
- the published values that §3's gate G1 reproduces;
- the label-free outputs of the extraction p7_0.

No new number that uses a label existed then. Phase 7 makes no sample, mutant
or graph. It reads the pairs, rows and retained graphs that Phases 4 and 5 left.

## 1. Why this analysis

**What Phase 5 left open.** Phase 5 (`PREREGISTRATION_MATCHED.md` §9) found that
the graph features separate a passing one-token mutant from a failing one at
identical length and matched code log-probability (Ac 0.641; S1c 0.639, p
0.005). It also found that the failing member's pruned graph is the larger. It
did not show that the graph knows more than cheaper signals:
- the rest of the output distribution separates the pairs as well (S2+c 0.649);
- the edit's surface identity separates them better (E 0.676);
- the graph adds +0.012 [−0.074, +0.098] to E.

These are side-by-side comparisons. No direct test of "beyond the cheap
signals" was ever run, and no probe from the same forward pass was ever
compared.

**What Phase 6 left.** Phase 6 (`PREREGISTRATION_OWNFIX.md` §9) could not build
pairs of the model's own errors. It found two things:
- 47% of checkable one-token repairs pass MBPP's asserts by luck;
- a post-hoc fit on Phase 5, β_5 = −0.461 (p 0.039), which is a lead from a
  linear adjustment, never size-checked on these pairs.

**Three questions the retained data can answer without a GPU:**
1. **The closing comparison.** Does the graph carry information about which
   member passes beyond a cheap baseline C, the edit's identity and the output
   distribution? Beyond a probe built from the same forward pass without
   attribution? Do its attribution-free columns alone come close to Ac?
2. **The label.** Phase 5's survivors pass three asserts. How many are correct
   beyond them, and does any signal tell a lucky survivor from a verified one?
3. **The resolution.** The 49 features summarise the whole graph. Do local
   quantities at the edit add to C? Without new graphs, this is the only test of
   "the summary is too coarse".

### A second look at analysed data, not a confirmation

The pairs and problems are Phase 5's, and the hypotheses come from its results.

**Looks already taken on Phase 5's pairs (259 pairs, 78 problems):**
- Ac, S2c, S2+c and S1c (§9a there);
- the G Wilcoxon (p 1.7 × 10⁻⁷) and the per-problem sign test (45 against 21,
  post hoc);
- the per-feature Wilcoxon with BH (23 of 49 features at q < 0.1);
- E 0.676; G − E −0.035 and GE − E +0.012 (`p5_6`);
- β_5 −0.461, CR2 [−0.897, −0.024], p 0.039, and the two frozen forests
  (`p6_3b_reference.py`).

**Looks already taken on Phase 4's pairs (98 analysed pairs):**
- A 0.469, S1, S2, Ac 0.674, S2c 0.694, S1c 0.592;
- E 0.667 (99 pairs);
- G/E/GE (GE − E +0.036).

**Computed while this plan was designed** (scripts and logs in `~/p7design`, not
in the repository). None of these compares a survivor with a killed member:
- The Spearman correlation of Δlog(1 + nodes) with each part of ΔC: at most
  0.17 in absolute value.
- The problem-grouped CV R² of each graph-feature difference from ΔC: median
  +0.005, Δnodes 0.10.
- The linear CV R² of the attribution-dependent differences from the
  attribution-free ones: median −0.013, column 44 0.66.
- Prefix bit-identity in 10 pairs, and one graph's layout (graph_68).
- How often the attribution differs on a shared prefix (3 pairs).
- A crude simulation of increment tests on the real covariates, with labels
  redrawn from C (`trap_sim`).
- Counts of verifiable problems from MBPP's Python references (no Rust compiled).

**Never computed on real labels before §3 runs:**
- A(C) for C as defined here;
- any forest with P or L;
- A(G_free) and A(G_attr);
- any conditional-independence test given C;
- any verification of a Phase 4 or 5 survivor, except the two-problem pipeline
  test in §2.

### What the simulation fixes in the design

Two findings of `trap_sim` fix the design.

**1. An increment A(G + C) − A(C) cannot be tested with the nulls tried before.**
- A label swap tests the false global null that nothing separates the pairs, and
  gives the wrong spread.
- A G flip or permutation breaks G's dependence on C. In the simulation it
  rejected 4 of 12 true nulls at nominal 0.05.
- So did the generalised covariance measure (GCM) when the simulation shared its
  folds with the design that drew the labels: it rejected 2 of 12, with a mean T
  of −1.15. That design coupled one problem's label to other problems' graphs;
  the simulation here avoids that coupling (§6).

**2. Under the null the increment is not centred at 0.** Adding the 49 columns
lowered A by about 0.01, because the extra columns dilute the forest. So:
- increments are estimated, and read after subtracting their null centre;
- what is tested is conditional independence, with a test whose size is checked
  on the real covariates before any real label is read.

## 2. Design — fixed

### Pairs and graphs

**Pairs.** Phase 5's 259 pairs over 78 problems (`matched_pairs.json`,
`matched_rows.jsonl`, `gemma_matched_features.jsonl`). None is dropped: a pair
that fails a check in §3 stops the run.

Phase 4's 98 pairs with both graphs (`mutant_*`) are analysed descriptively
only; pair 633 lacks its killed graph. Phase 4 is weaker evidence:
- 73 of its 98 problems are Phase 5's;
- its log-probabilities used sdpa attention, without the soft-capping the
  graphs used;
- it was not matched.

**Graphs.** The retained graphs are only read: `~/matched_graph_keep` (518 files)
and `~/mutant_graph_keep` (197 files; one is an orphan whose partner failed).
- They are unpickled with the installed circuit-tracer
  (0.5.3.dev1+g6018ed8d3, the one that built them). The scripts run from
  `data/`, never from the repository root.
- No graph is built, moved, relinked or deleted.
- Each file is dropped from the page cache after it is read.

**Orientation.** Every label-free step (p7_0, p7_2, p7_3) orders a pair's
members by a rule that uses no label: the member whose generated code sorts
first comes first ("canonical").
- Only `p7_common.orientation` reads which member passed MBPP's asserts. It is
  called by the reproduction gate G1 (published values only) and by p7_4.
- Pairs keep their places in the pair file. The survivor-first order is
  therefore Phases 4 and 5's own row order, and every published A reproduces
  exactly.

### Blocks

All blocks are per member. Before any forest, each pair's two members are
centred on their mean, as for Ac.

**C, the cheap baseline (65 columns),** in this order:
- **E,** the edit's identity, as encoded by `p5_5_edit_baseline.encode`: the
  replaced token, the new token, the family, and the position as a share of
  the code (58 columns).
- **O,** the output distribution:
  - S2+ as in `p5_4`: code log-probability, edit-token log-probability, lowest
    token log-probability, mean entropy, characters (5 columns);
  - the final position's top-logit probability and logit entropy, graph
    features 3 and 4 (2 columns).

Adding O's last two columns to the approved C was approved on 2026-10-14
(06:20 JST), before this commit.

S2+ is computed only at the code tokens. The final position's distribution
comes from the same forward pass, before any attribution. A label-free check
found that ΔC without it predicts Δf3 and Δf4 not at all (CV R² −0.03 and
−0.01). Leaving it out of C would let the test of the graph succeed on two
output-distribution columns.

**C63 = E + S2+** is the baseline approved on 2026-10-13. It is reported as a
sensitivity: its increments, and T without a p.

**C⁺** adds two columns, computed as in `p6_3b_reference.py` (the code is
copied into `p7_common.cplus_columns`). It is used only as a sensitivity:
- the tokens after the edit;
- the 7-level syntactic context.

**G, the graph summary: the 49 stored features.**
- The forests use all 49 as stored (Ac's input).
- **The tests** use G43, which leaves out six columns:
  - columns 42, 47 and 48, which are constant;
  - column 43, which is twice column 41;
  - columns 3 and 4, which are in C.
- Each column is replaced by the normal scores of its ranks over all members.
  This is label-free. It tames heavy tails: column 44's difference has a
  kurtosis of 22.
- **G_free** = {0, 3, 4}: active features, top-logit probability and logit
  entropy. The forward pass alone gives these.
- **G_attr** = the rest: 46 columns in the forests, 42 in the tests (G43 without
  column 0). They depend on the attribution, through its pruning or through the
  activations of the nodes it selected (8–10).
- **f0** = log(1 + feature 0) is a conditioning column.

**z, graph size.** z = Δlog(1 + feature 1)/c, first member minus second, with
c = 0.004917333931110322 (`results/ownfix/reference.json`).

**P, the internal probe (45 columns).** It is built from the forward-pass
record in each graph file: every active transcoder feature at every position,
not only those the attribution selected. The adjacency is not read.
- For each member, the activation of each feature (layer × 16384 + feature) is
  summed over positions and divided by the position count.
- A truncated SVD (ARPACK, start vector from stream 12) of the 259 pairs'
  differences gives the 45 leading right singular vectors. Each member's centred
  vector is projected on them.
- A pair's swap does not change these vectors, so the basis uses no label.
- **P⁺ = P + f0 + O (53 columns)** is the forward pass's whole cheap content:
  activations, the activation count, and the output distribution.
- **P_layer** (descriptive) gives, per layer, log(1 + active features) and
  log(1 + summed activation): 52 columns. Summed over layers, its counts give
  f0 exactly.

**L, local quantities at the edit (8 columns).** Both members are measured at the
same positions.
- **D** = the positions where the members' tokens differ: one position in 93
  Phase 5 pairs, two in 166.
- **W** = the positions 1–3 after a position of D, not in D and not the last
  position.
  - W is local on purpose: a window running to the end would reconstruct f0.
  - Its median size is 5 (Phase 5) and 3 (Phase 4).

L_attr uses the attribution:
- **b1:** D's share of the influence of all non-logit nodes. Influence comes from
  the installed `compute_node_influence` on the full adjacency, with the logit
  nodes weighted by their probabilities. There is no pruning.
- **b2:** error influence in D / (feature + error influence in D). It is 0 if
  both are 0, and the number of such cases is reported.
- **b3:** log(1 + Σ|w|) over the edges from D's nodes to nodes at later
  positions and to the logits.
- **b4:** the share of D's active features that the attribution selected.

L_act uses the forward pass and all active features:
- **a1:** log(1 + active features in D).
- **a2:** log(1 + summed activation in D).
- **a3:** log(1 + mean active features per position of W).
- **a4:** log(1 + mean summed activation per position of W).

The block order is [b1–b4, a1–a4]. The tests use the members' normal scores.

The graphs' logit targets are the tokens after the closing brace. L_attr
therefore measures how much the edit contributes to that end-of-function
prediction.

**N45 (implementation control).** Each pair gets 45 columns of N(0, 1) noise
(stream 11); half is added to the first member and half subtracted from the
second.

**G43sh (shadow control).** The shadow is m̂(C) of G43 plus another problem's
residual, with a random sign per pair (stream 11). Its nuisance is fitted afresh.
- Y is independent of it given C by construction.
- It depends on C the way G does, so a rejection reflects the nuisances' bias
  under the real P(Y | C).

### Verification of the survivors (`p7_1_verify_p45.py`)

It runs on the 103 problems of Phases 4 and 5 at once. It uses Phase 6's extra
inputs and judge (`p6_1b_verify.py`, imported, not changed):
- each assert's arguments are varied one at a time;
- MBPP's Python reference gives the expected values;
- an input is kept only if the problem's passing Gemma samples (whitespace-
  distinct, in sample order) all return the reference's value on it.

**Rule A** is Phase 6's: the first 10 passing samples calibrate.
- It is gate G4: on the problems Phase 6 verified, it must reproduce Phase 6's
  kept-input counts.
- Its survivor verdicts are descriptive.

**Rule B** gives the verdicts. The calibration samples leave out the pair's
source and its survivor (compared with whitespace removed). This is needed
because the source is among the first ten in 227 of Phase 5's 259 pairs and in
every Phase 4 pair. Rule A would verify the source by construction and pass its
blind spots to the survivor.

Each survivor gets one verdict:
- **verified:** right on every kept input.
- **lucky:** a wrong value or a panic on at least one; the inputs are recorded.
- **error:** no PASS line, a timeout or an abort. These are kept apart.
- **unverifiable:** any one of:
  - fewer than 10 candidate inputs;
  - no calibration sample is left, or none compiles and finishes;
  - fewer than 10 kept inputs.

The source's verdict comes from its own run on all candidate inputs, restricted
to the pair's kept inputs, with the same four values.

Descriptive: whether the survivor returns the source's values on the kept inputs
(p6_1's differential harness).

**Safety.** Gemma's Rust is compiled and run only in Pool workers (6), as in
Phases 4–6. Each worker runs:
- under an 8 GiB address-space limit;
- with the judge's time limits (compile 60 s, run 10 s);
- with an empty temporary directory as its working directory, removed at the
  end.

MBPP's references run in the same workers, so under the same limit: python -I,
their own temporary directory, 20 s.

Expected: about 185 of Phase 5's 259 survivors verifiable (175–193) and about 68
of Phase 4's 98. Every test keeps the label as recorded: passes MBPP's asserts.

### Resources

Measured 2026-10-14:
- **C:** 1,023 GB (952.6 GiB). About 453 GB (422 GiB) is free.
  - WSL's virtual disk lives on C:, so the larger free space WSL reports is not
    the limit.
  - The PC's second disk, a 2.0 TB HDD, holds another system's unmounted
    partitions and is not touched.
- **New outputs:** under 1 GB.
  - In the repository, under 50 MB (`data/results/closing`).
  - In `~/closing_work`, about 0.4 GB: the position means and the two SVD bases.
- **GPU:** none is used.
- **Retained graphs:** about 343 GB (320 GiB). They are kept until the write-up
  and the key scrub are done; rebuilding one costs 10–11 GPU minutes.

### Scripts

Driver: `data/run_closing.sh`, with `PHASE=extract|verify|prepare|sizesim|closing`.

| script | what it does | labels |
|---|---|---|
| `p7_common.py` | loaders, blocks, the seeded forest, the GCM, Holm, streams | — |
| `p7_0_extract.py` | graphs → P's raw material, P_layer, L; gate G2 | none |
| `p7_1_verify_p45.py` | the verification; gate G4 | compiles Gemma's Rust |
| `p7_2_prepare.py` | gate G1, the blocks, P's basis, the nuisances m̂ | G1 only (published values) |
| `p7_3_sizesim.py` | size, null bands, power; gate G3 | synthetic only |
| `p7_4_closing.py` | the analysis, run once | yes |

### Before this commit

Test outcomes:
- **Tokens.** p7_0's tokenizer mode passed on every pair.
  - Phase 5: |D| = 1 in 93 pairs and 2 in 166.
  - Phase 4: 56 and 42.
  - Every member's edit span meets D. W is never empty; the least distance from
    D to the last position is 2.
  - p5_2's edit token lies in D for every member but one: Phase 4's problem 113,
    whose survivor edit '..=' → '..' changes the next token. This is recorded,
    and is not a failure.
- **Two retained pairs per phase.** p7_0 read 8 graphs in 14 s (peak 2.3 GB).
  - Features 0, 3 and 4 were recomputed exactly; the bf16 arithmetic is
    graph_dataset's.
  - The prefix activations were bit-identical in all 4 pairs.
  - The attribution on the shared prefix differed in 2.7% and 3.7% of edges
    (medians).
- **G1, on the real stored features.** Every published value reproduced exactly:
  - Phase 5: G 0.640927, E 0.675676, GE 0.687259, S2+c 0.648649;
  - Phase 4: G 0.673469, E 0.642857, GE 0.678571.

  The seeded forest equals `w3.oof` bit for bit. C⁺ reproduced c, β_5 −0.4606
  [−0.8970, −0.0242] and S_U −13.183 (55 columns).
- **p7_2, p7_3 and p7_4 on random stand-ins** for p7_0's outputs (outside the
  repository), with p7_3 at 3 datasets per cell.
  - p7_3's designs are odd: swapping a random half of the pairs flips η there,
    to 1 × 10⁻¹⁵.
  - p7_4 ran end to end on synthetic orientations, never the real labels (36
    min with the 10 extra seeds). Over real G and C, with random stand-ins for L
    and P:
    - With no signal, nothing was rejected: H1 p 0.94, H2 0.25, S1–S3 0.80–0.93;
      G5a p 0.11, G5b 0.65.
    - With the synthetic killed member's feature 1 raised by one within-pair sd,
      H1 was rejected (max |T| 7.1, p 0.0001), and Romano–Wolf named f1 alone.
      S1 was rejected with T −7.2, and S2 and S3 were rejected too.
    - With b1 raised the same way, H2 was rejected (max |T| 6.7, p 0.0001), and H1
      was not.
    - A withdrawn test is never rejected: with the stand-in simulation's
      withdrawals, H1 and S1 stayed unrejected at p 0.0001.
- **The verification driver on two problems** (6 and 58: 6 Phase 5 pairs and 2
  Phase 4 pairs).
  - G4 passed: rule A reproduced Phase 6's kept inputs exactly on both problems.
  - Disclosure: the driver printed its verdict counts. Phase 5's 6 survivors
    were 2 verified and 4 lucky; Phase 4's 2 were 1 and 1. These 8 pairs are
    part of the full run, and §4's prediction on the lucky share was written
    after seeing them.
- **p7_0 ran on every analysed graph before this commit** (2026-10-14 05:17–05:32
  JST, 852 s, peak 2.7 GB). It reads no label. Its output hashes are in §8.
  - It read 714 graphs (518 + 196); the orphan is not read. G2 passed.
  - Prefix activations were bit-identical in every pair of both phases.
  - The attribution on the shared prefix differed in a median 3.8% (Phase 5) and
    3.9% (Phase 4) of edges; row sums differed by a median 1.6 × 10⁻⁴.
  - The two members' logit counts differ in 6.2% (Phase 5) and 11.2% (Phase 4)
    of pairs.
  - b2's denominator was never 0, and no member has an empty D.
  - Outputs: 2.9 MB in `data/results/closing` and 206 MB in `~/closing_work`.

## 3. Analysis — fixed

The problem is the unit of every interval and test.

### Forests

The forests are Phase 3's, unchanged (`w3_within_analysis.oof`):
- `RandomForestClassifier(n_estimators=100, random_state=42)`;
- 10 repeats × 5-fold `GroupKFold` over problems;
- fold permutations `default_rng(100 + rep)`.

A is the share of pairs whose survivor scores higher, with ties counted ½.
Blocks are concatenated tested block first, in the order named, because a
forest's result depends on column order.

### Gates, before any new number

- **G1, reproduction.** p7_2 stops (exit 2) unless the values listed in §2
  reproduce exactly and the seeded forest equals `w3.oof`. If C⁺ does not
  reproduce, C⁺ is not reported; nothing else stops.
- **G2, integrity.** p7_0 stops (exit 2) on any failure:
  - the graph counts;
  - a file name's row or expr_id;
  - input tokens other than the tokenizer's;
  - an active feature at position 0;
  - features 0, 3 or 4 not recomputed exactly;
  - |D| not 1 or 2, unequal lengths, or a member's edit span missing D;
  - an empty W;
  - the node layout.

  Recorded, never used to drop a pair:
  - the share of pairs whose prefix activations are bit-identical;
  - how often the attribution differs on the shared prefix;
  - the logit counts.
- **G3, size (§6).** Checked on synthetic labels before the real labels are
  read.
- **G4, verification.** Rule A must match Phase 6, or the verification is not
  used. A driver bug may be fixed once; the fix is recorded in §8.
- **G5a, implementation control.** If the test of N45 given C has p < 0.01,
  p7_4 stops before the primary family (exit 4) for an audit.
  - If the audit finds nothing, p7_4 is rerun once with the same streams. The
    rerun is recorded in §8.
- **G5b, shadow control.** If the test of G43sh given C has p < 0.01, H1's
  reading carries the note "nuisance bias suspected". Nothing stops.

### The test: a clustered generalised covariance measure

H0: Y ⟂ B | Z, where:
- B is a tested block;
- Z is a conditioning set;
- Y is which member passes.

Take the survivor as the first member, so Y = 1 for every pair. For each pair i:

**p̂_i, the probability that the first member passes given Z:**
- Phase 3's forest is fitted on the members' rows [centred Z, the pair's
  levels], out of fold.
- The levels are the pair means of Z's columns that vary over pairs, including
  E's one-hots. In 150 of the 259 pairs both members replace the same token,
  and the levels carry that shared token.
- The scores are symmetrised: π_i = (s_a + 1 − s_b)/2, clipped to [0.01, 0.99].
- They are recalibrated by a slope without intercept: p̂_i = σ(λ logit π_i).
  λ ∈ [0.1, 10] is fitted on the other folds of repeat 0.

**m̂_ij = E[ΔB_ij | Z]**, with no label:
- The model is `RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
  max_features=1/3, random_state=42)` on x = [ΔZ, levels].
- It is fitted on mirrored training pairs {(x, ΔB_j), (x⁻, −ΔB_j)}, where
  x⁻ = [−ΔZ, levels].
- It predicts (f(x) − f(x⁻))/2 out of fold on the same folds, averaged over the
  10 repeats.

**The statistic:**
- R_ij = (1 − p̂_i)(ΔB_ij − m̂_ij).
- U_gj = Σ_{i∈g} R_ij, summed within problem g.
- T_j = Σ_g U_gj / √(Σ_g U_gj²), and S = max_j |T_j|. For a single column, S is
  |T| and the test is two-sided.
- A column whose differences are all zero is left out and reported.

**The p-value:**
- p = (1 + #{S* ≥ S})/(1 + 9999), where S* = max_j |Σ_g ε_g U_gj| / √(Σ_g U_gj²).
- The ε_g are Rademacher signs, one per problem (stream 2). Every test uses the
  same draws.
- Within a rejected hypothesis, the Romano–Wolf step-down on the same draws
  names the columns.

The effect size of a column is ρ̂_j = Σ_i R_ij / √(Σ(1 − p̂_i)² · Σ(ΔB_ij − m̂_ij)²).

A pair's swap changes neither R nor the test, so the orientation enters only
through Y.

**Version 2.** If §6 fails version 1 for a conditioning set, version 2 replaces it
on that set:
- p̂, before the slope, is averaged with a logistic regression (C = 1);
- m̂ is averaged with a ridge (λ = 1).

Both are fitted without intercept, on standardised mirrored pairs, on the same
inputs and folds. If version 2 also fails a test, that test is withdrawn: T is
reported without a p, and the hypothesis counts as not rejected.

### Primary family (weighted Holm, family-wise α = 0.05)

- **H1, the closing comparison:** Y ⟂ G43 | C (43 columns), at α 0.035.
- **H2, the resolution:** Y ⟂ L | C (8 columns), at α 0.015.

If one is rejected, the other is tested at 0.05. A withdrawn test is not
rejected, and its level is not passed on.

### Secondary family (Holm over three, α = 0.05; read only as secondary)

- **S1, graph size: Y ⟂ z | C.** β_5's question, with a nonlinear adjustment and
  without β_5's extra columns. The sign of T is reported: negative means the
  killed member's graph is the larger.
- **S2, beyond the probe: Y ⟂ G_attr | P⁺** (42 columns).
- **S3, attribution: Y ⟂ G_attr | C + f0** (42 columns).

### Descriptive T (no p is read)

- z | C + f0;
- L | C + G43 and G43 | C + L;
- G45 | C63, L | C63 and z | C63;
- G43 | C⁺, L | C⁺ and z | C⁺.

Also: the range of H1's, H2's and S1's max |T| and p over 10 refits of p̂ with
other fold and forest seeds (stream 8). The decisions use the main seeds.

### Estimates (descriptive)

Every A and every difference gets a 95% interval from 5000 bootstrap draws of
problems (stream 1). The draws are the same throughout, so differences are
paired.

The intervals resample fixed out-of-fold scores, so they omit the forests'
refitting noise. That noise is shown beside each estimate: its range over 10
other fold permutations and forest seeds (stream 8), flagged when the main value
falls outside their 10–90% range.

**A of each block:** C, E, S2+, S2+ without the edit-token log-probability, O,
G, G_free, G_attr, P, P⁺, P_layer, L_act, L_attr, L, N45, C63, and [E, edit-token
log-probability].

**Increments.** Each is Δ = A(tested + conditioning) − A(conditioning):
- Δ(G | C), Δ(L | C), Δ(N45 | C), Δ(L | G + C);
- Δ(L_attr | L_act + C);
- Δ(G | P⁺), Δ(G | P + f0 + C);
- Δ(G_attr | G_free + C) = A(G + C) − A(G_free + C);
- Δ(G | C63), Δ(L | C63), Δ(G | C⁺), Δ(L | C⁺);
- the differences A(G) − A(G_free), A(P) − A(G) and A(P⁺) − A(G).

**Correction for dilution.** Each increment with a null band in §6 is read as
Δ̃ = Δ − c0, where c0 is the band's mean under design L. Its interval is shifted
by the same amount, and the band's 2.5–97.5% range is shown beside it. A(G) −
A(G_free) has no band and is read uncorrected.

**Same-site pairs (93).** Δ(G | C) and Δ(L | C) from the fixed scores, with their
own problem draws (stream 1, key 1).

### Verification (descriptive)

**Counts.** For each phase, survivors and sources as verified / lucky / error /
unverifiable, and the source × survivor table. Then, from the fixed scores, if
the verified pairs number at least 40 over at least 20 problems:
- every A and increment above, on the verified pairs;
- as a sensitivity, Δ(G | C) and Δ(L | C) refitted on those pairs alone;
- H1's T on them, with the all-pair nuisances.

**Verified − lucky.** For G, C, E, S2+, P, G_free, L, G + C, L + C and z (counted
right when the killed member has more nodes), A on verified pairs minus A on
lucky pairs:
- with 95% intervals from 5000 cluster draws of problems (stream 9);
- also within the problems that hold both kinds;
- only if each side has at least 20 pairs over 10 problems;
- within-problem figures only if at least 10 problems hold both.

An error is neither verified nor lucky. As a sensitivity, errors are counted as
lucky.

### Phase 4 (descriptive)

- The same estimates, with C4 = E + [sdpa code and edit-token log-probability,
  characters, final-position top probability and entropy].
- P from Phase 4's own SVD (stream 13).
- T, without p, for H1, H2 and S1, with version 1 nuisances.
- One pair per problem: the bootstrap draws problems (stream 14).
- Verification readout as above (stream 10).

C4 lacks the lowest log-probability and the mean entropy, and its log-probabilities
come from a different numerical model than the graphs'. Phase 4's increments
therefore lean towards the graph and are never read as support.

### Random streams

Every stream is `default_rng([20261014, s, …])`:

| s | use | s | use |
|---|---|---|---|
| 1 | bootstrap, Phase 5 (key 1: same-site) | 9 | verification intervals, Phase 5 |
| 2 | Rademacher signs, real labels | 10 | verification intervals, Phase 4 |
| 3 | synthetic labels and their signs | 11 | N45 and the shadow block |
| 4 | synthetic designs | 12 | SVD start, Phase 5 |
| 5 | reserved | 13 | SVD start, Phase 4 |
| 6 | power | 14 | bootstrap, Phase 4 |
| 7 | null bands | 15 | fresh size draws for a version-2 fallback |
| 8 | other fold and forest seeds (j = 0…9) | 16 | S_U reproduction (its p is discarded) |

Phase 3's fixed forest seeds (100 + rep, 42) are kept, so that every earlier
number reproduces.

Never reused: `[0, 10…61]`, `[20261012, 1…8]`, 20261005, 20261008,
20261001 + …, and the design simulation's `[77, …]`.

`p7_4` runs once. Any second run is a deviation recorded in §8, and needs
`RERUN_DEVIATION`.

## 4. Predictions

Written by the author before any number in §3 that uses a label. My record so
far: of five predicted nulls, four failed in the paper's favour. Phase 6's
prediction was never tested.

**H1: not rejected,** held weakly.
- A β_5-sized effect in one column is found inside a maximum over 43 correlated
  columns with probability of only about 0.1–0.3.
- Phase 5's GE − E was +0.012.

**H2: not rejected.** The activations at the edit mostly re-encode the edited
token, which E already holds.

**S1: not rejected at Holm's level,** with T negative. β_5's p of 0.039 came from
a linear fit with more baseline columns, and was never size-checked.

**S2 and S3: not rejected.**

**A(G) − A(G_free): the interval covers 0.**

**The lucky share among verifiable Phase 5 survivors: 25% or more.** This
prediction is not blind: it was written after the pipeline test showed 4 lucky
of 6 on two problems (§2).

## 5. What each outcome means

### The primary hypotheses

H1 (the graph) and H2 (the local quantities) are each read with their corrected
increment Δ̃(· | C):

| test | Δ̃(· \| C), 95% interval | reading |
|---|---|---|
| rejected | lower bound above 0 | it carries information about which member passes beyond C, and the forest ranks better with it. A second look at analysed pairs: new problems are needed to confirm it |
| rejected | lower bound ≤ 0 | dependence beyond C exists, in the columns named; a gain in ranking is not shown |
| not rejected | upper bound below +0.05 | beyond C it adds less than 0.05 in within-pair accuracy on these pairs: practically nothing |
| not rejected | upper bound ≥ +0.05 | not detected; a gain up to the upper bound is not excluded. The power of §6 is quoted beside it, and absence is not claimed |

H1 is read with three further notes:
- If Romano–Wolf names only f0, H1's dependence is read as forward-pass
  information, not attribution.
- If G5b noted bias, H1's reading carries the note.
- A rejection is a second test on the pairs that produced the lead, never a
  confirmation.

| H1 | H2 | together |
|---|---|---|
| rejected | rejected | information beyond C in the graph's summary and in the local quantities |
| not rejected | rejected | L shows dependence beyond C, and G43 is not shown to, at the power of §6. "The summary is too coarse" is written only if the descriptive T of L given C + G43 stays above 1.96 and Δ(L \| G + C) is above its band; otherwise the local quantities may carry the same information as the summary |
| rejected | not rejected | dependence beyond C in the graph's summary; the local quantities are not shown to add |
| not rejected | not rejected | the closing negative: on these pairs the graph is not shown to know more than the edit's identity and the output distribution, at the power of §6 |

Δ(L_attr | L_act + C) says, descriptively, whether attribution is needed at the
edit.

### The secondary hypotheses

**S1.**
- Rejected with T < 0: β_5's lead survives a nonlinear adjustment for C. Graph
  size carries information beyond C, in a second test on the pairs that produced
  the lead.
  - If the descriptive |T| given C + f0 falls below 1.96, the reading adds: "not
    shown to need attribution; the number of active features may carry it".
- Rejected with T > 0: the opposite sign to β_5; reported, not interpreted.
- Not rejected: β_5's result is not reproduced under a nonlinear adjustment for
  C. This is not a refutation:
  - the conditioning differs from β_5's;
  - the power is in §6;
  - T given C⁺ is shown beside it.

**S2.**
- Rejected: the attribution-dependent columns carry what the forward-pass probe
  P⁺ does not.
- Not rejected: beyond P⁺ the attribution is not shown to add. A(P⁺) − A(G) is
  reported with its interval, and no equivalence is claimed.

**S3.**
- Rejected: the attribution-dependent columns add to C and f0. The named columns
  are reported; feature 1 is a known lead.
- Not rejected: attribution is not shown to be needed beyond C and f0.

**A(G) − A(G_free).**
- Interval above 0: the attribution-dependent columns contribute to Ac.
- Interval below 0: the three attribution-free features rank better than all
  49; at this n the other 46 columns dilute the forest.
- Point estimate ≤ 0.03, interval covering 0: "three attribution-free features
  come within 0.03 of Ac at this n", with the interval. Equivalence is not
  claimed; the half-width is about 0.08.
- Otherwise: undecided.

**Δ(G | P + f0 + C)** is read like H1's row, descriptively.

### Verification (descriptive)

- A_lucky's interval covers 0.5, and the verified − lucky interval is above 0:
  the signal follows correctness, not the asserts.
- A_lucky is above 0.5, and the verified − lucky interval covers 0: the signal
  follows whether the edit changed behaviour on the asserts; correctness is not
  distinguished. E and C are read on the same rows.
- Lucky survivors are 20% or more of the verifiable ones: Phase 5's label is
  noisy, and the write-up says so.
- The verified-only increment differs from the all-pair one: this is read from
  their paired difference over the same problem draws, not from overlapping
  intervals.

Phase 4 is descriptive and is never called a replication.

### Gates

- G1 or G2 fails: stop.
- G3 fails a test: that test reports T only.
- G4 fails: no verification figures.
- G5a below 0.01: audit before reading.
- G5b below 0.01: a note on H1.

No threshold above is adjusted after the data are seen. A deviation, if one
proves necessary, is recorded in §8 with the original alongside.

## 6. Size and power (`p7_3_sizesim.py`)

The simulation uses synthetic labels only, on the real covariates, before the
real labels are read. It loads the label-free blocks and nuisances and never
calls the function that reads the labels.

The first member (canonical) passes with probability σ(kη). η is odd in the
pair differences: a pair's swap flips its sign. k sets the oracle accuracy,
mean σ(|kη|). Pairs are put passer first, p̂ is refitted on each dataset, and the
real m̂ is reused, since it uses no label.

### Designs, for each conditioning set

Every scale is an uncentred RMS: no mean, and so no orientation, enters.

**L: linear.**
- The RMS-scaled differences are grouped in blocks: E's one-hots or P's
  directions, and the continuous columns.
- Each block contributes a random linear combination (w ~ N(0, I)) with an equal
  share.
- Columns with fewer than 10 nonzero pairs are left out, so one rare token
  cannot carry the design.

**Q: nonlinear,** with each part scaled to unit RMS:
- Δ × standardised level, for the continuous columns;
- Δz_j·|Δz_k| for 10 random pairs (j, k);
- u|u| along a random direction;
- for C and C + f0, a random effect of each member's (replaced, new) token pair,
  differenced.

**W: adversarial,** one design per tested block.
- The target is the first right singular direction of the block's RMS-scaled
  differences (uncentred).
- It is predicted from the 5 columns of ΔZ most correlated with it, by OLS
  without intercept, fitted once on all pairs. One problem's label therefore
  depends on other problems' graphs only through 5 coefficients.
- η = m + m³/2.
- For C there are two: W_G (G43) and W_L (L). For P⁺ and for C + f0, the target
  is G_attr.

### Cells

| set | tests | designs | oracle accuracy |
|---|---|---|---|
| C | H1, H2, S1, N45, G43sh, and the primary family's weighted Holm | L, Q, W_G, W_L | 0.68 and 0.80 |
| P⁺ | S2 | L, Q, W | 0.72 |
| C + f0 | S3 | L, Q, W | 0.72 |

There are 400 datasets per cell, with labels from stream 3 and designs from
stream 4.

### Pass (G3)

A test fails on a set if any one of these holds:
- a cell rejects at nominal 0.05 in more than 9% of its datasets (36 of 400). A
  true size of 0.05 does this with probability about 0.0003.
- the rejections pooled over the set's cells exceed the binomial 99.5% point at
  size 0.05: 193 of 3200 for C (6.0%), 80 of 1200 for the others (6.7%).
- for S1 only, a cell's mean T lies outside ±0.15, or its sd outside 0.85–1.20.

**Version 2.** A set with a failing test is rerun with version 2, all its cells,
with new label draws. Then:
- a test that also fails version 2 is withdrawn;
- the size of version 2 is re-estimated on 200 fresh datasets per cell
  (stream 15) and reported.

**Time.** It is projected from the first 20 datasets, at 12 effective workers.
Above 6 hours, every cell is cut to 300 datasets, with a cell threshold of 27.
This rule is fixed now.

### Null bands (stream 7)

200 datasets of design L per set: C at 0.68, P⁺ and C + f0 at 0.72. For each
increment of §3 the band gives:
- c0, its mean, which §3 subtracts;
- its 2.5 and 97.5 percentiles.

### Power (stream 6)

200 datasets each, on top of design L for C at oracle 0.68:
- **z with coefficient −0.461 (β_5's) and −0.23,** on the logit scale: the power
  of H1 at 0.035 and 0.05, and of S1 at 0.0167 and 0.05. β_5 was estimated on
  these pairs, with more baseline columns. The power at −0.461 is therefore
  optimistic (winner's curse), and −0.23 is shown beside it.
- **The first direction of L's residuals,** with the coefficient that raises the
  oracle accuracy by 0.04: the power of H2 at 0.015 and 0.05.

### Planning figures, before the simulation, with 78 problems

- A single column at β_5's strength (|T| ≈ 2.1 under the linear fit) is found by
  S1 with probability about 0.55 at 0.05, and about 0.37 at Holm's first step.
- As one column inside H1's maximum over 43 correlated columns, it is found with
  probability about 0.1–0.3.
- Intervals for A have half-widths of about 0.08, and for differences about
  0.085.

A non-rejection of H1 therefore says little unless its interval is narrow, and
§5 reads it that way.

## 7. Limitations

**A second look at the same 78 problems.** The other phases' graphs are gone.
Phase 4 shares 73 of its 98 problems, used sdpa attention and was not matched. A
positive result needs new problems.

**The label is "passes MBPP's asserts".**
- The verification is a post-hoc subset, and about 30% of survivors cannot be
  verified.
- The calibration drops the inputs on which the reference and Gemma's passing
  samples disagree, so some boundary errors stay invisible.
- Problems with two calibration samples are protected less: Phase 5's 443, and
  Phase 4's 309, 404 and 736.

**The graphs explain the tokens after the closing brace,** not the edit and not
correctness.
- Only 8192 selected features have adjacency rows: 23% of the active features
  at graph_68's edit, and 0.3% one token later.
- The edit position's share of influence there was 0.019.
- L_attr measures the edit's part in that end-of-function prediction.

**The attribution is noisier than the forward pass.** Prefix activations were
bit-identical in every pair checked, but the attribution on the shared prefix
differed in about 3% of edges.
- The attribution-dependent columns (G_attr, L_attr, z) carry a GPU noise that
  P, G_free and L_act do not.
- Small differences therefore lean against the graph.

**P is one probe:** transcoder activations averaged over positions, compressed to
45 directions. The residual stream and the reconstruction errors are not stored
and are not used. "Not beyond P⁺" does not mean "not beyond every probe".

**No assumption-free test of conditional independence exists** (Shah and Peters,
2020).
- The size check covers designs L, Q and W on these covariates only.
- The version is chosen and its size estimated in the same simulation. Fresh
  draws are reported when version 2 is used.

**The intervals resample fixed out-of-fold scores.** Refitting noise is shown
separately. Increments are read after their null centre is subtracted, which is
exact only under design L.

**C was widened after the first approval.** C contains the final position's
output distribution (f3, f4), which the 2026-10-13 approval did not list. The
change was approved on 2026-10-14 before this commit, and C63 is reported
beside it.

**The errors are injected one-token mutants,** not the model's own. The 140-token
cap and the frontier problems are as before.

## 8. Deviations

None yet.

**The extraction's outputs** (p7_0, before the first commit; sha256):

| file | sha256 |
|---|---|
| `data/results/closing/extract.json` | `56b1087930b3b638d927d42d114d6a352453f33fff73a2ffae92fe37c2d7bf68` |
| `data/results/closing/local_matched.jsonl` | `36addc5eaa2036c2d8cc6fb9d43b92b1197e43ffe361b357056abcab750e5ac6` |
| `data/results/closing/local_mutant.jsonl` | `b5917a775439258ad15269de814c52df927dbc63db03d7ea08af1249896be3d3` |
| `data/results/closing/positions_matched.npz` | `37ce5ce80d3be20d7cc89e3b9b4d573a8d806b118e1bfa4441a77a3a59c56879` |
| `data/results/closing/positions_mutant.npz` | `20cd0188e7be41d9dbc9ef415176718c6caf8d895e7d3c304097b60e2a73e8cd` |
| `~/closing_work/posmean_matched.npz` | `ed083690c164a29f1ff095ff8b7530ebbc70446245c3844cea6bc61dae5dc0d4` |
| `~/closing_work/posmean_mutant.npz` | `052678a647beb6a1d48a75557bc72f66293ba6b51af60ce89f6b6ddcbbd4827c` |

**Recorded before `p7_4` runs:**
- the verification counts;
- the size simulation's results and the versions chosen;
- the line `sizesim sha256: <hex>`, which p7_4 checks against `sizesim.json`.

## 9. Results

Not yet.
