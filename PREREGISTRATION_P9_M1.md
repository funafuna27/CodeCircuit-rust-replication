# Pre-registration addendum — what does the graph add, between problems? (Phase 9, part ③, "M1")

Written 2026-10-22 (Japan time), before any output-distribution (C) or probe (P) value of the rows below
was computed, and committed with the code that computes and analyses them (`data/p9_m1_*.py`). Every M1
script refuses to run until this file, the other PREREGISTRATION_P9*.md, the four scripts and the modules
they rely on (`p9_common.py`, `5_problem_level.py`) are committed and unchanged, and records their hashes
in its output; the gates and the analysis also refuse an extraction made by other versions of them, or
whose sha256 differs from the one recorded when it was written. What existed before is listed in §9.

## 1. Question

In the original paper's between-problem setting (one score per problem from its line prefixes), does the
attribution graph (G) predict whether Gemma's code is correct beyond its output distribution (C), and
beyond an attribution-free activation probe (P)? Phase 7 asked this within matched pairs and found G
adding about nothing to C (+0.002 [−0.072, +0.074] as published; −0.015 [−0.095, +0.061] after the
erratum) with P ≥ G; it never asked it in the setting the paper reports. No graph is built: G is the 49
stored columns; C and P come from one forward pass per row.

## 2. Populations and labels (as Phase 2 fixed them)

| population | rows | problems | positive | role |
|---|---|---|---|---|
| **Rust strict** | 1,654 | 206 | 95 | primary |
| Rust loose | 1,734 | 214 | 99 | sensitivity (raw increment only) |
| Python 240 (generations that parse) | 1,137 | 240 | 114 | the Python arm (Phase 2's like-for-like comparator) |
| Python 256 | 1,337 | 256 | 114 | sensitivity (raw increment only; holds the 16 fence-corrupted negatives) |

Rust labels: compiles and passes every graded MBPP assert. Python labels: execution (`passes` in
`gemma_mbpp_python_correctness.jsonl`), not the judge's `step_labels`. A row is the problem's prose +
' ' + the first i lines of the greedy generation; step 0 is the first line (in Rust the prompt-dictated
signature + ` {` in all 214 problems; in Python the model's own first line). The 140-token cap (BOS
counted) and the subsets are applied as `5_problem_level.py` applies them; the rows, problems and positives
of the table are asserted whenever a population is loaded. Labels are constant within a problem; every
statistic uses one score per problem. Forward passes run on all 1,734 Rust and 1,420 Python texts.

## 3. Signals per row

**G**: the 49 stored columns as Phase 2 used them. Columns 0, 3 and 4 (G_free: active-feature count,
top-logit probability, entropy over at most 10 salient logits) come from the forward pass alone; of the
other 46, 43 are non-constant and all depend on the attribution.

**The forward pass** (`p9_m1_extract.py`): one per text, batch 1, on the local RTX 5070 in the
environment that built the stored graphs (torch 2.11.0+cu128; circuit-tracer 0.5.3.dev1+g6018ed8d3 from
site-packages, run from `data/`; the ReplacementModel loaded as `3_generate_graph.py` loads it; script 3's
allocator setting; no deterministic flags, as the graphs were built; the script refuses any other
circuit-tracer and records its location and version). The call is
`model.setup_attribution(input_ids)` — the first step of `attribute()` — on BOS + the text's tokens
(checked equal to `model.ensure_tokenized(text)`). Kept per text: token ids and offsets; log p of each
token and the entropy of the distribution that predicts it (float32 log-softmax of `ctx.logits`); the
final position's top-1 and top-2 probabilities and entropy; f0 = the active-feature count; the salient
set of `AttributionTargets(None, ctx.logits[0, −1], W_U, tokenizer, max_n_logits=10,
desired_logit_prob=0.95)` on the GPU, its probabilities moved to the CPU in bf16, f3 = the first,
f4 = −Σ p log(p + 1e−8); and two position means (below). No logits are written. Each row's C must be
finite and its code-token count positive and equal to the extraction's, or the analysis stops.

**C** (output distribution and length on the graphed text; 9 columns): (1) top-1 probability, (2)
full-vocabulary entropy and (3) top-1 minus top-2 probability at the final position; over the code
tokens (tokens starting at or after the space before the code, Phase 5's rule): (4) summed, (5) mean and
(6) minimum log p, (7) mean entropy of the predicting distributions; (8) the number of code tokens; (9)
the number of prose tokens. G's column 4 is not repeated in C. Length is in C on purpose: between
problems it predicts the label (Phase 2: −max prefix chars 0.694), and a graph that only measures length
should not count as adding to C. Descriptive splits: C_dist = columns 1, 2, 3, 5, 7; L = columns 8, 9.

**P and P_code** (45 columns each): the position mean of every active transcoder feature, by layer ×
16,384 + feature — summed in float64 from the stored bf16 values with `np.unique`/`np.bincount` in the
coalesced order, divided by the number of positions including BOS, cast to float32 (Phase 7's
`posmean`, `p7_0_extract.py` lines 137–150) — and the same over code positions only, divided by their
number (P_code). For each population, a label-free truncated SVD (k = 45) of these rows centred on their
mean, ARPACK with v0 from streams [20261022, 83, population] (P) and [20261022, 84, population] (P_code),
vectors sorted by singular value with the largest-magnitude entry positive; P = the centred rows
projected. The basis is fitted on all rows of the population, held-out problems included (no labels).
P and P_code are built for Rust strict and Python 240 only (the other two populations use G and C).
Unlike Phase 7's P, which was a basis of within-pair differences where the prose cancels, P is a level
and reads the prompt; P_code does not see the prose.

## 4. Learner and scores (Phase 2's)

`5_problem_level.oof_problem_scores`: RandomForestClassifier(100 trees, random_state 42) on rows, 10
repeats × 5-fold GroupKFold over problems, the problem-to-fold map permuted per repeat with
default_rng(100 + repeat); a problem's score is the mean of its rows' out-of-fold probabilities,
averaged over repeats. `p9_m1_common.oof` is that function with its two seeds as arguments; gate R
checks it equals the original bit for bit. A combined signal is the column concatenation, tested block
first.

## 5. Gates, run in this order

1. **X1** (`ROLE=matched p9_m1_extract.py`, then `STEP=x1 p9_m1_gates.py`; no M1 row involved): on the
   518 Phase 5 texts, the forward-only posmean equals Phase 7's `~/closing_work/posmean_matched.npz`
   (its sha256 checked first against PREREGISTRATION_CLOSING.md) row by row, indices and float32 bits;
   f0, f3, f4 are compared with the stored columns 0, 3, 4 of `gemma_matched_features.jsonl` (reported;
   the decision on f is X2's, per arm); and **X3**: the code log p (per token), minimum log p and mean
   entropy agree with Phase 5's `p5_2` values (HF, eager) within a mean absolute difference of 0.02 nats
   (amended by deviation 1, §10: the minimum log p is reported, not judged).
2. Extraction of the M1 texts (refused unless X1's C check passed).
3. **X2 and R** (`STEP=x2`): per arm, f0, f3, f4 against the stored G columns 0, 3, 4 on every row; the
   94 surviving Phase 1–2 graphs (`data/rust_graph`, 50; `data/python_graph`, 44) rebuilt by the same
   call and compared bit for bit (tokens, active features and values, logit targets and probabilities;
   reported, with the count compared per directory); R: stored G through the analysis code returns Phase 2's AUROCs to four decimals — Rust
   strict 0.7887 / 0.6540 (all rows / step 0), Rust loose 0.7953 / 0.6732, Python 240 0.7091 / 0.6928,
   Python 256 0.7269 / 0.6931 — and the seeded function equals `oof_problem_scores`.

Registered branches (no other exists):
- R fails: the analysis does not run. X3 fails: the M1 extraction is refused and nothing runs.
- X1's posmean fails for any of the 518: P and P_code are not computed; everything without them runs.
- f0 differs on an arm: if at most 1% of its rows differ and each by at most 0.1%, the arm runs, its
  problems are listed, and its primary (Rust strict or Python 240) is repeated without them — a raw paired
  difference on draws [20261022, 81, 10 + population], read with the population's own null centres;
  otherwise the arm does not run. f3/f4 differ: if at most 1% of rows differ and each by at most 1 bf16
  ulp, the arm runs and the counts are recorded; otherwise it does not. A Python failure does not stop
  Rust.

## 6. Statistics (`p9_m1_analysis.py`)

**Intervals.** One set of 5,000 bootstrap draws of the problems per population (stream [20261022, 81,
population]: 0 Rust strict, 1 Rust loose, 2 Python 256, 3 Python 240), shared by every contrast in that
population; AUROC is recomputed on each draw; single-class draws are dropped and counted (none expected).
Each difference is reported as the plug-in, the bootstrap mean and the 95% percentile interval. These
intervals are conditional on the fitted rule; for every reading against the SESOI each half-width is
widened by 1.28 about the plug-in (lo′ = d − 1.28(d − lo), hi′ = d + 1.28(hi − d)), the factor Phase 2
measured for a level by refitting.

**Null centres.** A forest can gain from redundant informative columns and lose on uninformative ones, so
an increment X + B − B is not centred at 0 when X adds nothing beyond B. Its null centre is estimated as
Phase 7 estimated it: refit [X, B] and B on synthetic problem labels that depend on B alone, with the real
rows. Design L: labels ~ Bernoulli(σ(k·η + b)), η = a random-weight combination (stream [20261022, 86,
id, r]) of the standardised problem means and maxima of B's columns, k and b set so that η's expected
AUROC (the ratio of the expected number of concordant positive–negative pairs, ties ½, to the expected
number of such pairs) equals the real out-of-fold AUROC(B) and the base rate equals the real one. Design
F: labels ~ Bernoulli(p̂), p̂ = B's out-of-fold problem score on the real labels, slope-recalibrated
(stream [20261022, 87, id, r]). id = 100 × population + increment (0 primary, 1 S0, 2 S1, 3 S2, 4 S2 for
P_code); r = the draw. c0 = the mean synthetic increment; its Monte Carlo SE and 2.5–97.5% band are
reported. The primary uses both designs with 200 draws each; the secondaries design L with 100.

**Reading** of an increment with plug-in d, widened interval [lo′, hi′] and null centres c0:
upper bracket U = hi′ − min(c0), lower bracket Lb = lo′ − max(c0); the corrected point is d − mean(c0).

| bracket | written as |
|---|---|
| U < 0 | "X + B ranks below B beyond the null centre: X costs this forest" |
| Lb ≤ 0, U < 0.05 | "X adds less than 0.05 (upper bound U); not distinguishable from 0" |
| Lb > 0, U < 0.05 | "X adds a little (point), less than 0.05" |
| Lb > 0, U ≥ 0.05 | "X adds (point); its size is not bounded below 0.05" |
| Lb ≤ 0, U ≥ 0.05 | "not determined at this n: [Lb, U]" |

Ten alternative seed pairs for folds and forest (stream [20261022, 85]) give the spread of the primary's
plug-in; if putting its 10th or 90th percentile in place of d (interval shifted with it) changes the row,
the reading carries "seed-dependent". Every reading is about this learner (a forest with mtry = √p).

**Primary (Rust strict 206).** Δ = AUROC(G + C) − AUROC(C), read by the table (SESOI 0.05, as an upper
bound). G contains its forward-only columns 0, 3, 4, so a positive primary is written "the graph's
features (including their forward-pass columns) add"; it is called attribution only if S0's lower bracket
is above 0. There is no p-value and no other primary.

**Secondary (Rust strict; same table; no multiplicity claim):**
- S0: AUROC(G + C) − AUROC(G_free + C): the attribution-dependent columns beyond C and G_free;
- S1: AUROC(G + C + P) − AUROC(G_free + C + P): the same beyond the probe as well;
- S2: AUROC(P + C) − AUROC(C), and the same for P_code. P beyond C is read as information about the
  code only if it also holds for P_code;
- S3: AUROC(P + G_free) − AUROC(G) (the probe with the graph's forward-only columns against the graph;
  paired interval, no null), with AUROC(P) − AUROC(G) and AUROC(P_code) − AUROC(G) descriptive;
- levels with intervals: G, C, G_free, P, P_code, G + C, G_free + C, C + P, G + C + P, G_free + C + P,
  P + G_free, P_code + C.

**Step 0 (reported as an equal of the primary).** For G, C, P, G + C, C + P, G + C + P: the all-rows C1 =
AUROC(all rows) − AUROC(step-0 rows), each its own refit, paired; it is not compared across signals,
because the step-0 refit trains on one row per problem and starves wide blocks more. The comparable form
is the one-row contrast AUROC(last row) − AUROC(step 0), one row per problem on each side
(`7_one_row_control.py`'s design), with each signal's one-row contrast minus C's, paired. A larger
contrast is read as more use of the generated code than C makes. Also the primary's raw Δ on step-0
rows. At Rust's step 0 the input is fixed by the prompt: a step-0 AUROC above 0.5 reads the problem as
the model holds it before writing a body (including its uncertainty about the first body token); in
Python, step 0 is the model's own first line.

**Python 240:** levels of G, C, G + C; the primary with a design-L null (100 draws) and the table; S0–S3 as
raw paired differences; the step-0 block. **Sensitivities (descriptive, raw paired differences):** Rust
loose and Python 256 (primary only); a logistic regression (inverse regularisation strength 1) on rank
normal scores fitted per training fold, in place of the forest; MBPP+ labels on Rust strict — "passes
MBPP and the usable MBPP+ cases" from `results/p9_state/a3_relabel_v2.jsonl.gz` (mode −O, Phase 2 greedy
programs; programs that failed MBPP were never re-judged), under the rule of PREREGISTRATION_P9_MUTSTATE.md
§7 (the 53 tasks below 20 cases and the 14 artefact tasks keep their MBPP label: 10 flips, 85/206
positive) and without exclusions (18 flips, 77/206), both counts asserted before any fit; C_dist and L
alone and G beyond each.

## 7. Predictions (written before any C or P value)

- Gates: X1 518/518 posmean, f0/f3/f4 and X3 pass; X2 passes on every row of both arms; the 94 graphs are
  equal; R passes.
- Levels (Rust strict): AUROC(C) 0.72–0.82; AUROC(P) 0.70–0.82; AUROC(P_code) 0.65–0.80.
- Primary: plug-in Δ between −0.02 and +0.05; c0_L and c0_F between −0.01 and +0.04 (a reviewer's
  label-free simulation on the real G with a stand-in C gave raw increments of +0.024 on average under the
  null); conditional half-width 0.035–0.05, widened 0.045–0.065. Reading: "not determined at this n"
  (most likely, about 0.6) or "less than 0.05; not distinguishable from 0" (about 0.35); a reading that G
  adds, about 0.05.
- S0: Lb ≤ 0 (no attribution-specific gain shown). S1: corrected point within ±0.03. S2: corrected
  point between −0.01 and +0.05 for P, smaller for P_code. S3: between −0.05 and +0.05.
- Step 0 (Rust): every signal's step-0 AUROC is above 0.6; the one-row contrast of G is within 0.05 of
  C's; the primary's Δ at step 0 within ±0.05.
- Python 240: the primary reads "less than 0.05" or "not determined"; every one-row contrast within
  ±0.06.

## 8. Execution and outputs

```
cd data
ROLE=matched python p9_m1_extract.py          # X1's forward passes (518 Phase 5 texts)
STEP=x1 python p9_m1_gates.py                 # results/p9_m1/gates_x1.json
ROLE=rust python p9_m1_extract.py             # refused unless X1's C check passed
ROLE=python python p9_m1_extract.py
STEP=x2 python p9_m1_gates.py                 # results/p9_m1/gates_x2.json (R, f0/f3/f4, 94 graphs)
WORKERS=16 N_JOBS=4 python p9_m1_analysis.py  # results/p9_m1/analysis.json
```
Extractions stay in `~/p9_work/m1/` (not committed; under 2 GB). Timing measured on the 20 longest Phase 5
texts before this commit: 0.43 s per text (median), 10.5 GB peak, so about 25–30 minutes for 3,672 texts (the smoke test on the 60 longest: 0.41 s).
The analysis runs on the CPU (about 1,800 out-of-fold fits, mostly the null bands).

## 9. What existed before this file

- The stored G of all rows and Phase 2's results on these rows and labels: G's levels and C1; surface and
  length baselines (strict: RF on 4 surface dimensions 0.6567, −max prefix chars 0.6942, −n_steps 0.6287);
  G residualised on length (0.7540); the one-row control (last row 0.7606, residualised 0.6871); C5 and its
  shuffled-label refits; the Python judge-label contrasts and the 240 and repaired results. The length
  results bear on C's columns 8–9, and the C predictions were written knowing them.
- Phase 7's posmean and basis of the 518 Phase 5 graphs and its within-pair results; Phases 4–7 and part ①
  computed code log p, entropies and position-mean probes on other samples of the same problems.
- Part ②'s MBPP+ verdicts of the greedy programs (10 and 18 strict flips). Part ②'s own addendum is not yet
  committed.
- The Colab speed test (bed2d9a): a local rebuild of 3 retained Phase 5 graphs reproduced their active
  features and values, selected features and logit probabilities bit for bit (6.6–9.1% of nonzero edges
  differed, relative gap 0.003); an A100 build did not. M1's extraction therefore runs locally.
- Before this commit, label-free work only: three reviewers read the draft, and two more read the code;
  one simulated the null on the real Rust-strict G with a stand-in C and synthetic labels (no real label
  read; the scratch script stays outside the repository); the analysis code was rehearsed end to end on
  stored G with the labels permuted across problems and label-free stand-ins for C and P; the extractor was
  timed on the 20 longest Phase 5 texts and run on the 60 longest as a smoke test through C, posmean and
  the SVD, where its posmean equalled Phase 7's bit for bit (60/60) and f0, f3, f4 the stored columns
  (60/60). No C, P, f0, f3 or f4 has been computed on any M1 text.

## 10. Deviations

**1. X3's minimum-log-p criterion (2026-10-22 21:31 JST, decided by the author before any M1 text was
extracted).** The first run of §8 (commit a2d57df, 20:23–20:26 JST) passed X1 everywhere — posmean
bit-equal 518/518, f0/f3/f4 equal 518/518 — but stopped at X3, as §5 required: the per-token code log p
agreed with p5_2 within 0.003 and the mean entropy within 0.002, while the minimum log p differed by 0.065
on average (median 0.050, 90th percentile 0.132, maximum 0.242; 63 of 518 rows above 0.125). The
minimum is one token's log p, about −14 to −25 here, so it carries the bf16 rounding of one logit of
magnitude 16–32, whose step is 0.125: the two implementations (HF eager in p5_2, TransformerLens here)
round it differently, by up to two steps. The 0.02 tolerance, meant for averaged quantities, was the wrong
scale for a single-token extreme. X3 exists to catch a misplaced code boundary or a wrong forward pass,
which the two averaged quantities test. Change: X3 is judged on the per-token code log p and the mean
entropy (each within 0.02); the minimum log p difference is reported and not judged. C keeps its minimum
log p column, computed from the same forward pass for every row. The gate script was changed accordingly
and, since its hash changed, the X1 extraction was run again before the M1 extraction. Nothing else
changed.
