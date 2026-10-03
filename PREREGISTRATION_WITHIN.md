# Pre-registration — the within-problem experiment (Phase 3)

**Version 2, 2026-10-01.** Version 1 was written and committed before a single
sample was drawn. Version 2 changes the sampling design after a yield pilot
that saw labels only, never a graph or a feature, and before the real draw;
§8 records what changed, why, and the numbers it was decided on.

---

## 1. Why this experiment

Phase 2 (`PREREGISTRATION.md` §9) found that in Rust the attribution graph of
the generated code predicts whether it runs beyond what the prompt predicts —
C1 +0.1350 [+0.0715, +0.2013] — and an exploratory control showed this is not
an artefact of comparing eight rows with one: with one row per side the last
row beats step 0 by +0.1066 [+0.0321, +0.1810]. But at that n what the code
adds **cannot be separated from its length**: residualised on length, the last
row's edge falls to +0.0320 [−0.0489, +0.1160].

Every comparison so far is *between* problems. There is one greedy generation
per problem and its label is constant within it, so any predictor is partly a
difficulty detector, and solution length differs systematically between
problems. No reanalysis of that data can fully undo this.

This experiment removes the between-problem axis by construction. It samples
several solutions to the same prompt, keeps the problems where some pass and
some fail, and asks of one passing and one failing solution **to the same
problem, of near-identical length**:

> **Does the attribution graph of a solution tell the one that passes from the
> one that fails?**

Within a pair the prompt is identical, so problem difficulty cannot help, and
the pair is matched on length by design.

## 2. Design — fixed

| | |
|---|---|
| problems | the 334 typeable MBPP tasks whose every assert is gradeable (`graded_all_tests`) |
| model, prompt | Gemma-2-2b-it and the Phase 2 Rust prompt, signature included; extraction and cleaning imported from `0_generate_responses_rust.py`, not copied |
| sampling | K = 20 per problem, temperature 0.8, top_p 0.95, max_new_tokens 384, seed 20261001 + task_id (v1: K = 10) |
| cleaning | the Phase 2 cleaner, then `within_clean.tidy`: keep top-level items only, drop prose and a second definition of the target function (v1: Phase 2 cleaner only). Raw text stored alongside |
| labels | `1_rust_correct.py`'s `judge`, unchanged: pass = compiles and every assert holds |
| eligible sample | compiles, not truncated, and prompt + the whole solution ≤ 140 tokens, so the graph sees the complete function |
| mixed problem | at least one eligible passing and one eligible failing sample |
| pair | one per mixed problem: the passing × failing combination with the smallest difference in tokens; ties go to the lowest sample indices |
| gate | at least **70 pairs**. Fewer: draw 20 more samples per problem once (seed + 1000, samples 20–39) and re-pair; still fewer: stop and report |
| cap | more than 150 pairs: keep the 150 with the smallest token difference |
| graphs | one per sample, on prompt + the complete solution; `BATCH_SIZE = 64`, the Phase 2 pipeline, the same 49 features |

**Compile failures are excluded on purpose.** A passing solution against one
that does not compile would test whether the graph sees a syntax or type error —
the Python arm's lesson, where the features predicted "did the extraction
survive" at 0.9995 — and not whether it sees a wrong answer.

Each sample enters the graph pipeline under its own id, `task_id × 100 +
sample`, so two solutions to one problem never share a ledger entry or a step
counter.

Scripts: `w0_sample_rust.py`, `w1_label_samples.py`, `w2_make_pairs.py`,
`w3_within_analysis.py`.

## 3. Analysis — fixed (`w3_within_analysis.py`)

- `RandomForestClassifier(n_estimators=100, random_state=42)` on the 49
  features, label pass/fail, **10 repeats × 5-fold `GroupKFold` over problems**
  — the two members of a pair are always in the same fold — with the
  out-of-fold P(pass) averaged over repeats.
- **Primary statistic A — within-pair accuracy:** the fraction of pairs whose
  passing member scores higher than its failing member; ties count ½. Chance is
  0.5.
- **Test:** one-sided, against K = 100 refits in which the two labels of each
  pair are swapped with probability ½ — features, pairs and folds untouched.
  p = (1 + #{A_k ≥ A}) / (1 + K). A 5000-draw bootstrap over pairs gives the
  interval. (A permutation of scores would not do: Phase 2 showed refitting on
  shuffled labels is the honest null for a fitted score.)
- **Secondary S1:** the same, with every feature first residualised on
  [tokens, tokens², chars, lines].
- **Secondary S2:** a length-only model on [tokens, chars, lines] — how much the
  residual length difference alone separates the pairs.
- Descriptive only: per-feature paired differences, Wilcoxon signed-rank with
  Benjamini–Hochberg.
- A pair missing either member's graph is dropped, and the count is reported.

`run_within.sh` runs everything: `PHASE=pilot` stops at the gate so the samples
can be inspected before two to three days of GPU are committed; `PHASE=graphs`
runs the rest.

## 4. Predictions

**A: its p is above 0.05.** Phase 2's code signal could not be separated from
length, and here length is matched. I hold this weakly — my Phase 2 prediction
for C1 was wrong.

**S2: within-pair accuracy near 0.5**, i.e. the matching worked.

## 5. What each outcome means

| A, p < 0.05 | S1, p < 0.05 | reading |
|---|---|---|
| yes | yes | the graph tells a correct solution from an incorrect one to the same problem, beyond length — the paper's claim, at the level of the code itself |
| yes | no | a within-problem signal that cannot be separated from the length difference that matching left |
| no | — | not detectable at this n; report the resolvable effect, and do not call it absence |

No threshold here is adjusted after a sample has been drawn; a deviation is
recorded in §8 with the original alongside.

## 6. Power

SE(A) ≈ √(0.25 / N): 0.05 at N = 100 pairs. One-sided α = 0.05 with 80% power
resolves A ≈ 0.62 at N = 100 and 0.60 at N = 150 (normal approximation;
fitting noise and the refit null make it somewhat worse).

How much worse, for a signal carried by one feature: on 40 synthetic pairs
with one of the 49 features shifted by 1.5 SD for the passing member (a
single-feature A near 0.86), `w3`'s random forest reached A = 0.625. The
forest spreads its splits over 48 noise features. A null here does not rule
out a signal confined to one or two features; it rules out one the forest
can find among all 49, which is the question Phase 2 asked.

## 7. Limitations

- Sampled at temperature 0.8, not greedy: these are not the solutions Phase 2
  graphed.
- Only problems the model sometimes solves can form a pair, so the population is
  the model's frontier, one pair per problem.
- Length is matched on tokens of the whole text; what else differs between a
  passing and a failing solution — algorithm, control flow — is exactly what
  the graph is allowed to read.
- The 140-token cap on prompt + whole solution restricts the experiment to
  short solutions.

## 8. Deviations

**v1 → v2 (2026-10-01, before the real draw): K, cleaning, second draw.**

*Why.* A smoke test (20 problems × 4 samples) produced no pair: within a
problem the compiling samples nearly always agreed, and half the samples did
not compile — many for wrapping rather than code (a duplicate definition of
the target function, E0428; prose or a stray fence line left after the Phase 2
cleaner). At that yield the gate would fail.

*Yield pilot.* The first 40 strict problems, K = 20, at temperature 0.8 and
1.0, with and without `within_clean.tidy`. Only compile rate, pass count,
eligible samples and mixed problems were measured; no graph was built and no
feature computed, so the choice cannot have been steered by the outcome.
The rule was stated before the numbers came in: adopt tidy if it raises the
projected pairs by ≥ 10%; take the temperature with more pairs, 0.8 if within
10%; K = 10 if it projects ≥ 100 pairs, else 20.

| T | cleaning | compile / 800 | pass | eligible | mixed / 40 | pairs over 334, K = 20 | K = 10 (samples 0–9) |
|---|---|---|---|---|---|---|---|
| 0.8 | Phase 2 | 410 (51.2%) | 192 | 372 | 13 | ~109 | ~67 |
| 0.8 | + tidy | 530 (66.2%) | 227 | 470 | 14 | ~117 | ~75 |
| 1.0 | Phase 2 | 316 (39.5%) | 145 | 286 | 13 | ~109 | ~58 |
| 1.0 | + tidy | 483 (60.4%) | 205 | 428 | 15 | ~125 | ~84 |

tidy changed 209 of 800 samples at 0.8 and 314 at 1.0.

*Decision.* The rule did not fix the order in which its three parts are
applied, and tidy's gain depends on the temperature (+7% at 0.8, +15% at 1.0).
Applied as a 2 × 2 main effect: tidy, pooled over both temperatures, gives 29
mixed problems against 26 (+12%) → **adopted**. With tidy, 1.0 against 0.8 is
125 against 117 (+7%, within 10%) → **0.8**. At 0.8 with tidy, K = 10 projects
~75 pairs, short of 100 → **K = 20** (~117 pairs). Every difference in the
"mixed" column is one or two problems out of 40, inside sampling noise; the
rule exists to make the choice without judgement, not because the differences
are real. tidy is kept on its own merit too: it turns 120 wrapping failures
into compiling samples, which also enlarges the eligible pool pairs are drawn
from.

*Consequences.* The gate's second draw becomes 20 more samples (20–39, seed +
1000). The 40 pilot problems are part of the real draw and, seeded the same,
yield the same samples; only their labels were seen. Graphs see the tidied
code, which is the code that was judged.

## 9. Results

### 9a. Pre-registered (`results/within/`, commit aa35c25)

Draw: 6680 samples over 334 problems; 56.3% compile, 24.9% pass. Eligible:
3096 samples over 284 problems; 89 mixed problems → **89 pairs, gate met** on
the first draw (projected ~117 from the pilot's first 40 problems, which
turned out richer than average). Every failing member fails an assertion; none
timed out. Token difference pass − fail: mean +0.00, |d| median 2, max 38.
178 graphs, none missing: generated 2026-10-01 13:42 → 10-02 20:05, about
10.2 min each on average; features extracted by 20:31.

| statistic | value | 95% CI | label-swap refits (K = 100) |
|---|---|---|---|
| **A**, graph features | **0.6236** | [0.5225, 0.7191] | mean 0.4942, sd 0.0596; **p = 0.0198** |
| S1, residualised on length | 0.5337 | [0.4270, 0.6348] | mean 0.4912, sd 0.0665; p = 0.2673 |
| S2, length only | 0.5449 | [0.4438, 0.6461] | — |
| per-feature Wilcoxon, BH q < 0.1 | 0 of 49 | | |

**Predictions.** A: p above 0.05 — **failed** (0.020). S2 near 0.5 — **held**
(0.545, the interval covers 0.5).

**Reading (§5):** A yes, S1 no → *a within-problem signal that cannot be
separated from the length difference that matching left.* A sits at the
resolvable effect for n = 89 (~0.632): the signal is weak, and the interval's
lower end is 0.52.

### 9b. Exploratory, not pre-registered (`results/exploratory/within_length_decomposition.*`)

Within the pairs, length differences are small and unsigned (chars pass − fail
median +2, |d| median 17; lines mean −0.15), yet S1 removed most of A. S1's
pooled fit removes two things at once; `8_within_length_decomposition.py`,
committed before it ran with its reading rule in the docstring, removes each
alone with the same covariates and the same label-swap null.

| | removes | A | 95% CI | p |
|---|---|---|---|---|
| A (9a) | nothing | 0.6236 | [0.5225, 0.7191] | 0.0198 |
| E1 | each member's length departure from its pair mean | 0.5899 | [0.4888, 0.6966] | 0.0594 |
| E2 | the pair's mean length among problems (within-pair differences unchanged, max change 2e-19) | 0.5674 | [0.4663, 0.6629] | 0.1584 |
| S1 (9a) | both | 0.5337 | [0.4270, 0.6348] | 0.2673 |

Within-pair length explains 31.3% of the within-pair feature variance, even
though the members differ by a median of two tokens.

**Reading, by the rule written beforehand: no clean split.** Of the 0.090 that
S1 removed, E1 alone removes 0.034 and E2 alone 0.056; each part takes some.
Both partial drops, and the 0.022 between E1 and E2, are smaller than one
null sd (0.060–0.066), so this n cannot apportion them. (Corrected: an
earlier version of this paragraph said 0.057, and that every gap in the
table was within one null sd; A − S1 is 0.090, which is not.) The data lean towards the first case — removing the members'
own length difference leaves A at 0.590 (p 0.059), so that difference does not
account for most of the signal — but a lean is all it is. What survives
unchanged is 9a's reading: a weak within-problem signal (A ≈ 0.6) that this
experiment cannot separate from length.
