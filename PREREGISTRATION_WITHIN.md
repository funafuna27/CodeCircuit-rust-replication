# Pre-registration — the within-problem experiment (Phase 3)

**Version 1, 2026-10-01.** Written and committed before a single sample was
drawn. Every script it names was committed with it.

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
| sampling | K = 10 per problem, temperature 0.8, top_p 0.95, max_new_tokens 384, seed 20261001 + task_id |
| labels | `1_rust_correct.py`'s `judge`, unchanged: pass = compiles and every assert holds |
| eligible sample | compiles, not truncated, and prompt + the whole solution ≤ 140 tokens, so the graph sees the complete function |
| mixed problem | at least one eligible passing and one eligible failing sample |
| pair | one per mixed problem: the passing × failing combination with the smallest difference in tokens; ties go to the lowest sample indices |
| gate | at least **70 pairs**. Fewer: draw 10 more samples per problem once (seed + 1000, samples 10–19) and re-pair; still fewer: stop and report |
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

None yet.
