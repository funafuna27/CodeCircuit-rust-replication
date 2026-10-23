# Pre-registration addendum — does Gemma's forward pass tell which of its passing programs MBPP+ overturns? (Phase 9, part ②)

Written 2026-10-23 (Japan time), before any signal of the programs below was computed, and committed with the
code that computes and analyses them (`data/p9_plus_*.py`) and the audit that preceded it
(`data/p9_c_audit.py`). `p9_plus_extract.py` and `p9_plus_analysis.py` refuse to run unless this file, the
other PREREGISTRATION_P9*.md and the scripts they rely on are committed and unchanged, and record their
hashes; the analysis also refuses an extraction made by other versions of them. What existed before is listed
in §8.

## 1. Question

MBPP's three asserts pass programs that MBPP+'s cases (about 100 per task) overturn: 506 of 1,936 distinct
passing programs in the registered label set (A3). Part ③ asked, between problems, what the graph adds to the
output distribution and an activation probe. This part asks the within-task question with the stronger
oracle:

> Within a task, among Gemma's programs that pass MBPP's asserts, does Gemma's own forward pass rank the
> programs MBPP+ confirms above the ones it overturns?

Within a task the prompt and the problem's difficulty are fixed, so a signal must read the program. A yes
means the model carries information about its own errors beyond the asserts; a no, that it is not shown to.
No graph is built (about 900 graphs; part ③ found the graph's attribution adds less than 0.05 beyond C + P
between problems); the signals are part ③'s forward-pass ones.

## 2. Population and labels

- **Programs:** every distinct program (whitespace removed, within a task) that Gemma wrote and that passes
  MBPP's graded asserts — the samples of Phases 3, 5 and 6 (T 0.8, top-p 0.95) and Phase 2's greedy
  generations — judged on MBPP+'s inputs and expected values by the finite-aware harness in mode `-O`
  (`results/p9_state/a3_relabel_v2.jsonl.gz`): **holds** (every case right) or **flips** (a case wrong or
  panicking). Timeouts are left out (3 programs here).
- **The registered label set** (PREREGISTRATION_P9_MUTSTATE.md §7): the 53 tasks with fewer than 20 usable
  cases and the 14 artefact tasks of the all-flip audit are left out.
- **Mixed tasks** (at least one hold and one flip): **44 tasks, 908 programs, 474 holds, 434 flips**; per
  task 1–44 holds (median 8) and 1–49 flips (median 4.5); 28 tasks have at least two of each, 3 have one of
  each. The counts are asserted whenever the population is built.
- **Text per program:** the task's prompt prose (one per task, the pool's `text`) + ' ' + the program's code
  as first sampled (its first member by phase, then sample index; one program's only member is Phase 2's
  greedy generation). Within a task the texts differ only in the code.
- The audit of §8 found 8 of 44 sampled flips to be MBPP+ artefacts. No case or program is removed (inputs
  are never calibrated on Gemma's samples); a sensitivity without those 8 tasks is reported (§5).

## 3. Signals (`p9_plus_extract.py`, `p9_plus_analysis.py`)

**The forward pass:** part ③'s call, unchanged (`p9_m1_extract.forward_one`: the ReplacementModel loaded as
`3_generate_graph.py` loads it, `model.setup_attribution` on BOS + the text's tokens, checked equal to
`ensure_tokenized`; installed circuit-tracer 0.5.3.dev1+g6018ed8d3; local RTX 5070; batch 1). Kept as part ③
keeps them; the code positions are the tokens starting at or after the space before the code.

- **C (8 columns):** part ③'s C without the prose-token count, which is constant within a task: (1) top-1
  probability, (2) entropy and (3) top-1 minus top-2 at the final position; over the code tokens, (4) summed,
  (5) mean and (6) minimum log p and (7) mean entropy; (8) the number of code tokens.
- **P_code (45 columns):** part ③'s code-position posmean, each row centred on its task's mean, then a
  label-free truncated SVD (k = 45, ARPACK, v0 from stream [20261023, 33]; vectors sorted by singular value,
  the largest-magnitude entry positive) of the centred rows; P_code = the centred rows projected. Fitted on all
  908 rows; no label.
- Every C column is centred on its task's mean as well. Centring uses all of a task's programs and no label:
  the learner sees only within-task differences.

## 4. Learner and statistic

- **Learner:** part ③'s forest (RandomForestClassifier, 100 trees, random_state 42) on rows, out of fold over
  tasks: 10 repeats × 5-fold GroupKFold, the task-to-fold map permuted per repeat with default_rng(100 +
  repeat). A program's score is its mean out-of-fold probability of holding. Every model uses the same folds.
- **Statistic:** AUC_t, the AUROC of the scores of task t's holds above its flips (ties count ½); d_t = AUC_t −
  0.5; **T = the mean of d_t over the 44 tasks** (one score per task, equal weights).
- **Intervals:** 5,000 bootstrap draws of the tasks (stream [20261023, 32]), shared by every contrast,
  conditional on the fitted scores; the one-sided 80% MDE = 2.486 × SD(d_t) / √44.

## 5. Tests

- **Q1 (primary): C + P_code; H1: T > 0; one-sided permutation test, α 0.05.** Within every task the labels
  are permuted (counts kept; streams [20261023, 31, k], k = 0..199), the whole pipeline is refitted, and p =
  (1 + #{T_k ≥ T}) / 201; its floor is 0.005. The task-level sign flip of d_t (99,999 flips plus the identity,
  stream 36) is reported beside it.
- **Secondary, only if Q1 rejects; Holm over the two at 0.05:**
  - **S1:** C alone, H1: T > 0, the same 200 permuted label sets.
  - **S2:** the increment AUC_t(C + P_code) − AUC_t(C), H1: mean > 0, one-sided task-level sign flip (stream
    35).
- **Descriptive:**
  - P_code alone.
  - The learner replaced by part ③'s logistic regression on rank normal scores, for C + P_code and for C.
  - Each raw single column's within-task AUROC (C's eight, f0, f3, f4). A value above 0.5 means that holds
    score higher; these read a direction.
  - Q1's d_t without the 8 tasks whose audited flip was an artefact: 36 tasks (sign flip stream 38,
    bootstrap 37).
  - Q1's d_t on the 28 tasks with at least two holds and two flips (bootstrap 39).
  - The pooled within-task AUROC (tasks weighted by their pairs).
  - The permutation null's mean and 95th percentile, for C + P_code and C.
- **Power** (label-free, on the 44 tasks' real counts, 1,000 simulations, stream 30): for a score whose true
  within-task AUROC is 0.55 / 0.58 / 0.60 / 0.65, the one-sided test detects with probability 0.34 / 0.62 /
  0.76 / 0.98. If 18% of the flips (the audit's artefact share) behave like holds, the probabilities are
  0.27 / 0.47 / 0.63 / 0.92. A fitted score reaches less than its signal's true AUROC.

## 6. Readings

| Q1 | S1 | S2 | written as |
|---|---|---|---|
| not rejected | — | — | "within a task, Gemma's forward pass is not shown to tell which of its passing programs MBPP+ overturns; the 80% MDE is X" |
| rejected | rejected | not | "it ranks the programs MBPP+ confirms above those it overturns; the output distribution and length suffice, and the activations are not shown to add" |
| rejected | either | rejected | "…, and the activations over the code add to the output distribution" |
| rejected | not | not | "the output distribution and the activations together do; neither part is shown alone" |

Every sentence carries T with its interval and the artefact share of §8. A rejected Q1 is "information about
errors beyond the asserts", not "knowledge of correctness": 18% of flips are artefacts, and 11 of the 36
genuine ones fail only on edge inputs.

## 7. Predictions (written before any signal of these programs exists)

- **Q1 is rejected,** with T between +0.05 and +0.15 (a mean within-task AUROC of 0.55–0.65).
  - Reason: many flips in the audit are wrong logic that passed three asserts by luck (tasks 6, 77, 128, 285,
    397, 594, 805). Such code is likely less typical than the task's usual correct program.
- **S1 is rejected**, with C within 0.02 of C + P_code.
- **S2 is not rejected:** the increment lies within ±0.03.
- **Single columns:**
  - The code's mean log p is above 0.5 (holds more probable), d between +0.05 and +0.15. It carries most of
    C's signal.
  - The number of code tokens lies within ±0.05 of 0.5.
- **Without the 8 artefact tasks,** Q1's T rises by 0.01–0.03.
- **The permutation null's mean** lies within ±0.01 of 0.

## 8. What existed before this file

- A3's verdicts, counts and power table (gate 1); the label decision of gate 2; the results of part ①
  (2e4f588), part ③ (8c4342e) and gate 3 (3a78ac9, f79fb0d, a6aa2f6).
- **The audit of flips in mixed tasks** (`p9_c_audit.py`, required by PREREGISTRATION_P9_MUTSTATE.md §7):
  - **Procedure:** one flip per mixed task, drawn at random. The classes were fixed in the script before any
    program was read. The 44 flips were read by eye, with all failing cases shown for the 14 programs where
    six did not settle the class.
  - **Result:** 36 genuine and 8 artefacts: 7 conventions (whitespace kept by the reference: 451, 800;
    punctuation dropped by it: 459; padded integers: 113; one-element lists: 472; the Armstrong definition:
    598; an unanchored match: 794) and 1 precondition (non-ASCII "ASCII values": 269). The artefact share is
    18% [Wilson 95%: 9.5%, 32%].
  - **Edge inputs:** 11 of the genuine flips fail only on edge inputs. These are an empty input that panics
    (5), negative numbers (2), non-ASCII text (1), and a naive intermediate beyond i64 or f64 (3).
  - **Records:** `results/p9_state/c_audit.json`, `c_audit_sample.txt`, `c_audit_sample_all.txt`.
- **Label-free work only:**
  - The analysis was rehearsed end to end with stand-in signals and labels permuted within task. No real
    association was read. 16 permutations took 51 s on 16 workers.
  - The extractor was timed on the 20 longest texts (0.56 s median; nothing saved).
  - The power simulation of §5 was run.
  - No signal of any of the 908 programs has been computed.

## 9. Deviations

None yet.

## 10. Execution

```
cd data
python p9_c_audit.py            # done before this commit (§8)
python p9_plus_extract.py       # local RTX 5070, about 10 minutes
python p9_plus_analysis.py      # CPU, 16 workers, about 10 minutes
python p9_plus_check.py         # independent recomputation
```

## 11. Results (2026-10-23, run 15:22–15:33 JST on commit e99c988)

**The run:** 908 texts at 0.33 s each on the RTX 5070 (`~/p9_work/plus/extract_plus.npz`, sha256
418b29bb…). P_code's 45 directions hold 0.789 of the within-task sum of squares.

**Checks:** `p9_plus_check.py`, written apart from the analysis, gives the same values to four
decimals for the population, Q1, S1, P_code alone and the code's mean log p.

**Outputs** (`results/p9_plus/`): `analysis.json`, `scores.jsonl.gz`, `permutation_null.npy`,
`check.json`.

| test | tasks | T = mean(AUC_t − 0.5) | 95% CI | p | decision |
|---|---|---|---|---|---|
| Q1: C + P_code | 44 | +0.086 | [−0.003, +0.173] | 0.040 (permutation, K 200; sign flip 0.032) | **rejected** |
| S1: C alone (Holm) | 44 | +0.028 | [−0.074, +0.127] | 0.249 | not rejected |
| S2: P_code beyond C (Holm) | 44 | +0.058 | [−0.013, +0.130] | 0.061 | not rejected |

Q1's permutation p rests on 7 of 200 permuted statistics at or above T. Its Monte Carlo SE at 0.04
is about 0.014. The registered decision stands on K = 200, and the result is close to the line.

**Reading (§6, row 4).** Within a task, the output distribution and the activations over the code
together rank the programs MBPP+ confirms above those it overturns. The mean within-task AUROC is 0.586
[0.497, 0.673]. Neither part is shown alone. This is information about errors beyond the asserts,
not knowledge of correctness: 18% of flips are artefacts, and 11 of the 36 genuine ones fail only on
edge inputs (§8).

**Descriptive:**
- **P_code alone:** +0.123 [+0.044, +0.204] (AUROC 0.623), above C + P_code. The forest on C alone
  (+0.028) is the weaker part.
- **Logistic learner:** C + P_code +0.094 [−0.004, +0.193]; C +0.075 [−0.023, +0.171].
- **Single columns** (positive means holds score higher):
  - number of code tokens −0.123 [−0.213, −0.028]: holds are shorter;
  - the code's mean log p −0.112 [−0.202, −0.020]: holds are less probable per token;
  - f0, the active-feature count, −0.116 [−0.208, −0.022];
  - summed log p +0.078 [−0.011, +0.167];
  - the final position's columns and the entropies lie within ±0.04.
- **Subsets:**
  - without the 8 artefact tasks (36): +0.088 [−0.017, +0.187], sign flip p 0.053;
  - tasks with at least two holds and two flips (28): +0.080 [+0.005, +0.155].
- **Pooled within-task AUROC:** C + P_code 0.567, C 0.490, P_code 0.592.
- **The permutation null:** C + P_code mean −0.003 (95th percentile 0.072); C +0.001 (0.063).

**Against §7.**
- **Held:** Q1 rejected, with T inside its predicted range (+0.05 to +0.15); S2 not rejected; the
  permutation null's mean within ±0.01.
- **Missed:**
  - S1 was predicted rejected. C is the weaker part: 0.058 below C + P_code, not within 0.02.
  - S2's increment (+0.058) lies outside the predicted ±0.03, though S2 is not rejected.
  - The code's mean log p points the other way: holds are less probable per token (−0.112), not
    +0.05 to +0.15.
  - Length is not neutral: holds are shorter (−0.123), not within ±0.05.
  - Without the artefact tasks, T barely moves (+0.002), not the predicted rise of 0.01–0.03.
