# Pre-registration — the Rust arm

**Version 2, 2026-09-22.** Version 1 was written the same day and red-teamed
before any Rust graph feature existed. Four independent attacks found nine
defects rated "would produce a wrong conclusion", every one of which I confirmed
by re-running it. This version fixes them. The Rust run is still in flight, no
Rust feature exists, and every number below comes from data that already did.

The reason for the document is specific. The Python arm produced three
successive wrong conclusions — a leakage claim, an execution-label indictment,
and a granularity explanation — and each survived until someone re-derived it
independently. Every one was a choice made *after* seeing a number. Version 1
tried to remove that freedom and left most of it in place; §8 lists what changed.

---

## 1. The question, restated

Version 1 asked: *do the graph features predict Rust correctness above chance and
above surface heuristics?* That question is **not answerable as asked**, and the
reason is the most important measurement in this project.

`original_expression` — what the attribution graph is computed on — is the
prompt plus a code prefix. At step 0 that prefix is one line: the function
signature. The graph sees the problem statement and `def f(x):` (Rust:
`fn f(a0: i64) -> bool {`) and **no function body whatsoever** — no logic, no
algorithm, nothing that could be right or wrong. In the Rust arm that line is
additionally dictated by the prompt, so it carries no information the model
chose. Scoring the locked protocol on step-0 rows only:

```
Python, 256 problems
  all rows          0.7269  95% CI [0.6619, 0.7901]
  step-0 rows only  0.6931  95% CI [0.6265, 0.7558]   <- signature, no body
  paired difference +0.0340 95% CI [-0.0160, +0.0846]  P(<=0) = 0.095

Python, the 240 whose generation parses (the clean subset)
  all rows          0.7091  95% CI [0.6402, 0.7737]
  step-0 rows only  0.6928  95% CI [0.6234, 0.7586]
  paired difference +0.0165 95% CI [-0.0341, +0.0684]  P(<=0) = 0.261
```

Removing the 16 corrupted generations makes the contrast **smaller**, not
larger: on clean data the body accounts for 0.017 of 0.709.

The interval includes zero. A graph computed before the model has written a line
of logic predicts the outcome as well as one computed over the whole generation.

**But the same contrast against the paper's own target is positive.** Run on the
gpt-4o-mini judge label the paper reports against:

```
Python, 256 problems, target = judge says every line is correct
  all rows          0.6992
  step-0 rows only  0.6114
  paired difference +0.0882 95% CI [+0.024, +0.154]   EXCLUDES zero
```

So the body is not inert. The features carry real information about **what the
judge will say** — unsurprising, since the judge reads the code — and no
separable information about **whether the code runs**. The execution-predicting
part of the signal is not distinguishable from a read on problem difficulty,
which is available before the answer exists.

That is consistent with the rest of what this dataset shows. Correctness here is
overwhelmingly a property of the problem, not the generation: on the 129
problems both arms share, simply copying the Python label predicts Rust
correctness at **AUROC 0.8669** (86.0% agreement, phi +0.731, P(Rust passes |
Python passed) = 0.794 against 0.066).

So the question this study can actually answer is narrower, and it is the one
that matters for the paper's claim:

> **Does anything in the attribution graph of the generated code predict whether
> that code RUNS, beyond what the problem statement alone already predicts?**

On Python the answer is no — the contrast is +0.0340 [−0.0160, +0.0846], and
+0.0165 [−0.0341, +0.0684] once the corrupted generations are removed. The Rust
arm asks the same question of a second language.

## 2. The protocol — fixed

`data/5_problem_level.py`, run identically on both arms.

| | |
|---|---|
| unit | one aggregated prediction per held-out **problem** |
| aggregator | mean of the problem's row probabilities |
| model | `RandomForestClassifier(n_estimators=100, random_state=42)` |
| scoring | 10 repeats × 5-fold `GroupKFold`, grouped by problem, every problem out of fold |
| interval | 5000-draw bootstrap resampling **problems** |
| population | the problems whose generation compiles (Rust: 214 of 350). The restriction is imposed upstream by `2_incremental_data_rust.py`, not chosen at analysis time; no graph exists or is planned for the 136 that do not compile |
| token cap | 140, applied to both arms |
| labels | execution: compiles **and** every graded assert passes |

**The primary result is a contrast, not a level.**

> **C1 (primary).** Paired bootstrap over problems of
> *AUROC(all rows) − AUROC(step-0 rows only)*.
> The paper's claim is supported **only if this CI excludes zero.**

Reporting the level alone is forbidden: a prompt-difficulty detector satisfies
it. Every table reports the step-0 number beside the headline.

Three further prohibitions, each the cause of an earlier error:

- **no row-level AUROC.** It weights each problem by its solution length, and
  length correlates with the label.
- **no seed-to-seed sd reported as an interval.** It measures split noise, not
  sampling error. (The size of the discrepancy is scheme-dependent — for the
  locked 10-repeat estimator the fold-seed sd is ~7× narrower than the bootstrap
  SE, but for a 70/30 `GroupShuffleSplit` it is ~1.25× *wider*. The prohibition
  is not about the factor; it is that the quantity is not a sampling error.)
- **no paired test over splits.** Its p-value is a function of the seed count.
  A reproducible demonstration on this data: `-max prefix chars` vs `-n_steps`
  on a true gap of +0.0216 gives p = 0.0020 at 10 seeds, 0.0001 at 20, <0.0001
  at 30. *(Version 1 quoted a 0.2305/0.0494/0.0010 triple that does not
  reproduce from anything the protocol records; it is withdrawn.)*

## 3. Locked baselines — all pre-data

### 3a. Python, graph features

```
256 problems, 114 positive (44.5%), 1337 rows after the 140-token cap
  level (all rows)          0.7269  95% CI [0.6619, 0.7901]
  step-0 only               0.6931  95% CI [0.6265, 0.7558]
  C1 contrast               +0.0340 95% CI [-0.0160, +0.0846]  does NOT exclude zero
  C2 vs best surface        +0.1026 95% CI [+0.0408, +0.1658]  excludes zero
  C4 residualised           0.7241, loss +0.0028               passes
```

**The Python arm's own C1 is null.** Whatever the Rust arm shows, the Python
comparator has already failed the primary test.

### 3b. Python, the like-for-like subset

The shipped `gemma_mbpp_responses.jsonl` is still the **pre-fix** file: 16 of
256 generations carry unstripped markdown fences and do not parse, and 147 of
1420 feature rows contain a literal ` ``` `. The extraction fix landed in the
script and was never applied to the data. Their Rust counterparts *are* deleted
(`2_incremental_data_rust.py` drops non-compiling generations), so the arms are
asymmetric until the Python graphs are rebuilt.

```
240 problems that parse, 114 positive (47.5%), 1137 rows
  level (all rows)          0.7091  95% CI [0.6402, 0.7737]
  step-0 only               0.6928  95% CI [0.6234, 0.7586]
  C1 contrast               +0.0165 95% CI [-0.0341, +0.0684]  does NOT exclude zero
  C2 vs best surface        +0.1243 95% CI [+0.0538, +0.1950]  excludes zero
  C4 residualised           0.6999, loss +0.0092               passes
  best surface              -max prefix chars 0.5846
```

**This is the like-for-like comparator for Rust**, and it is locked now so the
choice between 0.7269 and 0.7091 cannot be made after seeing a Rust number. Note
that the surface baselines fall much further than the graph on the clean subset
(0.6245 → 0.5846), because the corrupted generations were long *and* failing, so
length predicted them well. C2 therefore gets larger on clean data while C1 gets
smaller — the two contrasts move in opposite directions, which is itself the
finding: what the features have is a problem-difficulty read that beats length,
not a read on the code.

### 3c. Surface baselines — both arms, no graph features needed

```
                              Python              Rust (214 problems)
  trained RF on 4 surface   0.5751            0.6603 [0.5857, 0.7316]
  -n_steps                  0.6011            0.6219 [0.5455, 0.6973]
  -max prefix chars         0.6245            0.6970 [0.6251, 0.7653]   <- best in both
  -max prefix lines         0.6011            0.6219 [0.5456, 0.6968]
```

Two of the four are the same vector: rows are line prefixes, so the row count
*is* the line count (`-n_steps` ≡ `-max prefix lines`, identical on 256/256 and
214/214 problems). "Best surface" means the highest of the three distinct ones.

**The Rust length heuristic is 0.072 stronger than Python's.** Version 1's P1
("AUROC near 0.70") and P2 ("beats best surface by +0.05 to +0.15") were jointly
satisfiable only in [0.747, 0.80] — above P1's own central value — so whichever
way the number landed, one prediction passed and the other failed. Both are
restated in §4.

### 3d. The arms overlap less than stated, and not at random

```
  Python problems                      256
  Rust typeable (labelled)             350
  Rust analysed (compiling)            214
  Python  n  Rust-typeable             207
  Python  n  Rust-analysed             129   <- the real overlap
  Rust-analysed that Python never saw   85
```

Version 1 and the readme both used 207. The analysed overlap is **129**. Those
129 are easier in Python than the 127 that are not (52.7% vs 36.2% pass, Fisher
p = 0.0085), so Python's AUROC on them is **0.7816**, not 0.7269 — a +0.055
selection shift with no data change at all. Any cross-arm statement uses 0.7816
at n = 129 or is not made.

The arms are also drawn from different MBPP populations: Python is the test
split only; Rust spans train/test/validation/prompt (208/103/34/5).

## 4. Predictions

**C1 (primary).** The Rust paired contrast *AUROC(all rows) − AUROC(step-0)*.
*Prediction: its 95% CI includes zero, as Python's does.* If it excludes zero
and is positive, that is evidence the generated code contributes something in
Rust that it does not in Python, and it is the one result that would support the
paper's claim as the paper states it.

**C2.** Rust AUROC(all rows) vs the locked best Rust surface baseline (0.6970),
paired over problems. *Prediction: the CI includes zero.* Minimum resolvable
paired gap at n = 214 is about **0.06**, so a real effect below that will not be
detected; "CI includes zero" is reported as **unresolved**, never as "the claim
fails".

**C3.** Rust AUROC(all rows) is above chance: the 95% CI excludes 0.50.
*Prediction: yes, in [0.65, 0.76].* This band catches ~83% of outcomes if the
truth is 0.705, so it can miss. A miss is reported as a miss; it does not by
itself refute or support anything, because §1 establishes that the level is
satisfiable without any generated code.

**C4 (length).** Regress every feature at row level on
`[chars, log chars, chars², step, n_steps, lines, tokens, tokens²]` and re-score
the residuals. *Prediction: the loss is under 0.05.* On Python this gives
**0.7241 against 0.7269** (loss +0.0028; on the clean 240, 0.6999 against 0.7091,
loss +0.0092) — the real features pass cleanly.
*Version 1's drop-|r|>0.7 filter is withdrawn: a synthetic matrix built to be
100% length by construction passes it at every noise level tested, so it is a
test nothing can fail. It was also computed at the row unit while the claim is
at the problem unit — 23 of 49 dims exceed the threshold when recomputed at the
problem unit, against 10 at the row unit.*

**C5 (difficulty transfer).** Spearman correlation between the Python and Rust
per-problem out-of-fold scores, over the 129 shared problems, against a null
from permuting problems. *Prediction: ρ is clearly positive.* A strongly
positive ρ supports the reading that both arms read language-invariant problem
difficulty, and is **not** evidence that the effect generalises across languages.
The headline is additionally reported on the **85 Rust problems Python never
saw**, where no such transfer is possible.

## 5. What each outcome means

| C1 | C2 | reading |
|---|---|---|
| excludes zero, positive | — | the generated code contributes in Rust. The strongest result available, and stronger than Python's own. |
| includes zero | excludes zero | the features beat surface heuristics but cannot be shown to use the generated code. Consistent with a prompt-difficulty detector. |
| includes zero | includes zero | **unresolved at this n.** Not evidence of absence, and not support. |
| — | point estimate ≤ 0 | the graph adds nothing over length in Rust. |

**Equivalence bound.** If the Rust 95% CI upper bound is below **0.65**, the
effect is reported as materially weaker in Rust than the Python 0.7269. Without
this, no outcome the study can produce is evidence against the effect.

No threshold above is adjusted after seeing Rust numbers. A deviation, if one
proves necessary, is recorded in §8 with the original result alongside.

## 6. What this study cannot do

**It cannot separate "predicts code correctness" from "predicts problem
difficulty"** except through C1, and C1 is null on Python. This is the central
limitation and it applies to the paper's claim as much as to this replication.

**It is not powered to compare the languages.** Rust's half-width at n = 214
runs 0.061–0.076 across the predicted range, so the between-arm standard error
is near 0.05: a gap below ~0.10 would not have its CI exclude zero, and ~0.14 is
needed for 80% power. Pairing on the 129 shared problems does not rescue this —
the achievable score correlation bounds the paired MDE at 0.111 in the best case.

**The arms are not independent samples.** Copying the Python label scores 0.8669
on the Rust arm. Two arms drawn from the same problems with correlated outcomes
are not two tests of a hypothesis.

**The arms are not matched on information.** Rust generations were conditioned
on the function name *and* the full type signature; Python generations got
neither, and had the name repaired afterwards — aliasing fired on 210 of 256 and
manufactures 99 of the 114 positives (strict pass is 15/256). Under a matched
prompt the Python pass rate lies somewhere in [44.5%, 61.7%]. **No pass-rate or
AUROC difference between the arms is attributed to language.**

**On the analysed sets the base rates are nearly matched** — Rust 99/214
(46.3%) against Python 114/256 (44.5%) — so the "28.3% vs 44.5%" figure quoted
earlier compared the wrong populations. What does not match is composition: the
shared problems are easier in Python.

**Both arms are data-starved.** Python's learning curve was still climbing at
204 training problems, though its final increment is not significant
(+0.0152 ± 0.0107, 7/10 splits, paired p = 0.19). A null result means
underpowered, not absent.

**Intervals are conditional on the fitted scoring rule.** The bootstrap
resamples problems against a score vector computed once, so it omits
model-fitting variance. Refitting inside the bootstrap widens the interval by
about 1.28×. The power arithmetic above uses the reported widths and is
therefore optimistic in the same direction.

**The Rust label is graded on a subset of MBPP's asserts for 15 of 350 tasks.**
`derive_signature` silently skips asserts it cannot type — mostly the `== None`
invalid-input cases — and 4 of those tasks are labelled passing. They are
excluded from the headline and reported separately.

**Selection.** Rust problems are the 350 of 427 whose types can be read off
their asserts, then the 214 whose generations compile. Neither is a random
sample of MBPP.

## 7. Known defects not fixed before the data lands

- The Python graphs are built on 16 fence-corrupted generations (§3b). Fixing
  this needs the GPU, which is busy until ~2026-09-28. Until then §3b's
  240-problem number is the comparator.

  **The fix is prepared and costed, so it cannot be quietly skipped or quietly
  adopted.** `0b_repair_responses.py` repairs all 16 (0 remaining); the repaired
  code is shorter, so the rows to rebuild fall from 263 to 132, of which 124
  sit inside the 140-token cap — a measured **8.9 hours** of GPU.
  `run_python_topup.sh` runs the whole sequence and refuses to start while
  another graph run holds the card.

  Relabelling the repaired generations moves the Python pass rate from
  **114/256 to 117/256**: tasks 285, 395 and 405 flip to passing, each on 3/3
  asserts. They were failing because the extractor left fence lines in their
  code, not because the model got them wrong. When the top-up lands, the result
  is recorded in §8 as a deviation **alongside** the locked 240-problem number,
  not in place of it.
- Task 67's Rust harness cannot compile at all — its MBPP test literal exceeds
  `i64` — so it is recorded as a model compile failure. One problem of 350.
- The Rust float check uses `max(|e|, 1.0)` where `math.isclose` uses
  `max(|v|, |e|)`, and is applied to 10 tasks whose MBPP assert demands exact
  equality. No shipped label is affected; it is wrong for any re-run.

## 8. Deviations

**2026-09-22, version 1 → 2, before any Rust feature existed.** Nine
"would-produce-wrong-conclusion" defects found by an adversarial review of
version 1 and confirmed by re-running each:

1. Primary result changed from the AUROC level to the C1 contrast, because
   prompt-only graphs reproduce the headline (+0.0332 [-0.0172, +0.0826]).
2. P1 and P2 were jointly satisfiable only in the top quarter of P1's own range;
   restated as C2 and C3 against the now-locked Rust surface baselines.
3. P3's drop-|r|>0.7 filter replaced by residualisation (C4): a matrix built to
   be 100% length passes the old test at every noise level.
4. The arm overlap is 129, not 207; a second Python baseline is locked for it.
5. An equivalence bound added to §5; version 1 had no outcome that could be
   evidence against the effect.
6. The "4–10× too narrow" factor replaced with scheme-specific measurements, and
   the Wilcoxon triple withdrawn as unreproducible.
7. Power arithmetic recomputed at n = 214 and split into "CI excludes zero" and
   "80% power".
8. The compile restriction documented as imposed upstream, not chosen.
9. P4 (learning curve) demoted from a prediction to a descriptive figure: it had
   no decision rule and its final Python increment is not significant.
