# Pre-registration — the Rust arm

**Version 4, 2026-09-25.** Version 1 was written on 2026-09-22 and red-teamed
the same day, before any Rust graph feature existed: four independent attacks
found nine defects rated "would produce a wrong conclusion", every one of which
I confirmed by re-running it, and version 2 fixed them. Version 3 (2026-09-24)
re-derived the Rust labels twice and fixed three harness defects. Version 4
settles which problems the cross-arm statistics use and how, and commits the
whole analysis as a script. The Rust run is still in flight: when version 4 was
written, 600 of its 1734 feature rows had landed and **none had been scored
against a real label**. Every number below comes from data that existed before
them, except the shuffled-label figures in §8 (version 4, item 9), which use
the landed features with the labels permuted. §8 lists every change.

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
| population | **strict**: the problems whose generation compiles *and* every MBPP assert is gradeable — Rust 206 of 350, 95 positive (46.1%). The loose population (214 / 99) is reported alongside, never instead |
| token cap | 140, applied to both arms |
| labels | execution: compiles **and** every graded assert passes |

**Populations and evaluation sets are different things** (version 4). A
*population* — strict or loose Rust, the 256 or the 240 Python problems — is
what a model is fitted on: `5_problem_level.py` refits on it via `SUBSET_FILE`.
An *evaluation set* — the problems both arms share, the Rust problems Python
never saw — only restricts an arm's existing out-of-fold score vector, and
`6_cross_arm.py` never refits. Refitting on 80 or 120 problems would confound
the subset with a smaller training set while the learning curve is still
climbing (§6). Every set is written by `5a_make_subsets.py` from the label
files, with its size asserted, and `run_rust_analysis.sh` runs the whole
analysis below.

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

**Version 4 adds the strict population**, which is C2's actual comparator and
had no locked number:

```
                              Rust strict (206)          Rust loose (214)
  trained RF on 4 surface   0.6567 [0.5827, 0.7296]    0.6603 [0.5834, 0.7294]
  -n_steps                  0.6287 [0.5514, 0.7023]    0.6219 [0.5467, 0.6944]
  -max prefix chars         0.6942 [0.6229, 0.7653]    0.6970 [0.6217, 0.7650]   <- best in both
  -max prefix lines         0.6287 [0.5508, 0.7034]    0.6219 [0.5468, 0.6960]
```

These come from `5_problem_level.py` itself, run on a features file with the
Rust run's exact rows and every feature set to zero: its surface block depends
only on lengths and labels. The 600 rows already landed match that file row for
row, and the loose column reproduces the table above to four decimals. The
intervals differ from version 2's in the third decimal because version 2's came
from a different random stream; the analysis script consumes its stream
identically whatever the features are, so it will print these intervals.

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
selection shift with no data change at all. Any cross-arm statement is made on
problems both arms share, or is not made; §3e fixes which ones.

The arms are also drawn from different MBPP populations: Python is the test
split only; Rust spans train/test/validation/prompt (208/103/34/5).

### 3e. The cross-arm comparators — version 4, still pre-data

The 129 predate the strict rule, and two version-2 rules disagreed: §3b names
the 240 parsing problems "the like-for-like comparator for Rust", while 0.7816
above comes from the 256-problem fit. Version 4 applies both rules at once and
locks the Python side now (files in `data/results/python/`):

```
                          cross set                          Python fit  n    Python AUROC on it
  primary                 strict in Rust, parses in Python  240         120  0.7775  95% CI [0.6857, 0.8602]
  as defined in v2        compiles in Rust                  256         129  0.7816  95% CI [0.6965, 0.8611]
  after the top-up (§7)   strict in Rust                    repaired    126  when the top-up lands
```

Each number is the arm's own out-of-fold score vector **restricted** to the
set, with no refit — re-running the version-2 computation that way reproduces
0.7816 exactly, so that is also what it always was. The Rust side is computed
the same way, from the strict fit for the primary row and the loose fit for the
version-2 row. Of the primary 120, 67 pass in Python and 56 in Rust. The Rust
problems Python never saw number **80** under the strict rule (38 pass) and 85
under the loose one (41).

## 4. Predictions

**C1 (primary).** The Rust paired contrast *AUROC(all rows) − AUROC(step-0)*.
*Prediction: its 95% CI includes zero, as Python's does.* If it excludes zero
and is positive, that is evidence the generated code contributes something in
Rust that it does not in Python, and it is the one result that would support the
paper's claim as the paper states it.

**C2.** Rust AUROC(all rows) vs the locked best Rust surface baseline —
`-max prefix chars`, **0.6942** on the strict population (0.6970 on the loose;
§3c) — paired over problems. *Prediction: the CI includes zero.* Minimum
resolvable paired gap at n = 206 is about **0.06**, so a real effect below that
will not be detected; "CI includes zero" is reported as **unresolved**, never as
"the claim fails".

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
per-problem out-of-fold scores over the cross set, against a null from
refitting the Rust arm on shuffled labels. *Primary:* the 240-problem Python fit
and the strict Rust fit, on the 120 problems in both (§3e). *Alongside:* the
version-2 definition, the 256-problem fit and the loose Rust fit, on the 129.
The interval is a 5000-draw bootstrap over problems. The p-value is one-sided —
H1: ρ exceeds what a Rust model that learned nothing real about Rust
correctness produces — from K = 100 refits of the Rust population with its
labels permuted across its problems, with the +1 correction
(`6b_c5_null.py`). The problem-permutation p that version 2 specified is still
printed and is **not** the test (§8, version 4, item 9). *Prediction: ρ is
clearly positive,* meaning ρ > 0 with that p below 0.05. A strongly positive ρ
supports the reading that both arms read language-invariant problem
difficulty, and is **not** evidence that the effect generalises across
languages. The headline — level and C1 — is additionally
reported on the Rust problems Python never saw, **80 strict** (primary) and 85
loose, where no such transfer is possible; both are restrictions of the
population fit, not refits. Each arm's AUROC on the cross set and their paired
difference are printed beside C5 as description only — §6 shows the difference
cannot resolve a gap below ~0.11, and none is attributed to language.

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

**The Rust label was graded on a subset of MBPP's asserts for 16 of 350 tasks,
and 4 of them are labelled passing.** `derive_signature` skips asserts it cannot
type — mostly the `== None` invalid-input cases — and accepted the task anyway.
Task 744 is the clearest: both asserts that return `True` pass `None` inside a
tuple and are dropped, so it is graded on one remaining case, which a stub
passes by comparing each field to `0`. 297 is graded 2/3, 602 2/3, 755 1/4.

That is ~4% of the positive class, and it is not random — the dropped asserts
are systematically the edge cases, so the Rust arm was being measured on an
easier test set than the Python arm. **This is why the headline population is
the strict one.** The rule is applied uniformly and requires no judgement about
which subsets are "partial enough":

```
  loose  (as shipped)                    214 compile, 99 pass (46.3%)
  strict (every MBPP assert gradeable)   206 compile, 95 pass (46.1%)
  the 8 that drop out: 297, 395, 398, 602, 725, 744, 755, 803
  of those, positive: 297, 602, 744, 755
```

Their graphs are still built — the restriction is applied at analysis time, so
the in-flight run needs no change, and the 8 are reported separately.

**Selection.** Rust problems are the 350 of 427 whose types can be read off
their asserts, then the 214 whose generations compile. Neither is a random
sample of MBPP.

## 7. Known defects not fixed before the data lands

- The Python graphs are built on 16 fence-corrupted generations (§3b). Fixing
  this needs the GPU, which is busy until ~2026-09-30. Until then §3b's
  240-problem number is the comparator.

  **The fix is prepared and costed, so it cannot be quietly skipped or quietly
  adopted.** `0b_repair_responses.py` repairs all 16 (0 remaining); the repaired
  code is shorter, so the rows to rebuild fall from 263 to 132 (242 feature rows
  today — the 550-character filter had dropped 21 of the long corrupted ones),
  of which 124 sit inside the 140-token cap — a measured **8.9 hours** of GPU.
  `run_python_topup.sh` runs the whole sequence and refuses to start while
  another graph run holds the card. Since version 4 it runs to the end: it stops
  if the regenerated inputs differ from the committed ones, merges through
  `4b_merge_topup.py` — which copies the other 240 problems' rows byte for byte
  and refuses an incomplete, duplicated, non-finite or non-64 top-up — and scores
  the repaired arm alone and against Rust on the 126-problem cross set.
  Dropping the 550-character filter for the top-up loosens nothing: on this data
  the 140-token cap is strictly tighter (the longest row inside it is 486
  characters, and no repaired row is over 550 characters yet inside it).

  Relabelling the repaired generations moves the Python pass rate from
  **114/256 to 117/256**: tasks 285, 395 and 405 flip to passing, each on 3/3
  asserts. They were failing because the extractor left fence lines in their
  code, not because the model got them wrong. When the top-up lands, the result
  is recorded in §8 as a deviation **alongside** the locked 240-problem number,
  not in place of it.
*(Both Rust harness defects listed here in version 2 are now fixed — see §8.)*

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

**2026-09-24, version 2 → 3, still before any Rust feature existed.** The Rust
execution labels were independently re-derived twice, by deliberately different
mechanisms: one built a real cargo crate per problem with each assert as a
`#[test]` and the dev profile's overflow checks **on**, the other derived the
types from scratch under stricter rules and emitted floats as exact bit
patterns. Both reproduce the shipped labels on **350/350 for `compiles`,
350/350 for `passes`, and 350/350 for `n_pass`**. The Rust arm carries no
analogue of the two Python-arm label bugs, and the reason is structural: putting
the signature in the prompt eliminates the name-derivation bug, and handling
`math.isclose` at derivation time eliminates the `test_imports` bug. **The
in-flight graph run covers the right 214 problems and needed no change.**

Three harness defects were found and fixed. Re-labelling with the corrected
harness moved **no label at all** (350/350 unchanged), confirming all three were
latent — but each was live for any regenerated dataset:

  a. `rust_type` mapped every Python int to `i64` unchecked, so task 67's
     55-digit assert produced a literal `rustc` refuses. Now range-checked.
  b. The float tolerance `|v−e| ≤ 1e-3·max(|e|,1.0)` was applied to all 21
     f64-returning tasks; only 11 use `math.isclose`, and the formula was not
     isclose's. Now `|v−e| ≤ 1e-3·max(|v|,|e|)` and only where MBPP wrote it.
     (Verified to change 0 asserts: closest approach is 34× inside the threshold.)
  c. `_unwrap_call` never checked the comparison operator was `==`, so `!=`
     would have been graded as `==`. MBPP sanitized contains no such assert, so
     exposure was zero, but it is now checked.

The fourth defect is the one with live consequences and is handled by changing
the population rather than the harness — see §6. `graded_all_tests` and
`n_mbpp_asserts` are now recorded per task so the strict population is
reproducible from the file rather than from this document.

**2026-09-25, version 3 → 4, before any Rust feature was scored against a
label.** 600 of the 1734 Rust feature rows had landed. They had been read only
by integrity checks (batch size, duplicates, non-finite values) and by the
shuffled-label runs in items 6 and 9; no statistic relating a Rust feature to a
real Rust label had been computed.

1. **A correction to version 3's entry,** which said "still before any Rust
   feature existed". That was false: the run banks features every 200 graphs,
   and 400 rows had landed by then (chunks banked 2026-09-23 05:34 and
   2026-09-24 04:38; version 3 was committed 2026-09-24 14:05). The version-2
   header's "no Rust feature exists" was true when written and stayed in the
   document after it stopped being true. What held throughout, and still holds,
   is that no Rust feature had been scored against a label.
2. **The header still said "Version 2"** after version 3 was committed — the
   edit meant to change it did not match. Fixed.
3. **The cross-arm sets.** The overlap of 129 and the 85 Rust-only problems
   predate the strict rule, and §3b ("the like-for-like comparator for Rust" is
   the 240) disagreed with §3d (0.7816, from the 256-problem fit). Both rules now
   apply: the primary cross set is the 120 problems strict in Rust and parsing
   in Python, scored by the 240-problem fit, with the version-2 definition
   reported alongside; the Rust-only headline is 80 strict, 85 loose alongside.
   The Python side is locked in §3e.
4. **Restriction, not refit.** `SUBSET_FILE` refits, and its comment listed the
   129 and the 85 among its uses, while 0.7816 had in fact been computed by
   restricting the 256-problem score vector. Each kind of set now has one
   meaning (§2), and the version-2 number reproduces exactly under it.
5. **C5 was not computable from the protocol's output.** `5_problem_level.py`
   wrote no per-problem scores. `SCORES_OUT` now writes them; it consumes no
   randomness, and re-running both locked Python baselines with it reproduced
   every number in §3a and §3b to four decimals. `6_cross_arm.py` computes ρ
   and its interval; `6b_c5_null.py` computes its test (item 9).
6. **The analysis is a committed script.** `run_rust_analysis.sh` checks
   integrity (all 1734 rows and all 214 compiling problems, or it stops), then
   runs both Rust populations, both cross-arm definitions and C5's null. It was
   rehearsed end to end on a copy of the 600 landed rows with the pass/fail
   labels permuted across the 214 compiling problems (seed 12345), which
   exercises every step and says nothing about the real relation.
7. **The Python top-up is scripted to the end** (§7). `4b_merge_topup.py` was
   tested on a stand-in top-up: it accepts a complete one, and refuses a missing
   problem, a duplicated step, batch size 32, a non-finite feature, a stray
   problem, and an output path that is one of its inputs.
8. The GPU date in §7 moved from ~2026-09-28 to ~2026-09-30, the Rust run's
   measured pace; §7 also now counts the top-up's rows at both stages.
9. **C5's null was wrong, and the rehearsal showed it.** In the rehearsal the
   shuffled-label Rust scores correlated with the real Python scores at
   ρ = −0.41 over 69 problems, far outside the problem-permutation null
   (sd 0.12). Refitting the Rust model on the 600-row snapshot with its labels
   shuffled under six seeds gave ρ = −0.425, −0.413, −0.125, −0.083, −0.072 and
   +0.388: arbitrary sign, sd ≈ 0.30. The spread tracks each label-free score's
   alignment with length (its ρ with `-max prefix chars` ran −0.583 to +0.250,
   moving with the first column), and the Python score is itself aligned with
   length (+0.503 on those 69). So a model that has learned nothing about Rust
   correctness can match the Python ranking this well by riding a shared
   feature axis, and a permutation of problems — which destroys that axis —
   would call it significant. C5 is now decided against refits on shuffled
   labels (§4, `6b_c5_null.py`, K = 100), and "clearly positive" is defined as
   ρ > 0 with that p below 0.05. Before refitting, the script refits once on
   the real labels and demands `5_problem_level.py`'s scores bit for bit. These
   runs read Rust features with shuffled labels only, and length is already
   known to predict Rust correctness (§3c), so they reveal nothing the
   pre-registration did not already contain.
10. **C2 had no comparator on its own population.** §3c locked the best surface
   baseline on the loose 214 only, while C2 is decided on the strict 206. The
   strict column is now locked (`-max prefix chars` 0.6942), computed by the
   analysis script itself on zero features so that it will print the same
   numbers on the real run.

## 9. Results — the Rust arm

The graph run finished 2026-09-30 23:44 (ALL CHUNKS COMPLETE: 1734 rows, 214
problems, no failed graph, every integrity check passing). A watcher started
`run_rust_analysis.sh` two minutes later, unattended, and its outputs were
committed as produced (08a6663) before anything else was computed. Every locked
comparator printed exactly as locked: the four C2 surface baselines on both
populations, to the last digit of their intervals, and both Python cross-set
numbers.

### 9a. The pre-registered predictions — strict population, 206 problems, 95 passing

```
                           result                                   prediction            outcome
  C1  all rows - step 0    +0.1350  95% CI [+0.0715, +0.2013]       CI includes zero      FAILED: excludes zero
  C2  - best surface       +0.0942  95% CI [+0.0248, +0.1650]       CI includes zero      FAILED: excludes zero
      (-max prefix chars 0.6942)
  C3  level                0.7887   95% CI [0.7245, 0.8482]         > 0.50, in [0.65,0.76] above chance; band missed high
  C4  residualised         0.7540, loss +0.0347                     loss under 0.05       held
  C5  rho on the 120       +0.7581  95% CI [+0.6650, +0.8251]       clearly positive      held
      label-free refits: mean -0.028, sd 0.241, 95th pct +0.358; p 0.0099 (K = 100, the floor)
  equivalence bound        CI upper 0.8482                          below 0.65            not met: Rust is not weaker
```

**By the table in §5 this reads: the generated code contributes in Rust** —
"the strongest result available, and stronger than Python's own". The
prediction for the primary result was wrong, and wrong in the direction of the
paper's claim. Step 0 alone scores 0.6540 [0.5776, 0.7269].

The loose population (214, 99 passing) agrees: level 0.7953 [0.7316, 0.8524],
step 0 0.6732, C1 +0.1217 [+0.0639, +0.1831], C2 +0.0989 [+0.0253, +0.1719], C4
loss +0.0293. On the 80 strict problems Python never saw: level 0.7920
[0.6842, 0.8845], C1 +0.1257 [+0.0352, +0.2201]; on the 85 loose ones, C1
+0.0724 [−0.0153, +0.1619].

On the 120-problem cross set the two arms score 0.7775 (Python) and 0.7930
(Rust), a paired difference of +0.0149 [−0.0775, +0.1096], and their labels agree
on 85.8% of problems. Per §6 no difference between the arms is attributed to
language: the Python labels lean on name-aliasing (99 of its 114 positives), and
the Rust prompts carried the signature.

The learning curve (descriptive) still climbs: 0.7391 at 41 training problems
to 0.7949 at 164.

### 9b. Exploratory, not pre-registered: is C1 the code, or the extra rows?

This question was found only after 9a was seen, so it is handled the way the
pre-registered analysis was: `7_one_row_control.py` was written and committed
(dc7ed09) with its single question and its reading rule fixed in its docstring
before it was first run.

C1 compares a model fitted and scored on ~8 rows per problem with one fitted and
scored on a single row. The multi-row side has several times the training rows
and averages away test-time noise, which could raise its AUROC with no
information from the code at all. The control puts **one row per problem on
each side**: step 0 against the last row, the most code the 140-token cap keeps.

```
                       step 0   last row   last - step 0                 all rows - last row
  Rust strict (206)    0.6540   0.7606     +0.1066 [+0.0321, +0.1810]    +0.0281 [-0.0150, +0.0721]
  Rust loose (214)     0.6732   0.7579     +0.0849 [+0.0117, +0.1564]    +0.0371 [-0.0074, +0.0806]
  Python 256           0.6931   0.6993     +0.0054 [-0.0660, +0.0746]    +0.0277 [-0.0182, +0.0739]
  Python 240           0.6928   0.6814     -0.0112 [-0.0865, +0.0646]    +0.0278 [-0.0214, +0.0767]
```

By the rule fixed beforehand: **in Rust the generated code carries information
beyond the prompt with one row on each side** — about four-fifths of C1
survives, and the extra rows add a further +0.03 that is not distinguishable
from zero. In Python the last row adds nothing to step 0.

Printed alongside as context — not a test, and not something to lean on in
either direction: residualising the last row's features on C4's eight length and
position terms leaves it +0.0320 [−0.0489, +0.1160] above step 0 in Rust
(+0.0039 loose). So at this n what the code adds **cannot be separated from its
length**, although the last-row graph does beat the length heuristic itself,
+0.0655 [+0.0027, +0.1310]. Residualising also strips whatever is merely
correlated with length, so this says "not separable", not "is length".
