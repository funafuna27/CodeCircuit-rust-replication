# Pre-registration — the Rust arm

Written **2026-09-22**, while the Rust graph run is in flight and no Rust
feature has been computed. Everything below is fixed before the data exists.

The reason for writing it is specific to this project. The Python arm produced
three successive conclusions that were wrong — a leakage claim, an
execution-label indictment, and a granularity explanation — and each survived
until somebody re-derived it independently. Every one of those errors was a
choice made *after* seeing a number. This document removes that freedom for the
Rust arm.

---

## 1. The question

He et al. 2026 claim that topological features of an LLM's attribution graphs
predict whether its generated code is correct, and that this holds across
languages. The Python replication reproduced their headline (79.73% against a
reported 79.89%) and then established what that number measures: a
prompt-mandated step function, scored at the wrong unit, with intervals several
times too narrow.

The Rust arm asks one question:

> **Against execution truth, at the problem level, do the graph features predict
> Rust correctness above chance and above surface heuristics?**

It does **not** ask whether Rust and Python differ. See §6.

## 2. The protocol — fixed

`data/5_problem_level.py`, unchanged, run identically on both arms.

| | |
|---|---|
| unit | one aggregated prediction per held-out **problem** |
| aggregator | mean of the problem's row probabilities |
| model | `RandomForestClassifier(n_estimators=100, random_state=42)` |
| scoring | 10 repeats × 5-fold `GroupKFold`, grouped by problem, every problem scored out of fold |
| interval | 5000-draw bootstrap resampling **problems** |
| null test | 1000 label permutations across problems |
| token cap | 140, applied to both arms |
| labels | execution: compiles **and** every MBPP assert passes |

Three things are forbidden, because each produced a wrong conclusion on the
Python arm:

- **no row-level AUROC.** It weights each problem by its solution length, and
  length correlates with the label.
- **no seed-to-seed sd reported as an interval.** It measures split noise, not
  sampling error, and is 4–10× too narrow.
- **no paired t or Wilcoxon over splits.** Its p-value is a function of the seed
  count: one comparison ran p = 0.2305 at 20 seeds, 0.0494 at 50, 0.0010 at 100.

## 3. The Python baseline — locked

Measured 2026-09-22 with the script above, before any Rust feature existed.

```
gemma_mbpp_features.jsonl | 256 problems, 114 positive (44.5%) | 1337 rows
  graph topology            0.7269  95% CI [0.6619, 0.7901]   perm p < 0.001
  trained RF on 4 surface   0.5751  95% CI [0.5058, 0.6431]
  -n_steps                  0.6011  95% CI [0.5338, 0.6657]
  -max prefix chars         0.6245  95% CI [0.5556, 0.6894]
  graph - best surface      +0.1030 95% CI [+0.0395, +0.1662]
  drop 10 length-proxy dims 0.7147  95% CI [0.6516, 0.7751]
```

## 4. Predictions

Stated as intervals, not points, and in the order they will be checked.

**P1 — signal exists in Rust.** Rust problem-level AUROC is above chance:
permutation p < 0.05 and the 95% CI excludes 0.50.
*Prediction: AUROC in [0.62, 0.80], most likely near 0.70.*

**P2 — it beats surface heuristics.** The paired graph-minus-best-surface CI
excludes zero, as it does in Python (+0.1030 [+0.0395, +0.1662]).
*Prediction: a positive difference of +0.05 to +0.15.*

**P3 — it is not a length proxy.** Dropping every dimension correlating
|r| > 0.7 with prefix length costs less than 0.05 AUROC, as in Python
(0.7269 → 0.7147).

**P4 — the learning curve is still rising at the largest training set.** 214
problems will not saturate the question any more than 256 did.

**P5 — the compile/no-compile split matters.** Restricted to the 214 problems
that compile, the label is near-balanced (46.3% positive). Including the 136
that do not compile would add an easy, length-detectable negative class; the
headline is computed on compiling problems only, and the uncompilable set is
reported separately rather than folded in.

## 5. What would falsify what

| outcome | reading |
|---|---|
| CI excludes 0.50 **and** beats surface | P1+P2 hold; the paper's comparative claim generalises to Rust |
| CI excludes 0.50, does **not** beat surface | the signal is a surface artefact in Rust; the paper's central claim fails here |
| CI includes 0.50 | no evidence of the effect in Rust at this n; **not** evidence of absence (see §6) |
| dropping length proxies costs > 0.05 | P3 fails; report the features as partly a size proxy in Rust |

A result at the bottom of the predicted range is reported as such. No threshold
in §2 will be adjusted after seeing Rust numbers; if a deviation proves
necessary it is reported as a deviation, with the original result alongside.

## 6. What this study cannot do

**It is not powered to compare the languages.** Each arm's 95% CI has a
half-width near 0.065, so the difference between two independent arms carries a
standard error near 0.047 and only a gap beyond roughly **0.09–0.13** would be
resolvable. Any smaller Rust-vs-Python difference is noise and will be reported
as unresolvable, not as similarity.

**The arms are not matched on difficulty.** Rust passes 28.3% of problems
against Python's 44.5%, and the Rust prompts carry a required signature the
Python prompts did not. A difference in AUROC therefore cannot be attributed to
language alone even if it were large enough to resolve.

**Both arms are data-starved.** Python's learning curve was still climbing at
204 training problems (0.659 → 0.713 → 0.738 → 0.753). Neither arm has enough
problems to saturate the question, and the honest framing of any null result is
"underpowered", not "no effect".

**The Rust label inherits a selection step.** Problems are restricted to the 350
of 427 MBPP tasks whose argument and return types can be read off their asserts,
then to the 214 whose generations compile. Both restrictions are stated with the
result; neither is a random sample of MBPP.

## 7. Deviations

Any change to §2 after 2026-09-22 is recorded here with its date and reason.

*(none yet)*
