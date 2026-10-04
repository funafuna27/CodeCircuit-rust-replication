# Pre-registration — one-token mutants (Phase 4)

**Version 1, 2026-10-04.** Written and committed before any mutant was generated
or judged, together with every script it names. One count was made before
writing it, from labels only: 164 problems have at least one eligible passing
sample in Phase 3's draw (1,550 samples).

---

## 1. Why this experiment

Phase 3 (`PREREGISTRATION_WITHIN.md` §9) paired a passing and a failing solution
to the same problem, matched on length. The graph ranked the passing one higher
in 62% of pairs (A = 0.6236, label-swap p 0.0198), but with the features
residualised on length the signal was gone (0.5337, p 0.2673). Within a pair,
length explained 31% of the feature differences although the members differed
by a median of two tokens, and an exploratory split could not say whether the
members' own length difference or the pairs' placement among problems carried
the signal.

Phase 4 removes the members' length difference by construction. Each pair is
one passing solution in two copies, each with one token replaced, and the token
count of every copy is exactly the source's. One copy still passes every assert
(the mutant **survives**); the other fails at least one (it is **killed**).

Comparing two mutants, rather than the original against a mutant, matters: an
edit the model did not write may itself leave a trace in the graph, and both
members carry exactly one. What remains is how surprising each edit is to the
model, which may differ between a harmless and a harmful edit. That is measured
(the code's log-probability under the model) and removed in S1, and its own
power to separate the pairs is reported as S2.

The question:

> **Given two copies of a passing solution, each with one token changed and the
> same token count, does the attribution graph tell the copy that still passes
> from the copy that fails?**

This is narrower than Phase 3's. The errors are injected into code the model
wrote, not made by the model, and a one-token error is the smallest difference
two solutions can have.

## 2. Design — fixed

| | |
|---|---|
| source pool | Phase 3's draw (`within_samples.jsonl`, `within_labels.jsonl`): per problem, the eligible passing samples in index order — passes, not truncated, prose + `' '` + code ≤ 140 Gemma tokens |
| site | a token inside a function body (brace depth ≥ 1, so never a signature), not inside a string, char or comment, not a generic angle bracket, not a tuple index (`t.0`) |
| families | comparison `< <= > >= == !=` · arithmetic `+ - * / %` · compound `+= -= *= /= %=` · logical `&& \|\|` · boolean `true false` · integer literal `0 1 2` · range `.. ..=` |
| mutant | the site's token replaced by another member of its family; kept only if prose + `' '` + mutant has exactly the source's Gemma token count |
| judge | `1_rust_correct.py`'s `judge`, unchanged: compiles and every assert holds → **survives**; compiles and an assert fails → **killed**; does not compile, or the judge raises → dropped. Each worker runs under an 8 GiB address-space limit, inherited by rustc and the binary |
| source | per problem, the first sample in the pool with at least one survivor and one killed mutant |
| pair | a survivor drawn uniformly; then a killed mutant drawn uniformly from the survivor's family if it has one there, else from all the source's killed mutants. `default_rng(20261005 + task_id)` |
| cap | more than 150 pairs: 150 drawn uniformly, `default_rng(20261005)` |
| gate | at least **70 pairs**; fewer: stop and report. No relaxation of the length rule |
| graphs | one per member, on prose + `' '` + mutated code — the same text Phase 3 graphed — with `BATCH_SIZE = 64`, the Phase 2 pipeline and the same 49 features; id `task_id × 100` for the survivor, `+ 1` for the killed mutant |
| surprise | the summed log-probability Gemma-2-2b-it assigns to the code tokens, teacher-forced on the graphed text (`m2_logprob.py`), computed before the graphs |

Every judged mutant is written out (`mutants_judged.jsonl`), so the pairs can be
re-derived from the rules above.

Scripts: `m1_make_mutants.py`, `m2_logprob.py`, `m3_mutant_analysis.py`,
`run_mutant.sh` (`PHASE=pairs` stops at the gate so the pairs can be inspected
before the GPU is committed; `PHASE=graphs` runs the rest).

## 3. Analysis — fixed (`m3_mutant_analysis.py`)

- **Primary statistic A:** the fraction of pairs whose survivor scores higher
  than its killed mutant, ties ½, from Phase 3's procedure unchanged
  (`w3_within_analysis.test`): `RandomForestClassifier(n_estimators=100,
  random_state=42)` on the 49 features, 10 repeats × 5-fold `GroupKFold` over
  problems, out-of-fold P(pass) averaged over repeats; one-sided test against
  K = 100 refits with each pair's labels swapped with probability ½,
  p = (1 + #{A_k ≥ A}) / (1 + K); a 5000-draw bootstrap over pairs for the
  interval.
- **Secondary S1:** the same on features with the within-pair differences in
  surprise removed: each feature's fixed-effects slope on [code
  log-probability, chars], both centred within each pair, is subtracted. Phase
  3's S1 residualised on a pooled fit, which also strips the placement of whole
  pairs among problems (`PREREGISTRATION_WITHIN.md` §9b); a within-pair fit
  removes only what differs inside a pair. Token count does not differ inside a
  pair, so it is not a covariate.
- **Secondary S2:** the model's surprise alone, [code log-probability, chars,
  tokens], with the same test.
- Descriptive only: the log-probability gap, killed minus survivor; per-feature
  paired Wilcoxon with Benjamini–Hochberg; the share of same-family pairs.
- The script refuses a pair whose members differ in token count. A pair missing
  either graph is dropped, and the count is reported.

## 4. Predictions

**A: its p is above 0.05.** A one-token edit is the smallest difference two
solutions can have, the 49 features summarise whole graphs, and the forest
dilutes a signal carried by one feature (Phase 3's synthetic check). I predicted
a null three times before, for Phase 2's C1 and C2 and for Phase 3's A, and was
wrong all three times in the paper's favour, so I hold this weakly.

**S2: the killed mutant has the lower code log-probability in more than half
the pairs, and S2's A is above 0.5.** A harmful edit tends to break what the
model expected the code to do.

## 5. What each outcome means

| A, p < 0.05 | S1, p < 0.05 | reading |
|---|---|---|
| yes | yes | the graph tells a passing solution from a broken copy of it, at identical length and beyond the model's surprise at the edit |
| yes | no | a signal at identical length that cannot be separated from the model's surprise at the edit |
| no | — | not detectable at this n; report the resolvable effect, and do not call it absence |

S2 is read beside this table, not inside it. If S2 separates the pairs and A
does not, the model's output distribution carries the information and these 49
graph statistics do not. A positive A is evidence about injected errors; it
does not by itself show that the graph detects the model's own mistakes.

No threshold above is adjusted after the pairs are seen. A deviation, if one
proves necessary, is recorded in §8 with the original alongside.

## 6. Power

SE(A) ≈ √(0.25 / N): 0.041 at N = 150. One-sided α = 0.05 with 80% power
resolves A ≈ 0.60 at N = 150, 0.62 at 100 and 0.65 at 70 (normal approximation).
For a signal carried by one feature it is worse: on 40 synthetic pairs, one of
49 features shifted by 1.5 SD — alone it would rank about 86% of pairs
correctly — gave A = 0.625 through the forest (`PREREGISTRATION_WITHIN.md` §6).

## 7. Limitations

- The mutants are not the model's own errors, and they are minimal ones.
- "Survives" means passes MBPP's asserts, the label of Phases 2 and 3. Some
  survivors compute the same function as the source; others differ in ways the
  asserts do not test.
- The two edits sit at different places and may come from different families;
  the same-family share is reported.
- Solutions without an operator or small literal in the body cannot contribute
  (task 478's samples, for instance, only call methods).
- The source pool is Phase 3's sampled solutions under the same 140-token cap,
  so the population is again the model's frontier and short functions. Phase 3's
  graphs and features are not reused.

## 8. Deviations

None yet.
