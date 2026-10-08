# Pre-registration — one-token mutants (Phase 4)

**Version 1, 2026-10-04.** Written and committed before any mutant was generated
or judged, together with every script it names. One count was made before
writing it, from labels only: 164 problems have at least one eligible passing
sample in Phase 3's draw (1,550 samples). On 2026-10-07, before the analysis
ran, §8 added the deviations and two reading rules found by an audit.

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

**2026-10-07, before the analysis ran.** The graphs were still being built and
no feature file existed when these were written and committed.

*1. A graph that cannot be built, and the run that produces the analysis.* On
10-05 at 16:07, graph 147 (id 63301, the killed member of task 633) raised
`CUDA error: unknown error`. While `run_chunked.sh` was retrying it, the PC
stopped (Windows logged an unexpected shutdown at 16:47) and stayed down until
10-07 06:34. On the resume at 07:05, the same graph failed again at once.
Because the previous retry had come just before the crash, the run was resumed
again at 07:08 with `MAX_GRAPH_ATTEMPTS=2` (default 3), so the graph is given up
instead of being tried a third time. Pair 633 therefore has only one graph, and
§3 drops it: the analysis has 98 pairs. The 147 graphs built before the crash
were kept, and all of them load.

The committed `run_mutant.sh` stopped before the analysis once any graph was
given up, because `run_chunked.sh` then exits 4; that contradicts §3. It is
fixed now. The driver actually running is committed as `run_mutant_resume.sh`.
Every graph is also hard-linked outside the repository (`~/mutant_graph_keep/`),
because the pipeline deletes all but the first 50 after featurising them.

*2. How S1 and S2 can be read.* An audit run before the analysis found two
properties of the secondary analyses. Five independent reviewers read the code
and ran synthetic data only. Neither property is a coding error:
`m3_mutant_analysis.py` implements §3 as written. §3 stays as written, and S1
and S2 are computed and reported unchanged. What is fixed here, before any
result exists, is what each of them can show.

- **S1 absorbs signal whenever surprise differs systematically between the
  members.** The within-pair fit has no intercept and uses no label; that is
  what keeps the label-swap null valid. Let m and s be the mean and sd of the
  survivor − killed code log-probability. When m is not zero, the fitted slope
  picks up the product of the two means and removes about f = m² / (m² + s²) of
  every feature's survivor − killed difference. That includes a difference that
  has nothing to do with surprise.

  Synthetic check: a feature unrelated to surprise ranks the survivor higher in
  86% of pairs. After the correction it still does so in 86% of pairs when the
  killed member is the less probable one in 62% of pairs. It drops to 59% when
  the killed member is less probable in 89% of pairs (f ≈ 0.5), and to 52% at
  99%. §4 predicts such a gap.

  Rule: f is reported with S1 (by `m4_mutant_centred.py`). If f < 0.25, §5 is
  read as written. If f ≥ 0.25, S1 is reported but does not choose between §5's
  first two rows. A significant A is then read as "a signal at identical length;
  S1 cannot tell whether it goes beyond the model's surprise", and an S1 below
  0.5 is an artefact of the correction, not evidence.
- **S2 has little power for what §1 asks of it.** Its forest sees
  log-probability levels. These vary between problems far more than within a
  pair: the sd between pairs is about 15 nats, against a median within-pair
  difference of about 5. (These figures come from the audit and use no labels.)
  So the forest can hardly use the within-pair difference. On synthetic pairs
  whose killed member is the less probable one in 93% of pairs, it scored 0.52
  on levels and 0.91 on within-pair-centred values.

  Rule: whether the model's surprise separates the members is read from §3's
  descriptive measure, the share of pairs whose killed member has the lower code
  log-probability, with its Wilcoxon test. §4's S2 prediction is judged on its
  two parts separately.

*3. Exploratory, not pre-registered, committed now: `m4_mutant_centred.py`.* A
has the same weakness. The graph features vary between problems far more than a
one-token edit can move them. m4 repeats A and S2 on values centred within each
pair (Ac and S2c). Centring uses no label, so the same null holds. m4 also
prints f and the descriptive measures that m3 omits.

On synthetic data, the centred forest shows no bias when there is no signal: its
mean is 0.51 over 12 datasets, with an sd of 0.07–0.08. It recovers a planted
single-feature signal better than the forest on levels (0.91 against 0.79). Its
reading rule is in its docstring. It runs after m3.

*4. Smaller items.*
- m2 computed log-probabilities with the default `sdpa` attention of
  transformers 4.57.3, which omits Gemma-2's attention-logit soft-capping; the
  graphs do apply it. The audit estimates a per-member error of about 0.06 nats,
  against a median within-pair difference of about 4.8. About half a pair is
  expected to change order. Not recomputed.
- §3's share of same-family pairs, and the share of pairs whose two edits sit on
  the same token, are not printed by m3. m4 prints both over the analysed pairs.
  Of the 99 killed members, 7 fail through a panic or a timeout rather than a
  wrong value; the judge counts both as failing an assert. m4 prints the
  breakdown.
- The pooled within-pair R² that m3 prints ("surprise and chars explain …") is
  dominated by one raw-scale feature. m4 prints the per-feature median beside
  it. The same holds for the 31% quoted in §1 from Phase 3: 98.9% of that pooled
  sum of squares is feature 40. Feature by feature, length explains a median of
  10.5% of the within-pair variance (IQR 5.4–16.0%). This is corrected in
  `PREREGISTRATION_WITHIN.md` §9b.
- The resolvable effect m3 prints uses √(0.25/n). The label-swap null's own sd
  (0.06–0.07 in Phase 3) implies about 0.65; both are reported. With K = 100, p
  moves in steps of 1/101, with a Monte Carlo SE of about 0.02 near 0.05.
- m3's token-count check compares counts that m1 copied from the source.
  Equality itself was confirmed by recounting all 198 texts (0 mismatches, in
  the audit).
- m1's lexer takes the leading 0 of a hex literal as an integer site. Such
  mutants never compile, so no pair is affected; the audit re-derived all 99
  pairs.

## 9. Results

Times in this section are Japan time. The PC's clock came up 9 h slow at the
boot after the crash (real 10-07 15:34, logged 06:34) and stayed slow until
Windows' time service stepped it forward by 32,401 s at 01:09 on 10-08 (System
log: Time-Service event 52). Before the crash and after 01:09 it was right. So
§8's times for 10-05 (16:07, 16:47) are real, while its times for 10-07 (06:34,
07:05, 07:08), the "since 2026-10-07 07:08" in `run_mutant_resume.sh`, the
graph log of that period and the date of commit e630b37 (10-07 15:11) are 9 h
slow: the PC came back at 15:34 on 10-07, the run resumed at 16:08, and §8 was
committed at 00:11 on 10-08, before any feature existed. (The same 9 h error
followed a boot on 2026-09-10 and was corrected that day.)

### 9a. Pre-registered (`results/mutant/analysis.*`)

197 of the 198 graphs were built; the one given up (§8) drops pair 633, so 98
pairs were analysed. The last graph was featurised at 00:44 on 10-08; m3 ran
from 00:44 to 01:12, and m4 from 01:16 to 01:35. Every point estimate below
was recomputed afterwards by separate code that imports neither m3 nor w3; all
agree to four decimals.
Token counts are identical within every pair; chars differ by a mean of +0.02
(at most 2). The killed member has the lower code log-probability in 71.4% of
pairs: survivor − killed mean +3.03 nats, median +2.78, Wilcoxon p < 0.0001.

| statistic | value | 95% CI | label-swap refits (K = 100) |
|---|---|---|---|
| **A**, graph features | **0.4694** | [0.3673, 0.5714] | mean 0.5080, sd 0.0598; p = 0.7228 |
| S1, within-pair surprise removed | 0.4898 | [0.3878, 0.5918] | mean 0.5119, sd 0.0649; p = 0.6832 |
| S2, surprise alone | 0.5510 | [0.4541, 0.6480] | mean 0.5010, sd 0.0624; p = 0.2376 |
| per-feature Wilcoxon, BH q < 0.1 | 21 of 49 | | |

**Predictions.** A: p above 0.05 — **held** (0.7228), the first predicted null
of this study to hold. S2, judged in its two parts (§8): the killed member is
the less probable one in more than half the pairs — **held** (71.4%,
p < 0.0001); S2's A above 0.5 — 0.5510 is above 0.5 as a point estimate, but
its interval covers 0.5 (p 0.2376), so S2 itself shows no separation, as §8
expected of a forest fed levels.

**Reading (§5):** A not significant → *not detectable at this n.* 98 pairs
resolve A ≈ 0.63 by the normal approximation, ≈ 0.65 by the null's own sd
(0.0598). §8's rule for S1: f = 0.168, below 0.25, so S1 would be read as
written; with A not significant it has nothing to choose between.

The descriptive tests do not agree with A. Of the 49 features, 21 differ
between survivor and killed at q < 0.1: 12 are larger for the killed member
(for example feature 1, median survivor − killed −9; feature 2, −4.5; feature
40, −12,560), 4 are smaller, and 5 have a median difference of zero. The
forest, fed levels, did not use these differences, which is the weakness §8
described.

### 9b. Exploratory, committed before m3 first ran (`m4_mutant_centred.py`, e630b37; `results/mutant/centred.*`)

| | value | 95% CI | p |
|---|---|---|---|
| Ac, graph features centred within each pair | 0.6735 | [0.5816, 0.7653] | 0.0099 |
| S2c, [code log-probability, chars] centred within each pair | 0.6939 | [0.6020, 0.7857] | 0.0099 |

0.0099 is the floor at K = 100: none of the 100 refits reached either value
(null means 0.500 and 0.499, sds 0.068 and 0.063). Also from m4, over the 98
pairs: f = 0.168; same family 89.8%, same site 57.1%; the killed members fail
by a wrong value (91), a panic in a call (5) or a timeout (2); feature by
feature, [log-probability, chars] explain a median 4.0% (IQR 2.0–10.4%) of the
within-pair variance, while m3's pooled 0.168 is feature 40 alone.

**Reading, by the rule in m4's docstring:**
- Ac significant and A not: the graph carries a within-pair signal at identical
  length that the pre-registered forest, fed levels, could not use.
- S2c above 0.5: the model's surprise separates the members.
- Ac against S2c, descriptive and not tested: surprise separates them a little
  better, 0.694 against 0.674.

**What this does not settle** is whether the graph's within-pair signal is more
than the model's surprise at the edit. Ac and S2c were not tested against each
other, and nothing here removes surprise from the centred features. That needs
a new analysis, committed before it runs.

Put together: at identical length, the graph features of a copy that passes
differ from those of a copy that fails, but the pre-registered analysis could
not detect it, and the model's own surprise at the edit separates the pairs at
least as well.

### 9c. Post hoc, decided after 9a and 9b were seen (`m5_mutant_posthoc.py`, 998fa94; `results/mutant/posthoc.*`)

The per-feature differences track the model's surprise. Within pairs, the
differences in feature nodes, error nodes, total error influence, edges, summed
edge weight and maximum activation each correlate with the log-probability
difference (Spearman −0.33 to −0.49): the less probable the killed member, the
larger its graph. So Ac may be surprise read back through the graph. m5 asks
whether it is more: A on the centred features after S1's within-pair
fixed-effects removal of [code log-probability, chars]. It was written after
9a and 9b were seen, and committed with its reading rule in the docstring
before it first ran (01:37–01:46).

| statistic | value | 95% CI | label-swap refits (K = 100) |
|---|---|---|---|
| S1c, centred features, within-pair surprise removed | 0.5918 | [0.4898, 0.6837] | mean 0.5072, sd 0.0687; p = 0.1386 |

**Reading, by m5's rule:** p ≥ 0.05 → *the centred graph signal cannot be
separated from the model's surprise at this n.* This is not evidence that the
graph reads only surprise: the removal is linear, it also strips about f = 0.17
of any surprise-independent difference (§8), and the drop from Ac, 0.082, is
about 1.2 null sds. Part of the signal may survive the correction, as in Phase
3, but 98 pairs cannot show it.

Descriptive, uncorrected, and looked at only after the q-list was seen: in the
28 pairs whose killed member is not the less probable one, the killed member
has more feature nodes in 54% (74% over all pairs); a regression of the
feature-node difference on the log-probability difference, with an intercept,
leaves −5.0 nodes at no difference in surprise (t −3.0). Leads, not findings.

**Phase 4 in one paragraph.** At identical length, the pre-registered test
detects nothing: A = 0.469, and its interval, which ends at 0.571, excludes an
effect as large as Phase 3's 0.62. Compared within each pair, the graph
features of the passing and the failing copy do differ, but the model's own
surprise at the edit separates the pair at least as well, and with that
surprise removed the graph's separation is no longer significant. Whether the
graph carries anything beyond the surprise is open; a fresh draw,
pre-registered with the centred, surprise-removed A as its primary statistic,
could settle it.

### 9d. Post hoc, found while preparing Phase 5: the edits' surface identity

The two members of a pair carry different edits, and the kind of edit may tell
a surviving one from a killing one without any model: a boundary change such as
`>` → `>=` is often harmless. `p5_5_edit_baseline.py` (Phase 5, committed in
42aeb24) fits Phase 3's forest on the centred edit identity — one-hots of the
replaced token, the new token and the family, and the edit's position as a share
of the code. On Phase 4's 99 pairs, with K = 200: **E = 0.6667** [0.5758,
0.7576], label-swap refits mean 0.5035, sd 0.0719, p = 0.0050 (the floor). That
is about as high as the centred graph features (Ac 0.674) and the centred
surprise (S2c 0.694).

`p5_6_beyond_edit.py` adds the graph features to the edit identity, on the 98
analysed pairs: graph features alone 0.6735, edit identity 0.6429, both 0.6786;
both minus edit identity = +0.036 [−0.097, +0.168], from a bootstrap over
problems. So Phase 4's centred graph signal is not shown to exceed the edits'
surface identity, as it is not shown to exceed the model's surprise (§9c).
`results/mutant/edit_baseline.*`, `results/mutant/beyond_edit_posthoc.*`.
