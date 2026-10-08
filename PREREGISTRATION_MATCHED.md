# Pre-registration — surprise-matched one-token mutants (Phase 5)

Written 2026-10-08 (Japan time; the PC clock is synced), before any Phase 5
sample, mutant or log-probability exists. The planning figures in §1 and §6
come from Phase 3's labels and from Phase 4's own sources, which Phase 5
excludes; no graph was involved. The scripts named below are committed with
this document, after tests on Phase 4's sources and on synthetic data (§2).

## 1. Why this experiment

Phase 4 (`PREREGISTRATION_MUTANT.md` §9) paired two one-token mutants of one
passing solution, identical in token count, one passing every assert and one
failing. The pre-registered test found nothing (A = 0.469, p 0.72). Compared
within each pair, the graph features did separate the members (Ac = 0.674,
exploratory), but so did the model's own surprise at the edit (S2c = 0.694),
and when that surprise was removed by a linear within-pair fit, the graph's
separation was no longer significant (S1c = 0.592, p 0.14, post hoc).

That fit has two weaknesses. It also removes about f = m² / (m² + s²) of any
difference unrelated to surprise, and it removes only the linear part: a forest
can still use whatever nonlinear function of surprise the fit leaves in the
features. Phase 5 removes surprise by design instead, as Phase 4 removed length:
the two members of every pair have nearly the same code log-probability. Then a
graph that still ranks the passing member higher carries information about
correctness that the code's log-probability does not.

Planning pilot, on Phase 4's 99 usable sources only (log-probabilities with
eager attention, no graph). Pairing a random survivor with a random killed
mutant, as Phase 4 did, leaves a median gap of 5.23 nats. Taking the closest
pair within each source leaves a median of 0.33 (IQR 0.10–1.00). 76% of
sources have a pair within 1.0 nat and 58% within 0.5. In the closest pairs, the
killed member is the less probable one in 48%.

## 2. Design — fixed

**Sources.** Every eligible passing sample — passes, not truncated, prose +
' ' + code ≤ 140 Gemma tokens, as in Phase 4 — of Phase 3's samples 0–19
(round 0), except Phase 4's 99 pair sources. Within a problem, a sample whose
code with all whitespace removed equals that of a Phase 4 pair source or of an
earlier kept sample (in sample order) is a duplicate and is dropped. Planning
count: 624 sources over 149 problems; 154 of them were tried in Phase 4 and had
no survivor or no killed mutant (they are judged again, not skipped).

**More sources, in rounds.** If round 0 gives fewer than 250 pairs, round r = 1,
2, 3, 4 draws samples 20r to 20r + 19 for each problem with at least one
eligible passing sample among samples 0–19 (164 problems). The model, prompt,
extraction, decoding (temperature 0.8, top_p 0.95) and cleaning (tidy) are
Phase 3's, the seed is `torch.manual_seed(20261001 + task_id + 1000 r)`, and
Phase 3's judge labels the samples. Round 1's seed is the one Phase 3
pre-registered for samples 20–39, which were never drawn. The same eligibility
and deduplication apply, against every source kept so far.

**Mutants and judge.** Phase 4's, imported unchanged: one token inside a
function body replaced by another member of its family, kept only if the Gemma
token count is unchanged; judged under an 8 GiB address-space limit. Survives:
compiles and passes every assert; killed: compiles and fails one. Every
source's mutants are judged. A source is usable with at least one survivor and
one killed mutant.

**Surprise.** Phase 4's code log-probability — Gemma-2-2b-it in bf16,
teacher-forced on prose + ' ' + code, summed over the tokens that start at or
after the space before the code — with one change: eager attention, so the
attention-logit soft-capping that the graphs apply also applies here (Phase 4's
sdpa omitted it, §8 there). Computed for every survivor and killed mutant of a
usable source. Also recorded per mutant, for S2+c: the log-probability of the
token holding the edit, the lowest token log-probability in the code, and the
mean entropy of the next-token distribution over the code positions.

**Matching.** For each usable source, the candidates are all (survivor, killed)
pairs whose code log-probabilities differ by at most 1.0 nat. If any candidate
has both edits in the same family, only those are kept. Candidates are ordered
by the size of the difference, then the survivor's position, the killed
mutant's position and the replacement tokens. A source gives at most one pair.

**Per problem and overall.** Within a problem, sources are taken in order of
their best candidate (then sample index). Each takes its first candidate whose
two codes are not already used in that problem, by an earlier pair or by a
Phase 4 pair member. A problem gives at most 5 pairs. If more than 300 pairs
result, 300 are drawn at random (`default_rng(20261008)`).

**When to stop.** After round 0 and after each round: 250 pairs or more →
stop. Fewer, and r < 4 → draw the next round. After round 4: 150 or more →
proceed with what there is; fewer → the gate is not met, no graph is built, and
the result is reported as such. The rule reads only labels and
log-probabilities.

**Graphs.** Phase 4's pipeline unchanged (`run_chunked.sh`; BATCH_SIZE 64,
MAX_PROMPT_CHARS 99999, CHUNK_SIZE 200, `MAX_GRAPH_ATTEMPTS=2`) and the same 49
features. Member ids: problem · 10000 + sample · 10, + 0 for the survivor and
+ 1 for the killed mutant. A pair missing either graph is dropped, and the count
is reported. Every graph is hard-linked outside the repository while at least
150 GB stay free. The pairs, rows, judged mutants and log-probabilities are
committed before any graph is built.

**Scripts.** `p5_0_sample.py` (a round of samples), `w1_label_samples.py`
(Phase 3's labels, unchanged), `p5_1_mutants.py`, `p5_2_logprob.py`,
`p5_3_match.py`, `p5_4_analysis.py`, `run_matched.sh` (`PHASE=pairs` stops
once the pairs are written; `PHASE=graphs` runs the rest).

**Tests before this commit.** On 8 of Phase 4's own sources, `p5_1` re-judged
145 mutants with verdicts identical to Phase 4's in all 145, ran incrementally
without re-judging, and `p5_2`/`p5_3` produced pairs and the expected exit
codes. Without the exclusion of Phase 4's pair members, 6 of the 8 sources gave
a pair; with it, 1 — their nearest survivor was the one Phase 4 had used. The
analysis ran end to end on synthetic matched pairs (70 problems with 1–5 pairs
each, about 210 pairs, one graph missing, small K). With no signal: Ac 0.502
(p 0.54), G p 0.34, no feature at q < 0.1. With the killed member's feature 1
shifted so that it alone ranks 80% of pairs: Ac 0.782 (p at its floor), G
p < 10⁻⁴, S2c 0.46 and S2+c 0.52, and the script printed the first reading of
§5.

## 3. Analysis — fixed (`p5_4_analysis.py`)

Several pairs may share a problem. The folds keep a problem's pairs together,
as in Phases 3 and 4, and the interval resamples whole problems.

- **Primary Ac:** the 49 graph features centred within each pair; Phase 3's
  forest, folds and out-of-fold scores unchanged (`w3_within_analysis.oof`:
  `RandomForestClassifier(n_estimators=100, random_state=42)`, 10 repeats ×
  5-fold `GroupKFold` over problems); A = the share of pairs whose survivor
  scores higher, ties ½. One-sided test against K = 500 refits with each pair's
  labels swapped with probability ½, p = (1 + #{A_k ≥ A}) / (1 + K). 95%
  interval: 5000 bootstrap draws of problems.
- **Matching check S2c:** the same on centred [code log-probability, chars],
  K = 200.
- **Secondary S2+c:** the same on centred [code log-probability, edit-token
  log-probability, lowest token log-probability, mean entropy, chars], K = 200 —
  what the model's output distribution still says about the members once their
  summed log-probabilities match.
- **Secondary S1c:** Phase 4's m5 statistic — centred features minus a
  within-pair fixed-effects fit on [code log-probability, chars], with no
  intercept and no label — K = 200, with f.
- **Secondary G (graph size):** feature 1, the number of pruned feature nodes:
  paired Wilcoxon signed-rank test, one-sided, alternative "the killed member
  has more". Phase 4 suggested this post hoc (§9c there); here it is tested on
  new pairs. Reported with the share of pairs in which the killed member has
  more nodes and the median difference.
- Descriptive: the matched gap (median |difference|, share with the killed
  member less probable); per-feature paired Wilcoxon with Benjamini–Hochberg;
  same-family and same-site shares; how the killed members fail; pairs per
  problem.
- Ac is the only primary. The secondaries are reported unadjusted and read as
  secondary. Random streams: `default_rng([0, s])` with s = 10, 20, 30, 40 for
  the interval and s + 1 for the refits.

## 4. Predictions

**Ac: its p is above 0.05.** In Phase 4, a linear removal of surprise already
brought the centred signal down to 0.592, and matching should remove more. My
record argues for caution: of four nulls I predicted, three failed in the
paper's favour, so I hold this weakly.

**S2c is near 0.5, and its interval covers 0.5.** The matching leaves the code
log-probability nothing to rank by.

**G: its p is above 0.05,** held weakly. In Phase 4's 28 pairs whose killed
member was not the less probable one, that member had more feature nodes in only
54%.

No prediction for S2+c.

## 5. What each outcome means

| S2c | Ac, p < 0.05 | reading |
|---|---|---|
| p ≥ 0.05 | yes | the graph tells a passing one-token mutant from a failing one at identical length and matched code log-probability: it carries information about correctness that the code's log-probability does not |
| p ≥ 0.05 | no | not detectable at this n; report the resolvable effect, and do not call it absence. With Phase 4, the graph's within-pair signal is not shown to go beyond the model's surprise |
| p < 0.05 | — | the matching did not neutralise the code log-probability, so Ac is not read as going beyond it; S1c is read in its place, with Phase 4's f rule (`PREREGISTRATION_MUTANT.md` §8) |

S2+c is read beside the table. If Ac is significant and S2+c is too, the graph
is shown to go beyond the summed log-probability but not beyond the output
distribution as measured here; if S2+c is not, beyond these measures as well.
G is read beside it too: a difference in graph size at matched surprise,
whatever the forest makes of it. A positive Ac is evidence about injected
errors; it does not show that the graph detects the model's own mistakes.

No threshold above is adjusted after the pairs are seen. A deviation, if one
proves necessary, is recorded in §8 with the original alongside.

## 6. Power

Phase 4's centred null had an sd of 0.0675 at 98 pairs. Scaled by √(98 / n),
one-sided α = 0.05 with 80% power resolves A ≈ 0.605 at 250 pairs, 0.62 at 200
and 0.64 at 150. Phase 4's post-hoc S1c, 0.59, would be detected with
probability ≈ 0.68 at 250 pairs, 0.60 at 200 and 0.50 at 150. These figures
ignore that pairs share problems, which lowers them. For G: if the killed
member has more nodes in 58% of pairs, a one-sided sign test detects it with
probability ≈ 0.8 at 250 pairs; at 55%, ≈ 0.5.

Yield, projected from the planning counts: round 0, about 70 pairs (470 untried
sources, of which Phase 4 found about 20% usable, and 76% of those match within
1 nat); each later round, roughly 35–60 more, fewer each time as new samples
repeat old ones.
The target of 250 may not be reached by round 4; the gate is 150.

## 7. Limitations

- The mutants are not the model's own errors, and they are minimal ones.
- "Survives" means passes MBPP's asserts, not "is correct"
  (`PREREGISTRATION_MUTANT.md` §7).
- Matching selects killed mutants that the model finds about as probable as a
  survivor — plausible-looking bugs — so the population differs from Phase 4's
  randomly drawn killed mutants. That is the point of the design, but a result
  here describes these pairs.
- Matching fixes the summed code log-probability only; S2+c measures some of
  what else the output distribution carries.
- The problems are Phase 4's problems, with new solutions. Phase 4's pair
  sources are excluded, but the hypothesis came from pairs on the same problems.
- Several pairs share a problem; the analysis groups by problem.
- The same 140-token cap and frontier problems as Phases 3 and 4.

## 8. Deviations

None yet.

## 9. Results

Not yet run.
