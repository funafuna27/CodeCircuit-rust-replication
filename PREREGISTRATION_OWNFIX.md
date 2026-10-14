# Pre-registration — the model's own errors and their one-token repairs (Phase 6)

Version 1, written 2026-10-12 (Japan time), before any Phase 6 repair, control
or log-probability exists. The pool counts in §1 come from labels only. The
scripts for the pairs are committed with this document after tests on Phase 4's
own mutants (§2) and an adversarial review whose confirmed findings are worked
in. The analysis script (`p6_4_analysis.py`) follows in version 2, with the size
test and the reference values of §3 and §6, before any Phase 6 graph is built.

## 1. Why this experiment

Phase 5 (`PREREGISTRATION_MATCHED.md` §9) found that, at identical length and
matched code log-probability, the graph features separate a passing one-token
mutant from a failing one (S1c 0.639, p 0.005), and that the failing member's
pruned graph is larger (G: more feature nodes in 60% of pairs). But the errors
were injected, and the graph did not beat cheaper signals: the rest of the
output distribution (0.649) and the edit's surface identity (0.676).

G also has two readings that Phase 5 cannot separate. The failing member was
both the incorrect one and the one whose edit changed what the program does.
Phase 6 uses errors Gemma made itself and pulls the two readings apart:

- **S** — a failing solution Gemma wrote.
- **R** — S with one token replaced so that it passes every assert and a set of
  calibrated extra inputs: it is correct as far as these can tell, and its edit
  changed the behaviour.
- **C** — S with one other token replaced so that it returns exactly S's value on
  every assert: it keeps Gemma's bug, and its edit changed nothing the asserts
  can see.

Both members carry one edit of the same code, so being edited cannot tell them
apart, and they are matched on the model's surprise. If the graph reads
correctness, R's graph should be the smaller (as the passing member's was in
Phase 5). If it reads whether an edit mattered, R's should be the larger.

Pool (labels only): Phase 3's samples and Phase 5's round 1 hold 1551 distinct
samples that compile and fail, not truncated, ≤ 140 tokens, over 220 problems;
1458 fail with a wrong value, 62 with a panic, 27 time out. 535 pass at least
one assert. 209 problems have such a sample among samples 0–19.

## 2. Design — fixed

**Sources S.** Every sample Gemma drew in Phases 3 and 5 and in Phase 6's
rounds that compiles, fails through a wrong value or a panic (not a timeout),
is not truncated and fits prose + ' ' + code into 140 tokens. Within a problem,
a sample whose code with all whitespace removed equals an earlier kept one
(sample order) is dropped. S's verdict must be identical in 3 judgings (the
draw's label and two more: the same pass count and FAIL/PANIC line).

**Mutants.** Phase 4's (`m1_make_mutants.py`: sites in function bodies,
families, the exact token-count filter, the judge under an 8 GiB limit),
unchanged, on every stable S.

**R, a repair** (`p6_1_repairs.py`, `p6_1b_verify.py`): a mutant that passes
every assert in 3 judgings, and passes the problem's calibrated extra inputs.
Extra inputs: each assert's arguments with one argument changed at a time,
within its type and sign (integers ±1, ±2, 0, 1; floats ±0.5, ×2, 0; strings
reversed, shortened, extended; lists reversed, sorted, shortened, with the first
element duplicated or changed; tuples element by element; booleans flipped),
at most 40; the expected value from MBPP's Python reference solution. An input is
kept only if each of the problem's passing Gemma samples (the first 10
whitespace-distinct in sample order) returns that value on it, since the
reference and Gemma's passing samples disagree on some boundary inputs. A repair
is verified if it passes all kept inputs; with fewer than 10 kept, or no passing
sample, it is unverifiable and is not used. On 8 test sources, 5 of 17 stable
repairs failed the extra inputs.

**C, a control:** a mutant that compiles, fails with S's FAIL/PANIC line, and
returns S's value on every assert. S and the mutant are compiled into one
binary, in separate modules, and each assert's call is compared by its Debug
form (`{:?}`, so NaN equals NaN and −0.0 differs from 0.0; both panicking counts
as equal); all 3 runs must agree. C's site must be live: some other replacement
at the same position compiles and does not return S's values. The comparison
runs only for sources with a stable repair, and first compares S with itself;
where that does not come out equal, the source's comparisons count as unknown
(no control, no liveness witness). Whether each control also returns S's values
on the problem's calibrated extra inputs is recorded (descriptive). At a boolean or
logical site the only replacement is the control itself, so these sites are
never live and never give a control; range edits almost never keep the token
count (1 of 608 over the pool). Strict pairs therefore come from the comparison,
arithmetic, compound and integer families; boolean and logical repairs can pair
only under the fallback, and off-by-one range errors are not covered.

**Surprise and site.** As Phase 5 (`p5_2_logprob`; Gemma-2-2b-it, bf16, eager
attention): code log-probability, edit-token log-probability, lowest token
log-probability, mean entropy; plus tokens after the edit token; from S's own
forward pass at the edited position, the entropy there, the log-probability of
S's token there, and whether S's token was the most probable one (a "belief") or
not (a "slip"); and whether a mutant's own new token is the most probable one in
its own pass (descriptive). Each member's syntactic context: walking back from
the edited token at its own brace depth, the first of `if`, `while`, `for`,
`match`, `return`, `let`, else "other".

**Pairs** (`p6_3_match.py`). Per source, every (verified R, C) with the same
number of characters, |code log-probability R − C| ≤ 1.0 nat, and a tier: 0, C at
R's site with another replacement; 1, C makes R's replacement (same from → to)
elsewhere; 2, C's edit in R's family with the same change in length, elsewhere.
Distance d = |Δ code lp| + |Δ site entropy| + |Δ original-token lp| + |Δ tokens
to the end| / 10. Candidates are ordered by tier, d, R's position and new token,
C's position and new token. Within a problem, sources are taken in the order of
their best candidate (then sample index); each takes its first candidate none of
whose keys is used yet in the problem — the pair key (R's from → to, R's edited
line with whitespace removed), R's code, C's code (whitespace removed) — the
codes of Phase 4's and Phase 5's graph members being used from the start. At
most 5 pairs per problem; more than 300 in all are cut to 300 at random
(`default_rng(20261012)`), strict pairs kept first. Pairs whose R or C equals a
sample Gemma drew are flagged ("doubly natural" R, "natural" C).

**Rounds and gate.** Round 0 uses the samples drawn so far. Fewer than 250
pairs → round r = 1, …, 8 draws samples 20r–20r+19 for the 209 problems with a
failing source among samples 0–19 that do not yet hold 5 pairs, with Phase 3's
model, prompt, decoding and cleaning and seed 20261001 + task_id + 1000 r (Phase
5 drew round 1 for 89 of them with the same seeds; those samples are reused).
Repairs are re-verified every round with the passing samples drawn so far. Stop
at 250 pairs. After round 8 the gate is 150 pairs over 50 problems. If the
strict pairs miss it, the fallback is a second pass after the strict selection,
over the sources left without a pair, sharing the used keys and the cap of 5
per problem, so that it adds pairs and never displaces one: a control need only
be live and fail with S's FAIL/PANIC line, tier 3 (any family) is allowed, and an
output-identical control comes first whatever its tier. Each pair records its
rule and whether its control is output-identical. If the gate is still missed, no
graph is built and the yield is the result. The rule reads labels and
log-probabilities only. Every matching run writes a status file (round, rule,
final, and the hashes of the pairs and rows); `run_ownfix.sh` builds graphs only
if the pairs are final, unchanged since written, and committed.

**Graphs.** Phase 5's pipeline unchanged (BATCH_SIZE 64, the 49 features,
`MAX_GRAPH_ATTEMPTS=2`), for R and C; then, as a descriptive tail, for S. Ids:
problem · 100000 + sample · 10, + 0 (R), 1 (C), 2 (S). A pair missing a graph is
dropped. Graphs are hard-linked outside the repository while ≥ 150 GB stay free.
The pairs and every judged mutant, verification and log-probability are
committed before any graph is built, and re-derived by separate code (including
that no member appears twice).

**Scripts.** `p6_0_sample.py`, `w1_label_samples.py` (unchanged),
`p6_1_repairs.py`, `p6_1b_verify.py`, `p6_2_logprob.py`, `p6_3_match.py`,
`p6_natural.py`, `run_ownfix.sh`; `p6_4_analysis.py` in version 2.

**Tests before this commit.** On 8 of Phase 4's killed pair members (failing
programs with a known repair), `p6_1` found stable repairs in all 8 (17 in all),
and the harness returned "same" for S against itself and "diff" for S against its
repair and for a mutant failing differently; 77 same-failure mutants were all
output-identical, 25 on live sites. `p6_1b` kept 11–32 calibrated inputs per
problem and failed 5 of the 17 repairs. A second review pass found and fixed: a
NaN-returning source never equal to itself under `==` (task 252), a key-parsing
error that left every `||` repair unverified, and a fallback that could displace
strict pairs. `p6_2`/`p6_3` ran end to end; with these
injected bugs the repair restores the original, much more probable code (12–30
nats above the controls), so no pair met the caliper there. The pilot measures it
for Gemma's own errors.

## 3. Analysis — fixed (implemented in `p6_4_analysis.py`, version 2)

The problem is the unit of every test and interval. Random streams:
`default_rng([20261012, s])`, s = 1 primary bootstrap, 2 A_T, 3 Ac refits, 4 Ac
interval, 5 size test (i), 6 size test (ii), 7 power, 8 the reference fit on
Phase 5.

**Primary: the direction of graph size, β_G.** Per pair, z = (log(1 + nodes_R)
− log(1 + nodes_C)) / c, nodes = feature 1 (pruned feature nodes), c = the sd of
the same quantity (survivor minus killed) over Phase 5's 259 pairs, fixed with
version 2. A conditional logit for matched pairs — P(R is the passing member) =
σ(β_G z + γ·b), one row per pair, no intercept — with baseline terms b, each R
minus C: code log-probability, its sign, Δ·|Δ| of it, edit-token
log-probability, lowest token log-probability, mean entropy, site entropy,
original-token log-probability, tokens to the end; the squares of these seven
member-level quantities (each centred and scaled over all members), as R − C
differences; and the one-hot differences of the replaced token, the new token
and the syntactic context, all levels kept. The family block is left out: it is a
sum of token columns. Columns that are zero in every pair are dropped; a
continuous column that is an exact linear combination of the others is dropped,
last in this order first; dropped columns are reported.

Fit: maximise ℓ(β) − (λ/2)Σ_{j∈P} β_j² + ½ log det(XᵀWX + λP), λ = 1, P the
0/1 diagonal of the one-hot and squared-term columns (z is not penalised),
W = diag(p(1 − p)), by the modified-score Newton iteration.

Test of β_G = 0, two-sided α = 0.05, a wild score bootstrap over problems with
leverage-adjusted scores (CR2). The restricted fit (b only) gives p̂ and
w = p̂(1 − p̂); z̃ = z − B(BᵀWB + λP)⁻¹BᵀWz; H = W^½B(BᵀWB + λP)⁻¹BᵀW^½;
ẽ_i = (1 − p̂_i)/√w_i. The score of problem g is U_g = Σ_{i∈g} z̃_i(1 − p̂_i), its
adjusted form U_gᵃ = (z̃_g ∘ √w_g)ᵀ(I − H_gg)^{−½}ẽ_g (symmetric inverse square
root by eigendecomposition, eigenvalues floored at 10⁻⁸). The direction is the
sign of S_U = Σ_g U_g (negative: R's graph is the smaller).
p = (1 + #{|Σ_g ε_g U_gᵃ| ≥ |S_U|}) / (1 + B), ε_g Rademacher per problem,
B = 9999. If the version-2 size test rejects above 0.06, (I − H_gg)⁻¹ (CR3)
replaces the inverse square root; if CR3 also does, that is reported and CR3 is
used regardless. β̂_G is reported with a CR2 sandwich interval,
β̂_G ± t_{0.975, G−1}·√V (descriptive).

**Reference β_5** (version 2): the same model on Phase 5's 259 pairs (the
survivor as the passing member; without the site entropy and original-token terms,
which Phase 5 did not record), with its interval; committed with version 2.

**Secondary A_T, gatekept** (confirmatory at 0.05 only if the primary has
p < 0.05): Phase 5's forest, fitted on all 259 Phase 5 pairs (centred features),
is frozen and its hash committed with version 2. It scores the Phase 6 pairs from
problems outside Phase 5, on features centred within each pair: per pair r = 1
if R scores higher, 0 if lower, ½ if tied; A_T = mean r. Test:
S = Σ_g Σ_{i∈g}(r_i − ½), each problem's sum multiplied by a Rademacher sign,
B = 9999, two-sided, p = (1 + #{|S*| ≥ |S|})/(1 + B). With fewer than 20
problems outside Phase 5, A_T is not tested (reported only). A_T > 0.5: Phase
5's pattern transfers to the model's own errors; A_T < 0.5: it tracks whether
an edit mattered. Control scorer, descriptive: Phase 5's frozen edit-identity
forest.

**Other secondaries, descriptive.**
- Ac, Phase 5's procedure, one-sided against K = 500 label swaps made once per
  problem, p = (1 + #{A_k ≥ A})/(1 + K); interval from a problem bootstrap.
- β_G with features 0, 3 and 4 (all active features, top logit probability,
  logit entropy) added to b (the same column rules); and graph + edit identity
  against edit identity, as `p5_6`.
- Tier: β̂_G on tier 0 alone (if ≥ 40 pairs over ≥ 20 problems) and the score
  test of z × 1[tier ≥ 1] added to the model (same bootstrap).
- β̂_G within strata: belief vs slip; S passing ≥ 1 assert vs none; syntactic
  context; doubly-natural R (the pairs file's flag); C also returning S's values
  on the calibrated extra inputs or not; problems shared with Phase 5 or not (β_G
  only; A_T on shared problems is not computed, the forest having been trained
  there); under the fallback, output-identical C or not.
- Before any graph, the balance of the pairs: S2c, S2+c (with the site terms) and
  the edit-identity forest on the Phase 6 pairs; counts by tier, family and
  context. Reported, not used to rematch.
- S's own graphs (the tail): log(1 + nodes) of S against R and against C, and the
  frozen forest's ranking of S against R.
- Natural pairs (`p6_natural.py`): counted, every whitespace variant tried; if
  there are ≥ 40 member-unique pairs over ≥ 20 problems, building their graphs is
  proposed as a §8 deviation before any is built.

## 4. Predictions

**β_G: p above 0.05,** held weakly. If it is significant, I expect R's graph to
be the larger (S_U > 0), most of all for beliefs: Gemma wrote S as if it were
right, and R contradicts it. My record: of five nulls I predicted, four failed
in the paper's favour.

No prediction for A_T, Ac or the strata.

## 5. What each outcome means

The direction is that of S_U. A_T is read only when the primary has p < 0.05;
otherwise it is descriptive. Every row refers to verified repairs.

| primary | A_T | reading |
|---|---|---|
| p < 0.05, S_U < 0 | > 0.5, p < 0.05 | the graph reads the correctness of Gemma's own one-token errors beyond a linear adjustment, in R − C differences with their squares, for surprise, the site's uncertainty, the kind of edit and its syntactic context — the strongest support for the paper's claim so far, for this population and for "passes the asserts and the calibrated inputs" |
| p < 0.05, S_U < 0 | not significant, or not tested | graph size carries the signal; Phase 5's 49-feature pattern does not transfer (or was not tested) |
| p < 0.05, S_U < 0 | < 0.5, p < 0.05 | graph size points to correctness and Phase 5's pattern to whether an edit mattered: no claim about correctness beyond graph size |
| p < 0.05, S_U > 0 | not > 0.5 with p < 0.05 | the graph reads whether an edit mattered, not correctness; Phase 5's G is reinterpreted, and the strong claim is not supported (if A_T < 0.5 with p < 0.05, Phase 5's pattern tracks it as well) |
| p < 0.05, S_U > 0 | > 0.5, p < 0.05 | the two point opposite ways: the strong claim is not supported |
| p ≥ 0.05 | descriptive | not detected; whether the CR2 interval excludes β_5 (correctness at Phase 5's strength) and −β_5 (edit mattered at that strength) is reported; opposite effects cancelling cannot be ruled out, and the belief stratum is the closest look. Ac (descriptive) is read beside this row |
| gate missed | — | no graphs; the yield is the result |

If the version-2 size test (ii) fails, β_G is reported as descriptive only and
the first row cannot be claimed. If the score test of z × 1[tier ≥ 1] has
p < 0.05, the rows are read with "depends on the site composition". If pairs come
from the fallback, S_U > 0 reads "R against another wrong program that fails the
same asserts" unless β̂_G > 0 also on the pairs with an output-identical control;
the S_U < 0 readings are unchanged. A control is neutral on the asserts, not
necessarily elsewhere (on 8 test sources, 15 of 25 controls differed from S on
the extra inputs); for any S_U > 0 reading, β̂_G on the pairs whose control also
matches S on the extra inputs is reported beside it, and the reading becomes
"whether an edit changed the tested behaviour" if it does not agree in sign.

No threshold above is adjusted after the pairs are seen. Deviations go in §8,
with the original alongside.

## 6. Power

The design agents' post-hoc estimate, on Phase 5's pairs (injected errors, so
probably optimistic), put the graph-size effect at z ≈ −2.7 at 259 pairs, which
gives two-sided power of about 0.54 at 150 pairs, 0.76 at 250 and 0.83 at 300,
and about 0.3 at half that size. Version 2 replaces this with the power implied
by β_5 for the implemented test, by simulation on the real pairs (stream 7).

Size test (version 2, before any graph; 2000 replicates, Monte Carlo se 0.005, on
the real pairs and problems): (i) which member is R is redrawn from the fitted
restricted model; (ii) it is redrawn from the restricted fit plus even terms
(squares and products of the member-level quantities) with fixed coefficients,
scaled so that their part of the linear predictor has half the sd of the linear
part — not fitted to the labels. In both, z = the even terms × N(0, 1)
coefficients + noise of equal variance + a problem effect with sd half the
signal's. Pass: ≤ 0.06 in both. If (i) fails, CR3 is used and both are rerun; if
(ii) still fails, the consequence in §5 applies.

## 7. Limitations

- The errors are Gemma's, but only those one token away from passing: in the
  comparison, arithmetic, compound and integer families (boolean and logical
  repairs only under the fallback; range errors not covered). They are reported
  against all 1551 sources, with the families of all verified repairs.
- "Correct" means passing the asserts and the calibrated extra inputs; the
  inputs on which the reference and Gemma's passing samples disagree are dropped,
  so some boundary errors stay invisible. Doubly-natural repairs are reported.
- C is "test-neutral", not proven equivalent: it returns S's values on the
  asserts only.
- In tiers 1–2, R sits at the bug's site and C elsewhere; the syntactic context
  and the site terms are in b, and the tier analysis reports what remains.
- Several pairs may share a problem; every test and interval treats the problem
  as the unit.
- The same 140-token cap and frontier problems as before; problems without a
  passing Gemma sample cannot verify a repair and drop out.

## 8. Deviations

**2026-10-13, after the gate was missed.** Version 2 was never committed. Its
analysis script (`p6_4_analysis.py`) and size test (`p6_3c_sizetest.py`) were
drafted and reviewed while the pairs were being built, but no graph was built,
so they never ran on Phase 6 data and are not part of the record. Two pieces of
it are committed now because §9 reports a number they produce: the statistics
module (`p6_stats.py`) and the Phase 5 reference (`p6_3b_reference.py`). Before
committing, two findings of the version-2 review were fixed in `p6_stats.py`:
the score and the sandwich are computed in a numerically stable form (a pair
with a fitted logit beyond about 37 had turned the p-value into its minimum and
the interval into NaN), and the CR2 sandwich no longer carries a G/(G − 1)
factor that §3 does not name.

## 9. Results

**2026-10-13. The gate was missed: no graph was built, and the yield is the
result** (§2, §5 last row).

`run_ownfix.sh PHASE=pairs` ran from 01:05 to 10:06 (`results/ownfix/pairs.log`;
the data, gzipped, in `results/ownfix/data/`). Every round was drawn:

| round | stable failing sources | stable repairs | verified | failed the extra inputs | unverifiable | strict pairs (problems) |
|---|---|---|---|---|---|---|
| 0 | 1520 | 374 | 90 | 104 | 180 | 5 (4) |
| 1 | 2178 | 446 | 97 | 106 | 243 | 5 (4) |
| 2 | 3107 | 616 | 123 | 139 | 354 | 8 (5) |
| 3 | 3943 | 781 | 161 | 167 | 453 | 10 (5) |
| 4 | 4770 | 921 | 180 | 190 | 551 | 11 (5) |
| 5 | 5513 | 1039 | 238 | 230 | 571 | 17 (7) |
| 6 | 6274 | 1124 | 256 | 238 | 630 | 18 (7) |
| 7 | 7003 | 1237 | 276 | 258 | 703 | 20 (9) |
| 8 | 7691 | 1371 | 304 | 275 | 792 | 20 (9) |

After round 8: 525 sources over 78 problems had a stable one-token repair. The
strict rule gave 20 pairs over 9 problems; the fallback added 29, for 49 pairs
over 25 problems, short of 150 over 50. Natural pairs: 28 member-unique over 18
problems (below the §3 threshold); doubly-natural repairs among the pairs: 0.

Where the pairs were lost:
- **Repairs that pass the asserts by luck.** Of the 579 stable repairs that could
  be checked, 275 (47%) failed the calibrated extra inputs: they pass MBPP's
  asserts, usually three, and return a wrong value on a nearby input.
- **Repairs that could not be checked:** 792. Of the 78 problems with a repair,
  12 have no passing Gemma sample to calibrate with, and 26 keep fewer than 10
  extra inputs.
- **The control.** Among sources with a verified repair, a control that returns
  S's values on the asserts, sits on a live site, has the same length and the
  same kind of edit, and lies within 1 nat of the repair was rarely available.
  More samples barely helped: from round 5 to round 8 the strict pairs went from
  17 to 20.

Post hoc and descriptive (`p6_yield_whatif.py`, `results/ownfix/yield_whatif.log`):
the same data under relaxed rules, without graphs.

| caliper | unverifiable repairs allowed | fallback | pairs (problems) |
|---|---|---|---|
| 1.0 | no | no | 20 (9) |
| 1.0 | yes | yes | 116 (51) |
| 2.0 | yes | yes | 145 (58) |
| 3.0 | yes | yes | 150 (60) |
| 5.0 | yes | yes | 164 (62) |

Only by giving up both of this phase's safeguards — verified repairs and
matched surprise — would the gate have been reached.

**Post hoc, on Phase 5's injected errors** (`p6_3b_reference.py`,
`results/ownfix/reference.json`): §3's primary model on Phase 5's 259 pairs,
the survivor as the passing member and the baseline terms Phase 5 recorded
(surprise, its square terms, edit identity, syntactic context), gives
β_5 = −0.461, CR2 95% [−0.897, −0.024], score-test p 0.039: the killed member's
graph is larger beyond a linear adjustment for surprise and the kind of edit.
This method was designed after Phase 5's results were known, and `p5_6`'s
forest found no gain over edit identity (+0.012 [−0.074, +0.098]); it is a lead,
not a finding.

**Phase 6 in one paragraph.** Gemma-2-2b's own failing Rust solutions are often
one token from passing MBPP's asserts (1371 repairs in 525 of 7691 sources), but
about half of those repairs are right only on the asserts, and a clean contrast
— a verified repair against an output-identical, surprise-matched control of the
same code — exists for only 20 of them. At this scale the question Phase 6
asked, whether the graph reads the correctness of the model's own errors beyond
cheaper signals, cannot be answered this way.

## Erratum (2026-10-14)

The Rust judge passed ±inf on its `isclose` branch (readme, "Erratum"). 33 of the 304 verified repairs (tasks 137 and 248) only return inf: verified repairs are 271, repairs right only on the asserts 275 of 546 checkable (50%), pairs 41 (strict 20 unchanged). Three of its 120 samples labelled passing in these tasks (all task 248) only return inf too; they served only to calibrate the extra inputs, which stay the same without them. The gate is missed either way. β_5 without Phase 5's affected pair: −0.458 [−0.895, −0.022].
