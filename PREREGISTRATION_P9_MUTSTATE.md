# Pre-registration — does Gemma track that a binding is immutable? (Phase 9, part ①)

Written 2026-10-22 (Japan time) and committed before any contrast of the primary items was computed:
`p9_b_forward.py` and `p9_b_probe.py` refuse the primary role until this file is committed and unchanged.
What existed before it is listed in §7. Approval at gate 2 was delegated by the author to the assistant,
who decided from predicted outcomes; the changes this made to the gate-2 draft are listed in §7.

Parts ② (MBPP+ labels) and ③ (the graph against the output distribution and a probe, between
problems) get their own addenda, each committed before the data it reads. So do the graphs of part ①
(about 150, gate 3: does the attribution to q land on rustc's secondary span, against one-backward
GxA), committed before any graph is computed.

## 1. Why this experiment

Phases 1–8 asked whether the attribution graph knows more about correctness than cheap signals; it never
did (readme, "Conclusion"). Their labels were MBPP's three asserts, which miss a third of wrong programs
(Phase 7: 33% of verifiable survivors were wrong beyond them). This phase asks a narrower question with a
label that has no noise: rustc's borrow checker.

> When Gemma reads Rust in which a binding x is declared without `mut` and is later mutated, does the
> model, at the mutation, behave as if it knows x is immutable?

Rust's mutability is non-local state: whether `sum += 1` is legal depends on a declaration tens of tokens
earlier. If Gemma tracks it, a mutation of an immutable binding should be less probable than the same
mutation of a mutable one, beyond what any edit of that size does to the text.

## 2. Design — fixed

### Items (`p9_b_items.py`; `results/p9_state/b_items.jsonl.gz`)

**Sources.** Every Rust sample Gemma drew in Phases 3, 5 and 6 (41,560) that compiles in the judge's
harness and on its own (`rustc --crate-type lib`), first of its whitespace-free duplicates within a
problem: 13,018 sources over 318 problems (A2).

**Relocation.** In a source S, x is a `let mut` binding whose `mut` is necessary (removing it gives only
mutability errors naming x) and y is an immutable `let` binding. V moves the exact text `mut ` from x to
y: same characters, same Gemma token count and token multiset. rustc labels every V: only E0384, E0596
or E0594 errors naming x (A2).

**Primary items.** Relocations where both edits lie before q's span (below), minus two exclusions fixed at
gate 1: y named like x (38) and x's name bound again anywhere in the code (101). **3,457 items, 1,800
sources, 167 problems.** y binds in a function other than x's in 132 items (a covariate).

**q's span.** The first mutation site of x in V, from rustc's spans: q_place, the first token of the
statement's identifier, through q_op, the token that makes it a mutation (an assignment operator, or the
method name after `.`). Median 2 tokens. rustc puts the primary span on the declaration when x is borrowed
mutably more than once, so the site is the first span that does not cover x's declaration.

**Texts.** Each item gives five texts, scored as prose + ' ' + code (what every phase graphed):

| text | what it is | rustc |
|---|---|---|
| S | the source | compiles |
| V | x's `mut ` moved to y | mutability errors on x |
| A | S with y's `mut ` added, x's kept | compiles (unused_mut), all 3,457 |
| R | S with x's `mut ` removed | mutability errors on x |
| Pz | S with another necessary `let mut z`'s `mut ` moved to y, where z's own first violation lies after q's span | legal up to q; 1,012 items, 86 problems |

After both edits, every character sits at the same offset in V and Pz as in S, at +L in A and −L in R
(L = the moved text's length). Every target token must have the same id in all texts compared, or the
item is dropped and counted.

### The forward pass (`p9_b_forward.py`)

Gemma-2-2b-it in bf16, eager attention, one sequence per forward pass, log-softmax in float32 (as Phases
5–6). Per token: log p of the token given its prefix, and the entropy of that distribution.

### The primary statistic (`p9_b_analysis.py`)

For a token t after both edits, the contrast is c_t = log p_V(t) − log p_A(t). V and A carry the same
edit at y; they differ only in whether x keeps its `mut`.

**Why c is not tested against 0.** Gemma sampled every token of S under S's prefix. For any edit,
E[log p_edited(t) − log p_S(t)] = −KL ≤ 0 at that position, so c is negative without any tracking. The
pilot (§7) shows it: at background tokens c averages −0.061 nats per token, most of it next to the edit
(−0.29 per token 1–4 tokens after it, −0.003 at 20 or more).

**The matched null.** Background tokens of an item are the tokens of V after the later edit and before
q_place, other than occurrences of x's name, whose ids equal A's. For problem g, ĉ_g(t) is the mean c of
the background tokens of all other problems in the same cell of
- 20 bins of S-side entropy (quantiles of all background tokens' entropy), crossed with
- 5 bins of distance to the later edit (tokens: 1–4, 5–9, 10–19, 20–39, 40+).
A cell with fewer than 30 tokens is merged with its lower-entropy neighbour.

**Item:** d = Σ_{t ∈ span} (c_t − ĉ_g(t)). **Problem:** the mean over items within (source, x), then over
x within source, then over sources within the problem, each level weighted equally.

**Test (the one primary test).** One-sided problem-level sign flip of the mean problem score, H1: mean < 0,
**α = 0.05**; 99,999 flips plus the identity (stream [20261022, 11]). Interval: 5,000 bootstrap draws of
problems (stream [20261022, 12]).

### Secondary tests (only if the primary rejects; Holm over the two at 0.05)

The smaller one-sided p is compared with 0.025; if it rejects, the other is compared with 0.05.

- **S1, reads of x.** Occurrences of x's name after the later edit and before q_place, in x's function, not
  after `.`, not a mutation site, **at least 20 tokens after the later edit**, whose ids agree in V and A.
  d_read = (c_{q_place} − ĉ_g(q_place)) − the mean over reads of (c_r − ĉ_g(r)), with the matched null
  above. One-sided < 0. By a count from tokenization only (§7), 602 items in **61 problems** qualify.
  It separates a drop at the mutation from a drop at every mention of x after its declaration was edited.
- **S2, specificity.** Items with a placebo: d_z = Σ_span (log p_V − log p_Pz). x lost its `mut` in V, z
  lost its own in Pz, and both carry y's edit; under no x-specific tracking, d_z has no sign. One-sided < 0.
  86 problems. Both removal sites lie a median 48–49 tokens before q, where the background moves by
  −0.003 per token, so d_z is not matched.

### Descriptive (nothing here changes the tests)

- the raw Σ_span c and the background mean; the R arm (Σ_span log p_R − log p_S); the 2 × 2 of x's edit
  and y's edit (V, A, R, S);
- **the jump** (the Rust form of step 0): per item, the mean c of background tokens 1–4 tokens after the
  edit, the mean c of the last 5 background tokens before q, the raw per-token c over the span, and the
  span minus the last 5;
- q_place and q_op separately; E0384 and E0596 separately (90 / 38 / 48 problems hold only E0384, only
  E0596, both); S1's two parts (q_place's residual, the reads' residual);
- sensitivities: Phase 3's samples only (114 problems); without items whose y is in another function;
  20%-trimmed problem means; per-token rather than summed spans;
- **natural errors**, as context and not as a condition: Gemma's own failing samples whose only errors are
  mutability errors, on a let binding (47 problems) and on a prompt-dictated parameter (the behaviour gate,
  44 problems), each scored by adding the missing `mut ` and against its own background. Their text was
  sampled under the violating prefix, so their bias runs the other way (+KL); they are reported, never
  pooled with the primary;
- **the probes** (`p9_b_probe.py`; attribution-free, as Phase 7's P): transcoder activations at a
  position, forward only, as the difference between two texts. A logistic probe without intercept on the
  differences projected on 45 SVD directions (fitted without labels; the projections divided by one
  overall RMS), each row's orientation drawn at random, 5 folds grouped by problem, gives the share of
  rows it orients correctly, averaged as the primary (A). Two contrasts, each at two positions:
  - **probe 1, V against A**, at q_place and at one background token (≥ 20 tokens after the later edit,
    the entropy-nearest to q_place's). The edit changes activations at every later position: on the pilot
    both positions are at the ceiling (1.000 and 0.993), so probe 1 is reported and not read;
  - **probe 2, V against Pz**, at q_place and at one background token after all three edits (≥ 20 tokens).
    Both texts lose one necessary `mut`, so the edit's size is matched. **A2(q) − A2(bg)** is the quantity
    read, with a problem bootstrap. Only V edits x's own declaration, so A2(q) > A2(bg) can mean that x's
    token carries a change to its declaration, not that the model uses it;
- **cheap signals**, beside each probe at the same positions: from the output distribution there, the
  differences in log p of the next token, entropy and top probability, and KL(other‖V) − KL(V‖other),
  with the same orientation task.

## 3. Predictions (written before any primary contrast; after the pilot and the counts of §7)

- **Primary: rejected**, with d between −0.12 and −0.02 nats per item (pilot: −0.073 [−0.143, −0.020]).
- **S1: not rejected (weakly held).** d_read between −0.08 and 0.00; the expected size is near its MDE
  (§5). The pilot holds no S1 item.
- **S2: rejected**, with d_z between −0.40 and −0.05 (pilot, 7 problems: −0.30 [−0.68, −0.004]). On the
  pilot, removing z's `mut` raises the mutation of x (+0.23 nats against S), as if the statement were
  routed to the bindings still declared `mut`.
- So the expected reading is row 2 of §4: a drop beyond matched background and specific to which binding
  lost its `mut`, without a separation from mentions of x in general.
- **Jump:** c just after the edit −0.40 to −0.15 per token; just before q −0.02 to 0.00; at q −0.06 to
  −0.02 per token; at q minus just before q below 0.
- **Probes:** probe 1 at the ceiling at both positions (≥ 0.95; difference within ±0.03). Probe 2:
  A2(q) ≥ 0.95, A2(bg) between 0.6 and 0.9, difference above 0.05 (no confidence on its size). Cheap
  signals of probe 1: q above bg by 0.1 to 0.3.
- The raw Σ_span c is negative, between −0.15 and −0.03 nats.

## 4. Readings (how each outcome is written)

| primary | S1 / S2 (Holm) | probe 2: A2(q) − A2(bg) | reading | next step (the §5 rules agreed on 2026-10-21) |
|---|---|---|---|---|
| rejected | both rejected | — | Gemma lowers the probability of mutating x when x is immutable, beyond what the same edit does elsewhere, beyond mentions of x, and beyond another binding losing `mut` | probe and patching |
| rejected | one or neither rejected | — | a drop at the mutation beyond matched background; the specificity not shown is named | probe and patching, with the caveat |
| not rejected | — | interval covers 0 or point below 0.05 | **"not detected"**: at the mutation Gemma's output and its plain activations do not mark x as immutable beyond the edit's general effect; 80% MDE quoted | write it up; no further GPU (no graphs) |
| not rejected | — | interval above 0 and point ≥ 0.05 | x's token carries the change, the output does not use it | patching becomes the main follow-up (proposed at gate 3) |

A non-rejection is always "not detected; the 80% MDE is X nats", never "Gemma does not track state".

## 5. Power

From the pilot (§7), the problem-level SD of d is 0.149 nats. With 167 problems the standard error of the
primary mean is about 0.012 nats, so the one-sided 80% MDE at α 0.05 is about 0.029 nats per item.
S1 (61 problems): with an assumed SD of 0.12 (the pilot's q_place SD is 0.11; no pilot item has a read),
the MDE is about 0.043 at 0.025 and 0.038 at 0.05. S2 (86 problems): with the pilot's SD of 0.52 (7
problems), about 0.16 at 0.025 and 0.14 at 0.05. The matched null's own estimation error is not in these
figures.

## 6. Limitations

- One model, Gemma-2-2b-it; MBPP's short functions; the population is Gemma's frontier samples.
- The mutation sites are where Gemma itself mutated x; under S the span is nearly certain (p 0.995 at
  q_place in the pilot), so the test asks whether that certainty drops, not whether Gemma would avoid the
  mutation when writing.
- The matched null assumes span tokens respond to the edit like background tokens of the same entropy and
  distance. S1 and S2 test that assumption from two sides.
- Phases 5 and 6 drew their samples for other questions; 48 of the 167 problems enter only through Phase
  6's extra rounds, which were drawn for problems with failing samples.
- The probes cannot tell information that is present from information that is used; patching can.

## 7. What existed before this file

- **Stage A** (commit 93863eb): rustc diagnostics for all 41,560 samples, the relocation candidates, the
  MBPP+ relabel. No log-probability.
- **The item table** (`b_items.jsonl.gz`, built 2026-10-22 09:55 JST), with every text rustc-checked
  (commit bf766f8).
- **The pilot**: the 83 relocations of the 20 problems that hold no primary item (y's edit lies after q,
  so at q the contrast is R − S). These problems are not in the primary set.
  - `b_pilot.json` (first look): Σ_span c −0.073 [−0.141, −0.022]; the mean of all background tokens −0.061
    per token [−0.090, −0.039], by entropy tercile −0.020 / −0.064 / −0.081; p_S at q_place 0.995 (median),
    at q_op 0.967.
  - `b_analysis_pilot.json` (`p9_b_analysis.py` on the pilot role, the code committed with this file):
    d −0.073 [−0.143, −0.020], one-sided p 0.003; q_place alone −0.053, q_op alone −0.019; S2 (7
    problems) −0.300 [−0.681, −0.004]; no S1 item. The jump: −0.322 per token just after the edit, −0.008
    just before q, −0.031 at q, at q minus before −0.024 [−0.052, −0.004]. (A first, unmatched comparison
    against the mean of all background tokens gave +0.030 per token; it was wrong, because the
    background's drop sits next to the edit.)
  - The background by distance from the edit (tokens 1–4, 5–9, 10–19, 20–39, 40+): −0.291, −0.110, −0.058,
    −0.004, −0.003 nats per token. Every pilot span lies 20 or more tokens after the edit.
  - `b_probe_pilot.json`: probe 1 1.000 at q and 0.993 at bg (19 problems); probe 2 1.00 and 0.75 (6
    problems). Cheap signals of probe 1: 0.871 at q, 0.674 at bg.
- **The behaviour gate** (natural parameter errors, 286 sources, 44 problems): adding `mut ` to the
  parameter raises the span's log p by +0.636 nats [+0.388, +0.931] (one-sided p 1e-5), but the background
  rises too (+0.134 per token), and the span's per-token rise above it is +0.044 [−0.035, +0.138].
- **Counts on the primary items from tokenization only** (no forward pass; 2026-10-22 12:20 JST): items
  with any read of x 1,585 (95 problems); with a read of q_place's token id 439 (57); reads lie a median
  17 tokens after the later edit and q_place 41. Reads ≥ 20 tokens after the edit: 602 items (61
  problems) for any id, 226 (38) for q_place's id. Placebos: 1,192 (item, z) rows; x's `mut` lies a median
  49 tokens before q, z's 48.
- **Timing**: 43 ms per text for log-probabilities; about 0.23 s per row for the probes (two texts; pilot, 108 rows).

**Changes to the gate-2 draft (decided at gate 2, before this commit).**
- S1 compared raw c at q_place with raw c at reads of q_place's id at any distance. Reads lie closer to the
  edit than q does, where the background drops more; from the pilot's distance gradient the raw form is
  biased by +0.060 per item, more than the pilot's whole q_place effect (−0.053), so it could not have
  rejected under tracking. S1 now uses the matched residual at both and reads ≥ 20 tokens after the edit,
  of any id (61 problems instead of 38 with the id condition).
- The fixed sequence primary → S1 → S2 (gate 1) became primary → Holm over S1 and S2, because S1's low
  power would have decided whether S2 is ever tested.
- The jump (step 0) was missing from the draft and is added as a descriptive.
- The probe: the draft's probe was V against A only; on the pilot it sits at the ceiling at both
  positions, so its q − bg difference cannot inform. Probe 2 (V against Pz) is added and read. The first
  pilot run of the probe gave 0.21 at q because of a bug (each column scaled by its training-fold SD,
  while some SVD directions vary only in held-out problems); one overall RMS replaces it.
- §4's probe column is now probe 2's difference.

**Part ②'s labels (decided at gate 2; binding on the addenda of ② and ③).** The MBPP+ label set leaves out
the 53 tasks with fewer than 20 usable cases and the 14 artefact tasks of the audit (reference
conventions 593, 806, 790, 771, 559, 109; inputs against the prompt's preconditions 633, 564, 119, 142,
753; i64 limits or Rust's remainder 409, 274, 435). In ③'s MBPP+ sensitivity, a left-out task keeps its
MBPP label (no problem is dropped). The set without the exclusions is reported beside it. Flips inside
mixed tasks may hold the same artefacts; a random sample of them is audited before part ②'s addendum,
to estimate that rate, not to remove cases (inputs are never calibrated on Gemma's samples).

## 8. Deviations

None in the execution (appended 2026-10-22 after the run). Two statements written before it turned out
wrong and are kept as written: §5's spreads (the problem-level SD of d is 0.628, not 0.149; S1's 0.533,
not the assumed 0.12; S2's 0.815, not 0.52), and §3's sizes (below).

## 9. Execution

```
cd data
ROLE=primary python p9_b_forward.py      # refuses unless this file is committed and unchanged
ROLE=primary python p9_b_analysis.py
ROLE=primary python p9_b_probe.py        # refuses likewise
```

Run 2026-10-22 12:50–13:10 JST on commit 66a884e (guard hash 6fd6d99018aefa50): 10,334 texts, 41 ms per
text; probes 2,932 rows in 639 s. Outputs: `results/p9_state/b_analysis_primary.json`,
`results/p9_state/b_probe_primary.json`. `p9_b_check.py`, written apart from the analysis, gives the same
primary, S1 and S2 means to four decimals.

### Results (appended 2026-10-22, after the run)

All 3,457 items kept (no id mismatch), 167 problems. Means over problems, 95% bootstrap intervals,
one-sided sign-flip p.

| test | problems | mean (nats) | 95% interval | p | decision | §3 prediction |
|---|---|---|---|---|---|---|
| primary d | 167 | −0.290 | [−0.390, −0.200] | 1e-5 | **rejected** | rejected, −0.12 to −0.02 (size missed) |
| S2 d_z (Holm, first) | 86 | −0.607 | [−0.787, −0.441] | 1e-5 | **rejected** (< 0.025) | rejected, −0.40 to −0.05 (size missed) |
| S1 d_read (Holm, second) | 61 | −0.060 | [−0.184, +0.078] | 0.20 | not rejected (≥ 0.05) | not rejected, −0.08 to 0.00 |

**Reading (§4, row 2).** At the mutation of x, Gemma's probability of the mutation drops beyond what the
same edit does to background tokens of the same entropy and distance, and the drop depends on which
binding lost its `mut`: x's own, not another necessary binding's. Not shown: that it is specific to the
mutation rather than to every mention of x — reads of x 20 or more tokens after the edit drop too
(residual −0.113 [−0.227, −0.034]; q_place's residual on the same 61 problems −0.173 [−0.265, −0.098]),
and S1's 80% MDE turned out to be 0.17 nats, not the 0.04 of §5.

Descriptive (registered):
- raw Σ_span c −0.320 [−0.431, −0.229] (predicted −0.15 to −0.03); background −0.062 per token
  [−0.080, −0.048]; q_place alone −0.205 [−0.271, −0.145], q_op alone −0.042 [−0.080, −0.014];
- the 2 × 2 on the span: R − S −0.272, A − S −0.214, V − S −0.535 [−0.662, −0.419]. **Adding `mut` to y
  alone lowers the mutation of x by 0.21 nats** (on the pilot, where y's edit lies after q, A − S was
  −0.002). The two edits roughly add: Gemma predicts which binding a statement mutates from which
  bindings are declared `mut`;
- the jump: −0.260 per token 1–4 tokens after the edit, −0.018 over the last 5 tokens before q, −0.122 per
  token at q, at q minus before −0.104 [−0.138, −0.071] (predicted at q −0.06 to −0.02; the rest held);
- sensitivities all below 0: 20%-trimmed −0.171 [−0.197, −0.145]; per token −0.110; Phase 3 only (110
  problems) −0.278; without y in another function (157) −0.315; E0384 sites (130) −0.326, E0596 (81)
  −0.207;
- probe 1 (V against A; 2,285 rows, 143 problems): A(q) 0.998, A(bg) 0.902, difference +0.097
  [+0.073, +0.122] (predicted both ≥ 0.95 and within ±0.03: missed — the background is not at the
  ceiling on the primary items). Its cheap signals: 0.780 at q, 0.726 at bg, difference +0.055
  [−0.001, +0.110] (predicted 0.1 to 0.3);
- **probe 2 (V against Pz; 647 rows, 67 problems): A2(q) 0.958, A2(bg) 0.572, difference +0.385
  [+0.301, +0.466]** (predicted ≥ 0.95, 0.6–0.9, above 0.05). Its cheap signals at q (the distribution
  that predicts q_op) are at chance: 0.517 [0.432, 0.608]. At x's token the activations separate x's
  declaration losing `mut` from z's; the next-token distribution there does not. As §2 says, this shows
  the change is carried at x's token, not that it is used.

Next step under §4 row 2: probe and patching, with the caveat that specificity to the mutation (S1) is
not shown. Patching and the graphs of part ① are planned in their own addendum at gate 3.
