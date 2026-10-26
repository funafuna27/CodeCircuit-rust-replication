# Pre-registration addendum — where does Gemma read that x is immutable? (Phase 9, part ①, gate 3)

Written 2026-10-22 and 2026-10-23 (Japan time); the design was approved by the author at gate 3 on
2026-10-23. Committed with its code (`data/p9_mech_*.py`) before any patched forward pass or graph of a
primary item exists. `p9_mech_patch.py` (primary role) and `p9_mech_analysis.py` refuse to run unless
this file, the other PREREGISTRATION_P9*.md and the scripts they rely on are committed and unchanged.
Part ①'s results (PREREGISTRATION_P9_MUTSTATE.md §9) and part ③'s (PREREGISTRATION_P9_M1.md §11) are
known; what else existed is listed in §8.

## 1. Questions

Part ① found that Gemma's probability of mutating x drops beyond matched background when x's own `mut`
is removed, and not when another necessary binding loses its `mut` (S2), with the drop not separable from
other mentions of x (S1). rustc names the cause of every violation: x's binding name at its declaration,
where its `mut ` suggestion points (E0384, E0594, E0596; E0384 also labels it "first assignment to `x`").
Call these tokens D.

- **P (causal; PREREGISTRATION_P9_MUTSTATE.md §4 row 2):** is the information that x is mutable carried
  at D, and used at the mutation? Activation patching from A (x mutable) into V (x immutable) at D.
- **G (explanatory; the plan's secondary test):** does the attribution graph of the mutation's first
  token put more of its attribution on D than one backward pass of gradient × activation (GxA) does?

## 2. Items and positions (`p9_mech_spans.py`; `results/p9_mech/spans.jsonl.gz`)

- P: all 3,457 primary items of part ① (167 problems). G: one item per problem, the first of each problem
  in `b_items.jsonl.gz` order: 167 graphs (the plan said about 150).
- D: V's tokens overlapping x's binding name at its declaration, which must be where rustc's `mut `
  suggestion for x points (V compiled again, `--error-format=json`); otherwise the item is dropped. K: the
  token after D (inside the declaration). W: D through the last token starting before the end of D's line
  (the whole declaration; descriptive). Readouts as part ①: the span (q_place..q_op); reads of x at
  least 20 tokens after the later edit (counted in V's tokens) and before q; background tokens (after
  the later edit, before q, not a mention of x).
- Each V token is mapped to A through its last character (V and A differ only by x's `mut `), and the ids
  must agree. Built 2026-10-23: 3,457 primary and 83 pilot items, none dropped (rustc's suggestion sits
  at x's declaration in every V; D is 1–6 tokens); 602 primary items have a read.

## 3. P — patching (`p9_mech_patch.py`; local RTX 5070; batch 1)

The ReplacementModel's TransformerLens model loaded as `3_generate_graph.py` loads it (circuit-tracer
0.5.3.dev1+g6018ed8d3 from site-packages; its hooks change no forward value), run as a plain model. For
each item: run A and cache `blocks.{l}.hook_resid_pre` at every layer; run V; run V with D's positions
replaced at every layer by A's cached values at D's A positions (the all-layer patch); likewise K and W.
On the 167 G items, also D patched from layer l on (a hook at layer l only), l = 0..25. Log p from a
float32 log-softmax.

- d_D = Σ over the span of (log p_V|D − log p_V); d_K, d_W likewise; d_R = the change at each read; d_bg =
  the change in the mean log p of the background tokens under D's patch.
- Problem scores as part ①: items → (source, x) → source → problem, equal weights.
- **P1 (this addendum's primary): one-sided problem-level sign flip of the mean of d_D, H1: mean > 0, α
  0.05** (99,999 flips plus the identity, stream [20261023, 1]; interval: 5,000 problem bootstraps,
  stream [20261023, 2]).
- **Secondary (only if P1 rejects; Holm over the two at 0.05):** P2: d_D − d_K > 0 (stream 3); P3:
  d_D / |span| − mean(d_R) > 0 on items with a read (stream 5).
- Descriptive: the fraction restored, mean(d_D) / mean(span log p_A − span log p_V) over problems, and the
  same for W; d_K, d_W; d_bg; **the span's change per token minus the background's** (whether D's patch
  restores the mutation more than any later token); the reads' change per token; E0384 and E0596
  separately; the layer sweep with problem bootstraps.

## 4. G — graphs and GxA (`p9_mech_graphs.py`; Colab A100-SXM4-80GB, all 167 on one GPU type)

The environment of the 10-22 bench (torch 2.11.0+cu130; circuit-tracer from commit 6018ed8d; transformers
4.57.3; transformer-lens 3.2.1); the script refuses another GPU, records its own sha256 and the
environment, and writes each item as soon as it exists (a disconnected session resumes). The analysis
refuses outputs not made on an A100 by the committed script. No G number is compared with a graph built
on another GPU.

- Input: V's full-text token ids (BOS + prose + ' ' + code, as part ① scored them) up to and excluding
  q_place's token; target: that token's id. `attribute(ids, model, attribution_targets=[id],
  batch_size=64, max_feature_nodes=8192, offload='cpu')`, script 3's allocator setting, after an untimed
  warm-up graph. The graph is not saved. Kept: each node's influence on the target
  (`compute_node_influence`, weighted by the target's probability) summed by position, separately for
  features, error nodes and token nodes; the target's probability; the active and selected counts; the
  49 features of `graph_dataset.py` (script 4's featuriser with the igraph override; a failure is
  recorded, not fatal).
- GxA: one backward pass of the target's demeaned logit (its logit minus the mean logit) through a plain
  TransformerLens model of the same weights (`from_pretrained_no_processing`). The ReplacementModel is
  not used for GxA: its permanent hooks stop the gradient through attention patterns and normalisation
  scales, so its gradient is not the model's. Kept: per layer and position, Σ_d grad · act on
  `blocks.{l}.hook_resid_pre`; layer 0 (input × gradient) is GxA.
- s_m(D) = method m's share on D among all positions except BOS (graph: the summed influence; GxA: the
  absolute value at each position).
- **G1: d_G = s_graph(D) − s_GxA(D) per problem, one-sided sign flip, H1: mean > 0, α 0.05** (stream
  [20261023, 11]; bootstrap 12). One item per problem.
- The target is x's name, so any method credits earlier mentions of x for supplying it, and D is one of
  them. **The reading of G1 therefore rests on its specificity row:** per token, the share on D minus the
  share on x's other mentions before q, per method and graph minus GxA (problem bootstraps). Other
  descriptives: the share on other `let` bindings' names, on the final position; the 49 features; the
  Spearman correlations of each method's s(D) with the same item's d_D and their difference (bootstrap,
  stream 13): which explanation tracks the causal effect better.

## 5. Readings

| P1 | P2 / P3 | written as |
|---|---|---|
| rejected | both | "the information that x is mutable is carried at x's binding name and read at the mutation more than at other mentions of x" |
| rejected | P2 only | "…carried at x's binding name more than at the next token; not shown to be specific to the mutation" |
| rejected | P3 only | "restoring x's name restores the mutation more than other mentions of x, but the next token does so as much" |
| rejected | neither | "restoring x's name restores part of the mutation, as does the next token; not shown specific" |
| not rejected | — | "not detected by patching x's binding name; the 80% MDE is X nats" |

If the span's change per token does not exceed the background's, every P1 sentence adds "and the
background as much". G1 rejected: "the graph places more of the mutation's attribution on rustc's cause
than GxA does", followed by the specificity row (whether D stands out among mentions of x for either
method); not rejected: "not shown", with its interval.

## 6. Predictions (written after the pilot of §8, before any primary patch or graph)

- P1 rejected; mean d_D between +0.03 and +0.20 nats per item; the fraction restored by D 10–40%, by W
  70–110%.
- P2 not rejected (on the pilot K restored as much as D). P3 about even odds.
- The span's change per token does not exceed the background's by more than 0.01 (the pilot's
  background rose by 0.020 per token).
- The sweep: patching from layer 0 does nothing (the same token), the effect rises over the early and
  middle layers and is near its all-layer value by layer 15.
- G1 rejected; s_graph(D) between 0.03 and 0.25, s_GxA(D) below 0.03. D's specificity over x's other
  mentions is positive for the graph and smaller for GxA. Both correlations with d_D within ±0.3.

## 7. Budget and schedule

- Patching: 1.3 s per pilot item (five forward passes through the ReplacementModel), so about 75 minutes for
  3,457 items plus about 20 for the sweep, on the local 5070.
- Graphs: prefixes of 57–225 tokens (median 103); on the A100 bench 40.5 / 47.3 / 53.9 s at 100 / 120 /
  140 tokens: about 2.0 h for 167 graphs by a linear fit, 2.5–3.5 h with longer prefixes and the session's
  setup; about 19–26 compute units. Colab Pro has no background execution, so a browser tab stays open.
- Windows Update's pause ends 2026-11-11 06:40 JST (2026-11-10 21:40 UTC); both runs fit well before.

## 8. What existed before this file

- Parts ① and ③ (above), part ①'s item table and forward pass, part ②'s labels, the Colab bench (bed2d9a).
- The positions of §2, built from the item table and rustc (no model).
- **The pilot** (83 items, 20 problems, never in the primary set), patched with the committed code on the
  local 5070: d_D +0.014 per problem [+0.004, +0.026] (one-sided p 0.009); the pilot's own gap is small
  (span log p_A − log p_V +0.074); restored by D 19%, by W 94%; d_K +0.029, so D − K −0.015 [−0.051,
  +0.013]; background +0.020 per token; the pilot has no read at 20 tokens or more.
- **Two pilot items** (one problem, 53 tokens) through the graph script on the local 5070 (FORCE=1, a
  code test): the graph's share on D 0.142, GxA's 0.0085; on x's other mentions 0.027 and 0.002 per
  token; the 49 features computed. Nothing from these graphs enters G.

## 9. Deviations

1. **The graphs ran on an A100-SXM4-40GB, not the 80GB card §4 names** (the bench's). Colab assigned the
   40GB variant; all 167 graphs and GxA passes came from one session (one `run_*.json`), with the
   committed script (sha256 8e7a627e…) and the §4 environment (torch 2.11.0+cu130, circuit-tracer
   0.5.3.dev1+g6018ed8d3, transformers 4.57.3, transformer-lens 3.2.1). Both cards are the same GA100 chip
   with 108 SMs; only the memory differs. The analysis's check ("A100" and the committed script) passes,
   "all on one GPU type" holds, and no G number is compared with a graph from another card. Noticed when
   the outputs were copied, recorded with the results.

## 10. Execution

```
cd data
python p9_mech_spans.py                         # done before this commit (§8)
ROLE=primary python p9_mech_patch.py            # local 5070
PART=P python p9_mech_analysis.py
# Colab (A100): the bundle's notebook runs p9_mech_graphs.py on the 167 G items; outputs copied to
# ~/p9_work/mech/colab/out
PART=G python p9_mech_analysis.py
```

## 11. Results — P (patching; 2026-10-23, 3,457 items, 167 problems, local RTX 5070, 76 minutes)

`results/p9_mech/patch_primary.jsonl.gz` (sha256 a8a72d94…), `analysis_P.json`; recomputed independently from
the item file (same means to four decimals). Problem means with 95% problem-bootstrap intervals.

- **P1 rejected:** d_D +0.121 nats [+0.099, +0.145], one-sided p 1e-5 (the floor of 99,999 flips); the
  80% MDE was 0.030. E0384 +0.136 (130 problems), E0596 +0.100 (81).
- **P2 not rejected:** d_D − d_K +0.018 [−0.015, +0.047], p 0.137 (d_K +0.103 [+0.066, +0.148]).
- **P3 not rejected:** d_D / |span| − mean(d_R) −0.008 [−0.049, +0.028], p 0.635 (61 problems with a read;
  the reads' own change +0.046 per token [+0.020, +0.081]).
- Descriptive: the gap (span log p_A − log p_V) +0.321 [+0.230, +0.425]; restored by D 38%, by W 98%; d_W
  +0.314; the background +0.010 per token [+0.007, +0.014]; **the span's change per token minus the
  background's +0.038 [+0.029, +0.048]**; 519 of 3,457 items have d_D < 0.
- The sweep (167 G items, D patched at layer l only): 0 at layers 0–3, rising from layer 4 (+0.017) to
  +0.047 at 11, +0.068 at 14, +0.115 at 18 and its peak +0.132 [+0.103, +0.165] at 22, then +0.054 at 23
  and 0 at 24–25. Read: what D carries about x's `mut` is written there over layers 4–21 and is read from
  D by later positions by layer 23; a patch at 24 or later comes too late.
- **Reading (§5, row "rejected / neither"):** restoring x's name restores part of the mutation (38% of
  the gap, more per token than the background), as does the next token; not shown to be specific to the
  mutation over x's other reads.
- Against §6: P1, its size, both fractions (D near the top of 10–40%), P2 and the layer-0 null as
  predicted; P3 not rejected (predicted even odds). Wrong: the span exceeds the background by 0.038 per
  token, not ≤ 0.01; and the sweep is about half its peak at layer 15 and peaks at 22, not near its
  all-layer value by 15.

## 12. Results — G (graphs and GxA; 2026-10-23, 167 problems, Colab A100-SXM4-40GB)

`results/p9_mech/analysis_G.json`; the raw outputs in `results/p9_mech/colab_out.tar.gz`. All 167 graphs had
the expected tokens and node count and their 49 features; attribute 42.4 s median (max 80.0). The target's
probability agrees between the graph and the plain model to 0.0017 (median). The shares on D were
recomputed independently from the raw outputs (same means). Problem means with 95% problem-bootstrap
intervals.

- **G1 rejected:** s_graph(D) − s_GxA(D) +0.059 [+0.052, +0.065], one-sided p 1e-5 (80% MDE 0.008);
  s_graph(D) 0.085 (median 0.091), s_GxA(D) 0.026 (median 0.019); the graph's share is below GxA's on 13
  of 167 items.
- **Specificity row** (57 problems whose G item mentions x again before q): per token, D minus x's other
  mentions is +0.013 [+0.005, +0.022] for the graph and −0.001 [−0.005, +0.004] for GxA; graph minus GxA
  +0.014 [+0.005, +0.023]. Per token over all 167, the graph puts 0.076 on D and 0.011 on other `let`
  names; GxA 0.022 and 0.010. On the 57, x's other mentions get 0.027 (graph) and 0.013 (GxA) per token.
- The final position: graph 0.113, GxA 0.033.
- **Spearman with the same item's d_D:** graph −0.33, GxA −0.07; graph minus GxA [−0.43, −0.09]. Neither
  method's share on D rises with the causal effect of patching D; the graph's falls with it.
- **Reading (§5):** the graph places more of the mutation's attribution on rustc's cause than GxA does;
  among mentions of x, D stands out for the graph and not for GxA. But the graph's share on D is lower
  where patching D matters more, so its concentration on D is not evidence that D carries what the model
  uses about x's mutability.
- Against §6: G1, both shares and the specificity row as predicted; the graph's correlation (−0.33) lies
  just outside the predicted ±0.3, in the direction opposite to what a faithful explanation would show.

## 13. Post hoc: what the graph-vs-patching correlation goes with (2026-10-26)

§12 reported, as a registered descriptive, the Spearman correlation between each method's share on D and the
same item's patching effect: graph −0.33, GxA −0.07. The write-up read it as "the graph's share is lower
where patching D matters more".

**How this came up.** Reviewers of the write-up (2026-10-26; three independent recomputations) found that
most of the correlation goes with lpV, the mutation span's log p in V. When V already gives the mutation a
high probability:
- patching can restore little (ρ(d_D, lpV) −0.59);
- the graph puts more on D (ρ(graph share, lpV) +0.42; GxA +0.03).

`p9_mech_covariates.py` records this (`results/p9_mech/covariates.json`). It uses partial Spearman
correlations: ranks residualised on the covariates' ranks. Intervals come from 2,000 bootstrap draws of the
167 items, one per problem (stream [20261023, 50]).

| covariates | graph | GxA | graph − GxA |
|---|---|---|---|
| none (§12, registered) | −0.328 [−0.471, −0.176] | −0.068 | −0.260 [−0.431, −0.091] |
| lpV | −0.107 [−0.267, +0.051] | −0.066 | −0.041 [−0.242, +0.149] |
| gap (lpA − lpV) | −0.209 [−0.368, −0.051] | +0.077 | −0.286 [−0.459, −0.103] |
| lpV + gap | −0.102 [−0.261, +0.050] | +0.038 | −0.140 [−0.324, +0.041] |
| prefix length + the final position's share | −0.354 [−0.498, −0.208] | −0.098 | −0.256 [−0.424, −0.094] |
| all five (lpV, gap, prefix length, final share, span length) | −0.157 [−0.307, −0.014] | +0.025 | −0.181 [−0.353, −0.007] |

The gap is small on these items: median 0.038 nats, and 40 of 167 items have a gap ≤ 0.

**Reading (replaces the write-up's sentence; §12's numbers stand).**
- **G1 stands.** The graph puts more of the mutation's attribution on D than GxA does.
- **Not shown:** that this concentration follows the causal effect of D.
- **Not separable from lpV:** whether it runs against that effect. With lpV in the model, the graph's
  correlation is −0.10 to −0.16, and its difference from GxA ranges from −0.04 to −0.18 depending on the
  covariate set; several intervals include 0.
- lpV may be a mediator rather than a confounder. If so, conditioning on it removes part of what was asked.
  That is why the claim is weakened, not reversed.
