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

None yet.

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
