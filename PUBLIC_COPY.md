# About this public copy

This repository is the public copy of a private working repository. The two
differ in one respect. Every commit contained two OpenAI API keys in
`data/1_code_correct.py`, and those keys came with the upstream repository
([bruno686/CodeCircuit](https://github.com/bruno686/CodeCircuit)). Before
publishing, the whole history was rewritten with
`git filter-repo --replace-text`. Every match of `sk-[A-Za-z0-9_-]{20,}` became
`REDACTED_OPENAI_KEY`. As a result, that script needs your own key to run.

The rewrite kept everything else: every other file, every author and
committer name and date, and every commit message. The one exception is six
messages that quoted an abbreviated commit hash; filter-repo updated each of
those to the new hash. Because one file changed in every commit, every commit
hash changed too.

## Hashes cited in the documents

The pre-registrations cite commits by their hash in the working repository.
Each cited commit is how a document shows that it was committed before the
result it governs. The dates carried over unchanged, so the order can still be
checked here under the new hashes:

| Document | Cited hash | Hash here | Committed |
|---|---|---|---|
| `PREREGISTRATION.md` | `08a6663` | `0725cb8` | 2026-10-01 11:13 +0900 |
| `PREREGISTRATION.md` | `dc7ed09` | `3c8c3f0` | 2026-10-01 11:17 +0900 |
| `PREREGISTRATION_WITHIN.md` | `aa35c25` | `fc6f4fc` | 2026-10-02 22:58 +0900 |
| `PREREGISTRATION_MUTANT.md` | `42aeb24` | `76e14f2` | 2026-10-08 11:44 +0900 |
| `PREREGISTRATION_MUTANT.md` | `998fa94` | `f1ea921` | 2026-10-08 01:36 +0900 |
| `PREREGISTRATION_MUTANT.md` | `e630b37` | `058054f` | 2026-10-07 15:11 +0900 |
| `PREREGISTRATION_MATCHED.md` | `42aeb24` | `76e14f2` | 2026-10-08 11:44 +0900 |
| `PREREGISTRATION_CLOSING.md` | `554f053` | `0b6ce46` | 2026-10-14 08:24 +0900 |
| `PREREGISTRATION_CLOSING.md` | `7fc2a48` | `4d73a96` | 2026-10-14 11:18 +0900 |
| `PREREGISTRATION_P8_INTERVENTION.md` | `7f04e70` | `657e377` | 2026-10-18 11:16 +0900 |

One recorded date is wrong: the PC's clock was 9 h slow when `e630b37`
(`058054f` here) was committed, so it was really committed at 2026-10-08
00:11 +0900. `PREREGISTRATION_MUTANT.md` §9 explains.

The full old-to-new map for the 63 rewritten commits is in [`commit-map.txt`](commit-map.txt),
as written by `git filter-repo`. The last of them, `e6db624`, is `66c448b` here.
The commits that added this note and later extended it have no
counterpart in the working repository.

## Commits added after the rewrite

Later commits of the working repository were copied here with `git cherry-pick`,
after a scan for keys. Each keeps its message, its author and its author date;
its committer date is when it was copied. None of them touches
`data/1_code_correct.py`.

| Working repository | Here | Authored | Subject |
|---|---|---|---|
| `3eda320` | `97ebb31` | 2026-10-14 19:55 +0900 | Erratum: the Rust judge passed infinite results on its isclose branch |
| `ec592e0` | `8773602` | 2026-10-15 00:54 +0900 | Erratum: add Phase 6's own samples to the inf audit |
| `7f04e70` | `657e377` | 2026-10-18 11:16 +0900 | Phase 8 pre-registration: graph-ranked feature ablation at the model's own bugs |
| `56f0d16` | `57029c4` | 2026-10-19 05:24 +0900 | Phase 8 results: graph-ranked ablation beats random features, not a token ban |
| `5869555` | `c105bc0` | 2026-10-21 17:05 +0900 | Close the study: conclusion, audit corrections, inputs for a clone |
| `93863eb` | `431b4c6` | 2026-10-21 23:11 +0900 | Phase 9 stage A: rustc diagnostics, mut relocations, MBPP+ relabel |
| `bf766f8` | `76b37fd` | 2026-10-22 10:17 +0900 | Phase 9 stage B: typed MBPP+ harness, flip audit, items, pilot |
| `66a884e` | `c781019` | 2026-10-22 12:50 +0900 | Phase 9 part 1: pre-registration, analysis and probe code |
| `2e4f588` | `e54ed0c` | 2026-10-22 13:11 +0900 | Phase 9 part 1: primary results |
| `bed2d9a` | `b5b916c` | 2026-10-22 16:50 +0900 | Colab speed test: three retained Phase 5 graphs on an A100 and the RTX 5070 |
| `a2d57df` | `29f5ea2` | 2026-10-22 20:22 +0900 | Phase 9 part 3 (M1): pre-registration addendum and its code |
| `aac9ac3` | `140bf49` | 2026-10-22 21:31 +0900 | Phase 9 M1: deviation 1, X3 judged on the averaged quantities |
| `8c4342e` | `2554bf2` | 2026-10-22 22:56 +0900 | Phase 9 M1: results |
| `1655be0` | `987cb5c` | 2026-10-23 11:06 +0900 | Phase 9 gate 3: pre-registration of patching and graphs, with its code |
| `3a78ac9` | `a22a830` | 2026-10-23 12:23 +0900 | Phase 9 gate 3: patching results (P1 rejected, P2 and P3 not) |
| `f79fb0d` | `fb0f66c` | 2026-10-23 15:01 +0900 | Phase 9 gate 3: graph results (G1 rejected) and deviation 1 |
| `a6aa2f6` | `be674c0` | 2026-10-23 15:01 +0900 | Phase 9 gate 3: write deviation 1 into section 9 (missed in f79fb0d) |
| `e99c988` | `d8d5954` | 2026-10-23 15:22 +0900 | Phase 9 part 2: pre-registration of the within-task MBPP+ test, with its code |
| `67e4483` | `0399695` | 2026-10-23 15:34 +0900 | Phase 9 part 2: within-task MBPP+ results (Q1 rejected at p 0.040; S1, S2 not) |
| `8ea59c3` | `d9f7739` | 2026-10-23 16:05 +0900 | Phase 9 part 2: post hoc checks (more permutations, length) |
| `4f3e2e3` | `2ef04ac` | 2026-10-23 16:06 +0900 | readme: Phase 9 section (rustc and MBPP+ as labels) |
