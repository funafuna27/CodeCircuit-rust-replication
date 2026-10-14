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

One recorded date is wrong: the PC's clock was 9 h slow when `e630b37`
(`058054f` here) was committed, so it was really committed at 2026-10-08
00:11 +0900. `PREREGISTRATION_MUTANT.md` §9 explains.

The full old-to-new map for all 63 commits is in [`commit-map.txt`](commit-map.txt),
as written by `git filter-repo`. The last commit of the working repository,
`e6db624`, is `66c448b` here. The commit that adds this note has no counterpart
in the working repository.
