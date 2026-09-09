"""Free the .pt files of the chunk that script 4 has just featurized.

A whole language is ~1420 graphs at ~400MB, so keeping every graph would cost
~500GB and make a four-language study impossible. Run this after script 4 to
reclaim the chunk, then let script 3 generate the next one.

Ordering matters more than it looks. The ledger is the ONLY record that stops
script 3 from regenerating a graph, because once the .pt is gone a finished
graph and a never-started one look identical. So a name is written to the
ledger and fsynced BEFORE its file is unlinked. The reverse order leaves a
window where the graph is gone and nothing remembers it was ever made -- an
interrupted cleanup would then cost a full chunk of regeneration, about ten
GPU-hours.

A graph is only ever ledgered once its (expr_id, step_number) is present in the
features file, so an unverified graph is never deleted.
"""

import argparse
import json
import os
from pathlib import Path

GRAPH_DIR = Path(os.environ.get('GRAPH_DIR', 'python_graph'))
FEATURES_PATH = Path(os.environ.get('FEATURES_FILE', 'gemma_mbpp_features.jsonl'))
METADATA_PATH = GRAPH_DIR / 'graph_metadata.jsonl'
LEDGER_PATH = GRAPH_DIR / 'done_graphs.txt'

# Keep the .pt of the first N graphs on disk so there is always a sample to
# debug feature extraction against without a regeneration run. Set to 0 to
# exercise the delete path on a short validation run.
KEEP_FIRST_N = int(os.environ.get('KEEP_FIRST_N', 50))


def load_jsonl(path: Path) -> list:
    """Tolerates a truncated final record: these files are append-only, so a
    kill mid-write leaves one, and raising here would stop an unattended run."""
    if not path.exists():
        return []
    out = []
    with open(path, "rb") as f:
        lines = f.read().splitlines(keepends=True)
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if n == len(lines):
                print(f"WARNING: ignoring truncated final line of {path}")
                break
            raise
        if not line.endswith(b"\n") and n == len(lines):
            print(f"WARNING: ignoring unterminated final line of {path}")
            break
        out.append(rec)
    return out


def graph_index(name: str) -> int:
    """graph_<i>_<task>_<step>.pt -> i"""
    return int(name.split("_")[1])


def append_ledger(names: list):
    """Durably record the names before anything is deleted."""
    if not names:
        return
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be freed, delete nothing")
    args = ap.parse_args()

    metadata = load_jsonl(METADATA_PATH)
    featurized = {(r["expr_id"], r["step_number"]) for r in load_jsonl(FEATURES_PATH)}
    ledger = set()
    if LEDGER_PATH.exists():
        ledger = {l.strip() for l in open(LEDGER_PATH, encoding="utf-8") if l.strip()}

    print(f"graph dir      : {GRAPH_DIR}")
    print(f"features file  : {FEATURES_PATH}")
    print(f"chunk metadata : {len(metadata)} graphs")
    print(f"featurized     : {len(featurized)} rows")
    print(f"ledger (before): {len(ledger)}")

    verified, unverified = [], []
    for entry in metadata:
        name = Path(entry["graph_path"]).name
        if (entry["expr_id"], entry["step_number"]) in featurized:
            verified.append(name)
        else:
            unverified.append(name)

    # Ledger first, then delete. A crash between the two costs nothing: the
    # graphs are recorded as done and their features are already durable.
    if not args.dry_run:
        append_ledger([n for n in verified if n not in ledger])

    freed = kept = missing = 0
    for name in verified:
        pt = GRAPH_DIR / name
        if not pt.exists():
            missing += 1
            continue
        if graph_index(name) < KEEP_FIRST_N:
            kept += pt.stat().st_size
            continue
        freed += pt.stat().st_size
        if not args.dry_run:
            pt.unlink()

    verb = "would free" if args.dry_run else "freed"
    print(f"\n{verb}       : {freed / 1024**3:.1f} GB ({len(verified)} verified)")
    print(f"kept (first {KEEP_FIRST_N}): {kept / 1024**3:.1f} GB")
    print(f"already gone   : {missing}")
    print(f"ledger (after) : {len(ledger | set(verified))}")

    if unverified:
        # Script 3 truncates the metadata at the start of the next chunk, so
        # this is the last moment these names are recorded anywhere. Stopping
        # here also prevents the driver from looping on graphs it can never
        # make progress on.
        print(f"\nERROR: {len(unverified)} graph(s) in this chunk have no features. "
              f"They were NOT deleted. Investigate before rerunning:")
        for name in unverified[:20]:
            print(f"  {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
