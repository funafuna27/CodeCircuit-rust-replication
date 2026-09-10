import json
import os
from pathlib import Path
from tqdm import tqdm
from graph_dataset import GraphDataset

# METADATA_PATH = "/home/kaiyu/CodeCircuit/data/graph/graph_metadata.jsonl"
# OUTPUT_PATH = "gemma_mbpp_features.jsonl"

# METADATA_PATH = "/home/kaiyu/CodeCircuit/data/cpp_graph/graph_metadata.jsonl"
# OUTPUT_PATH = "gemma_mbpp_cpp_features.jsonl"

# Same GRAPH_DIR / FEATURES_FILE knob as script 3 and chunk_cleanup, so a
# language switch cannot leave the three stages pointing at different places.
GRAPH_DIR = os.environ.get("GRAPH_DIR", "python_graph")
METADATA_PATH = f"{GRAPH_DIR}/graph_metadata.jsonl"
OUTPUT_PATH = os.environ.get("FEATURES_FILE", "gemma_mbpp_features.jsonl")
# Shared with script 3: an unloadable graph is recorded here so the same
# give-up counter bounds it.
FAILED_PATH = f"{GRAPH_DIR}/failed_graphs.tsv"


def load_metadata(jsonl_path: str):
    graph_entries = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                graph_entries.append(entry)
    return graph_entries


def append_jsonl(path: str, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_keys(path: str) -> set:
    """(expr_id, step_number) already present in the output, so re-running a
    chunk after a crash appends nothing twice.

    A run killed mid-append leaves a truncated last line. This file is only
    ever appended to, so nothing would repair it later: the partial line is
    cut here instead. Its row is then simply treated as not yet extracted and
    gets rewritten. Letting json.loads raise would instead brick every later
    chunk of an unattended multi-day run.
    """
    done = set()
    if not Path(path).exists():
        return done

    with open(path, "rb") as f:
        raw = f.read()

    good_bytes = 0
    for line in raw.splitlines(keepends=True):
        if not line.strip():
            good_bytes += len(line)
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            break
        if not line.endswith(b"\n"):
            break  # complete JSON but no terminator: the write was cut short
        done.add((r["expr_id"], r["step_number"]))
        good_bytes += len(line)

    if good_bytes != len(raw):
        print(f"WARNING: {path} ends in a partial record; truncating "
              f"{len(raw) - good_bytes} trailing bytes")
        with open(path, "rb+") as f:
            f.truncate(good_bytes)

    return done


def fsync_file(path: str):
    """Force the features file to disk before chunk_cleanup deletes the ~400MB
    graphs it was derived from. Without this the rows exist only in the page
    cache, and a power loss could take them while the graphs are already gone.
    """
    if not Path(path).exists():
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def extract_and_save_features():
    print("Loading metadata...")
    metadata = load_metadata(METADATA_PATH)

    print(f"Loaded {len(metadata)} graph entries.")

    done = load_done_keys(OUTPUT_PATH)
    print(f"Already extracted: {len(done)} (will be skipped)")

    dataset = GraphDataset(
        graph_paths=metadata,
        feature_type='advanced_graph_features',
        use_cpu=True
    )

    print("Start extracting features...")
    n_written = 0
    n_failed = 0

    for idx in tqdm(range(len(dataset))):
        if (metadata[idx]["expr_id"], metadata[idx]["step_number"]) in done:
            continue

        sample = dataset[idx]

        if not sample["success"]:
            # An unloadable .pt is unusable and nothing else will ever repair
            # it: script 3 resumes on bare exists(), so leaving the file in
            # place makes this chunk fail identically forever. Delete it so the
            # graph is regenerated, and record the attempt in the same ledger
            # script 3 uses to give up, so a prompt that keeps producing bad
            # graphs is bounded instead of looping.
            path = Path(metadata[idx]["graph_path"])
            print(f" Failed to load {path.name} "
                  f"(expr_id={sample['expr_id']} step={sample['step_number']}) "
                  f"-- deleting so it is regenerated")
            path.unlink(missing_ok=True)
            with open(FAILED_PATH, "a", encoding="utf-8") as fh:
                fh.write(f"{path.name}\tUnloadableGraph\tscript 4 could not load it\n")
                fh.flush()
                os.fsync(fh.fileno())
            n_failed += 1
            continue

        feature = sample.get("features")
        if feature is not None:
            feature = feature.tolist()
        else:
            feature = []

        output_record = {
            # Position of this sample in the source dataset. Rows are appended
            # in whatever order the chunks happen to produce them -- a graph
            # that failed once and succeeded on a later chunk lands far from
            # its neighbours -- and script 5 splits the file positionally, so
            # the canonical order has to be recoverable downstream.
            "row_index": metadata[idx].get("row_index"),
            "expr_id": sample["expr_id"],
            "step_number": sample["step_number"],
            "before_after": sample["before_after"],
            "graph_path": metadata[idx]["graph_path"],
            "step_labels": sample["step_labels"],
            "original_expression": sample["original_expression"],
            "batch_size": metadata[idx].get("batch_size"),
            "feature": feature
        }

        append_jsonl(OUTPUT_PATH, output_record)
        n_written += 1

    fsync_file(OUTPUT_PATH)

    print(" Feature extraction complete.")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"rows written this chunk: {n_written}, failed to load: {n_failed}")

    if n_failed:
        # Not fatal any more: the bad files are gone, so the next chunk will
        # regenerate them and the give-up counter bounds how often that repeats.
        # Exiting here instead would stop the driver before cleanup and strand
        # the whole chunk.
        print(f"{n_failed} graph(s) were unloadable, deleted, and will be "
              f"regenerated; see {FAILED_PATH}")


if __name__ == "__main__":
    extract_and_save_features()
