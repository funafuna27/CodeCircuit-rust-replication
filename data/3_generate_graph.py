from pathlib import Path
import os
import time
import torch
import json
from circuit_tracer import ReplacementModel, attribute
from tqdm import tqdm

# --- Configuration and Model Loading ---
MODEL_NAME = 'google/gemma-2-2b-it'
TRANSCODER_NAME = "gemma"
model = ReplacementModel.from_pretrained(MODEL_NAME, TRANSCODER_NAME, dtype=torch.bfloat16).to('cuda')

# --- Graph Generation Parameters ---
MAX_N_LOGITS = 10
DESIRED_LOGIT_PROB = 0.95
MAX_FEATURE_NODES = 8192
# NOT a performance-only knob. attribute() picks which MAX_FEATURE_NODES of the
# active features enter the graph by greedy frontier expansion, and re-ranks
# only once per update_interval*batch_size features -- so batch_size changes the
# graph itself. The cap binds hard here (8192 selected out of 34k-99k active),
# so a run must keep one value throughout, and it should be the upstream
# default that the reference numbers were produced with. Measured on row 3:
# batch 16 = 88.3s, 32 = 45.3s, 64 = 26.2s, 128 = 75.4s, all at 9.08 GiB peak.
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 64))
OFFLOAD = 'cpu'
VERBOSE = False

# Stop after this many NEW graphs so a full run fits on disk: a whole language
# is ~1420 graphs x ~400MB (~500GB). Run 3 -> 4 -> chunk_cleanup.py in a loop;
# cleanup frees each chunk before the next one starts. 0 disables the cap.
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 200))

# --- Input/Output Paths ---
# Set by environment, not by editing: scripts 3, 4 and chunk_cleanup all read
# GRAPH_DIR / DATA_FILE / FEATURES_FILE, and run_chunked.sh exports them
# together. Each script used to carry its own commented-out per-language block,
# and editing only some of them left cleanup pointing at another language's
# graphs -- freeing nothing while the run filled the disk, the exact failure
# chunking exists to prevent.
#
#   java: DATA_FILE=gemma_mbpp_java_incremental.jsonl GRAPH_DIR=java_graph \
#         FEATURES_FILE=gemma_mbpp_java_features.jsonl
#   cpp:  DATA_FILE=gemma_mbpp_cpp_incremental.jsonl  GRAPH_DIR=cpp_graph  \
#         FEATURES_FILE=gemma_mbpp_cpp_features.jsonl
DATA_FILE_PATH = os.environ.get('DATA_FILE', 'gemma_mbpp_incremental.jsonl')
OUTPUT_DIR = Path(os.environ.get('GRAPH_DIR', 'python_graph'))

OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Graph files will be saved in: {OUTPUT_DIR.resolve()}")

# Output metadata file (jsonl append mode). Rewritten every run, so it always
# describes exactly the current chunk -- which is what script 4 should consume.
METADATA_FILE_PATH = OUTPUT_DIR / 'graph_metadata.jsonl'

# Graph names whose features are already extracted. Completion has to be
# tracked here rather than by checking for the .pt, because cleanup deletes
# the .pt and an existence check would regenerate it on the next chunk.
LEDGER_FILE_PATH = OUTPUT_DIR / 'done_graphs.txt'

# Graphs whose attribution raised. Recorded rather than fatal, so one bad
# prompt cannot end a multi-day run.
FAILED_FILE_PATH = OUTPUT_DIR / 'failed_graphs.tsv'

# torch.save streams ~400MB straight to the destination path, so a kill during
# a save leaves a truncated file under the real name -- which the resume logic
# would then accept as a finished graph forever. Write here first, rename after.
PARTIAL_SUFFIX = '.partial'


def load_data(file_path: str) -> list:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_ledger() -> set:
    if not LEDGER_FILE_PATH.exists():
        return set()
    with open(LEDGER_FILE_PATH, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}


def append_metadata(metadata: dict):
    """Append one metadata entry to the jsonl file."""
    with open(METADATA_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(metadata) + "\n")


def generate_and_save_graphs(data_list: list, output_dir: Path, model: ReplacementModel):
    task_step_counters = {}
    ledger = load_ledger()
    n_new = 0
    n_failed = 0
    # Serializing ~400MB per graph is a real share of the run, and the graphs
    # are discarded after featurization -- worth knowing what it costs.
    t_attr = t_save = 0.0

    # Leftovers from a kill during a previous save. They are always incomplete,
    # and leaving them would slowly fill a disk the study is already tight on.
    for stale in output_dir.glob('*' + PARTIAL_SUFFIX):
        print(f"Removing stale partial write: {stale.name}")
        stale.unlink()

    for i, item in enumerate(tqdm(data_list, desc="Generating Graphs")):
        prompt = item['text']
        task_id = item['task_id']
        current_step = task_step_counters.get(task_id, 0)

        graph_name = f'graph_{i}_{task_id}_{current_step}.pt'
        graph_path = output_dir / graph_name

        # --- Already featurized in an earlier chunk ---
        # No metadata: script 4 must only see the current chunk.
        if graph_name in ledger:
            task_step_counters[task_id] = current_step + 1
            continue

        if CHUNK_SIZE > 0 and n_new >= CHUNK_SIZE:
            print(f"\nChunk limit reached ({CHUNK_SIZE} new graphs). Stopping.")
            break

        # --- Skip if graph exists ---
        if graph_path.exists():
            print(f"\nSkipping generation for existing file: {graph_name}")

            metadata = {
                'row_index': i,
                'expr_id': task_id,
                'step_number': current_step,
                'before_after': 'after',
                'graph_path': str(graph_path.resolve()),
                'step_labels': item['label'],
                'original_expression': item['text'],
                'batch_size': BATCH_SIZE,
            }
            append_metadata(metadata)

            n_new += 1
            task_step_counters[task_id] = current_step + 1
            continue

        # --- Generation ---
        if len(prompt) > 550:
            continue

        partial_path = graph_path.with_name(graph_name + PARTIAL_SUFFIX)
        t_start = time.time()
        try:
            graph = attribute(
                prompt=prompt,
                model=model,
                max_n_logits=MAX_N_LOGITS,
                desired_logit_prob=DESIRED_LOGIT_PROB,
                batch_size=BATCH_SIZE,
                max_feature_nodes=MAX_FEATURE_NODES,
                offload=OFFLOAD,
                verbose=VERBOSE
            )
            t_mid = time.time()
            graph.to_pt(partial_path)
            os.replace(partial_path, graph_path)
            t_attr += t_mid - t_start
            t_save += time.time() - t_mid
        except Exception as exc:
            # VRAM headroom is only ~4%, so an OOM on one prompt is plausible.
            # The step counter is advanced either way: numbering must depend
            # only on (i, task_id) or a later retry would rename every
            # following graph of the same task.
            print(f"\nFAILED {graph_name}: {type(exc).__name__}: {exc}")
            with open(FAILED_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(f"{graph_name}\t{type(exc).__name__}\t{exc}\n")
            partial_path.unlink(missing_ok=True)
            torch.cuda.empty_cache()
            n_failed += 1
            task_step_counters[task_id] = current_step + 1
            continue

        metadata = {
            'row_index': i,
            'expr_id': task_id,
            'step_number': current_step,
            'before_after': 'after',
            'graph_path': str(graph_path.resolve()),
            'step_labels': item['label'],
            'original_expression': item['text'],
            'batch_size': BATCH_SIZE,
        }

        append_metadata(metadata)
        n_new += 1
        task_step_counters[task_id] = current_step + 1

    return n_new, n_failed, t_attr, t_save


if __name__ == "__main__":
    import sys

    # Clear metadata file if already exists
    open(METADATA_FILE_PATH, 'w').close()

    data_list = load_data(DATA_FILE_PATH)
    print(f"Already done in earlier chunks: {len(load_ledger())} / {len(data_list)}")

    _t0 = time.time()
    n_new, n_failed, t_attr, t_save = generate_and_save_graphs(data_list, OUTPUT_DIR, model)
    _el = time.time() - _t0

    if n_new:
        print(f"\n[MEASURE] BATCH_SIZE={BATCH_SIZE}: {_el:.1f}s total, "
              f"{_el / n_new:.1f}s/graph over {n_new} graphs")
        print(f"[MEASURE]   attribute: {t_attr / n_new:6.1f}s/graph "
              f"({t_attr / _el * 100:4.1f}%)")
        print(f"[MEASURE]   to_pt    : {t_save / n_new:6.1f}s/graph "
              f"({t_save / _el * 100:4.1f}%)")

    print("\n--- Process Complete ---")
    print(f"Metadata appended to: {METADATA_FILE_PATH.resolve()}")
    print(f"new graphs this chunk: {n_new}, failed: {n_failed}")

    # A sweep where nothing succeeded looks identical to a finished dataset if
    # only n_new is consulted, so a GPU fault would be reported as completion.
    # Exit 3 (the driver's "stop, we are done" sentinel) requires that nothing
    # failed either; otherwise fail loudly and let the driver halt.
    if n_failed:
        print(f"See {FAILED_FILE_PATH.resolve()} for the failures.")
        sys.exit(1)

    if n_new == 0:
        print("Nothing left to generate.")
        sys.exit(3)
