import os

# Must precede the torch import: the allocator reads this when CUDA starts.
# The card sits at ~96% VRAM with a different prompt shape every graph, and
# letting the allocator collect early rather than fragment cut a three-graph
# chunk from 1389s to 764s. expandable_segments is faster still but trips an
# allocator INTERNAL ASSERT here, taking the rest of the run with it.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8')

from pathlib import Path
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
EMPTY_CACHE = os.environ.get('EMPTY_CACHE', '1') != '0'

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

# Graphs whose attribution raised, one line per attempt. This file is READ at
# startup, not just written: it is the only thing that distinguishes a
# transient fault (a fresh process fixes it) from a prompt that always fails.
FAILED_FILE_PATH = OUTPUT_DIR / 'failed_graphs.tsv'

# Give up on a single graph after this many recorded attempts and move past it.
# Without this the run cannot make progress at all: a deterministic failure
# breaks the chunk, the retry rebuilds identical state, reaches the same row and
# dies the same way, and after run_chunked.sh exhausts its retries the whole
# chunk -- ~80GB and ~14 GPU-hours -- is left neither featurized nor ledgered.
MAX_GRAPH_ATTEMPTS = int(os.environ.get('MAX_GRAPH_ATTEMPTS', 3))

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


def load_failure_counts() -> dict:
    """How many times each graph has already raised."""
    counts = {}
    if not FAILED_FILE_PATH.exists():
        return counts
    with open(FAILED_FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.split('\t', 1)[0].strip()
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def append_metadata(metadata: dict):
    """Append one metadata entry to the jsonl file."""
    with open(METADATA_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(metadata) + "\n")


def generate_and_save_graphs(data_list: list, output_dir: Path, model: ReplacementModel):
    task_step_counters = {}
    ledger = load_ledger()
    failure_counts = load_failure_counts()
    n_new = 0        # graphs entering this chunk's metadata, generated or not
    n_generated = 0  # graphs this process actually computed
    n_failed = 0
    given_up = []
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

        # --- Tried and failed too many times already ---
        # A fresh process repairs a corrupted CUDA context, so the first
        # failures are worth retrying. One that survives that many restarts is
        # a property of the prompt, and retrying it forever would stall the run
        # instead of losing one sample.
        if failure_counts.get(graph_name, 0) >= MAX_GRAPH_ATTEMPTS:
            given_up.append(graph_name)
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
        # Every prompt is a different shape and the card runs at ~96% VRAM, so
        # blocks cached for the previous graph rarely fit the next one. Handing
        # them back keeps the allocator from falling into cudaMalloc/cudaFree,
        # which synchronises the device. Set EMPTY_CACHE=0 to compare.
        if EMPTY_CACHE:
            torch.cuda.empty_cache()
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
            # os.replace is atomic for the NAME, not for the ~400MB behind it.
            # Without these fsyncs a host-level crash (a Windows Update reboot
            # takes the whole WSL2 VM down) can leave the final name present
            # with its data still only a page-cache promise -- and the resume
            # check is bare exists(), so that torn file would be accepted
            # forever and script 4 would fail on it every single chunk.
            fd = os.open(partial_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(partial_path, graph_path)
            dir_fd = os.open(output_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            t_attr += t_mid - t_start
            t_save += time.time() - t_mid
        except Exception as exc:
            # Stop the whole chunk, do not try the next prompt. attribute()
            # reloads its offloaded modules in a finally block, but each
            # reload_handle is a bare module.to('cuda'): once the CUDA context
            # is unhappy the first one raises and the rest never run, leaving
            # the transcoders stranded on the CPU. Every later prompt then dies
            # with "indices should be either on cpu or on the same device".
            # Observed for real: one allocator fault at graph 8 was followed by
            # 1412 identical failures. A fresh process is the only reliable
            # repair, so exit and let run_chunked.sh retry. The attempt is
            # recorded so that after MAX_GRAPH_ATTEMPTS restarts this graph is
            # skipped instead of stalling the run forever.
            attempts = failure_counts.get(graph_name, 0) + 1
            print(f"\nFAILED {graph_name} (attempt {attempts}/{MAX_GRAPH_ATTEMPTS}): "
                  f"{type(exc).__name__}: {exc}")
            with open(FAILED_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(f"{graph_name}\t{type(exc).__name__}\t{exc}\n")
                f.flush()
                os.fsync(f.fileno())
            partial_path.unlink(missing_ok=True)
            n_failed += 1
            break

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
        n_generated += 1
        task_step_counters[task_id] = current_step + 1

    if given_up:
        print(f"\nGIVEN UP on {len(given_up)} graph(s) after {MAX_GRAPH_ATTEMPTS} "
              f"attempts each; they will be missing from the dataset:")
        for name in given_up:
            print(f"  {name}")

    return n_new, n_generated, n_failed, t_attr, t_save, given_up


if __name__ == "__main__":
    import sys

    # Clear metadata file if already exists
    open(METADATA_FILE_PATH, 'w').close()

    data_list = load_data(DATA_FILE_PATH)
    # Rows over the length filter are never generated, so counting against the
    # raw row count would make a finished run read as 27 graphs short -- which
    # is exactly how a genuinely lossy run looks.
    n_target = sum(1 for r in data_list if len(r['text']) <= 550)
    print(f"Already done in earlier chunks: {len(load_ledger())} / {n_target} "
          f"({len(data_list) - n_target} rows are over the length filter)")

    _t0 = time.time()
    n_new, n_generated, n_failed, t_attr, t_save, given_up = generate_and_save_graphs(
        data_list, OUTPUT_DIR, model)
    _el = time.time() - _t0

    # Divide by what this process actually computed. n_new also counts graphs
    # that were already on disk from an interrupted pass, and those did no
    # work -- averaging them in understates the per-graph cost, which is the
    # number BATCH_SIZE was chosen on.
    if n_generated:
        print(f"\n[MEASURE] BATCH_SIZE={BATCH_SIZE}: {_el:.1f}s total, "
              f"{n_generated} generated ({n_new - n_generated} already on disk)")
        print(f"[MEASURE]   attribute: {t_attr / n_generated:6.1f}s/graph")
        print(f"[MEASURE]   to_pt    : {t_save / n_generated:6.1f}s/graph")

    print("\n--- Process Complete ---")
    print(f"Metadata appended to: {METADATA_FILE_PATH.resolve()}")
    print(f"new graphs this chunk: {n_new}, failed: {n_failed}, "
          f"given up: {len(given_up)}")

    # A sweep where nothing succeeded looks identical to a finished dataset if
    # only n_new is consulted, so a GPU fault would be reported as completion.
    # Exit 3 (the driver's "stop, we are done" sentinel) requires that nothing
    # failed either; otherwise fail loudly and let the driver halt.
    if n_failed:
        print(f"See {FAILED_FILE_PATH.resolve()} for the failures.")
        sys.exit(1)

    if n_new == 0:
        if given_up:
            # Exit 3 would have the driver print ALL CHUNKS COMPLETE and finish
            # with status 0 over a dataset that is quietly short, which is the
            # one outcome this pipeline must never produce. A separate code
            # lets the driver finish the work and still report the loss.
            print(f"\nNothing left to generate, but {len(given_up)} graph(s) were "
                  f"permanently skipped. The dataset is INCOMPLETE.")
            sys.exit(4)
        print("Nothing left to generate.")
        sys.exit(3)
