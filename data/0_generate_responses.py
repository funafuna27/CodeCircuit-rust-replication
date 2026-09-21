from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ast
import json
import re

model_name = "google/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

dataset = load_dataset("google-research-datasets/mbpp", "sanitized")
data = dataset["test"]   # 你可以改 train / validation

# 固定的 system_prompt：强制模型只输出纯代码
SYSTEM_PROMPT = (
    "You are a Python coding assistant. "
    "You must output ONLY valid Python code. "
    "Do not include comments, markdown, explanations, or text. "
    "Output only code."
)

def build_prompt(example):
    user_prompt = example["prompt"]
    return SYSTEM_PROMPT + "\n\n" + user_prompt


FENCE_LINE = re.compile(r"^[ \t]*```[a-zA-Z0-9+#_-]*[ \t]*$", re.MULTILINE)


def extract_code(text: str) -> str:
    """Pull the code out of a reply that may carry markdown fences.

    The earlier version stripped a fence only at the very start and the very end
    of the reply, so a reply holding more than one fenced block kept every
    interior ``` as a line of "code". 16 of the 256 Python generations came out
    of that path unparseable, and because the stray fences inflate the line
    count they account for 242 of 1420 feature rows -- 17% of the dataset. The
    C++ script already carried a working version of this; it was never brought
    back to Python or Java.

    Fences are treated as SEPARATORS rather than delimiters. Pairing them is
    what breaks: this model leaves blocks unclosed and emits ``` immediately
    followed by ```python, so a paired regex reads the opening fence of the
    real block as the closing fence of an empty one and drops everything
    outside. Splitting is unambiguous no matter how the fencing is malformed.
    """
    # Strip newlines only. A fence can interrupt a single logical block -- task
    # 405 put `def check_element(...):` before the fence and its indented body
    # after it -- so stripping leading whitespace here would destroy the
    # indentation that makes the two halves join back up.
    segments = [s.strip("\r\n") for s in FENCE_LINE.split(text)]
    segments = [s for s in segments if s.strip()]
    if not segments:
        return ""

    # The model sometimes emits the same block several times over.
    seen, kept = set(), []
    for s in segments:
        if s not in seen:
            seen.add(s)
            kept.append(s)

    # It also abandons a block part-way and starts again -- task 304 emitted a
    # bare `def` line, then the same function with a body. Keeping both leaves
    # an empty function definition that will not parse.
    kept = [s for i, s in enumerate(kept)
            if not any(later.startswith(s) for later in kept[i + 1:])]

    code = "\n".join(kept)
    if len(kept) > 1 and not parses(code):
        # A prose preamble ("Here you go:") is the usual cause, and this model
        # never produced one -- but a different model or prompt will. Drop
        # leading segments until what remains parses. Never trim from the end:
        # that is where the answer is.
        for i in range(1, len(kept)):
            candidate = "\n".join(kept[i:])
            if parses(candidate):
                return candidate
    return code


def parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def clean_output(text: str):
    text = extract_code(text.strip())

    text = re.sub(r'("""|\'\'\')(.*?)\1', '', text, flags=re.DOTALL)
    lines = text.splitlines()
    code_lines = [line for line in lines if not line.strip().startswith("#")]
    code_lines = [line for line in code_lines if line.strip() != ""]
    return "\n".join(code_lines).strip()

def run_example(example, max_new_tokens=256):
    prompt = build_prompt(example)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return clean_output(generated)

results = []
unparsable = []
for i, ex in enumerate(data):
    code = run_example(ex)
    ok = parses(code)
    if not ok:
        unparsable.append(ex.get("task_id", i))
    print(f"===== Example {i} ===== {'' if ok else '[DOES NOT PARSE]'}")
    print(code)
    print("\n")
    results.append({
        "task_id": ex.get("task_id", i),
        "text": ex["prompt"],
        "generated_code": code,
        "parses": ok,
    })

with open("gemma_mbpp_responses.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# A generation that does not parse is usually an extraction failure rather than
# a model failure, and it poisons everything downstream: it survives the length
# filter, it inflates the step count, and the graph features predict it at 0.9995
# AUROC -- which looks like signal about correctness and is not. Fail loudly.
print(f"\n{len(results) - len(unparsable)}/{len(results)} generations parse.")
if unparsable:
    print(f"DOES NOT PARSE ({len(unparsable)}): {unparsable}")
    print("Inspect these before running 2_incremental_data.py -- do not featurise them.")

