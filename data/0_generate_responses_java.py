import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "google/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

INPUT_FILE = "gemma_mbpp_responses.jsonl"
OUTPUT_FILE = "gemma_mbpp_java_responses.jsonl"

SYSTEM_PROMPT_JAVA = (
    "You are a professional Java developer. "
    "You must output ONLY valid Java code. "
    "Do NOT output Python, comments, explanations, markdown, or any text. "
    "Output only Java code that can compile."
)

def build_prompt_java(text):
    return SYSTEM_PROMPT_JAVA + "\n\n" + text

FENCE_LINE = re.compile(r"^[ \t]*```[a-zA-Z0-9+#_-]*[ \t]*$", re.MULTILINE)


def extract_code(text: str) -> str:
    """Pull the code out of a reply that may carry markdown fences.

    Same bug, and same fix, as 0_generate_responses.py -- see the long comment
    there. The previous version stripped a fence only at the very start and the
    very end of the reply, so every interior ``` survived as a line of "code".
    On the Python run that corrupted 16 of 256 generations and 17% of the
    feature rows. Fences are treated as separators, not delimiters, because this
    model leaves blocks unclosed and emits ``` immediately followed by ```java.

    There is no cheap validity gate for Java here the way ast.parse is for
    Python, so the prose-preamble fallback in the Python version is absent.
    Check the output before featurising it.
    """
    segments = [s.strip("\r\n") for s in FENCE_LINE.split(text)]
    segments = [s for s in segments if s.strip()]
    if not segments:
        return ""

    seen, kept = set(), []
    for s in segments:
        if s not in seen:
            seen.add(s)
            kept.append(s)
    kept = [s for i, s in enumerate(kept)
            if not any(later.startswith(s) for later in kept[i + 1:])]
    return "\n".join(kept)


def clean_output(text: str):
    text = extract_code(text.strip())

    text = re.sub(r'("""|\'\'\')(.*?)\1', '', text, flags=re.DOTALL)
    lines = text.splitlines()
    lines = [l for l in lines if not l.strip().startswith("//")]
    lines = [l for l in lines if l.strip() != ""]
    return "\n".join(lines).strip()

def generate_java(text, max_new_tokens=256):
    prompt = build_prompt_java(text)
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


# 实时写入文件
with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

    for i, line in enumerate(f_in):
        item = json.loads(line)
        text = item["text"]
        text = text.replace("python", "Java").replace("Python", "Java")

        java_code = generate_java(text)

        result = {
            "task_id": item.get("task_id", i),
            "text": text,
            "generated_code": java_code,
        }

        # 写入文件并立即 flush
        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
        f_out.flush()  # 确保立刻写入磁盘
        print(f"Generated example {i} -> written to file")
