"""Generate Rust solutions for the typeable MBPP problems.

Mirrors 0_generate_responses.py -- same model, same decoding, same extraction --
with two deliberate differences, both recorded in the readme's plan:

  1. The required signature goes in the prompt. MBPP does not state the function
     name, and in Rust a name or arity mismatch is a compile error, so without
     this the label would measure naming rather than reasoning.
  2. max_new_tokens is higher, because Rust says the same thing in more
     characters. The token count of every generation is recorded so the length
     filter can be matched on tokens or on drop rate rather than on a character
     threshold that would cut the two languages differently.
"""
import ast
import json
import os
import re
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mbpp_rust import load_typeable

MODEL_NAME = "google/gemma-2-2b-it"
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 384))
OUTPUT_PATH = os.environ.get('OUTPUT', 'gemma_mbpp_rust_responses.jsonl')
LIMIT = int(os.environ.get('LIMIT', 0))          # 0 = every typeable problem

SYSTEM_PROMPT_RUST = (
    "You are a Rust coding assistant. "
    "You must output ONLY valid Rust code. "
    "Do not include comments, markdown, explanations, or text. "
    "Output only code."
)

FENCE_LINE = re.compile(r"^[ \t]*```[a-zA-Z0-9+#_-]*[ \t]*$", re.MULTILINE)


def extract_code(text: str) -> str:
    """Fences are separators, not delimiters -- see 0_generate_responses.py."""
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


def drop_stray_semicolon(code: str) -> str:
    """The prompt ends on a bare signature, and the model sometimes answers it
    with `;` -- completing it as a declaration -- before writing the real
    function. A lone `;` is never a meaningful item, so it is noise from the
    prompt shape rather than anything the model meant. 14 of 350 generations
    failed to compile on this alone.

    An A/B over three prompt endings on 40 problems found no pass-rate
    difference between the bare signature and a wordier version that never
    provokes the `;` (15/40 vs 13/40), and opening the body for the model was
    much worse (5/40), so the prompt is left alone and the artefact removed
    here.
    """
    lines = code.splitlines()
    while lines and lines[0].strip() in (';', ''):
        lines.pop(0)
    return "\n".join(lines)


def trim_to_balanced(code: str) -> str:
    """Cut back to the last point where braces balance.

    The model finishes the function and then starts writing it over again; the
    token cap cuts the restart off mid-way and leaves the whole file unparseable.
    Keeping the complete first attempt is the same rule the Python extractor
    applies to a restarted fenced block. 9 of 350 generations end unbalanced.
    """
    depth, last, in_str, in_chr, esc = 0, None, False, False, False
    for i, ch in enumerate(code):
        if esc:
            esc = False
            continue
        if ch == '\\' and (in_str or in_chr):
            esc = True
            continue
        if ch == '"' and not in_chr:
            in_str = not in_str
            continue
        if ch == "'" and not in_str:
            in_chr = not in_chr
            continue
        if in_str or in_chr:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                last = i
    if depth == 0:
        return code
    return code[:last + 1] if last is not None else code


def clean_output(text: str) -> str:
    text = extract_code(text.strip())
    lines = text.splitlines()
    lines = [l for l in lines if not l.strip().startswith("//")]
    lines = [l for l in lines if l.strip() != ""]
    return trim_to_balanced(drop_stray_semicolon("\n".join(lines))).strip()


def to_rust_prose(prompt: str) -> str:
    """MBPP prompts say 'python'. Say Rust instead, and nothing else."""
    out = re.sub(r'\bpython\b', 'Rust', prompt, flags=re.IGNORECASE)
    return re.sub(r'\ba Rust\b', 'a Rust', out)


def build_prompt(sig: dict) -> str:
    return (
        SYSTEM_PROMPT_RUST + "\n\n"
        + to_rust_prose(sig['prompt']) + "\n\n"
        + "Use exactly this signature:\n" + sig['signature']
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16,
    )

    typeable, rejected = load_typeable()
    task_ids = sorted(typeable)
    if LIMIT:
        task_ids = task_ids[:LIMIT]
    print(f"typeable {len(typeable)} of {len(typeable) + len(rejected)} MBPP tasks; "
          f"generating for {len(task_ids)}")

    results = []
    for n, tid in enumerate(task_ids, 1):
        sig = typeable[tid]
        prompt = build_prompt(sig)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True)
        code = clean_output(raw)

        results.append({
            "task_id": tid,
            "text": to_rust_prose(sig['prompt']),
            "signature": sig['signature'],
            "generated_code": code,
            "n_new_tokens": int(new_ids.shape[0]),
            "truncated": int(new_ids.shape[0]) >= MAX_NEW_TOKENS,
        })
        if n % 25 == 0 or n == len(task_ids):
            print(f"  {n}/{len(task_ids)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_trunc = sum(r['truncated'] for r in results)
    lens = sorted(r['n_new_tokens'] for r in results)
    print(f"\nwrote {len(results)} -> {OUTPUT_PATH}")
    print(f"tokens generated: median {lens[len(lens)//2]}, "
          f"p90 {lens[int(len(lens)*0.9)]}, max {lens[-1]}")
    print(f"hit the {MAX_NEW_TOKENS}-token cap: {n_trunc} "
          f"({n_trunc/len(results)*100:.1f}%)")
    if n_trunc:
        print("A truncated generation cannot compile. Raise MAX_NEW_TOKENS or "
              "exclude these before featurising.")


if __name__ == '__main__':
    main()
