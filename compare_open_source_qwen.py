"""Colab-ready comparison script: open-source LLM (Qwen2.5-7B-Instruct).

Uses 4-bit loading on GPU when possible for Colab memory efficiency.
"""

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-3B-Instruct"

SCHEMA = {
    "sets": {"nodes": ["n1", "n2"], "products": ["manure", "fertilizer"]},
    "suppliers": [{"id": "s1", "node": "n1", "product": "manure", "bid": -1.0, "cap_max": 1000.0}],
    "consumers": [{"id": "d1", "node": "n2", "product": "fertilizer", "bid": 25.0, "cap_max": 200.0}],
    "transporters": [{"id": "l1", "from_node": "n1", "to_node": "n2", "product": "manure", "bid": 2.0, "cap_max": 900.0}],
    "technologies": [{"id": "t1", "node": "n2", "ref_product": "manure", "bid": 3.0, "cap_max": 800.0, "yields": {"manure": -1.0, "fertilizer": 0.3}}],
    "settings": {"allow_negative_bids": True},
}

PROMPT = """
You are an optimization extraction agent.
Return ONLY valid JSON inside <json>...</json> tags.
Keep LP-only fields and numeric values.

Schema pattern:
[[SCHEMA]]

Problem:
[[PROBLEM]]
""".strip()


def extract_json(raw_text: str):
    txt = (raw_text or "").strip().replace("```json", "```")
    m = re.search(r"<json>([\s\S]*?)</json>", txt, flags=re.IGNORECASE)
    if m:
        txt = m.group(1).strip()
    if "```" in txt:
        parts = [p.strip() for p in txt.split("```") if p.strip()]
        if parts:
            txt = max(parts, key=len)
    i = txt.find("{")
    j = txt.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("No JSON object found in model output")
    return json.loads(txt[i : j + 1])


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
            return tokenizer, model, model_name
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
            return tokenizer, model, model_name

    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model, model_name


def run(problem_text: str, model_name: str):
    try:
        tok, model, loaded_name = load_model(model_name)
    except Exception:
        tok, model, loaded_name = load_model(FALLBACK_MODEL)

    prompt = PROMPT.replace("[[SCHEMA]]", json.dumps(SCHEMA, indent=2)).replace("[[PROBLEM]]", problem_text)

    if hasattr(tok, "apply_chat_template"):
        input_text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt

    inputs = tok(input_text, return_tensors="pt", truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=1200, do_sample=False)

    full = tok.decode(out_ids[0], skip_special_tokens=True)
    raw = full[len(input_text) :].strip() if len(full) > len(input_text) else full
    parsed = extract_json(raw)
    return {"provider": "open_source", "model": loaded_name, "raw": raw, "json": parsed}


def default_problem() -> str:
    return (
        "At node n1, farms supply manure with cap 1000 and bid -1. "
        "Manure is transported to node n2 with cap 900 and bid 2. "
        "A technology at n2 converts manure to fertilizer with cap 800, bid 3, yields manure -1 and fertilizer +0.3. "
        "Demand for fertilizer at n2 is up to 200 with bid 25."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--problem", default=default_problem())
    ap.add_argument("--problem-file", default="")
    args = ap.parse_args()

    problem = args.problem
    if args.problem_file:
        with open(args.problem_file, "r", encoding="utf-8") as f:
            problem = f.read()

    out = run(problem, args.model)
    print("Provider:", out["provider"])
    print("Model:", out["model"])
    print(json.dumps(out["json"], indent=2))


if __name__ == "__main__":
    main()
