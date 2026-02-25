"""Colab-ready comparison script: Gemini cheap model (gemini-2.5-flash-lite)."""

import argparse
import json
import os
import re

from google import genai

DEFAULT_MODEL = "gemini-2.5-flash-lite"

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


def run(problem_text: str, model: str):
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY in environment or Colab secrets")

    client = genai.Client()
    prompt = PROMPT.replace("[[SCHEMA]]", json.dumps(SCHEMA, indent=2)).replace("[[PROBLEM]]", problem_text)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": 0.0, "max_output_tokens": 1200},
    )
    raw = getattr(response, "text", "") or ""
    parsed = extract_json(raw)
    return {"provider": "gemini", "model": model, "raw": raw, "json": parsed}


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
