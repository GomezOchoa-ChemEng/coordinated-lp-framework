# Coordinated LP Framework

LP-only, size-agnostic framework for coordinated waste supply chain market clearing.

## What this project does
- Parses prose optimization problems into a structured LP dispatch schema.
- Solves a social-welfare LP with:
  - suppliers
  - consumers
  - transport providers
  - transformation providers
- Generates:
  - model equations (indexed + expanded)
  - solution allocations
  - operational analysis report
  - optional nodal dual prices (if solver/runtime exposes them)

## Model scope
This project is strictly **linear programming (LP)**:
- Continuous nonnegative decision variables.
- No binary/integer variables.
- No nonlinear constraints/objectives.

## Main file
- `coordinated_lp_framework.py`

## Installation
```bash
pip install -r requirements.txt
```

## Quick use
```python
from coordinated_lp_framework import run_dispatch_lp_framework

problem_text = """
Coordinated regional waste market with manure supply at n1,
processing at n2, fertilizer demand at n3, and power demand at n4.
"""

out = run_dispatch_lp_framework(problem_text, use_llm=False)
print(out["solve_result"]["status"])
print(out["solve_result"]["objective_value"])
print(out["equations"]["expanded"]["objective"])
print(out["analysis_markdown"])
```

## Run built-in tests
```bash
python coordinated_lp_framework.py
```
This runs a smoke test and a small LP regression suite.

## Optional Gemini usage
To use the 4-agent Gemini pipeline (ExtractorA/B, Critic, Adjudicator), install Google Gemini client libraries and configure your API credentials in your environment.

Example:
```python
out = run_dispatch_lp_framework(problem_text, gemini_model="gemini-1.5-pro", use_llm=True)
```

## Output structure
`run_dispatch_lp_framework(...)` returns:
- `model_json`
- `equations`
- `solve_result`
- `analysis_markdown`
- `logs`

## Notes
- Negative bids are supported when `settings.allow_negative_bids = true`.
- If duals are unavailable in the active solver runtime, the framework still returns valid primal allocations and analysis.
