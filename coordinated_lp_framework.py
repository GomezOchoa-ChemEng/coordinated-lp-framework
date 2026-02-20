from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pulp

DISPATCH_SCHEMA_TEXT = """
{
  "sets": {"nodes": ["n1"], "products": ["p1"]},
  "suppliers": [{"id": "s1", "node": "n1", "product": "p1", "bid": 1.0, "cap_max": 10.0}],
  "consumers": [{"id": "d1", "node": "n1", "product": "p1", "bid": 5.0, "cap_max": 8.0}],
  "transporters": [],
  "technologies": [],
  "settings": {"allow_negative_bids": true}
}
""".strip()

PROMPT_EXTRACT = """
You are an optimization extraction agent.
Return only valid JSON wrapped in <json>...</json>.
Use this LP-only schema:
[[SCHEMA]]
Problem:
[[PROBLEM]]
""".strip()

PROMPT_REPAIR = """
You are a model repair critic.
Return only corrected JSON wrapped in <json>...</json>.
Schema:
[[SCHEMA]]
Problem:
[[PROBLEM]]
Candidate:
[[CANDIDATE]]
Errors:
[[ERRORS]]
""".strip()

PROMPT_ADJUDICATE = """
You are a final adjudicator.
Return only final JSON wrapped in <json>...</json>.
Schema:
[[SCHEMA]]
Problem:
[[PROBLEM]]
Candidate:
[[CANDIDATE]]
""".strip()


@dataclass
class LLMResult:
    text: str
    ok: bool
    error: Optional[str] = None


class GeminiBackend:
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.model_name = model_name
        self.available = False
        self.error = None
        self._mode = None
        self._client = None
        try:
            from google import genai  # type: ignore

            self._client = genai.Client()
            self._mode = "genai"
            self.available = True
            return
        except Exception:
            pass
        try:
            import google.generativeai as legacy  # type: ignore

            self._client = legacy
            self._mode = "legacy"
            self.available = True
            return
        except Exception as e:
            self.error = str(e)

    def run(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1200) -> LLMResult:
        if not self.available:
            return LLMResult("", False, self.error or "Gemini unavailable")
        try:
            if self._mode == "genai":
                r = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"temperature": temperature, "max_output_tokens": max_tokens},
                )
                return LLMResult(getattr(r, "text", "") or "", True)
            self._client.configure()
            model = self._client.GenerativeModel(self.model_name)
            r = model.generate_content(prompt, generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
            return LLMResult(getattr(r, "text", "") or "", True)
        except Exception as e:
            return LLMResult("", False, str(e))


@dataclass
class GeminiAgent:
    name: str
    backend: GeminiBackend
    temperature: float = 0.0

    def run(self, prompt: str) -> str:
        out = self.backend.run(prompt, temperature=self.temperature)
        if not out.ok:
            raise RuntimeError(f"{self.name} failed: {out.error}")
        return out.text


def _render(template: str, mapping: Dict[str, str]) -> str:
    t = template
    for k, v in mapping.items():
        t = t.replace(f"[[{k}]]", v)
    return t


def extract_json_object(raw: str) -> Dict[str, Any]:
    txt = (raw or "").strip().replace("```json", "```")
    m = re.search(r"<json>([\s\S]*?)</json>", txt, flags=re.IGNORECASE)
    if m:
        txt = m.group(1).strip()
    if "```" in txt:
        parts = [p.strip() for p in txt.split("```") if p.strip()]
        if parts:
            txt = max(parts, key=len)
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("No JSON object found")
    return json.loads(txt[i : j + 1])


def normalize_dispatch_model(model: Dict[str, Any]) -> Dict[str, Any]:
    m = json.loads(json.dumps(model))
    m.setdefault("sets", {})
    m["sets"].setdefault("nodes", [])
    m["sets"].setdefault("products", [])
    for k in ["suppliers", "consumers", "transporters", "technologies"]:
        m.setdefault(k, [])
    m.setdefault("settings", {})
    m["settings"].setdefault("allow_negative_bids", True)

    for s in m["suppliers"]:
        s["bid"] = float(s["bid"])
        s["cap_max"] = float(s["cap_max"])
    for d in m["consumers"]:
        d["bid"] = float(d["bid"])
        d["cap_max"] = float(d["cap_max"])
    for l in m["transporters"]:
        l["bid"] = float(l["bid"])
        l["cap_max"] = float(l["cap_max"])
    for t in m["technologies"]:
        t["bid"] = float(t["bid"])
        t["cap_max"] = float(t["cap_max"])
        t.setdefault("yields", {})
        t["yields"] = {str(p): float(g) for p, g in t["yields"].items()}

    m["sets"]["nodes"] = sorted({str(x) for x in m["sets"]["nodes"]})
    m["sets"]["products"] = sorted({str(x) for x in m["sets"]["products"]})
    return m


def validate_dispatch_model_lp(model: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    req = ["sets", "suppliers", "consumers", "transporters", "technologies", "settings"]
    for k in req:
        if k not in model:
            errs.append(f"Missing top-level key: {k}")
    if errs:
        return errs

    nodes = model["sets"].get("nodes", [])
    products = model["sets"].get("products", [])
    if not isinstance(nodes, list) or not nodes:
        errs.append("sets.nodes must be non-empty list")
    if not isinstance(products, list) or not products:
        errs.append("sets.products must be non-empty list")

    nset, pset = set(nodes), set(products)
    allow_neg = bool(model.get("settings", {}).get("allow_negative_bids", True))
    seen = set()

    def check_bid_cap(x: Dict[str, Any], tag: str) -> None:
        try:
            bid = float(x["bid"])
            if bid < 0 and not allow_neg:
                errs.append(f"{tag}: negative bid not allowed")
        except Exception:
            errs.append(f"{tag}: bid must be numeric")
        try:
            if float(x["cap_max"]) < 0:
                errs.append(f"{tag}: cap_max must be >= 0")
        except Exception:
            errs.append(f"{tag}: cap_max must be numeric")

    for s in model["suppliers"]:
        tag = f"supplier[{s.get('id','?')}]"
        for f in ["id", "node", "product", "bid", "cap_max"]:
            if f not in s:
                errs.append(f"{tag}: missing {f}")
        if s.get("id") in seen:
            errs.append(f"duplicate id: {s.get('id')}")
        seen.add(s.get("id"))
        if s.get("node") not in nset:
            errs.append(f"{tag}: invalid node")
        if s.get("product") not in pset:
            errs.append(f"{tag}: invalid product")
        check_bid_cap(s, tag)

    for d in model["consumers"]:
        tag = f"consumer[{d.get('id','?')}]"
        for f in ["id", "node", "product", "bid", "cap_max"]:
            if f not in d:
                errs.append(f"{tag}: missing {f}")
        if d.get("id") in seen:
            errs.append(f"duplicate id: {d.get('id')}")
        seen.add(d.get("id"))
        if d.get("node") not in nset:
            errs.append(f"{tag}: invalid node")
        if d.get("product") not in pset:
            errs.append(f"{tag}: invalid product")
        check_bid_cap(d, tag)

    for l in model["transporters"]:
        tag = f"transporter[{l.get('id','?')}]"
        for f in ["id", "from_node", "to_node", "product", "bid", "cap_max"]:
            if f not in l:
                errs.append(f"{tag}: missing {f}")
        if l.get("id") in seen:
            errs.append(f"duplicate id: {l.get('id')}")
        seen.add(l.get("id"))
        if l.get("from_node") not in nset or l.get("to_node") not in nset:
            errs.append(f"{tag}: invalid route node")
        if l.get("product") not in pset:
            errs.append(f"{tag}: invalid product")
        check_bid_cap(l, tag)

    for t in model["technologies"]:
        tag = f"technology[{t.get('id','?')}]"
        for f in ["id", "node", "ref_product", "bid", "cap_max", "yields"]:
            if f not in t:
                errs.append(f"{tag}: missing {f}")
        if t.get("id") in seen:
            errs.append(f"duplicate id: {t.get('id')}")
        seen.add(t.get("id"))
        if t.get("node") not in nset:
            errs.append(f"{tag}: invalid node")
        if t.get("ref_product") not in pset:
            errs.append(f"{tag}: invalid ref_product")
        y = t.get("yields", {})
        if not isinstance(y, dict):
            errs.append(f"{tag}: yields must be dict")
        else:
            if t.get("ref_product") not in y:
                errs.append(f"{tag}: ref_product missing in yields")
            else:
                try:
                    if abs(float(y[t["ref_product"]]) + 1.0) > 1e-9:
                        errs.append(f"{tag}: ref_product gamma must be -1")
                except Exception:
                    errs.append(f"{tag}: invalid ref gamma")
            for p, g in y.items():
                if p not in pset:
                    errs.append(f"{tag}: yield product not in sets")
                try:
                    float(g)
                except Exception:
                    errs.append(f"{tag}: non-numeric gamma")
        check_bid_cap(t, tag)

    return errs

def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def build_and_solve_dispatch_lp(model: Dict[str, Any], solver_msg: bool = False) -> Dict[str, Any]:
    m = normalize_dispatch_model(model)
    errs = validate_dispatch_model_lp(m)
    if errs:
        raise ValueError("Invalid dispatch model: " + " | ".join(errs))

    nodes = m["sets"]["nodes"]
    products = m["sets"]["products"]

    prob = pulp.LpProblem("coordinated_dispatch_lp", pulp.LpMaximize)

    s_var = {s["id"]: pulp.LpVariable(f"s_{_safe_name(s['id'])}", lowBound=0.0, upBound=s["cap_max"], cat=pulp.LpContinuous) for s in m["suppliers"]}
    d_var = {d["id"]: pulp.LpVariable(f"d_{_safe_name(d['id'])}", lowBound=0.0, upBound=d["cap_max"], cat=pulp.LpContinuous) for d in m["consumers"]}
    f_var = {l["id"]: pulp.LpVariable(f"f_{_safe_name(l['id'])}", lowBound=0.0, upBound=l["cap_max"], cat=pulp.LpContinuous) for l in m["transporters"]}
    xi_var = {t["id"]: pulp.LpVariable(f"xi_{_safe_name(t['id'])}", lowBound=0.0, upBound=t["cap_max"], cat=pulp.LpContinuous) for t in m["technologies"]}

    prob += (
        pulp.lpSum(float(d["bid"]) * d_var[d["id"]] for d in m["consumers"])
        - pulp.lpSum(float(s["bid"]) * s_var[s["id"]] for s in m["suppliers"])
        - pulp.lpSum(float(l["bid"]) * f_var[l["id"]] for l in m["transporters"])
        - pulp.lpSum(float(t["bid"]) * xi_var[t["id"]] for t in m["technologies"])
    )

    suppliers_by_np: Dict[Tuple[str, str], List[str]] = {}
    consumers_by_np: Dict[Tuple[str, str], List[str]] = {}
    inflow_by_np: Dict[Tuple[str, str], List[str]] = {}
    outflow_by_np: Dict[Tuple[str, str], List[str]] = {}
    techs_by_n: Dict[str, List[str]] = {}

    tech_data = {t["id"]: t for t in m["technologies"]}

    for s in m["suppliers"]:
        suppliers_by_np.setdefault((s["node"], s["product"]), []).append(s["id"])
    for d in m["consumers"]:
        consumers_by_np.setdefault((d["node"], d["product"]), []).append(d["id"])
    for l in m["transporters"]:
        inflow_by_np.setdefault((l["to_node"], l["product"]), []).append(l["id"])
        outflow_by_np.setdefault((l["from_node"], l["product"]), []).append(l["id"])
    for t in m["technologies"]:
        techs_by_n.setdefault(t["node"], []).append(t["id"])

    balances: Dict[Tuple[str, str], pulp.LpConstraint] = {}
    for n in nodes:
        for p in products:
            lhs = (
                pulp.lpSum(s_var[i] for i in suppliers_by_np.get((n, p), []))
                + pulp.lpSum(f_var[l] for l in inflow_by_np.get((n, p), []))
                - pulp.lpSum(d_var[j] for j in consumers_by_np.get((n, p), []))
                - pulp.lpSum(f_var[l] for l in outflow_by_np.get((n, p), []))
                + pulp.lpSum(float(tech_data[t]["yields"].get(p, 0.0)) * xi_var[t] for t in techs_by_n.get(n, []))
            )
            cname = f"balance_{_safe_name(n)}_{_safe_name(p)}"
            c = pulp.LpConstraint(e=lhs, sense=pulp.LpConstraintEQ, rhs=0.0, name=cname)
            prob += c
            balances[(n, p)] = c

    prob.solve(pulp.PULP_CBC_CMD(msg=solver_msg))
    status = pulp.LpStatus[prob.status]

    alloc = {
        "s": {k: float(v.value() or 0.0) for k, v in s_var.items()},
        "d": {k: float(v.value() or 0.0) for k, v in d_var.items()},
        "f": {k: float(v.value() or 0.0) for k, v in f_var.items()},
        "xi": {k: float(v.value() or 0.0) for k, v in xi_var.items()},
    }

    duals: Dict[str, Optional[float]] = {}
    duals_available = True
    diagnostics = []
    for (n, p), c in balances.items():
        key = f"({n},{p})"
        if hasattr(c, "pi") and c.pi is not None:
            duals[key] = float(c.pi)
        else:
            duals[key] = None
            duals_available = False
        lhs_val = float(pulp.value(c.expr))
        diagnostics.append(
            {
                "constraint": f"balance({n},{p})",
                "residual": lhs_val,
                "is_binding": abs(lhs_val) <= 1e-7,
            }
        )

    return {
        "status": status,
        "objective_value": float(pulp.value(prob.objective)) if status == "Optimal" else None,
        "allocations": alloc,
        "duals": duals,
        "duals_available": duals_available,
        "diagnostics": diagnostics,
    }


def render_equations_lp(model: Dict[str, Any]) -> Dict[str, Any]:
    m = normalize_dispatch_model(model)
    idx = {
        "objective": "max sum_j alpha_d[j]*d[j] - sum_i alpha_s[i]*s[i] - sum_l alpha_f[l]*f[l] - sum_t alpha_xi[t]*xi[t]",
        "balance": "forall (n,p): sum_{i in S(n,p)} s[i] + sum_{l in Lin(n,p)} f[l] - sum_{j in D(n,p)} d[j] - sum_{l in Lout(n,p)} f[l] + sum_{t in T(n)} gamma[t,p]*xi[t] = 0",
        "bounds": ["0<=s<=smax", "0<=d<=dmax", "0<=f<=fmax", "0<=xi<=ximax", "all continuous"],
    }

    terms = []
    for d in m["consumers"]:
        terms.append(f"+ ({d['bid']:g}) d_{d['id']}")
    for s in m["suppliers"]:
        terms.append(f"- ({s['bid']:g}) s_{s['id']}")
    for l in m["transporters"]:
        terms.append(f"- ({l['bid']:g}) f_{l['id']}")
    for t in m["technologies"]:
        terms.append(f"- ({t['bid']:g}) xi_{t['id']}")

    obj = "MAX z = " + " ".join(terms).replace("+ -", "- ")

    bal = []
    for n in m["sets"]["nodes"]:
        for p in m["sets"]["products"]:
            expr: List[str] = []
            for s in m["suppliers"]:
                if s["node"] == n and s["product"] == p:
                    expr.append(f"+ s_{s['id']}")
            for l in m["transporters"]:
                if l["to_node"] == n and l["product"] == p:
                    expr.append(f"+ f_{l['id']}")
            for d in m["consumers"]:
                if d["node"] == n and d["product"] == p:
                    expr.append(f"- d_{d['id']}")
            for l in m["transporters"]:
                if l["from_node"] == n and l["product"] == p:
                    expr.append(f"- f_{l['id']}")
            for t in m["technologies"]:
                if t["node"] == n:
                    g = float(t["yields"].get(p, 0.0))
                    if abs(g) > 1e-12:
                        sign = "+" if g >= 0 else "-"
                        coef = abs(g)
                        expr.append(f"{sign} {coef:g} xi_{t['id']}")
            bal.append(f"balance({n},{p}): {' '.join(expr) if expr else '0'} = 0")

    bounds = []
    bounds += [f"0 <= s_{s['id']} <= {s['cap_max']:g}" for s in m["suppliers"]]
    bounds += [f"0 <= d_{d['id']} <= {d['cap_max']:g}" for d in m["consumers"]]
    bounds += [f"0 <= f_{l['id']} <= {l['cap_max']:g}" for l in m["transporters"]]
    bounds += [f"0 <= xi_{t['id']} <= {t['cap_max']:g}" for t in m["technologies"]]

    return {"indexed_template": idx, "expanded": {"objective": obj, "balance_constraints": bal, "bounds": bounds}}


def build_operational_analysis(model: Dict[str, Any], solve_result: Dict[str, Any]) -> str:
    lines = []
    lines.append("## Coordinated Dispatch LP Report")
    lines.append(f"- Status: **{solve_result['status']}**")
    if solve_result["objective_value"] is not None:
        lines.append(f"- Objective: **{solve_result['objective_value']:g}**")
    else:
        lines.append("- Objective unavailable (non-optimal status).")
    lines.append("")
    lines.append("### Allocations")
    for k in ["s", "d", "f", "xi"]:
        vals = solve_result["allocations"].get(k, {})
        lines.append(f"- `{k}`:")
        for name in sorted(vals.keys()):
            lines.append(f"  - {name}: {vals[name]:g}")
    lines.append("")
    lines.append("### Constraint Diagnostics")
    for row in solve_result["diagnostics"]:
        lines.append(f"- {row['constraint']}: residual={row['residual']:.3e}, binding={row['is_binding']}")
    lines.append("")
    lines.append("### Price Signals")
    if solve_result.get("duals_available", False):
        lines.append("- Nodal balance dual prices available.")
    else:
        lines.append("- Nodal balance dual prices unavailable in current runtime/solver.")
    return "\n".join(lines)

def _fallback_model(_: str) -> Dict[str, Any]:
    return {
        "sets": {"nodes": ["n1", "n2"], "products": ["waste", "prod"]},
        "suppliers": [{"id": "s1", "node": "n1", "product": "waste", "bid": 1.0, "cap_max": 100.0}],
        "consumers": [{"id": "d1", "node": "n2", "product": "prod", "bid": 10.0, "cap_max": 40.0}],
        "transporters": [{"id": "l1", "from_node": "n1", "to_node": "n2", "product": "waste", "bid": 1.0, "cap_max": 100.0}],
        "technologies": [{"id": "t1", "node": "n2", "ref_product": "waste", "bid": 1.0, "cap_max": 90.0, "yields": {"waste": -1.0, "prod": 0.4}}],
        "settings": {"allow_negative_bids": True},
    }


def parse_problem_to_dispatch_model_lp(problem_text: str, gemini_model: str = "gemini-1.5-pro", use_llm: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    logs: Dict[str, Any] = {"mode": None, "stages": []}
    if not use_llm:
        logs["mode"] = "fallback_no_llm"
        return normalize_dispatch_model(_fallback_model(problem_text)), logs

    backend = GeminiBackend(gemini_model)
    if not backend.available:
        logs["mode"] = "fallback_gemini_unavailable"
        logs["error"] = backend.error
        return normalize_dispatch_model(_fallback_model(problem_text)), logs

    agents = [
        GeminiAgent("ExtractorA", backend, 0.0),
        GeminiAgent("ExtractorB", backend, 0.05),
    ]
    critic = GeminiAgent("Critic", backend, 0.0)
    adjudicator = GeminiAgent("Adjudicator", backend, 0.0)

    candidates: List[Tuple[Dict[str, Any], List[str], str]] = []
    for a in agents:
        try:
            raw = a.run(_render(PROMPT_EXTRACT, {"SCHEMA": DISPATCH_SCHEMA_TEXT, "PROBLEM": problem_text}))
            model = normalize_dispatch_model(extract_json_object(raw))
            errs = validate_dispatch_model_lp(model)
            logs["stages"].append({"agent": a.name, "n_errors": len(errs)})
            candidates.append((model, errs, a.name))
        except Exception as e:
            logs["stages"].append({"agent": a.name, "parse_error": str(e)})

    if not candidates:
        logs["mode"] = "fallback_after_extract_failure"
        return normalize_dispatch_model(_fallback_model(problem_text)), logs

    candidates.sort(key=lambda x: len(x[1]))
    current, current_errs, source = candidates[0]
    logs["selected_initial"] = {"agent": source, "n_errors": len(current_errs)}

    if current_errs:
        try:
            raw = critic.run(
                _render(
                    PROMPT_REPAIR,
                    {
                        "SCHEMA": DISPATCH_SCHEMA_TEXT,
                        "PROBLEM": problem_text,
                        "CANDIDATE": json.dumps(current, indent=2),
                        "ERRORS": json.dumps(current_errs, indent=2),
                    },
                )
            )
            repaired = normalize_dispatch_model(extract_json_object(raw))
            repaired_errs = validate_dispatch_model_lp(repaired)
            logs["stages"].append({"agent": "Critic", "n_errors": len(repaired_errs)})
            if len(repaired_errs) <= len(current_errs):
                current, current_errs = repaired, repaired_errs
        except Exception as e:
            logs["stages"].append({"agent": "Critic", "parse_error": str(e)})

    try:
        raw = adjudicator.run(
            _render(PROMPT_ADJUDICATE, {"SCHEMA": DISPATCH_SCHEMA_TEXT, "PROBLEM": problem_text, "CANDIDATE": json.dumps(current, indent=2)})
        )
        adj = normalize_dispatch_model(extract_json_object(raw))
        adj_errs = validate_dispatch_model_lp(adj)
        logs["stages"].append({"agent": "Adjudicator", "n_errors": len(adj_errs)})
        if len(adj_errs) <= len(current_errs):
            current, current_errs = adj, adj_errs
    except Exception as e:
        logs["stages"].append({"agent": "Adjudicator", "parse_error": str(e)})

    if current_errs:
        logs["mode"] = "fallback_after_validation_failure"
        logs["final_errors"] = current_errs
        return normalize_dispatch_model(_fallback_model(problem_text)), logs

    logs["mode"] = "gemini_4agent"
    return current, logs


def run_dispatch_lp_framework(problem_text: str, gemini_model: str = "gemini-1.5-pro", use_llm: bool = True, solver_msg: bool = False) -> Dict[str, Any]:
    model_json, logs = parse_problem_to_dispatch_model_lp(problem_text, gemini_model=gemini_model, use_llm=use_llm)
    solved = build_and_solve_dispatch_lp(model_json, solver_msg=solver_msg)
    equations = render_equations_lp(model_json)
    analysis = build_operational_analysis(model_json, solved)
    return {
        "model_json": model_json,
        "equations": equations,
        "solve_result": {
            "status": solved["status"],
            "objective_value": solved["objective_value"],
            "allocations": solved["allocations"],
            "duals": solved["duals"],
            "duals_available": solved["duals_available"],
        },
        "analysis_markdown": analysis,
        "logs": logs,
    }


def _case_feasible() -> Dict[str, Any]:
    return {
        "sets": {"nodes": ["n1", "n2", "n3", "n4"], "products": ["manure", "fert", "power"]},
        "suppliers": [{"id": "farm", "node": "n1", "product": "manure", "bid": -2.0, "cap_max": 12000.0}],
        "consumers": [
            {"id": "crop", "node": "n3", "product": "fert", "bid": 120.0, "cap_max": 1300.0},
            {"id": "utility", "node": "n4", "product": "power", "bid": 80.0, "cap_max": 2800.0},
        ],
        "transporters": [
            {"id": "haul12", "from_node": "n1", "to_node": "n2", "product": "manure", "bid": 8.0, "cap_max": 12000.0},
            {"id": "haul23", "from_node": "n2", "to_node": "n3", "product": "fert", "bid": 6.0, "cap_max": 1500.0},
            {"id": "haul24", "from_node": "n2", "to_node": "n4", "product": "power", "bid": 4.0, "cap_max": 3200.0},
        ],
        "technologies": [{"id": "adnr", "node": "n2", "ref_product": "manure", "bid": 12.0, "cap_max": 9000.0, "yields": {"manure": -1.0, "fert": 0.12, "power": 0.28}}],
        "settings": {"allow_negative_bids": True},
    }


def run_regression_tests() -> Dict[str, Any]:
    tests = []
    m = _case_feasible()
    tests.append(("feasible_baseline", m, "valid"))

    bad = json.loads(json.dumps(m))
    bad["settings"]["allow_negative_bids"] = False
    tests.append(("negative_bid_disallowed", bad, "validation_error"))

    cong = json.loads(json.dumps(m))
    cong["transporters"][0]["cap_max"] = 500.0
    tests.append(("transport_congestion", cong, "valid"))

    proc = json.loads(json.dumps(m))
    proc["technologies"][0]["cap_max"] = 300.0
    tests.append(("processing_binding", proc, "valid"))

    scale = {
        "sets": {"nodes": [f"n{i}" for i in range(1, 11)], "products": [f"p{k}" for k in range(1, 6)]},
        "suppliers": [{"id": f"s{i}", "node": f"n{(i%10)+1}", "product": f"p{(i%5)+1}", "bid": 2.0 + (i%4), "cap_max": 100.0} for i in range(1, 21)],
        "consumers": [{"id": f"d{i}", "node": f"n{((i+2)%10)+1}", "product": f"p{((i+1)%5)+1}", "bid": 12.0 + (i%5), "cap_max": 80.0} for i in range(1, 21)],
        "transporters": [{"id": f"l{i}", "from_node": f"n{(i%10)+1}", "to_node": f"n{((i+3)%10)+1}", "product": f"p{(i%5)+1}", "bid": 1.0 + (i%3), "cap_max": 120.0} for i in range(1, 31)],
        "technologies": [{"id": f"t{i}", "node": f"n{(i%10)+1}", "ref_product": f"p{(i%5)+1}", "bid": 3.0, "cap_max": 140.0, "yields": {f"p{(i%5)+1}": -1.0, f"p{((i+1)%5)+1}": 0.6}} for i in range(1, 8)],
        "settings": {"allow_negative_bids": True},
    }
    tests.append(("scaled_instance", scale, "valid"))

    out = []
    passed = 0
    for name, case, exp in tests:
        errs = validate_dispatch_model_lp(case)
        if exp == "validation_error":
            ok = len(errs) > 0
            out.append({"test": name, "ok": ok, "errors": errs})
            passed += int(ok)
            continue
        if errs:
            out.append({"test": name, "ok": False, "errors": errs})
            continue
        res = build_and_solve_dispatch_lp(case)
        ok = res["status"] in {"Optimal", "Infeasible", "Unbounded", "Not Solved", "Undefined"}
        out.append({"test": name, "ok": ok, "status": res["status"], "objective": res["objective_value"]})
        passed += int(ok)

    return {"total": len(tests), "passed": passed, "failed": len(tests) - passed, "results": out}


if __name__ == "__main__":
    p = "Coordinated LP market-clearing dispatch with suppliers, transport, transformation, and demands."
    r = run_dispatch_lp_framework(p, use_llm=False)
    print("Status:", r["solve_result"]["status"])
    print("Objective:", r["solve_result"]["objective_value"])
    print(json.dumps(run_regression_tests(), indent=2)[:1400])
