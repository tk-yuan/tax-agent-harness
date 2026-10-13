"""税筹域工具：把毕设的税负模型与政策库包成 Agent 可调用的工具。

复用的是毕设里**已经跑通、且写在论文里**的模型：
  tax_model.simulate()   —— 8 种政策机制的企业所得税模拟
  methods.optimize()     —— 全部合法处理组合的穷举最优（T3）
  methods.baseline_*()   —— 5 种对照方法（人工经验法、贪心、局部搜索、随机搜索）

工具只做「翻译层」：把模型接口包成 JSON Schema，让模型能调。
计算逻辑一行没改——这也是为什么这些数字可信：它们和你论文里的结果同源。

工具返回都给**精简视图**：算例有 292 个、每个含 10 年财务数据，
原样塞进上下文会立刻撑爆。这里只返回判断所需的关键字段，
完整数据留给计算工具自己去读。这是上下文工程的基本功。
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

from ..config import CACHE_DIR, DATA_DIR
from ..runtime.tools import Tool, ToolRegistry
from .retrieval import PolicyIndex

# 毕设的模块是扁平导入（methods.py 里 `from tax_model import ...`），
# 所以 tools 目录本身要在 sys.path 上
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import methods  # noqa: E402
import tax_model  # noqa: E402


# —— 数据加载（只读一次）——
@lru_cache(maxsize=1)
def cases() -> list[dict]:
    return json.loads((DATA_DIR / "cases_real.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def cases_by_id() -> dict[str, dict]:
    return {c["id"]: c for c in cases()}


@lru_cache(maxsize=1)
def policy_library() -> list[dict]:
    return json.loads((DATA_DIR / "policy_library.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def policy_index() -> PolicyIndex:
    return PolicyIndex(DATA_DIR / "corpus.jsonl", CACHE_DIR / "policy_index")


# —— 工具实现 ——
def policy_search(query: str, k: int = 3) -> dict:
    """语义检索政策原文。"""
    hits = policy_index().search(query, k=k)
    return {
        "query": query,
        "results": [
            {
                "rank": h["rank"],
                "doc_no": h["doc_no"],
                "title": h["title"],
                "issued": h["issued"],
                "valid_to": h["valid_to"] or "未标注",
                "score": h["score"],
                "text": h["text"][:600],
            }
            for h in hits
        ],
    }


def list_policies(keyword: str = "", mechanic: str = "", limit: int = 15) -> dict:
    """按关键词/机制检索政策库（结构化字段，比语义检索更精确）。"""
    out = []
    for p in policy_library():
        blob = json.dumps(p, ensure_ascii=False)
        if keyword and keyword not in blob:
            continue
        if mechanic and mechanic not in str(p.get("mechanic", "")):
            continue
        out.append(
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "issue_no": p.get("issue_no"),
                "category": p.get("category"),
                "mechanic": p.get("mechanic"),
                "rate_or_factor": p.get("rate_or_factor"),
                "valid_from": p.get("valid_from"),
                "valid_to": p.get("valid_to"),
                "validity_type": p.get("validity_type"),
            }
        )
        if len(out) >= limit:
            break
    return {"count": len(out), "policies": out}


def case_lookup(industry: str = "", size_group: str = "", limit: int = 8) -> dict:
    """按行业/规模查找真实企业算例（292 个）。返回精简视图。"""
    out = []
    for c in cases():
        if industry and industry not in str(c.get("industry", "")):
            continue
        if size_group and size_group != c.get("group"):
            continue
        out.append(
            {
                "case_id": c["id"],
                "company": c.get("company_name", ""),
                "industry": str(c.get("industry", ""))[:40],
                "employees": c.get("employees"),
                "total_assets_yuan": c.get("total_assets_yuan"),
                "rate_tier": c.get("rate_tier"),
                "assets_count": len(c.get("assets", [])),
                "years": f"{min(c.get('years', [0]))}-{max(c.get('years', [0]))}",
            }
        )
        if len(out) >= limit:
            break
    return {"count": len(out), "cases": out, "total_cases": len(cases())}


def get_case(case_id: str) -> dict:
    """取某个算例的完整输入（资产清单 + 逐年财务数据）。"""
    c = cases_by_id().get(case_id)
    if c is None:
        similar = [k for k in cases_by_id() if case_id[:6] in k][:5]
        return {"error": f"未找到算例 {case_id}", "maybe": similar}
    return {
        "case_id": c["id"],
        "company": c.get("company_name", ""),
        "employees": c.get("employees"),
        "total_assets_yuan": c.get("total_assets_yuan"),
        "rate_tier": c.get("rate_tier"),
        "years": c["years"],
        "assets": c["assets"],
        "profit_before_deductions_yuan": c["profit_before_deductions_yuan"],
        "r_and_d_expense_yuan": c["r_and_d_expense_yuan"],
    }


def tax_calc(case_id: str, treatments: dict | None = None, mode: str = "current") -> dict:
    """按给定的资产处理方式，模拟各年应纳税额。

    treatments 形如 {"A1": "one_time", "A2": "straight"}；
    不传则默认全部直线折旧 + 申报研发加计扣除（即「人工经验法」）。
    mode="current" 只允许使用有效期内的政策；"stale" 忽略有效期（模拟规则库未更新）。
    """
    c = cases_by_id().get(case_id)
    if c is None:
        return {"error": f"未找到算例 {case_id}"}
    if treatments is None:
        treatments = {a["id"]: "straight" for a in c["assets"]}
    bad = [k for k in treatments if k not in {a["id"] for a in c["assets"]}]
    if bad:
        return {"error": f"未知资产编号：{bad}", "valid_ids": [a["id"] for a in c["assets"]]}

    r = tax_model.simulate(c, treatments, mode=mode)
    ok, problems = tax_model.audit_plan(c, treatments)
    return {
        "case_id": case_id,
        "mode": mode,
        "total_tax_yuan": round(r["total_tax"], 2),
        "by_year": [
            {"year": x["year"], "taxable_income": round(x["taxable_income"], 2),
             "rate": x["rate"], "tax": round(x["tax"], 2)}
            for x in r["records"]
        ],
        "compliant": ok,
        "compliance_issues": problems,
    }


def feasible_options(case_id: str) -> dict:
    """列出某算例每项资产可选的处理方式（政策约束后）。"""
    c = cases_by_id().get(case_id)
    if c is None:
        return {"error": f"未找到算例 {case_id}"}
    return {
        "case_id": case_id,
        "assets": [
            {
                "id": a["id"],
                "cost_yuan": a["cost_yuan"],
                "purchase_year": a["purchase_year"],
                "statutory_life_years": a["statutory_life_years"],
                "accelerable": a.get("accelerable", False),
                "options": list(tax_model.feasible_treatments(a, c, mode="current")),
            }
            for a in c["assets"]
        ],
        "option_labels": tax_model.TREATMENT_LABELS,
    }


def optimize_plan(case_id: str, mode: str = "current") -> dict:
    """穷举全部合法组合，给出累计税额最小的方案，并与人工经验法对照。"""
    c = cases_by_id().get(case_id)
    if c is None:
        return {"error": f"未找到算例 {case_id}"}

    best = methods.optimize(c, mode=mode)
    exp = methods.baseline_experience(c, mode=mode)
    t0 = methods.baseline_no_planning(c, mode=mode)

    return {
        "case_id": case_id,
        "mode": mode,
        "optimal": {
            "total_tax_yuan": round(best["total_tax"], 2),
            "treatments": best["asset_treatments"],
            "combinations_evaluated": best.get("evaluated_combinations"),
        },
        "baseline_experience": {"total_tax_yuan": round(exp["total_tax"], 2)},
        "baseline_no_planning": {"total_tax_yuan": round(t0["total_tax"], 2)},
        "saving_vs_experience_yuan": round(exp["total_tax"] - best["total_tax"], 2),
    }


def submit_plan(case_id: str, plan_path: str, summary: str = "") -> dict:
    """把筹划方案正式提交归档。

    这是本域里**唯一有对外后果**的动作——方案一旦提交，就算正式交付。
    所以它被标记为高风险，每次调用都要过人工审批门。
    其余工具（检索、测算、写草稿）都是只读或可撤销的，不需要审批。
    """
    import time as _t

    rec = {
        "case_id": case_id,
        "plan_path": plan_path,
        "summary": summary[:500],
        "submitted_at": _t.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = DATA_DIR.parent / "workspace" / "submissions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"submitted": True, **rec, "registry": str(out)}


# —— 注册到工具表 ——
def register_tax_tools(registry: ToolRegistry) -> None:
    registry.register(Tool(
        name="policy_search",
        description="语义检索税收政策原文。传入业务化的问题描述（如「研发费用加计扣除比例」），返回最相关的政策片段与文号。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索问题，用业务语言描述"},
                "k": {"type": "integer", "description": "返回条数，默认 3", "default": 3},
            },
            "required": ["query"],
        },
        handler=policy_search,
        read_only=True,
        category="policy",
    ))
    registry.register(Tool(
        name="list_policies",
        description="按关键词或政策机制查结构化政策库（89 条，含适用条件、税率、有效期）。比语义检索更精确，适合已知政策名称时使用。",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "关键词，如「小微」「加计扣除」"},
                "mechanic": {"type": "string", "description": "政策机制，如「加计扣除」「税额抵免」"},
                "limit": {"type": "integer", "default": 15},
            },
        },
        handler=list_policies,
        read_only=True,
        category="policy",
    ))
    registry.register(Tool(
        name="case_lookup",
        description="按行业或规模查找真实企业算例（共 292 个），返回精简列表。用于找到可用于验证筹划方案的企业画像。",
        parameters={
            "type": "object",
            "properties": {
                "industry": {"type": "string", "description": "行业关键词，如「制造业」「农业」"},
                "size_group": {"type": "string", "description": "规模分组，如 size_small"},
                "limit": {"type": "integer", "default": 8},
            },
        },
        handler=case_lookup,
        read_only=True,
        category="case",
    ))
    registry.register(Tool(
        name="get_case",
        description="取指定算例的完整输入（资产清单与逐年财务数据）。计算前必须先调用它拿到资产编号。",
        parameters={
            "type": "object",
            "properties": {"case_id": {"type": "string", "description": "算例编号"}},
            "required": ["case_id"],
        },
        handler=get_case,
        read_only=True,
        category="case",
    ))
    registry.register(Tool(
        name="feasible_options",
        description="列出某算例每项资产的合法处理方式（已按政策有效期过滤）。设计筹划方案前应先看这个，避免给出超期适用的方案。",
        parameters={
            "type": "object",
            "properties": {"case_id": {"type": "string"}},
            "required": ["case_id"],
        },
        handler=feasible_options,
        read_only=True,
        category="compute",
    ))
    registry.register(Tool(
        name="tax_calc",
        description="按指定的资产处理方式计算各年应纳税额，并校验方案是否合规（如是否超期适用一次性扣除）。",
        parameters={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "treatments": {
                    "type": "object",
                    "description": "资产编号到处理方式的映射，如 {\"A1\":\"one_time\",\"A2\":\"straight\"}。取值：one_time / accelerated / straight",
                },
                "mode": {"type": "string", "enum": ["current", "stale"], "default": "current"},
            },
            "required": ["case_id"],
        },
        handler=tax_calc,
        read_only=True,
        category="compute",
    ))
    registry.register(Tool(
        name="submit_plan",
        description=(
            "把筹划方案正式提交归档。这是有对外后果的动作，会触发人工审批——"
            "提交前请确认方案已经过独立复核且合规校验通过。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "plan_path": {"type": "string", "description": "方案文件在工作区中的路径"},
                "summary": {"type": "string", "description": "一句话说明方案要点与节税额"},
            },
            "required": ["case_id", "plan_path"],
        },
        handler=submit_plan,
        read_only=False,
        requires_approval=True,   # 高风险：每次调用都要人批
        category="submit",
    ))
    registry.register(Tool(
        name="optimize_plan",
        description="穷举全部合法组合，给出税负最小的筹划方案，并与人工经验法对照出节税额。",
        parameters={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "mode": {"type": "string", "enum": ["current", "stale"], "default": "current"},
            },
            "required": ["case_id"],
        },
        handler=optimize_plan,
        read_only=True,
        category="compute",
    ))
