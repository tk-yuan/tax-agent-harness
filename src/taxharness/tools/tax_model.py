"""企业所得税组合筹划模拟核心（v2：扩展政策机制）。

支持的政策机制（按税法规定顺序适用）：
    1. 免税收入      —— 从收入中剔除
    2. 减计收入      —— 按比例计入（如综合利用资源减按90%）
    3. 加计扣除      —— 研发费用等额外扣除
    4. 固定资产折旧   —— 一次性扣除 / 加速折旧 / 直线折旧
    5. 所得减免      —— 项目所得免征或减半
    6. 弥补以前年度亏损
    7. 减免所得税额   —— 小型微利 / 高新技术 / 区域优惠税率
    8. 税额抵免      —— 专用设备投资额按比例抵免应纳税额

政策时效（本项目创新点）：mode="current" 只允许使用有效期内的政策；
mode="stale" 忽略有效期（模拟规则库未更新）。

**政策参数来源**：关键政策的执行期限由 `policy_data/policy_library.json` 读取
（该库由 official 目录构建并逐条核验），使"政策库 → 模型约束"链条闭合；
读取失败时回退到下方默认值。
"""

import json
from pathlib import Path

# —— 项目根与模块搜索路径（文件已按功能分层，此处统一引导）——
import sys as _sys
try:                       # 控制台统一 UTF-8，避免特殊符号导致打印崩溃
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
_SRC = Path(__file__).resolve().parents[1]
for _d in ("core", "pipeline", "policy", "experiments", "demos"):
    _p = _SRC / _d
    if _p.exists() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

NORMAL_RATE = 0.25
HNTECH_RATE = 0.15
SME_RATE = 0.05

SHORTENED_LIFE_RATIO = 0.6

TECH_TRANSFER_FREE = 5_000_000
TECH_TRANSFER_HALF_RATE = 0.5

LOSS_CARRY_GENERAL = 5
LOSS_CARRY_QUALIFIED = 10

# 研发费用加计扣除的「可加计比例」：并非全部研发支出都符合加计条件
# （部分为资本化支出、部分不符合加计归集口径）。
# 该比例由「模型测算税负 = 企业披露所得税费用」的总量校准确定，属校准参数。
RD_DEDUCTION_RATIO = 0.56


def load_model_params():
    """读取由 RAG 从政策原文抽取的模型参数（policy_data/model_params.json）。

    每个参数都可溯源到具体文号；缺项时**不覆盖**，由下方默认值兜底，
    以避免静默改变口径。生成脚本：src/policy/build_rag_params.py。
    """
    p = Path(__file__).resolve().parents[3] / "data" / "policy_data" / "model_params.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


_PARAMS = load_model_params()

# 政策参数：优先取 RAG 抽取值（带文号依据），否则用默认值
SME_TAXABLE_MAX = float(_PARAMS.get("sme_income_max", 3_000_000))
SME_EMPLOYEE_MAX = float(_PARAMS.get("sme_employee_max", 300))
SME_TOTAL_ASSETS_MAX = float(_PARAMS.get("sme_assets_max", 50_000_000))

ONE_TIME_MAX_UNIT_VALUE = float(_PARAMS.get("one_time_unit_max", 5_000_000))
ONE_TIME_FIRST_YEAR = int(_PARAMS.get("one_time_year_from", 2024))


def load_policy_validity():
    """读取关键政策的执行期限。

    取值优先顺序：
      ① RAG 时间线 `policy_data/rag_timeline.json`——由政策**原文**抽取并归并的
         版本沿革，每版带文号与起止年，**可溯源**；
      ② 政策库 `policy_data/policy_library.json` 的 `valid_to_verified`（人工核验）；
      ③ 代码内默认值。

    返回 {'sme_last_year': int, 'one_time_last_year': int, 'source': str}；
    读取失败时返回 None，调用方回退到默认值。
    """
    # ---- ① RAG 时间线（可溯源到文号）----
    tl_path = Path(__file__).resolve().parents[3] / "data" / "policy_data" / "rag_timeline.json"
    if tl_path.exists():
        try:
            tl = json.loads(tl_path.read_text(encoding="utf-8"))
            out = {}
            for fam, key in (("小型微利企业优惠", "sme_last_year"),
                             ("设备器具一次性扣除", "one_time_last_year")):
                items = [x for x in (tl.get(fam) or []) if x.get("valid_to")]
                if items:
                    cur = max(items, key=lambda x: x["valid_to"])   # 现行版＝截止年最晚
                    out[key] = int(cur["valid_to"])
                    out[key + "_doc"] = cur.get("doc_num", "")
            if out:
                out["source"] = "policy_data/rag_timeline.json（RAG 抽取，含文号）"
                return out
        except Exception:
            pass

    # ---- ② 政策库（人工核验）----
    try:
        path = Path(__file__).resolve().parents[3] / "data" / "policy_data" / "policy_library.json"
        lib = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    out = {}
    for rec in lib:
        name = rec.get("name", "")
        to = rec.get("valid_to_verified")
        if not to or rec.get("validity_type") != "阶段性":
            continue
        if "小型微利" in name and "sme_last_year" not in out:
            out["sme_last_year"] = int(to)
        if ("500" in name and "设备" in name) and "one_time_last_year" not in out:
            out["one_time_last_year"] = int(to)
    if out:
        out["source"] = "policy_data/policy_library.json"
    return out or None


_POLICY = load_policy_validity() or {}
# 小型微利企业优惠的适用截止年（政策库优先，回退）
SME_POLICY_LAST_YEAR = _POLICY.get("sme_last_year", 2027)
# 设备器具一次性扣除的适用截止年（政策库优先，回退）
ONE_TIME_LAST_YEAR = _POLICY.get("one_time_last_year", 2027)

TREATMENTS = ("one_time", "accelerated", "straight")
TREATMENT_LABELS = {
    "one_time": "一次性扣除",
    "accelerated": "加速折旧(缩短至60%年限)",
    "straight": "直线折旧",
}
POLICY_MODES = ("current", "stale")


def feasible_treatments(asset, case, mode="current"):
    options = ["straight"]
    if asset.get("accelerable", False):
        options.append("accelerated")
    in_window = ONE_TIME_FIRST_YEAR <= asset["purchase_year"] <= ONE_TIME_LAST_YEAR
    if mode == "stale":
        in_window = True
    if asset["cost_yuan"] <= ONE_TIME_MAX_UNIT_VALUE and in_window:
        options.append("one_time")
    return tuple(options)


def depreciation_schedule(asset, treatment):
    cost = float(asset["cost_yuan"])
    start_year = int(asset["purchase_year"])
    start_month = int(asset["put_into_use_month"]) + 1
    if start_month > 12:
        start_month -= 12
        start_year += 1

    if treatment == "one_time":
        return {start_year: cost}

    life_years = int(asset["statutory_life_years"])
    if treatment == "accelerated":
        life_years = max(1, round(life_years * SHORTENED_LIFE_RATIO))

    total_months = life_years * 12
    monthly = cost / total_months
    schedule = {}
    months_done = 0
    year, month = start_year, start_month
    while months_done < total_months:
        take = min(13 - month, total_months - months_done)
        schedule[year] = schedule.get(year, 0.0) + take * monthly
        months_done += take
        year += 1
        month = 1
    return schedule


def load_year_rates():
    """读取「逐年有效的小微优惠参数」（policy_data/rag_year_rates.json）。

    由 RAG 从各版本政策原文抽取并归并，生成脚本 src/policy/build_rag_year_rates.py。
    小微优惠的门槛与税率**随版本变动**，故必须逐年取该年有效那一版的参数。
    """
    p = Path(__file__).resolve().parents[3] / "data" / "policy_data" / "rag_year_rates.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


_YEAR_RATES = load_year_rates()


def sme_rate_for(year, taxable_income):
    """该年有效的小微实际税负；该年无有效版本或超过门槛时返回 None。"""
    rec = (_YEAR_RATES.get("小微") or {}).get(str(year))
    if not rec:
        return None
    for cap, rate in (rec.get("bands") or []):
        if cap is None or taxable_income <= cap:
            return float(rate)
    return None


def applicable_rate(taxable_income, case, year, mode="current"):
    """确定适用税率。

    两种口径：
    · **默认**：若算例给出 `rate_tier`（由企业披露的实际税率反推的税率档），直接用该档。
      说明：实际税率已含各项优惠净效果，直接乘「扣除后的所得」会重复扣除，
      故只借它反推**档位**，扣除项仍由模型按政策逐一建模。
    · **版本感知**（`case["use_rag_versions"]=True`）：不使用 `rate_tier`，
      改为按该年**有效的小微政策版本**判定门槛与税率（参数来自 RAG 抽取），
      并逐年检查资格。此口径使政策版本信息成为模型结论的必要输入。
    """
    if case.get("use_rag_versions"):
        years = case.get("years") or [year]
        sme_ok = (
            taxable_income <= SME_TAXABLE_MAX
            and case["employees"] <= SME_EMPLOYEE_MAX
            and case["total_assets_yuan"] <= SME_TOTAL_ASSETS_MAX
        )
        if sme_ok:
            r = sme_rate_for(year, taxable_income)
            if r is not None:
                return r
        candidates = [NORMAL_RATE]
        if year in case.get("high_tech_years", []):
            candidates.append(HNTECH_RATE)
        region = case.get("region_rate")
        if region and year in case.get("region_rate_years", []):
            rv = case.get("region_rate_valid_to")
            expired = mode == "current" and rv is not None and year > rv
            if not expired:
                candidates.append(region)
        return min(candidates)

    tier = case.get("rate_tier")
    if tier is not None:
        return float(tier)

    sme_ok = (
        taxable_income <= SME_TAXABLE_MAX
        and case["employees"] <= SME_EMPLOYEE_MAX
        and case["total_assets_yuan"] <= SME_TOTAL_ASSETS_MAX
    )
    if mode == "current" and year > SME_POLICY_LAST_YEAR:
        sme_ok = False
    if sme_ok:
        return SME_RATE

    candidates = [NORMAL_RATE]
    if year in case.get("high_tech_years", []):
        candidates.append(HNTECH_RATE)

    region = case.get("region_rate")
    if region and year in case.get("region_rate_years", []):
        region_valid_to = case.get("region_rate_valid_to")
        expired = (
            mode == "current" and region_valid_to is not None and year > region_valid_to
        )
        if not expired:
            candidates.append(region)

    return min(candidates)


def _series(case, key, year):
    d = case.get(key)
    if not d:
        return 0.0
    v = d.get(str(year), 0.0)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def simulate(case, asset_treatments, claim_rd=True, mode="current"):
    """模拟多期应纳税额（按扩展的政策机制）。"""
    years = case["years"]
    dep = {y: 0.0 for y in years}
    for a in case["assets"]:
        sched = depreciation_schedule(a, asset_treatments[a["id"]])
        for y, v in sched.items():
            if y in dep:
                dep[y] += v

    carry_limit = (
        LOSS_CARRY_QUALIFIED if case.get("high_tech_years") else LOSS_CARRY_GENERAL
    )
    # 亏损结转开关：合并报表口径下，不同法人主体的盈亏不能互相弥补，
    # 故对由合并报表构建的算例应设为 False（case["loss_offset"]）。
    use_loss_offset = bool(case.get("loss_offset", True))
    losses = []
    records = []
    total_tax = 0.0

    for y in years:
        losses = [e for e in losses if e[1] >= y and e[0] > 1e-9]

        profit = float(case["profit_before_deductions_yuan"].get(str(y), 0))
        rd_ratio = float(case.get("rd_deduction_ratio", RD_DEDUCTION_RATIO))
        rd = (
            float(case["r_and_d_expense_yuan"].get(str(y), 0)) * rd_ratio
            if claim_rd else 0.0
        )
        d = dep.get(y, 0.0)

        # 1) 免税收入
        exempt = _series(case, "exempt_income_yuan", y)
        # 2) 减计收入：按比例计入，减计额 = 基数 × (1 - 比例)
        reduce_base = _series(case, "reduce_income_yuan", y)
        reduce_factor = float(case.get("reduce_income_factor", 0.9))
        reduce_adj = reduce_base * (1 - reduce_factor)
        # 5) 所得减免（项目所得）
        project_income = _series(case, "project_exempt_income_yuan", y)
        project_ratio = float(case.get("project_exempt_ratio", 0.0))  # 0 免征 / 0.5 减半
        project_adj = project_income * (1 - project_ratio)
        # 技术转让所得独立优惠
        tt = float(case["tech_transfer_income_yuan"].get(str(y), 0))
        tt_taxable = (
            max(0.0, tt - TECH_TRANSFER_FREE) * TECH_TRANSFER_HALF_RATE if tt > 0 else 0.0
        )

        pre_loss = profit - exempt - reduce_adj - project_adj - rd - d

        used_loss = 0.0
        if use_loss_offset and pre_loss > 0 and losses:
            for e in losses:
                take = min(e[0], pre_loss - used_loss)
                e[0] -= take
                used_loss += take
                if pre_loss - used_loss <= 1e-9:
                    break
        post_loss = pre_loss - used_loss

        new_loss = 0.0
        if use_loss_offset and post_loss < 0:
            new_loss = -post_loss
            losses.append([new_loss, y + carry_limit])
            post_loss = 0.0
        elif post_loss < 0:
            post_loss = 0.0

        taxable = post_loss + tt_taxable
        rate = applicable_rate(taxable, case, y, mode=mode)
        tax_before_credit = taxable * rate
        # 8) 税额抵免（不超过当年应纳税额）
        credit = _series(case, "tax_credit_yuan", y)
        tax = max(0.0, tax_before_credit - credit)
        total_tax += tax

        records.append({
            "year": y,
            "profit": profit,
            "exempt_income": exempt,
            "reduce_income_adj": reduce_adj,
            "project_exempt_adj": project_adj,
            "rd_deduction": rd,
            "depreciation": d,
            "loss_used": used_loss,
            "new_loss": new_loss,
            "tech_transfer_taxable": tt_taxable,
            "taxable_income": taxable,
            "rate": rate,
            "tax_before_credit": tax_before_credit,
            "tax_credit": credit,
            "tax": tax,
        })

    return {
        "records": records,
        "total_tax": total_tax,
        "asset_treatments": dict(asset_treatments),
        "mode": mode,
    }


def audit_plan(case, asset_treatments):
    problems = []
    for a in case["assets"]:
        t = asset_treatments.get(a["id"])
        if t == "one_time" and not (
            ONE_TIME_FIRST_YEAR <= a["purchase_year"] <= ONE_TIME_LAST_YEAR
        ):
            problems.append(
                f"{a['id']}：{a['purchase_year']}年购置，采用一次性扣除，"
                f"但政策有效期仅至{ONE_TIME_LAST_YEAR}年12月31日——超期适用"
            )
        if t == "accelerated" and not a.get("accelerable", False):
            problems.append(f"{a['id']}：不符合加速折旧适用条件")
    return (len(problems) == 0), problems
