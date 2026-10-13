"""求解方法与对照方法。"""

import itertools

from tax_model import feasible_treatments, simulate


def optimize(case, mode="current"):
    """T3 本研究的组合优化：枚举全部合法处理组合，取累计税额最小者。"""
    assets = case["assets"]
    ids = [a["id"] for a in assets]
    options = [feasible_treatments(a, case, mode=mode) for a in assets]

    best = None
    evaluated = 0
    for combo in itertools.product(*options):
        treatments = dict(zip(ids, combo))
        result = simulate(case, treatments, mode=mode)
        evaluated += 1
        if best is None or result["total_tax"] < best["total_tax"] - 1e-9:
            best = result
    best["evaluated_combinations"] = evaluated
    return best


def baseline_no_planning(case, mode="current"):
    """T0 基准：全部资产直线折旧，且不申报研发费用加计扣除。"""
    treatments = {a["id"]: "straight" for a in case["assets"]}
    return simulate(case, treatments, claim_rd=False, mode=mode)


def baseline_experience(case, mode="current"):
    """T1 人工经验法：资产按常规直线折旧，申报研发费用加计扣除。"""
    treatments = {a["id"]: "straight" for a in case["assets"]}
    return simulate(case, treatments, claim_rd=True, mode=mode)


def baseline_greedy_single_period(case, mode="current"):
    """T2 单期贪心法：逐项资产选择使其投入使用年度应纳税额最小的处理方式。

    只看当期、不看跨期与跨资产影响，是「每年想少缴税」的自然做法。
    """
    treatments = {a["id"]: "straight" for a in case["assets"]}
    for a in case["assets"]:
        best_t, best_val = None, None
        for t in feasible_treatments(a, case, mode=mode):
            trial = dict(treatments)
            trial[a["id"]] = t
            result = simulate(case, trial, mode=mode)
            year_tax = next(
                row["tax"] for row in result["records"] if row["year"] == a["purchase_year"]
            )
            if best_val is None or year_tax < best_val - 1e-9:
                best_val, best_t = year_tax, t
        treatments[a["id"]] = best_t
    return simulate(case, treatments, mode=mode)


def baseline_local_search(case, mode="current", starts=None):
    """T4 局部搜索法（爬山）：每次只改动一项资产的处理，若累计税额改善则接受，直到无法再改进。

    starts 为多起点列表；为 None 时使用单起点（全部直线折旧）。
    """
    assets = case["assets"]
    ids = [a["id"] for a in assets]
    by_id = {a["id"]: a for a in assets}

    if starts is None:
        starts = [{aid: "straight" for aid in ids}]

    best = None
    for start in starts:
        current = {aid: start.get(aid, "straight") for aid in ids}
        current_val = simulate(case, current, mode=mode)["total_tax"]
        improved = True
        while improved:
            improved = False
            for aid in ids:
                for t in feasible_treatments(by_id[aid], case, mode=mode):
                    if t == current[aid]:
                        continue
                    trial = dict(current)
                    trial[aid] = t
                    val = simulate(case, trial, mode=mode)["total_tax"]
                    if val < current_val - 1e-9:
                        current, current_val = trial, val
                        improved = True
        result = simulate(case, current, mode=mode)
        if best is None or result["total_tax"] < best["total_tax"] - 1e-9:
            best = result
    return best


def baseline_local_search_multi(case, mode="current"):
    """T4b 多起点局部搜索：从每个资产的各种处理组合出发生成多个初始解。"""
    assets = case["assets"]
    ids = [a["id"] for a in assets]
    starts = [{aid: "straight" for aid in ids}]
    for var in ("one_time", "accelerated"):
        starts.append({aid: var for aid in ids})
    for i, a in enumerate(assets):
        s = {aid: "straight" for aid in ids}
        for t in feasible_treatments(a, case, mode=mode):
            if t != "straight":
                s[a["id"]] = t
        starts.append(s)
    return baseline_local_search(case, mode=mode, starts=starts)


def baseline_random_search(case, mode="current", samples=800, seed=20260921):
    """T5 随机搜索：在合法组合空间内均匀随机采样，取最好结果。"""
    import itertools as _it
    import random

    rng = random.Random(seed)
    assets = case["assets"]
    ids = [a["id"] for a in assets]
    options = [feasible_treatments(a, case, mode=mode) for a in assets]
    space_size = 1
    for o in options:
        space_size *= len(o)
    n = min(samples, space_size)

    best = None
    for _ in range(n):
        combo = tuple(rng.choice(o) for o in options)
        treatments = dict(zip(ids, combo))
        result = simulate(case, treatments, mode=mode)
        if best is None or result["total_tax"] < best["total_tax"] - 1e-9:
            best = result
    best["sampled"] = n
    best["space_size"] = space_size
    return best


def legalize(case, asset_treatments, mode="current"):
    """把一个（可能违规的）方案修正为合法方案：对违规处理回退为直线折旧，再逐项求最优。

    用于回答：「若一开始用了过期的政策规则，事后被迫改正，代价是多少？」
    """
    from tax_model import audit_plan

    fixed = dict(asset_treatments)
    ok, _ = audit_plan(case, fixed)
    if ok:
        return simulate(case, fixed, mode=mode), fixed

    # 先把违规项回退为直线折旧
    for a in case["assets"]:
        t = fixed.get(a["id"])
        if t == "one_time" and not (2024 <= a["purchase_year"] <= 2027):
            fixed[a["id"]] = "straight"
        if t == "accelerated" and not a.get("accelerable", False):
            fixed[a["id"]] = "straight"

    # 再在合法空间内对每项资产做逐项改进，逼近「改正后的最优」
    improved = True
    while improved:
        improved = False
        base = simulate(case, fixed, mode=mode)["total_tax"]
        for a in case["assets"]:
            for t in feasible_treatments(a, case, mode="current"):
                trial = dict(fixed)
                trial[a["id"]] = t
                ok, _ = audit_plan(case, trial)
                if not ok:
                    continue
                val = simulate(case, trial, mode=mode)["total_tax"]
                if val < base - 1e-9:
                    fixed = trial
                    base = val
                    improved = True
    return simulate(case, fixed, mode=mode), fixed
