"""内置 workflow：把固定流程写进代码，而不是让模型每轮自己决定。

「政策核查 → 方案测算 → 独立复核」这三步的顺序在执行前就定死了，
让模型自由发挥只会更慢、更不稳定。写成脚本后：
  · 顺序由宿主保证，模型没法跳过复核
  · 每步结果落 journal，中断可续跑
  · 模型只提供 case_id，不给它执行代码的机会
"""

from __future__ import annotations

from .workflow import WorkflowMeta, WorkflowRuntime

META_TAX_REVIEW = WorkflowMeta(
    name="tax_planning_review",
    description="对指定算例跑完整筹划流程：政策核查 → 方案测算 → 独立复核",
    phases=["政策核查", "方案测算", "独立复核"],
)


def tax_planning_review(ctx, args: dict) -> dict:
    case_id = args.get("case_id", "")
    if not case_id:
        raise ValueError("workflow 需要参数 case_id")

    ctx.phase("政策核查")
    policy = ctx.agent(
        "查政策时效",
        f"算例 {case_id} 涉及设备器具一次性扣除与小型微利企业优惠。"
        f"请查明这两类政策的现行文号、执行期限，以及是否存在已过期或尚未生效的版本。"
        f"输出：政策清单（文号 / 名称 / 有效期），以及时效风险提示。",
        subagent="policy_researcher",
    )

    ctx.phase("方案测算")
    calc = ctx.agent(
        "算最优方案",
        f"对算例 {case_id}：先看每项资产的合法处理方式，再给出税负最小的组合方案，"
        f"并算出该方案与人工经验法（全部直线折旧 + 申报研发加计扣除）的差额。"
        f"所有数字必须来自工具。输出：最优处理方式、累计应纳税额、节税额、合规校验结果。",
        subagent="tax_calculator",
    )

    ctx.phase("独立复核")
    verify = ctx.agent(
        "独立复核",
        f"复核下面这份筹划结论是否成立：\n{calc}\n\n"
        f"请自己动手用 tax_calc / optimize_plan 对算例 {case_id} 复算关键数字，"
        f"并检查它引用的政策是否真实有效。按「通过项 / 存疑项 / 错误项」输出。",
        subagent="verifier",
    )

    return {"case_id": case_id, "policy": policy, "calculation": calc, "verification": verify}


def register_builtin_workflows(runtime: WorkflowRuntime) -> None:
    runtime.register(META_TAX_REVIEW, tax_planning_review)
