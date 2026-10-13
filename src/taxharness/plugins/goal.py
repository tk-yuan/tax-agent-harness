"""Goal Loop（对应教程 s17）：模型说停不算数，独立判断器说了算。

主循环原本的退出条件很简单：模型不再调工具，就返回。
这对闲聊够用，对「给出可复核的筹划方案」这种任务远远不够——
模型可能只做了一半就认为完成了。

`/goal` 的做法是在真正返回前，**再加一次独立判断**：
把完成条件交给一个独立判断器（与主模型分开的一次调用），它说没完成就继续。

本项目做了两级判断，先便宜后昂贵：

  一级（确定性）任务清单是否还有未完成项 —— 零成本，先挡掉大部分情况
  二级（模型判定）把目标条件与实际产出交给独立判断器 —— 只在有一级判断
                   通过但条件本身是开放式时才调用（如"节税额测算完成且有依据"）
"""

from __future__ import annotations

import json

JUDGE_SYSTEM = (
    "你是任务完成度的独立判断器。你不参与任务执行，只做判断。"
    "你只输出 JSON，不输出解释。宁可判未完成，也不要放过半成品。"
)

JUDGE_PROMPT = """完成条件：
{condition}

Agent 的最终回答：
{answer}

Agent 实际执行过的动作记录：
{actions}

请判断：完成条件是否已经真正达成？

判「已达成」的要求：
- 条件里要求的每一项产出都确实存在（不是"打算做"或"建议做"）
- 数字有工具计算支撑，不是估算
- 政策依据有文号

输出 JSON：
{{"achieved": true/false, "reason": "判断理由，40 字以内", "missing": ["还缺什么"]}}"""


class GoalChecker:
    """独立的完成度判断器。"""

    def __init__(self, llm, condition: str = "") -> None:
        self.llm = llm
        self.condition = condition
        self.history: list[dict] = []

    def check_deterministic(self, board) -> dict | None:
        """一级判断：任务清单未清空就不算完成。零成本。"""
        if board is None or not board.items:
            return None
        open_items = board.open_items()
        if not open_items:
            return None
        return {
            "achieved": False,
            "reason": "任务清单仍有未完成项",
            "missing": [f"{t.id} {t.subject}" for t in open_items],
            "level": "deterministic",
        }

    def check_with_llm(self, answer: str, actions: list[str]) -> dict:
        """二级判断：交给独立模型调用判定。"""
        if not self.condition:
            return {"achieved": True, "reason": "未设置完成条件，跳过模型判定", "level": "none"}
        prompt = JUDGE_PROMPT.format(
            condition=self.condition,
            answer=answer[:3000],
            actions="\n".join(actions[-40:]) or "（无记录）",
        )
        try:
            raw = self.llm.complete(prompt, system=JUDGE_SYSTEM, max_tokens=400)
            verdict = _parse(raw)
        except Exception as e:  # noqa: BLE001
            return {"achieved": True, "reason": f"判定器失败，保守放行：{e}", "level": "error"}
        verdict["level"] = "llm"
        self.history.append(verdict)
        return verdict


def _parse(raw: str) -> dict:
    import re

    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            return {
                "achieved": bool(data.get("achieved", False)),
                "reason": str(data.get("reason", ""))[:200],
                "missing": data.get("missing", []) or [],
            }
        except json.JSONDecodeError:
            pass
    return {"achieved": False, "reason": f"判定输出无法解析：{raw[:80]}", "missing": []}
