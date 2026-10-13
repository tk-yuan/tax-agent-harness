"""标准中间件集合：把各机制挂到主循环的四个节点上。

这是「能力用中间件扩展」这句话的落地处。写完它你会发现，
主循环 loop.py 一行 if 都不用加，但这些能力全都有了：

    before_model       注入记忆与任务清单 → 重写 system prompt → 触发上下文压缩
    after_model        记录用量
    before_tool        权限校验 → 审计 → 前置条件检查
    after_tool         超长结果落盘，上下文只留摘要与路径
    before_task_state  状态机合法性（前置未完成不允许开工）
    before_finish      目标闭环（s17）：任务清单没做完就不许收工
"""

from __future__ import annotations

from ..config import CONTEXT_BUDGET_CHARS, WORKSPACE_DIR
from ..context.compaction import compact_messages, spill_tool_result
from ..context.memory import LayeredMemory
from ..observability.trace import Trace
from ..runtime.hooks import HookManager
from ..runtime.permissions import PermissionDenied, audit
from ..tasks.todo import TaskBoard


def build_hooks(
    *,
    base_prompt: str,
    memory: LayeredMemory,
    board: TaskBoard,
    skill_catalog: str = "",
    subagent_catalog: str = "",
    llm=None,
    trace: Trace | None = None,
    budget: int = CONTEXT_BUDGET_CHARS,
    max_goal_rounds: int = 3,
    goal_checker=None,
) -> HookManager:
    hooks = HookManager()
    spill_dir = WORKSPACE_DIR / "artifacts"
    state = {"goal_rounds": 0, "compactions": 0}

    # —— 节点 1：模型调用前 ——
    @hooks.hook("before_model", "assemble_and_compact")
    def _before_model(ctx: dict) -> dict:
        msgs = list(ctx["messages"])

        # 动态块：技能目录（第一层渐进披露）、子 Agent 目录、任务清单、记忆
        #
        # **作用域很关键**：任务清单、子 Agent 目录、目标闭环都是
        # **主 Agent 专属**的。子 Agent 是独立上下文，它既不该看见主清单、
        # 也不该被告知"你可以委派"（它的工具集里根本没有 task），
        # 更不该被主清单的未完成项拦住收工。
        #
        # 本项目实测踩过两次：子 Agent 因此跑满 max_turns 后返回空结论；
        # 以及被告知有 task 工具、实际调用却失败，白白浪费轮次。
        is_main = getattr(ctx.get("state"), "depth", 0) == 0

        blocks = [base_prompt]
        if skill_catalog:
            blocks.append(skill_catalog)
        if is_main and subagent_catalog:
            blocks.append(subagent_catalog)

        if is_main and board.items:
            blocks.append("【任务清单】\n" + board.render()
                          + "\n（用 task_board 工具更新状态；前置未完成的不要开工）")
        mem_block = memory.context_block(ctx.get("goal", ""))
        if mem_block:
            blocks.append(mem_block)

        sys_text = "\n\n".join(blocks)
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {"role": "system", "content": sys_text}
        else:
            msgs.insert(0, {"role": "system", "content": sys_text})

        new_msgs, info = compact_messages(msgs, budget, llm=llm)
        if info.get("compacted"):
            state["compactions"] += 1
            if trace is not None:
                trace.event(
                    "context", "上下文压缩",
                    before=info["chars_before"], after=info["chars_after"],
                    removed=info["removed_messages"], strategy=info["strategy"],
                )
            # 压缩掉的是历史，把结论沉淀进工作记忆，避免"压没了"
            memory.note(f"上下文已压缩 {state['compactions']} 次；此前结论摘要见对话中的压缩块。")
        return {"messages": new_msgs}

    # —— 节点 2：模型调用后 ——
    @hooks.hook("after_model", "record_usage")
    def _after_model(ctx: dict) -> None:
        return None

    # —— 节点 3：工具调用前（权限 + 审批 + 审计）——
    @hooks.hook("before_tool", "permission_and_approval")
    def _before_tool(ctx: dict) -> dict:
        tool, args = ctx["tool"], ctx["args"]
        policy = ctx["policy"]
        try:
            policy.check_tool(tool, args)
        except PermissionDenied as e:
            return {"deny": str(e)}

        # 人工审批门：高风险操作在这里**阻塞**等人批。
        # 阻塞是刻意的——有后果的动作不能先做了再补审批。
        gate = ctx.get("approval_gate")
        if gate is not None:
            need, reason = gate.requires(tool, args)
            if need:
                req = gate.request(tool.name, args, reason)
                if ctx.get("trace") is not None:
                    ctx["trace"].event("approval", f"{tool.name} 审批",
                                       status=req.status, waited=req.waited_sec)
                audit("approval_decision", tool=tool.name, request=req.id,
                      status=req.status, decider=req.decider, reason=reason)
                if req.status != "approved":
                    return {
                        "deny": f"人工审批未通过（{req.status}）：{reason}。"
                                f"审批人备注：{req.note or '无'}"
                    }
        return None

    # —— 节点 4：工具调用后（超长结果落盘）——
    @hooks.hook("after_tool", "spill_large_result")
    def _after_tool(ctx: dict) -> dict:
        name, result = ctx["name"], ctx.get("result")
        import json

        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        summary, path = spill_tool_result(text, name, spill_dir)
        if path:
            ctx["state"].artifacts[path] = f"{name} 的完整结果"
            return {"result": summary}
        return None

    # —— 节点 5：收工前（目标闭环，对应教程 s17）——
    @hooks.hook("before_finish", "goal_check")
    def _before_finish(ctx: dict) -> dict:
        answer = ctx.get("answer", "")

        # 目标闭环只作用于主 Agent。子 Agent 的收工条件就是「模型不再调工具」，
        # 用主清单去卡子 Agent 会让它永远收不了工（见 before_model 处的注释）。
        if getattr(ctx.get("state"), "depth", 0) > 0:
            return None

        # 一级判断：任务清单未清空就不许收工。零成本，先挡掉大部分情况。
        verdict = goal_checker.check_deterministic(board) if goal_checker else None
        if verdict is None:
            open_items = board.open_items()
            if open_items:
                verdict = {
                    "achieved": False,
                    "reason": "任务清单仍有未完成项",
                    "missing": [f"{t.id} {t.subject}" for t in open_items],
                }
        # 二级判断：条件本身是开放式时，交给独立的模型判定
        if verdict is None and goal_checker is not None and goal_checker.condition:
            actions = [f"{s['kind']}:{s['name']}" for s in (trace.spans if trace else [])]
            verdict = goal_checker.check_with_llm(answer, actions)
            if trace is not None:
                trace.event("goal", "独立判断", achieved=verdict.get("achieved"),
                            reason=verdict.get("reason", "")[:80])

        if verdict is None or verdict.get("achieved"):
            return None

        if state["goal_rounds"] >= max_goal_rounds:
            # 兜底：避免无限续跑
            ctx["state"].notes.append(
                f"未能完成的完成条件：{verdict.get('missing')}"
                f"（已达最大续跑轮数 {max_goal_rounds}）"
            )
            return None

        state["goal_rounds"] += 1
        missing = verdict.get("missing") or []
        lines = [f"- {m}" for m in missing] or ["- （未列出具体缺口）"]
        return {
            "continue": True,
            "reason": (
                f"完成条件尚未达成：{verdict.get('reason', '')}\n"
                "仍缺：\n" + "\n".join(lines)
                + "\n请继续执行；确实无法完成的，说明原因后再收工。"
            ),
        }

    return hooks
