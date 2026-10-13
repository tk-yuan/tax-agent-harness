"""Agent 主循环（对应教程 s01，并把各机制串进同一个循环）。

主循环本身**只做一件事**：推进对话。
其他所有能力——权限、审计、上下文压缩、记忆注入、Skills 组装、目标检查——
都通过 Hook 挂在四个节点上，主循环一行都不用改。

这是本项目最核心的设计主张：**能力用中间件扩展，不用 if 分支堆叠**。
对照教程 s15 的说法：「多种机制，一个循环」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import MAX_TURNS
from ..llm import LLMClient, parse_tool_arguments
from . import messages as M
from .hooks import HookManager
from .permissions import PermissionPolicy
from .tools import ToolRegistry

# 工具结果超过这个长度就交给 after_tool Hook 处理（落盘 + 只留摘要）
INLINE_LIMIT = 2000


@dataclass
class RunState:
    """一次运行的可变状态。Hook 与工具共享它。"""

    goal: str = ""
    todo: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)      # 从历史任务沉淀的经验
    artifacts: dict[str, str] = field(default_factory=dict)  # 落到虚拟文件系统的产出
    depth: int = 0                                      # 子 Agent 深度
    parent_goal: str | None = None

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "todo": [t.get("subject") for t in self.todo],
            "notes": len(self.notes),
            "artifacts": list(self.artifacts),
            "depth": self.depth,
        }


@dataclass
class RunResult:
    goal: str
    answer: str
    turns: int
    stop_reason: str
    trace_id: str
    usage: dict
    state: RunState


class Agent:
    """一个可运行的 Agent 实例。

    registry / hooks / policy / llm 都可替换——不同角色（主 Agent、子 Agent）
    的差别只在这四个依赖的组装方式，不在循环逻辑。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        hooks: HookManager | None = None,
        policy: PermissionPolicy | None = None,
        llm: LLMClient | None = None,
        system_prompt: str = "",
        allow: set[str] | None = None,
        max_turns: int = MAX_TURNS,
        state: RunState | None = None,
        trace: Any = None,
        approval_gate: Any = None,
        name: str = "main",
    ) -> None:
        self.registry = registry
        self.hooks = hooks or HookManager()
        self.policy = policy or PermissionPolicy()
        self.llm = llm or LLMClient()
        self.system_prompt = system_prompt
        self.allow = allow
        self.max_turns = max_turns
        self.state = state or RunState()
        self.trace = trace
        self.approval_gate = approval_gate
        self.name = name

    # —— 供子 Agent 复用循环逻辑 ——
    def clone(self, **overrides) -> "Agent":
        cfg = dict(
            registry=self.registry,
            hooks=self.hooks,
            policy=self.policy,
            llm=self.llm,
            system_prompt=self.system_prompt,
            allow=self.allow,
            max_turns=self.max_turns,
            state=self.state,
            trace=self.trace,
            name=self.name,
        )
        cfg.update(overrides)
        return Agent(**cfg)

    def run(self, goal: str, max_turns: int | None = None) -> RunResult:
        limit = max_turns or self.max_turns
        self.state.goal = goal
        convo: list[dict] = [M.system(self.system_prompt), M.user(goal)]
        answer, stop_reason, turns = "", "max_turns", 0
        # 兜底：模型每轮可能同时输出文字和工具调用。跑满轮数时，
        # 最后一次说的话就是它当时的结论——不返回空字符串，
        # 否则上层只能看到一个「成功但没结论」的调用（本项目实测踩过）。
        last_text = ""

        for turn in range(1, limit + 1):
            turns = turn

            # —— 节点 1：模型调用前（组装系统提示、压缩上下文、注入记忆）——
            ctx = self.hooks.run(
                "before_model",
                {"messages": convo, "state": self.state, "turn": turn,
                 "trace": self.trace, "agent": self, "goal": goal},
            )
            if ctx.get("deny"):
                stop_reason, answer = "denied", f"[中止] {ctx['deny']}"
                break
            convo = ctx["messages"]

            resp = self._call_model(convo, turn)
            if resp is None:
                stop_reason, answer = "model_error", f"[中止] 模型调用失败（已有结论：{last_text[:200]}）"
                break

            ctx = self.hooks.run(
                "after_model",
                {"response": resp, "state": self.state, "turn": turn, "trace": self.trace},
            )
            resp = ctx.get("response", resp)

            # —— 模型不再调工具：默认认为这一轮结束 ——
            if not resp["tool_calls"]:
                fin = self.hooks.run(
                    "before_finish",
                    {"answer": resp["content"], "state": self.state,
                     "messages": convo, "trace": self.trace, "turn": turn},
                )
                if fin.get("continue"):
                    convo.append(M.user(fin["reason"]))
                    continue
                answer = fin.get("answer", resp["content"])
                stop_reason = "completed"
                break

            convo.append(M.assistant(resp["content"], resp["tool_calls"]))
            if resp["content"].strip():
                last_text = resp["content"].strip()
            for call in resp["tool_calls"]:
                convo.append(M.tool_result(call["id"], self._run_tool(call)))

        else:
            stop_reason = "max_turns"
            answer = last_text

        return RunResult(
            goal=goal,
            answer=answer,
            turns=turns,
            stop_reason=stop_reason,
            trace_id=getattr(self.trace, "trace_id", ""),
            usage=self.llm.usage,
            state=self.state,
        )

    # —— 内部：一次模型调用（带 trace span）——
    def _call_model(self, convo: list[dict], turn: int) -> dict | None:
        """返回 None 表示调用失败，由调用方统一处理成 model_error。

        注意：异常必须**无条件**捕获。曾有版本把 try/except 写在
        `if self.trace is not None` 分支里，导致不开链路追踪时
        一次网络抖动就把整个任务崩掉——而"崩掉"和"优雅降级"是两回事。
        """
        specs = self.registry.specs(self.allow)
        span_ctx = (
            self.trace.span("model", f"{self.name}@turn{turn}",
                            ctx_chars=M.approx_chars(convo), tools=len(specs))
            if self.trace is not None
            else _null_span()
        )
        with span_ctx as span:
            try:
                resp = self.llm.chat(convo, tools=specs)
            except Exception as e:  # noqa: BLE001 —— 任何模型侧异常都降级处理
                span["fields"]["error"] = repr(e)
                return None
            span["fields"]["tool_calls"] = [c["function"]["name"] for c in resp["tool_calls"]]
            span["fields"]["chars"] = len(resp["content"])
            return resp

    # —— 内部：一次工具调用（四个 Hook 节点中的两个 + 权限 + 审计）——
    def _run_tool(self, call: dict) -> str:
        fn = call.get("function", {})
        name = fn.get("name", "")
        args = parse_tool_arguments(fn.get("arguments", ""))
        tool = self.registry.get(name)

        span_ctx = (
            self.trace.span("tool", name, args=str(args)[:160])
            if self.trace is not None
            else _null_span()
        )
        with span_ctx:
            if tool is None:
                return f"[错误] 未知工具：{name}"

            ctx = self.hooks.run(
                "before_tool",
                {"tool": tool, "name": name, "args": args, "policy": self.policy,
                 "state": self.state, "trace": self.trace,
                 "approval_gate": self.approval_gate},
            )
            if ctx.get("deny"):
                return f"[被拒绝] {ctx['deny']}"
            args = ctx.get("args", args)

            try:
                result = self.registry.call(name, args)
            except Exception as e:  # noqa: BLE001 —— 工具异常不应中断整个任务
                return f"[工具错误] {type(e).__name__}: {e}"

            ctx = self.hooks.run(
                "after_tool",
                {"tool": tool, "name": name, "args": args, "result": result,
                 "state": self.state, "trace": self.trace},
            )
            return self._stringify(ctx.get("result", result))

    @staticmethod
    def _stringify(value: Any, limit: int = INLINE_LIMIT) -> str:
        import json

        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:  # noqa: BLE001
                text = str(value)
        if len(text) > limit:
            text = text[:limit] + f"\n…[结果过长已截断，原长 {len(text)} 字符]"
        return text


class _null_span:
    def __enter__(self):
        return {"fields": {}}

    def __exit__(self, *a):
        return False
