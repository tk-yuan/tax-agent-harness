"""Agent 主循环（对应教程 s01，并把各机制串进同一个循环）。

主循环本身**只做一件事**：推进对话。
其他所有能力——权限、审计、上下文压缩、记忆注入、Skills 组装、目标检查——
都通过 Hook 挂在四个节点上，主循环一行都不用改。

这是本项目最核心的设计主张：**能力用中间件扩展，不用 if 分支堆叠**。
对照教程 s15 的说法：「多种机制，一个循环」。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

# 一轮内并发执行工具调用的线程上限
MAX_PARALLEL_TOOLS = 4


@dataclass
class RunState:
    """一次运行的可变状态。Hook 与工具共享它。

    带锁：一轮内的多个工具调用会并发执行，共享状态必须能安全写入。
    """

    goal: str = ""
    todo: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)      # 从历史任务沉淀的经验
    artifacts: dict[str, str] = field(default_factory=dict)  # 落到虚拟文件系统的产出
    depth: int = 0                                      # 子 Agent 深度
    parent_goal: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

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
            convo.extend(self._run_tools(resp["tool_calls"]))

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

    # —— 内部：一次工具调用（四个 Hook 节点中的两个 + 权限 + 审批 + 缓存 + 重试）——
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
        t0 = time.perf_counter()
        with span_ctx as span:
            text = self._execute_tool(tool, name, args)
            # 成败写进 span：回归测试要断言「这次运行没出工具错误」，
            # 没这个字段就只能去翻对话文本，断言很难写。
            span["fields"]["ok"] = not text.startswith("[")
            span["fields"]["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            if text.startswith("["):
                span["fields"]["outcome"] = text[:60]
            return text

    def _execute_tool(self, tool, name: str, args: dict) -> str:
        """真正的执行体。返回文本形式的工具结果（错误也以 [..] 开头返回）。"""
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

        # 短路：缓存命中直接返回，不再执行。策略由 Hook 决定，循环只认约定。
        if "short_circuit" in ctx:
            if self.trace is not None:
                self.trace.event("cache", f"{name} 命中缓存")
            return self._stringify(ctx["short_circuit"])

        # 重试策略同样来自 Hook。只在**抛异常**时重试——
        # 返回错误字符串属于业务失败，重试多少次都一样。
        max_retry = int(ctx.get("max_retry", 0) or 0)
        backoff = float(ctx.get("retry_backoff", 0.4) or 0.4)
        result = None
        for attempt in range(max_retry + 1):
            try:
                result = self.registry.call(name, args)
                break
            except Exception as e:  # noqa: BLE001 —— 工具异常不应中断整个任务
                if attempt >= max_retry:
                    return f"[工具错误] {type(e).__name__}: {e}（已重试 {attempt} 次）"
                if self.trace is not None:
                    self.trace.event("retry", f"{name} 第 {attempt + 1} 次重试",
                                     error=type(e).__name__)
                time.sleep(backoff * (2 ** attempt))

        ctx = self.hooks.run(
            "after_tool",
            {"tool": tool, "name": name, "args": args, "result": result,
             "state": self.state, "trace": self.trace},
        )
        return self._stringify(ctx.get("result", result))

    # —— 内部：一轮内的全部工具调用（并发执行）——
    def _run_tools(self, calls: list[dict]) -> list[dict]:
        """执行模型本轮发出的所有工具调用，返回对应的 tool 结果消息。

        **为什么并发**：模型一轮可以返回多个 tool_call。它们之间必然没有依赖
        （有依赖的话模型会在拿到结果后的下一轮再调），所以串行执行纯属浪费——
        N 个网络往返会线性叠加。并发之后，一轮里的 3 个检索/测算工具
        只花最慢那一个的时间。

        **为什么结果要保持顺序**：API 是按 tool_call_id 匹配的，理论上乱序也能跑；
        但上下文里 tool 结果与 assistant 的调用顺序错位，会让模型对不上号，
        长任务里容易引发混乱。所以用 map（保序）而不是 as_completed。

        **单调用不走线程池**：省掉线程创建开销，也让最常见的路径保持简单。
        """
        if not calls:
            return []
        if len(calls) == 1:
            c = calls[0]
            return [M.tool_result(c["id"], self._run_tool(c))]

        workers = min(MAX_PARALLEL_TOOLS, len(calls))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(self._run_tool, calls))
        return [M.tool_result(c["id"], r) for c, r in zip(calls, results)]

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
