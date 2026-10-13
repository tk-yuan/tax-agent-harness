"""四节点可插拔中间件（对应教程 s04）。

挂载点覆盖主循环的四个关键节点：

    模型调用前后   before_model / after_model     —— 改写提示、缓存、PII 检测
    工具调用前后   before_tool  / after_tool      —— 参数校验、权限、审计、重试
    文件读写前后   before_fs    / after_fs        —— 路径白名单、版本留痕
    任务状态变更前后 before_task_state / after_task_state —— 状态机合法性、通知

Hook 的契约：收到一个 ctx dict，返回 None 表示不改动，返回 dict 表示合并进 ctx。
若 ctx 里出现 `deny` 键，流水线立即中止并把它作为原因上报。

**为什么要有这一层**：扩展不该改主循环。加审计、加限流、加敏感信息过滤，
都只挂 Hook。主循环保持只有一件事——推进对话。
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

HOOK_POINTS = (
    "before_model",
    "after_model",
    "before_tool",
    "after_tool",
    "before_fs",
    "after_fs",
    "before_task_state",
    "after_task_state",
    "before_finish",
)

HookFn = Callable[[dict], dict | None]


class HookManager:
    def __init__(self) -> None:
        self._hooks: dict[str, list[tuple[str, HookFn]]] = defaultdict(list)
        self._log: list[dict] = []

    def register(self, point: str, fn: HookFn, name: str | None = None) -> None:
        if point not in HOOK_POINTS:
            raise ValueError(f"未知挂载点：{point}（可选 {HOOK_POINTS}）")
        self._hooks[point].append((name or fn.__name__, fn))

    def hook(self, point: str, name: str | None = None):
        def deco(fn: HookFn):
            self.register(point, fn, name)
            return fn

        return deco

    def run(self, point: str, ctx: dict) -> dict:
        """依次执行该挂载点的所有 Hook。任一 Hook 设置 deny 即中止。"""
        ctx = dict(ctx)
        for name, fn in self._hooks.get(point, []):
            t0 = time.perf_counter()
            try:
                out = fn(ctx)
            except Exception as e:  # noqa: BLE001 —— Hook 出错不应拖垮主流程
                self._log.append(
                    {"point": point, "hook": name, "error": repr(e),
                     "ms": round((time.perf_counter() - t0) * 1000, 2)}
                )
                if point.startswith("before_"):
                    ctx["deny"] = f"Hook {name} 执行失败：{e}"
                continue
            if out:
                ctx.update(out)
            self._log.append(
                {"point": point, "hook": name,
                 "ms": round((time.perf_counter() - t0) * 1000, 2)}
            )
            if "deny" in ctx:
                break
        return ctx

    @property
    def stats(self) -> list[dict]:
        return list(self._log)

    def describe(self) -> dict[str, list[str]]:
        return {p: [n for n, _ in fns] for p, fns in self._hooks.items() if fns}
