"""Workflow Runtime（对应教程 s16）：模型决定单步，脚本决定编排。

有些流程是**固定的**——「查政策时效 → 算税负 → 独立复核 → 出报告」，
顺序在执行前就已知。让模型每轮自己决定下一步，既慢又不稳定。

把它们写成脚本，由宿主注册，模型只提供名字和参数。

三条关键设计：

1. **安全边界**：脚本由宿主提供，模型**不能提交可执行代码**。
   模型能给的只有 workflow 名、参数、以及可选的续跑 run_id。
   否则这就是一个远程代码执行漏洞。

2. **journal 续跑**：每一步的结果落进磁盘 journal。
   跑到一半断了（超时、限额、人工中断），下次带 run_id 重启，
   已完成的步骤直接复用 journal 里的结果，不重复烧钱。

3. **步骤间并行**：不相互依赖的步骤并发跑，而不是串着等。
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import CACHE_DIR

JOURNAL_DIR = CACHE_DIR / "workflows"


@dataclass
class StepResult:
    step: str
    ok: bool
    value: str = ""
    error: str = ""
    cached: bool = False
    duration_ms: float = 0.0


@dataclass
class WorkflowMeta:
    name: str
    description: str
    phases: list[str] = field(default_factory=list)


class WorkflowContext:
    """脚本运行时的上下文：脚本通过它调 Agent、并行、写日志。

    脚本拿到的是这个对象，不是裸的 Agent——所以它能做的只有
    「发起一次子任务」「并行发起多个」「记日志」，做不了别的。
    """

    def __init__(self, runner, journal: dict, run_id: str, events: list[dict]) -> None:
        self._runner = runner
        self._journal = journal
        self.run_id = run_id
        self._events = events

    def phase(self, name: str) -> None:
        self._event("phase", name)

    def _event(self, kind: str, name: str, **kw) -> None:
        self._events.append({"kind": kind, "name": name, "ts": time.time(), **kw})

    def agent(self, step: str, prompt: str, subagent: str = "tax_calculator") -> str:
        """执行一步。若 journal 里已有该步结果且未被标记失效，直接复用。"""
        key = f"{subagent}::{step}"
        if key in self._journal and not self._journal[key].get("stale"):
            self._event("step", step, cached=True)
            return self._journal[key]["value"]

        t0 = time.time()
        self._event("step", step, cached=False, agent=subagent)
        rec = self._runner.run(subagent, prompt)
        value = rec.get("conclusion", "") or rec.get("error", "")
        self._journal[key] = {"value": value, "ts": time.time()}
        self._event("step_done", step, ms=round((time.time() - t0) * 1000, 1))
        return value

    def parallel(self, jobs: list[tuple[str, str, str]]) -> list[str]:
        """并行执行多个步骤。jobs = [(step, prompt, subagent), ...]"""
        with ThreadPoolExecutor(max_workers=min(4, len(jobs) or 1)) as ex:
            futures = [ex.submit(self.agent, *j) for j in jobs]
            return [f.result() for f in futures]

    def log(self, message: str) -> None:
        self._event("log", message)


class WorkflowRuntime:
    def __init__(self, runner, journal_dir: Path = JOURNAL_DIR) -> None:
        self.runner = runner
        self.journal_dir = Path(journal_dir)
        self._workflows: dict[str, tuple[WorkflowMeta, Callable]] = {}

    def register(self, meta: WorkflowMeta, fn: Callable) -> None:
        self._workflows[meta.name] = (meta, fn)

    def catalog(self) -> list[dict]:
        return [
            {"name": m.name, "description": m.description, "phases": m.phases}
            for m, _ in self._workflows.values()
        ]

    def _journal_path(self, run_id: str) -> Path:
        return self.journal_dir / f"{run_id}.json"

    def run(self, name: str, args: dict | None = None, run_id: str | None = None) -> dict:
        if name not in self._workflows:
            return {"error": f"未知 workflow：{name}", "available": list(self._workflows)}
        meta, fn = self._workflows[name]
        args = args or {}

        run_id = run_id or f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
        jpath = self._journal_path(run_id)
        journal = json.loads(jpath.read_text(encoding="utf-8")) if jpath.exists() else {}

        events: list[dict] = []
        ctx = WorkflowContext(self.runner, journal, run_id, events)
        t0 = time.time()
        try:
            result = fn(ctx, args)
            ok, error = True, ""
        except Exception as e:  # noqa: BLE001
            result, ok, error = None, False, repr(e)

        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps(journal, ensure_ascii=False, indent=2), encoding="utf-8")

        reused = sum(1 for e in events if e["kind"] == "step" and e.get("cached"))
        fresh = sum(1 for e in events if e["kind"] == "step" and not e.get("cached"))
        return {
            "workflow": name,
            "run_id": run_id,
            "ok": ok,
            "error": error,
            "result": result,
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "steps": {"executed": fresh, "reused_from_journal": reused},
            "events": events,
            "journal": str(jpath),
        }
