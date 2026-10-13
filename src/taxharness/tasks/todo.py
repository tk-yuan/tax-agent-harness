"""任务清单与状态机（对应教程 s05 / s10）。

主 Agent 把目标拆成任务清单并持续追踪。状态：
    pending → in_progress → completed / failed

本项目在教程基础上加了**依赖图**：任务可以有 blocked_by 关系。
只有前置任务完成后，后续任务才是「可执行」的。

为什么需要：税筹流程天然有先后约束——「核查政策时效」必须在
「设计扣除方案」之前，「算完方案」才能在「独立复核」里被验证。
靠模型自觉遵守顺序是不可靠的；把它编码成依赖关系，调度层就能强制。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

STATUSES = ("pending", "in_progress", "completed", "failed")
_MARK = {"pending": " ", "in_progress": "~", "completed": "x", "failed": "!"}


@dataclass
class TaskItem:
    id: str
    subject: str
    status: str = "pending"
    blocked_by: list[str] = field(default_factory=list)
    owner: str = ""
    result: str = ""

    @property
    def done(self) -> bool:
        return self.status in ("completed", "failed")


class TaskBoard:
    def __init__(self) -> None:
        self.items: list[TaskItem] = []
        self._seq = 0

    # —— 变更（每次变更都应经过 Hook，供审计与校验）——
    def add(self, subject: str, blocked_by: list[str] | None = None) -> TaskItem:
        self._seq += 1
        t = TaskItem(id=f"T{self._seq}", subject=subject, blocked_by=list(blocked_by or []))
        self.items.append(t)
        return t

    def _find(self, tid: str) -> TaskItem | None:
        return next((t for t in self.items if t.id == tid), None)

    def start(self, tid: str) -> dict:
        t = self._find(tid)
        if t is None:
            return {"error": f"未知任务 {tid}"}
        pending = [
            b for b in t.blocked_by if not (self._find(b) and self._find(b).done)
        ]
        if pending:
            return {"error": f"任务 {tid} 的前置任务未完成：{pending}"}
        t.status = "in_progress"
        return {"ok": True, "task": asdict(t)}

    def complete(self, tid: str, result: str = "") -> dict:
        t = self._find(tid)
        if t is None:
            return {"error": f"未知任务 {tid}"}
        t.status, t.result = "completed", result[:500]
        return {"ok": True, "task": asdict(t)}

    def fail(self, tid: str, reason: str = "") -> dict:
        t = self._find(tid)
        if t is None:
            return {"error": f"未知任务 {tid}"}
        t.status, t.result = "failed", reason[:500]
        return {"ok": True, "task": asdict(t)}

    # —— 查询 ——
    def ready(self) -> list[TaskItem]:
        """可执行的任务：还在 pending，且所有前置已完成。"""
        out = []
        for t in self.items:
            if t.status != "pending":
                continue
            if all((self._find(b) and self._find(b).done) for b in t.blocked_by):
                out.append(t)
        return out

    def open_items(self) -> list[TaskItem]:
        return [t for t in self.items if not t.done]

    def progress(self) -> dict:
        counts = {s: 0 for s in STATUSES}
        for t in self.items:
            counts[t.status] += 1
        ready_ids = {t.id for t in self.ready()}
        return {
            "total": len(self.items),
            "counts": counts,
            "ready": sorted(ready_ids),
            "blocked": [
                t.id for t in self.items
                if t.status == "pending" and t.id not in ready_ids
            ],
        }

    def render(self) -> str:
        """渲染成 Markdown，供注入上下文。"""
        if not self.items:
            return "（任务清单为空）"
        lines = []
        for t in self.items:
            dep = f" ←依赖 {','.join(t.blocked_by)}" if t.blocked_by else ""
            owner = f" @{t.owner}" if t.owner else ""
            lines.append(f"- [{_MARK[t.status]}] {t.id} {t.subject}{owner}{dep}")
        return "\n".join(lines)

    def to_dict(self) -> list[dict]:
        return [asdict(t) for t in self.items]
