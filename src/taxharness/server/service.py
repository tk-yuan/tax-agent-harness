"""服务层：把 Agent 包成可异步执行、可观测、可审批的服务。

三件事：

1. **任务在后台线程里跑，不阻塞 HTTP 请求。**
   Agent 一次任务要跑几十秒到几分钟，同步等待必然超时。

2. **执行过程通过队列推给 SSE**。
   Trace 每产生一个 span/事件就塞进队列，`/runs/{id}/events` 端点在另一头取。
   这就是「执行链路可视化」的实时版——离线跑时同一份数据还会落成 HTML。

3. **审批是跨线程握手的**。
   Agent 线程在审批门上阻塞；HTTP 线程调用 `resolve()` 放行。
   两个线程共用一个 `ApprovalGate`，靠它的 `threading.Event` 同步。
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field

from ..assemble import build_harness
from ..observability.trace import Trace
from ..runtime.approval import ApprovalGate
from ..storage.store import load_checkpoint, save_checkpoint, shared_store


@dataclass
class RunRecord:
    run_id: str
    goal: str
    status: str = "running"        # running / awaiting_approval / completed / failed
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    answer: str = ""
    trace_id: str = ""
    error: str = ""
    turns: int = 0
    usage: dict = field(default_factory=dict)
    board: list[dict] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    pending_approvals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "status": self.status,
            "created": self.created,
            "finished": self.finished,
            "elapsed_sec": round((self.finished or time.time()) - self.created, 1),
            "answer": self.answer,
            "trace_id": self.trace_id,
            "error": self.error,
            "turns": self.turns,
            "usage": self.usage,
            "board": self.board,
            "artifacts": self.artifacts,
            "pending_approvals": self.pending_approvals,
        }


class RunManager:
    def __init__(self, max_runs: int = 50) -> None:
        self.runs: dict[str, RunRecord] = {}
        self._queues: dict[str, queue.Queue] = {}
        self._gates: dict[str, ApprovalGate] = {}
        self._bundles: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._max_runs = max_runs

    # —— 生命周期 ——
    def start(self, goal: str) -> RunRecord:
        rec = RunRecord(run_id=f"run-{uuid.uuid4().hex[:8]}", goal=goal)
        with self._lock:
            self.runs[rec.run_id] = rec
            self._queues[rec.run_id] = queue.Queue()
            if len(self.runs) > self._max_runs:
                for rid in sorted(self.runs, key=lambda r: self.runs[r].created)[: -self._max_runs]:
                    self._forget(rid)
        threading.Thread(target=self._execute, args=(rec,), daemon=True).start()
        return rec

    def _forget(self, run_id: str) -> None:
        self.runs.pop(run_id, None)
        self._queues.pop(run_id, None)
        self._gates.pop(run_id, None)
        self._bundles.pop(run_id, None)

    def get(self, run_id: str) -> RunRecord | None:
        return self.runs.get(run_id)

    def queue_of(self, run_id: str) -> queue.Queue | None:
        return self._queues.get(run_id)

    # —— 事件推送 ——
    def _emit(self, run_id: str, payload: dict) -> None:
        q = self._queues.get(run_id)
        if q is not None:
            q.put(payload)

    def _on_approval_request(self, rec: RunRecord, req) -> None:
        """审批门回调：在 Agent 线程里被调用，推给前端并挂起状态。"""
        rec.status = "awaiting_approval"
        rec.pending_approvals.append(req.id)
        self._checkpoint(rec)          # 审批断点也落盘：进程被杀也恢复得回来
        self._emit(
            rec.run_id,
            {
                "type": "approval_request",
                "request": {
                    "id": req.id, "tool": req.tool, "args": req.args,
                    "reason": req.reason, "risk": req.risk,
                    "timeout_sec": self._gates[rec.run_id].timeout,
                },
            },
        )

    # —— 断点 ——
    def _checkpoint(self, rec: RunRecord) -> None:
        gate = self._gates.get(rec.run_id)
        save_checkpoint(
            shared_store(),
            rec.run_id,
            {
                "goal": rec.goal,
                "status": rec.status,
                "answer": rec.answer,
                "board": rec.board,
                "artifacts": rec.artifacts,
                "turns": rec.turns,
                "trace_id": rec.trace_id,
                "approvals": gate.history() if gate else [],
            },
        )

    def resume(self, run_id: str) -> RunRecord | None:
        """从断点续跑：把已完成的任务与结论带进新任务的背景，避免重复烧 token。

        注意口径：这是「恢复任务上下文与已得结论后重新推进」，
        不是恢复精确的消息栈——后者需要序列化整个对话，代价大且收益有限。
        真正省下的是重复的检索与测算。
        """
        cp = load_checkpoint(shared_store(), run_id)
        if cp is None:
            return None
        done = [t for t in cp.get("board", []) if t.get("status") == "completed"]
        lines = [
            f"- {t.get('id')} {t.get('subject')}：{(t.get('result') or '（无结论）')[:200]}"
            for t in done
        ]
        goal = (
            f"这是一次断点续跑（原任务 {run_id}）。\n"
            f"原目标：{cp.get('goal')}\n\n"
            f"此前已完成的任务与结论：\n{chr(10).join(lines) or '（无）'}\n\n"
            f"已落盘产出：{cp.get('artifacts') or '（无）'}\n\n"
            f"请勿重复已完成的工作，从中断处继续，补齐剩余任务并给出最终结论。"
        )
        return self.start(goal)

    def resolve_approval(self, run_id: str, req_id: str, approved: bool,
                         decider: str = "", note: str = "") -> bool:
        gate = self._gates.get(run_id)
        rec = self.runs.get(run_id)
        if gate is None or rec is None:
            return False
        ok = gate.resolve(req_id, approved, decider=decider, note=note)
        if ok:
            if req_id in rec.pending_approvals:
                rec.pending_approvals.remove(req_id)
            if not rec.pending_approvals:
                rec.status = "running"
            self._emit(
                run_id,
                {
                    "type": "approval_resolved",
                    "id": req_id,
                    "approved": approved,
                    "decider": decider,
                    "note": note,
                },
            )
        return ok

    # —— 执行体 ——
    def _execute(self, rec: RunRecord) -> None:
        try:
            trace = Trace(rec.goal, on_event=lambda p: self._emit(rec.run_id, p))
            rec.trace_id = trace.trace_id
            gate = ApprovalGate(notifier=lambda req: self._on_approval_request(rec, req))
            with self._lock:
                self._gates[rec.run_id] = gate

            bundle = build_harness(trace=trace, approval_gate=gate)
            with self._lock:
                self._bundles[rec.run_id] = bundle

            self._emit(
                rec.run_id,
                {
                    "type": "start",
                    "trace_id": trace.trace_id,
                    "model": bundle["model"],
                    "tools": bundle["registry"].names(),
                    "skills": list(bundle["skills"].skills),
                    "subagents": list(bundle["runner"].specs),
                    "mcp_tools": bundle["mcp_tools"],
                },
            )

            result = bundle["agent"].run(rec.goal)

            rec.answer = result.answer
            rec.turns = result.turns
            rec.usage = result.usage
            rec.board = bundle["board"].to_dict()
            rec.artifacts = list(result.state.artifacts)
            rec.status = "completed" if result.stop_reason == "completed" else result.stop_reason
            rec.finished = time.time()
            self._checkpoint(rec)

            json_path, html_path = trace.save(
                {"answer": result.answer, "board": rec.board, "status": rec.status}
            )
            self._emit(
                rec.run_id,
                {
                    "type": "done",
                    "status": rec.status,
                    "answer": rec.answer,
                    "turns": result.turns,
                    "usage": result.usage,
                    "board": rec.board,
                    "artifacts": rec.artifacts,
                    "trace_id": rec.trace_id,
                    "trace_html": html_path.name,
                    "approvals": gate.history(),
                    "elapsed_ms": trace.elapsed_ms,
                },
            )
        except Exception as e:  # noqa: BLE001 —— 任何异常都要落成终态，不能留个假 running
            rec.status = "failed"
            rec.error = f"{type(e).__name__}: {e}"
            rec.finished = time.time()
            self._checkpoint(rec)      # 失败也要留快照，否则查不出崩在哪
            self._emit(rec.run_id, {"type": "error", "error": rec.error})
