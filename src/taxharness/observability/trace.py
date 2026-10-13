"""链路追踪（对应 JD 里的「执行链路可视化」，不上公司平台，落到本地）。

一次任务 = 一个 trace_id，每个关键步骤 = 一个 span：
    用户请求 → Agent 规划 → 子 Agent 委派 → 记忆访问 → 工具调用
    → 文件操作 → 模型调用 → 收尾

产出两份东西：
  traces/<trace_id>.json   原始事件流，可程序化断言（回归测试用）
  traces/<trace_id>.html   人类可读的时间线，面试时可以当场打开

**为什么要有这个**：Agent 是概率系统，出了问题如果只能看到最终答案，
根本没法定位是「检索没找对」还是「模型没用检索结果」。
有了 span 级记录，每一步的输入输出、耗时、token 都在，才谈得上可观测。
"""

from __future__ import annotations

import html
import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..config import TRACE_DIR

SPAN_COLORS = {
    "model": "#4c6ef5",
    "tool": "#12b886",
    "subagent": "#f59f00",
    "task": "#ae3ec9",
    "memory": "#0ca678",
    "fs": "#868e96",
    "hook": "#e8590c",
    "run": "#343a40",
}


class Trace:
    """一次任务的完整事件流。

    on_event 可选回调：每产生一个 span 就往外推一次。
    服务模式下用它把执行过程实时送到前端（SSE）；
    离线跑时传 None，就没有任何额外开销。
    """

    def __init__(self, goal: str, on_event: Callable[[dict], None] | None = None) -> None:
        self.trace_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.goal = goal
        self.started = time.time()
        self.spans: list[dict] = []
        self.counters: dict[str, int] = {}
        self._stack: list[dict] = []
        self.on_event = on_event

    def _emit(self, payload: dict) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(payload)
        except Exception:  # noqa: BLE001 —— 推送失败绝不能影响任务执行
            pass

    # —— span 记录 ——
    def start_span(self, kind: str, name: str, **fields) -> dict:
        span = {
            "kind": kind,
            "name": name,
            "start": time.time(),
            "fields": fields,
            "children_start": len(self.spans),
        }
        self.counters[kind] = self.counters.get(kind, 0) + 1
        self._stack.append(span)
        return span

    def end_span(self, span: dict, **result) -> None:
        span["end"] = time.time()
        span["duration_ms"] = round((span["end"] - span["start"]) * 1000, 1)
        span["result"] = result
        span["depth"] = len(self._stack) - 1
        self.spans.append(span)
        if self._stack and self._stack[-1] is span:
            self._stack.pop()
        self._emit(
            {
                "type": "span",
                "trace_id": self.trace_id,
                "kind": span["kind"],
                "name": span["name"],
                "duration_ms": span["duration_ms"],
                "fields": span["fields"],
                "elapsed_ms": self.elapsed_ms,
            }
        )

    @contextmanager
    def span(self, kind: str, name: str, **fields):
        s = self.start_span(kind, name, **fields)
        try:
            yield s
        finally:
            self.end_span(s)

    def event(self, kind: str, name: str, **fields) -> None:
        """记录一个瞬时事件（没有持续时间，比如"触发了一次压缩"）。"""
        self.counters[kind] = self.counters.get(kind, 0) + 1
        now = time.time()
        self.spans.append(
            {
                "kind": kind,
                "name": name,
                "start": now,
                "end": now,
                "duration_ms": 0.0,
                "depth": 0,
                "fields": fields,
                "result": {},
            }
        )
        self._emit(
            {
                "type": "event",
                "trace_id": self.trace_id,
                "kind": kind,
                "name": name,
                "fields": fields,
                "elapsed_ms": self.elapsed_ms,
            }
        )

    @property
    def elapsed_ms(self) -> float:
        return round((time.time() - self.started) * 1000, 1)

    def summary(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "goal": self.goal,
            "elapsed_ms": self.elapsed_ms,
            "span_counts": dict(sorted(self.counters.items())),
            "total_spans": len(self.spans),
        }

    # —— 落盘 ——
    def save(self, extra: dict | None = None) -> tuple[Path, Path]:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            **self.summary(),
            "spans": [{k: v for k, v in s.items() if k != "children_start"} for s in self.spans],
            **(extra or {}),
        }
        json_path = TRACE_DIR / f"{self.trace_id}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        html_path = TRACE_DIR / f"{self.trace_id}.html"
        html_path.write_text(render_html(payload), encoding="utf-8")
        return json_path, html_path


def _fmt_fields(fields: dict) -> str:
    if not fields:
        return ""
    parts = []
    for k, v in fields.items():
        s = str(v)
        if len(s) > 120:
            s = s[:120] + "…"
        parts.append(f"{html.escape(k)}={html.escape(s)}")
    return " ".join(parts)


def render_html(payload: dict) -> str:
    """把事件流渲染成可读的甘特图式时间线。"""
    total = max(s["duration_ms"] for s in payload["spans"]) if payload["spans"] else 1
    rows = []
    for s in payload["spans"]:
        color = SPAN_COLORS.get(s["kind"], "#adb5bd")
        indent = "  " * s.get("depth", 0)
        width = max(0.5, s["duration_ms"] / 4)
        rows.append(
            f'<div class="row">'
            f'<div class="meta"><span class="kind" style="background:{color}">{s["kind"]}</span>'
            f'<span class="name">{indent}{html.escape(s["name"])}</span></div>'
            f'<div class="bar-wrap"><div class="bar" style="background:{color};width:{width}px"></div>'
            f'<span class="dur">{s["duration_ms"]}ms</span></div>'
            f'<div class="fields">{_fmt_fields(s["fields"])}</div>'
            f"</div>"
        )
    counts = " ".join(
        f'<span class="pill">{k} × {v}</span>' for k, v in payload["span_counts"].items()
    )
    return f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>Trace {payload['trace_id']}</title>
<style>
 body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;padding:24px;
      background:#0f1115;color:#e6e8eb}}
 h1{{font-size:17px;margin:0 0 4px}} .goal{{color:#9aa4b2;font-size:13px;margin-bottom:14px}}
 .pill{{display:inline-block;background:#1b1f27;border:1px solid #2a3039;border-radius:11px;
       padding:2px 9px;margin:0 5px 5px 0;font-size:12px;color:#adb5bd}}
 .row{{display:grid;grid-template-columns:230px 200px 1fr;gap:10px;align-items:center;
      padding:5px 8px;border-bottom:1px solid #191d24;font-size:12.5px}}
 .kind{{display:inline-block;color:#fff;border-radius:4px;padding:1px 6px;font-size:11px;
       margin-right:7px}}
 .name{{color:#e6e8eb}} .bar{{height:9px;border-radius:4px;display:inline-block;
       vertical-align:middle}}
 .bar-wrap{{position:relative}} .dur{{color:#7a8494;font-size:11px;margin-left:7px}}
 .fields{{color:#6b7480;font-size:11.5px;overflow:hidden;text-overflow:ellipsis;
         white-space:nowrap}}
</style>
<h1>执行链路 · {payload['trace_id']}</h1>
<div class="goal">目标：{html.escape(payload['goal'])}　|　总耗时 {payload['elapsed_ms']}ms　|
 共 {payload['total_spans']} 个 span</div>
<div>{counts}</div>
<div style="margin-top:14px">{''.join(rows)}</div>
</html>"""
