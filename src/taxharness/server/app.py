"""FastAPI + ASGI 服务层。

对应 JD 原话：「后端采用 FastAPI + ASGI 实现高性能服务」。

端点：
    GET  /                          单页控制台（实时事件流 + 任务清单 + 审批面板）
    GET  /health                    健康检查
    GET  /tools                     当前工具池（含是否需要审批）
    POST /runs                      提交任务，返回 run_id
    GET  /runs                      历史任务列表
    GET  /runs/{id}                 任务详情与结论
    GET  /runs/{id}/events          SSE 实时事件流
    GET  /runs/{id}/approvals       待审批列表
    POST /runs/{id}/approvals/{aid} 提交审批决定
    GET  /traces/{trace_id}         离线链路时间线（HTML）

为什么是 SSE 而不是 WebSocket：执行事件是**单向**的（服务端 → 浏览器），
SSE 足够且能自动重连、走普通 HTTP、不引入额外协议负担。
"""

from __future__ import annotations

import json
import queue
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..assemble import build_harness
from ..config import TRACE_DIR
from ..storage.store import list_checkpoints, load_checkpoint, shared_store
from .service import RunManager

app = FastAPI(title="税筹决策 Agent Harness", version="0.2.0")
MANAGER = RunManager()


# —— 请求体 ——
class RunIn(BaseModel):
    goal: str = Field(..., min_length=2, description="企业税负筹划目标")
    subagent: str | None = None


class ApprovalIn(BaseModel):
    approved: bool
    decider: str = "operator"
    note: str = ""


# —— 基础 ——
@app.get("/health")
def health() -> dict:
    return {"ok": True, "runs": len(MANAGER.runs),
            "active": sum(1 for r in MANAGER.runs.values()
                          if r.status in ("running", "awaiting_approval"))}


@app.get("/tools")
def tools() -> dict:
    """把工具池连同「是否需要审批」一起列出来——审批规则是可查的，不是黑箱。"""
    bundle = build_harness()
    out = []
    for name in bundle["registry"].names():
        t = bundle["registry"].get(name)
        out.append({
            "name": t.name, "category": t.category,
            "read_only": t.read_only, "requires_approval": t.requires_approval,
            "description": t.description.split("\n")[0][:80],
        })
    return {"count": len(out), "tools": out}


# —— 任务 ——
@app.post("/runs")
def create_run(body: RunIn) -> dict:
    rec = MANAGER.start(body.goal)
    return {"run_id": rec.run_id, "status": rec.status}


@app.get("/runs")
def list_runs() -> dict:
    items = sorted(MANAGER.runs.values(), key=lambda r: -r.created)
    return {"count": len(items), "runs": [r.to_dict() for r in items[:20]]}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    rec = MANAGER.get(run_id)
    if rec is None:
        raise HTTPException(404, f"未知任务 {run_id}")
    return rec.to_dict()


@app.get("/runs/{run_id}/approvals")
def list_approvals(run_id: str) -> dict:
    rec = MANAGER.get(run_id)
    if rec is None:
        raise HTTPException(404, f"未知任务 {run_id}")
    return {"pending": rec.pending_approvals}


@app.post("/runs/{run_id}/approvals/{approval_id}")
def decide_approval(run_id: str, approval_id: str, body: ApprovalIn) -> dict:
    ok = MANAGER.resolve_approval(run_id, approval_id, body.approved,
                                  decider=body.decider, note=body.note)
    if not ok:
        raise HTTPException(409, "该审批不存在或已处理")
    return {"ok": True, "approval_id": approval_id, "approved": body.approved}


# —— 断点与续跑（短期存储）——
@app.get("/checkpoints")
def checkpoints() -> dict:
    """列出所有任务快照。断点信息落的是本地 sqlite，生产环境应换 Redis。"""
    items = list_checkpoints(shared_store())
    items.sort(key=lambda c: -c.get("saved_at", 0))
    return {
        "count": len(items),
        "checkpoints": [
            {
                "run_id": c["run_id"], "goal": c.get("goal", "")[:80],
                "status": c.get("status"), "turns": c.get("turns"),
                "open_tasks": [t["id"] for t in c.get("board", [])
                               if t.get("status") not in ("completed", "failed")],
                "saved_at": c.get("saved_at"),
            }
            for c in items[:20]
        ],
    }


@app.get("/runs/{run_id}/checkpoint")
def get_checkpoint(run_id: str) -> dict:
    cp = load_checkpoint(shared_store(), run_id)
    if cp is None:
        raise HTTPException(404, f"未找到任务 {run_id} 的快照")
    return cp


@app.post("/runs/{run_id}/resume")
def resume_run(run_id: str) -> dict:
    """从断点续跑，返回新任务 id。"""
    rec = MANAGER.resume(run_id)
    if rec is None:
        raise HTTPException(404, f"未找到任务 {run_id} 的快照，无法续跑")
    return {"resumed_from": run_id, "run_id": rec.run_id, "status": rec.status}


# —— 事件流 ——
def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/runs/{run_id}/events")
def run_events(run_id: str) -> StreamingResponse:
    q = MANAGER.queue_of(run_id)
    if q is None:
        raise HTTPException(404, f"未知任务 {run_id}")

    def gen():
        yield _sse({"type": "hello", "run_id": run_id})
        while True:
            try:
                item = q.get(timeout=15)
            except queue.Empty:
                yield ": keepalive\n\n"   # 心跳，防中间层掐断空闲连接
                continue
            yield _sse(item)
            if item.get("type") in ("done", "error"):
                break

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/traces/{trace_id}")
def trace_html(trace_id: str) -> HTMLResponse:
    p: Path = TRACE_DIR / f"{trace_id}.html"
    if not p.exists():
        raise HTTPException(404, f"未找到链路 {trace_id}")
    return HTMLResponse(p.read_text(encoding="utf-8"))


# —— 单页控制台 ——
PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>税筹决策 Agent Harness</title>
<style>
:root{--bg:#f7f8fa;--panel:#ffffff;--line:#e5e7eb;--fg:#1f2328;--dim:#6b7280;
      --blue:#2563eb;--green:#059669;--orange:#d97706;--red:#dc2626;--purple:#7c3aed;
      --shadow:0 1px 2px rgba(16,24,40,.06),0 1px 3px rgba(16,24,40,.08)}
*{box-sizing:border-box}
body{margin:0;font:13.5px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
     background:var(--bg);color:var(--fg)}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
       align-items:center;gap:14px;background:var(--panel);box-shadow:var(--shadow)}
h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.01em}
#status{margin-left:auto;font-size:12px;color:var(--dim)}
.wrap{display:grid;grid-template-columns:1fr 380px;height:calc(100vh - 57px)}
.left,.right{overflow:auto;padding:16px 20px}
.right{border-left:1px solid var(--line);background:#fbfcfd}
.bar{display:flex;gap:8px;margin-bottom:14px}
input{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:8px;
      padding:9px 12px;color:var(--fg);font-size:13px;box-shadow:var(--shadow)}
input:focus{outline:none;border-color:var(--blue);box-shadow:0 0 0 3px rgba(37,99,235,.12)}
button{background:var(--blue);border:0;border-radius:8px;padding:9px 18px;color:#fff;
       font-size:13px;font-weight:500;cursor:pointer;box-shadow:var(--shadow)}
button:hover{filter:brightness(1.08)} button:disabled{opacity:.45;cursor:default}
.sec{font-size:11.5px;letter-spacing:.06em;color:var(--dim);text-transform:uppercase;
     font-weight:600;margin:18px 0 8px}
.ev{display:grid;grid-template-columns:66px 84px minmax(0,1fr);gap:9px;align-items:baseline;
    padding:5px 8px;border-radius:6px;font-size:12.5px;overflow:hidden}
.ev:hover{background:#f1f3f6}
.t{color:#9aa1ab;font-variant-numeric:tabular-nums;font-size:11.5px}
.k{border-radius:4px;padding:1px 6px;font-size:11px;color:#fff;text-align:center;
   font-weight:500}
.k-model{background:var(--blue)} .k-tool{background:var(--green)}
.k-subagent{background:var(--orange)} .k-task{background:var(--purple)}
.k-context{background:#0d9488} .k-memory{background:#0d9488} .k-fs{background:#94a3b8}
.k-approval{background:var(--red)} .k-goal{background:#ea580c}
.k-cache{background:#0891b2} .k-retry{background:#b45309}
/* min-width:0 是关键：grid 子项默认 min-width:auto，长内容会把列撑开、
   整个列表出现横向滚动条。这一行是修那个的。 */
.d{color:#3f4652;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ans{background:var(--panel);border:1px solid var(--line);border-radius:10px;
     padding:14px 16px;white-space:pre-wrap;font-size:12.5px;max-height:340px;
     overflow:auto;box-shadow:var(--shadow);line-height:1.7}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
      padding:11px 13px;margin-bottom:9px;font-size:12.5px;box-shadow:var(--shadow)}
.card.warn{border-color:#fcd9a5;background:#fffbf3}
.card b{color:var(--orange)}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:11.5px;color:var(--dim)}
.pill{display:inline-block;background:#f1f3f6;border:1px solid var(--line);border-radius:11px;
      padding:2px 10px;margin:0 5px 5px 0;font-size:11.5px;color:#5b6472}
.ok{color:var(--green)} .no{color:var(--red)}
.pending{color:var(--orange)}
</style>
<header>
  <h1>税筹决策 Agent Harness</h1>
  <span class="pill" id="meta">连接中…</span>
  <span id="status">空闲</span>
</header>
<div class="wrap">
  <div class="left">
    <div class="bar">
      <input id="goal" placeholder="例如：帮 size_large_104_920008 做税负筹划，给出最优方案与节税金额"
             value="帮 size_large_104_920008 这家企业做税负筹划，给出最优方案和节税金额">
      <button id="go">运行</button>
    </div>
    <div class="sec">执行链路（实时）</div>
    <div id="events"></div>
    <div class="sec">结论</div>
    <div class="ans" id="answer">（尚未运行）</div>
  </div>
  <div class="right">
    <div class="sec">待审批</div>
    <div id="approvals"><div class="mono">无</div></div>
    <div class="sec">任务清单</div>
    <div id="board"><div class="mono">无</div></div>
    <div class="sec">落盘产出</div>
    <div id="artifacts"><div class="mono">无</div></div>
    <div class="sec">运行统计</div>
    <div id="stats"><div class="mono">—</div></div>
    <div class="sec">链路文件</div>
    <div id="trace"><div class="mono">—</div></div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
// t0 必须初始化成当前时间：页面加载时会先拉一次 /tools 并写一条事件，
// 那时若 t0 还是 0，时间戳会显示成 17 亿秒这种荒唐值。
let es=null, t0=Date.now();

function meta(d){
  $('meta').textContent = `工具 ${d.tools.length} · 技能 ${d.skills.length} · `
    + `子Agent ${d.subagents.length} · MCP ${d.mcp_tools.length} · ${d.model}`;
}
function ev(kind,name,extra=''){
  const el=document.createElement('div'); el.className='ev';
  el.innerHTML=`<span class="t">${((Date.now()-t0)/1000).toFixed(1)}s</span>`
    +`<span class="k k-${kind}">${kind}</span>`
    +`<span class="d">${name}${extra?' <span class="mono">'+extra+'</span>':''}</span>`;
  $('events').appendChild(el); $('events').scrollTop=1e9;
}
function board(items){
  if(!items.length){$('board').innerHTML='<div class="mono">无</div>';return;}
  const m={pending:'·',in_progress:'▶',completed:'✓',failed:'✗'};
  $('board').innerHTML=items.map(t=>`<div class="card"><b>${m[t.status]||'·'}</b> `
    +`${t.id} ${t.subject}${t.blocked_by.length?' <span class="mono">←'+t.blocked_by.join(',')+'</span>':''}`
    +(t.result?`<div class="mono">${t.result.slice(0,120)}</div>`:'')+`</div>`).join('');
}
function approval(req){
  const d=document.createElement('div'); d.className='card warn'; d.id='ap-'+req.id;
  d.innerHTML=`<b>需人工审批</b> · 风险 ${req.risk}<div>工具 <span class="mono">${req.tool}</span></div>`
    +`<div class="mono">${JSON.stringify(req.args).slice(0,220)}</div>`
    +`<div>${req.reason}</div><div class="mono">${req.timeout_sec}s 内未批复将自动拒绝</div>`
    +`<div style="margin-top:8px;display:flex;gap:8px">`
    +`<button onclick="decide('${req.id}',true)">批准</button>`
    +`<button style="background:#fff;color:#dc2626;border:1px solid #f3c9c9" `
    +`onclick="decide('${req.id}',false)">拒绝</button></div>`;
  const box=$('approvals'); if(box.querySelector('.mono')) box.innerHTML='';
  box.appendChild(d);
}
async function decide(id,ok){
  const r=await fetch(`/runs/${window.rid}/approvals/${id}`,{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({approved:ok,decider:'console',note:ok?'批准':'拒绝'})});
  if(r.ok){const el=$('ap-'+id); if(el) el.remove();}
}
$('go').onclick=async()=>{
  const goal=$('goal').value.trim(); if(!goal) return;
  $('events').innerHTML=''; $('answer').textContent='（运行中…）';
  $('board').innerHTML='<div class="mono">无</div>'; $('approvals').innerHTML='<div class="mono">无</div>';
  $('artifacts').innerHTML='<div class="mono">无</div>'; $('stats').innerHTML='<div class="mono">—</div>';
  $('go').disabled=true; $('status').textContent='运行中'; t0=Date.now();
  const r=await fetch('/runs',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({goal})});
  const {run_id}=await r.json(); window.rid=run_id;
  if(es) es.close();
  es=new EventSource(`/runs/${run_id}/events`);
  es.onmessage=e=>{
    const d=JSON.parse(e.data);
    if(d.type==='start'){meta(d); ev('task','任务启动',`trace ${d.trace_id}`);}
    else if(d.type==='span'){ev(d.kind,d.name,`${d.duration_ms}ms`);}
    else if(d.type==='event'){ev(d.kind,d.name);}
    else if(d.type==='approval_request'){approval(d.request); ev('approval','等待人工审批',d.request.tool);}
    else if(d.type==='approval_resolved'){ev('approval',d.approved?'已批准':'已拒绝',d.decider);}
    else if(d.type==='done'){
      $('status').textContent=`完成 · ${d.turns} 轮 · ${(d.elapsed_ms/1000).toFixed(1)}s`;
      $('answer').textContent=d.answer||'（无结论）';
      board(d.board||[]);
      const u=d.usage||{};
      $('stats').innerHTML=`<div class="mono">轮数 ${d.turns} · 调用 ${u.calls} · `
        +`token ${u.total_tokens}（提示 ${u.prompt_tokens} / 生成 ${u.completion_tokens}）</div>`;
      $('artifacts').innerHTML=(d.artifacts||[]).length
        ? d.artifacts.map(a=>`<div class="mono">${a}</div>`).join('')
        : '<div class="mono">无</div>';
      $('trace').innerHTML=`<a style="color:#4c6ef5" href="/traces/${d.trace_id}" target="_blank">`
        +`${d.trace_id}.html</a>`;
      $('go').disabled=false;
    }
    else if(d.type==='error'){$('status').textContent='失败';$('answer').textContent=d.error;$('go').disabled=false;}
  };
};
fetch('/tools').then(r=>r.json()).then(d=>{
  const ap=d.tools.filter(t=>t.requires_approval).map(t=>t.name);
  if(!$('meta').textContent.startsWith('工具')) $('meta').textContent=`工具 ${d.count} 个`;
  if(ap.length) ev('approval','需要审批的工具',ap.join(', '));
}).catch(()=>{});
</script>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(PAGE)
