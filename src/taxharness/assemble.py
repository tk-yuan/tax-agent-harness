"""装配：把所有零件拼成一个可运行的主 Agent。

这是唯一一处"知道全局"的代码。其余模块彼此不知道对方存在，
只通过 ToolRegistry / HookManager / RunState 三个接口耦合。
换掉任何一个零件（比如把自实现检索换成 Milvus），只改这里。
"""

from __future__ import annotations

from .config import DATA_DIR, MODEL, SKILLS_DIR, VIRTUAL_FS, WORKSPACE_DIR
from .context.memory import LayeredMemory
from .llm import LLMClient
from .observability.trace import Trace
from .plugins.builtin_workflows import register_builtin_workflows
from .plugins.goal import GoalChecker
from .plugins.mcp import build_demo_servers
from .plugins.workflow import WorkflowRuntime
from .runtime.hooks_impl import build_hooks
from .runtime.loop import Agent, RunState
from .runtime.middleware import LatencyTracker, ToolCache
from .runtime.permissions import PermissionPolicy
from .runtime.tools import Tool, ToolRegistry
from .skills.loader import SkillLibrary, register_skill_tools
from .tasks.subagent import SubagentRunner, default_specs, register_task_tool
from .tasks.todo import TaskBoard
from .tools.builtin import register_fs_tools
from .tools.tax import register_tax_tools

MAIN_SYSTEM = """你是企业税务筹划决策 Agent。你的职责是：在**合法合规**的前提下，
为企业找到税负最小的政策组合方案，并给出可追溯的依据。

工作原则：

1. **先建清单再动手**。遇到多步任务，先用 task_board 建任务清单；
   任务间有先后依赖时用 blocked_by 表达（例如"设计扣除方案"依赖"核实时效"）。
   有前置未完成的任务不要提前开工。

2. **数字必须来自工具**。任何税额都不能自己估算——用 tax_calc / optimize_plan 算。

3. **依据必须可追溯**。引用政策要写文号；查不到就说查不到，不要用常识填充。

4. **时效是红线**。任何"能不能用某项优惠"的判断，都要核实政策有效期。

5. **大产出落盘**。报告、明细用 write_file 写进工作区对应的分区，
   对话里只留摘要和路径——上下文要留给推理，不是留给搬运。
   报告较长时**分节追加**（第一节用覆盖，之后每节 append=true），
   一次写太长会被截断。

   工作区布局（请把文件放到对应分区，别都堆在根目录）：

{fs_layout}

6. **该委派就委派**。查政策、跑测算、独立复核这类可独立的子任务，
   用 task 工具交给子 Agent，只收回结论。

7. **收工前自查**。任务清单没做完不许收工；完成了就如实标记。
"""

# 把虚拟文件系统布局渲染进系统提示，让模型知道文件该放哪
_FS_LAYOUT = "\n".join(f"     workspace/{k}/   —— {v}" for k, v in VIRTUAL_FS.items())
MAIN_SYSTEM = MAIN_SYSTEM.replace("{fs_layout}", _FS_LAYOUT)


def register_board_tool(registry: ToolRegistry, board: TaskBoard) -> None:
    registry.register(Tool(
        name="task_board",
        description=(
            "任务清单操作。action: add(新建) / start(开工) / complete(完成) / fail(失败) / view(查看)。"
            "新建时 blocked_by 传前置任务 id 列表。前置未完成的任务不能 start。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "start", "complete", "fail", "view"]},
                "task_id": {"type": "string", "description": "任务 id，add 时可省略"},
                "subject": {"type": "string", "description": "任务描述，add 时必填"},
                "blocked_by": {"type": "array", "items": {"type": "string"},
                               "description": "前置任务 id 列表，add 时可选"},
                "result": {"type": "string", "description": "complete/fail 时的结论或原因"},
            },
            "required": ["action"],
        },
        handler=lambda action, task_id="", subject="", blocked_by=None, result="": _board_action(
            board, action, task_id, subject, blocked_by, result
        ),
        read_only=True,
        category="task",
    ))


def _norm_id(raw) -> str:
    """把任务编号归一化。

    模型经常会写 "1" 而不是 "T1"（任务清单里显示的是 T1，但它记住了序号）。
    不归一化的话，start("T2") 里 blocked_by=["1"] 永远匹配不上，
    任务被永久阻塞——实测中模型因此把整张清单重建了一遍。
    """
    s = str(raw or "").strip().upper()
    if not s:
        return ""
    return f"T{s}" if s[0].isdigit() else s


def _board_action(board: TaskBoard, action: str, task_id: str, subject: str,
                  blocked_by, result: str) -> dict:
    """任务清单操作。每次都把整张清单连同结果一起返回，
    让模型始终看到真实状态——否则它会凭记忆猜，然后重复建任务。"""
    task_id = _norm_id(task_id)
    deps = [_norm_id(b) for b in (blocked_by or [])]

    if action == "add":
        if not subject:
            return {"error": "add 需要 subject", "board": board.render()}
        unknown = [d for d in deps if board._find(d) is None]
        if unknown:
            return {
                "error": f"blocked_by 引用了不存在的任务：{unknown}。"
                         f"现有任务：{[t.id for t in board.items]}",
                "board": board.render(),
            }
        t = board.add(subject, deps or None)
        return {"ok": True, "task_id": t.id, "board": board.render(),
                "progress": board.progress()}

    if action == "view":
        return {"board": board.render(), "tasks": board.to_dict(),
                "progress": board.progress()}

    if not task_id:
        return {"error": f"{action} 需要 task_id（形如 T1）", "board": board.render()}

    if action == "start":
        out = board.start(task_id)
    elif action == "complete":
        out = board.complete(task_id, result)
    elif action == "fail":
        out = board.fail(task_id, result)
    else:
        return {"error": f"未知 action：{action}", "board": board.render()}

    out["board"] = board.render()
    out["progress"] = board.progress()
    return out


def build_harness(goal: str = "", trace: Trace | None = None,
                  goal_condition: str = "", approval_gate=None) -> dict:
    """组装一个完整的主 Agent。

    goal_condition 传入后启用「目标闭环」的模型判定（对应教程 s17）：
    模型说完成任务不算数，另起一次独立调用判定是否真达成。

    approval_gate 传入后启用「人工审批门」：高风险工具调用会阻塞等人批复。
    CLI 模式传一个控制台提示的 gate，服务模式传一个经 SSE 推给前端的 gate。
    """
    registry = ToolRegistry()
    policy = PermissionPolicy()
    board = TaskBoard()
    memory = LayeredMemory()
    skills = SkillLibrary(SKILLS_DIR)
    llm = LLMClient()

    # —— 基础工具 ——
    register_tax_tools(registry)
    register_fs_tools(registry, policy)
    register_skill_tools(registry, skills)
    register_board_tool(registry, board)

    agent = Agent(
        registry, policy=policy, llm=llm, system_prompt=MAIN_SYSTEM,
        trace=trace, state=RunState(), approval_gate=approval_gate,
    )

    # —— 子 Agent 委派 ——
    runner = SubagentRunner(agent, default_specs())
    register_task_tool(registry, runner)

    # —— 插件一：MCP 外部工具（动态并入工具池）——
    mcp_clients = build_demo_servers(DATA_DIR)
    mcp_added: list[str] = []
    for client in mcp_clients:
        mcp_added += client.connect(registry)

    # —— 插件二：Workflow 确定性编排 ——
    workflows = WorkflowRuntime(runner)
    register_builtin_workflows(workflows)
    registry.register(Tool(
        name="run_workflow",
        description=(
            "运行一个预定义的固定流程（宿主注册的可信脚本，模型只能选名字和参数）。"
            "适用于已知步骤顺序的任务，比逐轮自行决策更快更稳，且中断可续跑。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "workflow 名称"},
                "args": {"type": "object", "description": "参数，如 {\"case_id\": \"...\"}"},
                "run_id": {"type": "string", "description": "续跑时传入此前的 run_id"},
            },
            "required": ["name"],
        },
        handler=lambda name, args=None, run_id=None: workflows.run(name, args, run_id),
        read_only=False,
        category="workflow",
    ))

    # —— 插件三：目标闭环判定器 ——
    goal_checker = GoalChecker(llm, condition=goal_condition)
    goal_rounds_hint = (
        "【可用 workflow】\n"
        + "\n".join(f"- {w['name']}：{w['description']}" for w in workflows.catalog())
        + "\n（步骤顺序固定的任务优先用 run_workflow，而不是自己一步步来）"
    )

    # —— 补充中间件：缓存、耗时统计 ——
    tool_cache = ToolCache()
    latency = LatencyTracker()

    hooks = build_hooks(
        base_prompt=MAIN_SYSTEM,
        memory=memory,
        board=board,
        skill_catalog=skills.catalog(),
        subagent_catalog=runner.catalog() + "\n\n" + goal_rounds_hint,
        llm=llm,
        trace=trace,
        goal_checker=goal_checker,
        tool_cache=tool_cache,
        latency=latency,
    )
    agent.hooks = hooks

    return {
        "agent": agent, "registry": registry, "policy": policy, "board": board,
        "memory": memory, "skills": skills, "runner": runner, "llm": llm,
        "hooks": hooks, "workflows": workflows, "goal_checker": goal_checker,
        "approval_gate": approval_gate, "cache": tool_cache, "latency": latency,
        "mcp_tools": mcp_added, "workspace": WORKSPACE_DIR, "model": MODEL,
    }
