"""子 Agent 委派（对应教程 s06 / s13）。

主 Agent 遇到可独立完成的子任务时，开一个子 Agent 去做，只收回结论。

**核心价值是上下文隔离，不是"多几个 Agent 显得厉害"**：
子 Agent 在自己的上下文里翻几十页政策原文、试十几种资产组合，
主 Agent 只收到一段结论。不这么做，主上下文二十轮内就会被原始材料淹没。

**代价必须说清楚**：子 Agent 看不到主对话历史。所以委派时，
必要背景必须写进 goal。这不是缺陷——它逼你把任务边界想清楚，
写不清楚 goal 就说明任务本身没定义好。

权限收窄：每个子 Agent 只拿到完成自己职责所需的最小工具集。
复核员刻意不给写权限，避免"自己改自己验"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import MAX_SUBAGENT_TURNS
from ..runtime.loop import Agent, RunState
from ..runtime.tools import Tool, ToolRegistry

POLICY_TOOLS = {"policy_search", "list_policies", "read_file", "load_skill"}
CASE_TOOLS = {"case_lookup", "get_case", "feasible_options", "read_file"}
CALC_TOOLS = {"tax_calc", "optimize_plan", "get_case", "feasible_options"}
WRITE_TOOLS = {"write_file", "read_file", "list_dir"}


@dataclass
class SubagentSpec:
    name: str
    description: str
    system_prompt: str
    allow: set[str] | None = None


def default_specs() -> dict[str, SubagentSpec]:
    """三个角色，对应 JD 里的「特征分析 / 规则配置 / 测试验证」类子 Agent。"""
    return {
        "policy_researcher": SubagentSpec(
            name="policy_researcher",
            description="政策研究员：查政策原文、核实有效期与适用条件，输出可追溯的政策依据",
            system_prompt=(
                "你是税收政策研究员。你的任务是把某个业务问题对应的政策依据查清楚。\n"
                "要求：\n"
                "1. 每条结论都给出文号和有效期（用 policy_search / list_policies 查证）。\n"
                "2. 查不到就明说查不到，不要用常识推测政策条文。\n"
                "3. 输出结构化结论：适用政策清单（文号 / 名称 / 有效期 / 适用条件），"
                "以及任何时效上的坑（已过期 / 尚未生效 / 未标注期限）。\n"
                "4. 不要输出检索到的原文长段落，只给结论和文号。"
            ),
            allow=POLICY_TOOLS | {"write_file"},
        ),
        "tax_calculator": SubagentSpec(
            name="tax_calculator",
            description="税负测算员：按给定方案算税负、跑穷举优化、给出与基线的差额",
            system_prompt=(
                "你是税负测算员。你的任务是把筹划方案算准。\n"
                "要求：\n"
                "1. 先 feasible_options 看合法选项，不要凭记忆猜。\n"
                "2. 所有数字必须来自工具返回，不得自行估算。\n"
                "3. 输出：最优处理方式、累计应纳税额、与人工经验法的差额、"
                "以及合规校验结果（audit_plan 的 problems）。\n"
                "4. 只输出结论与关键数字，不要复述逐年明细（明细可落盘）。"
            ),
            allow=CALC_TOOLS | CASE_TOOLS | WRITE_TOOLS | {"load_skill"},
        ),
        "verifier": SubagentSpec(
            name="verifier",
            description="独立复核员：用不同角度验证已有结论是否成立，刻意不给写权限",
            system_prompt=(
                "你是独立复核员。你的职责是**证伪**，不是附和。\n"
                "要求：\n"
                "1. 对给定的结论，逐条检查：数字是否可复算、政策文号是否真的存在且有效、"
                "推理链是否有跳跃。\n"
                "2. 自己动手用 tax_calc / optimize_plan 复算关键数字，不要相信转述。\n"
                "3. 输出格式：\n"
                "   通过项 / 存疑项 / 错误项 三节。存疑和错误项必须写清理由与证据。\n"
                "4. 找不到问题的，明确写「未发现问题」，不要为了凑数硬找。"
            ),
            allow=CALC_TOOLS | POLICY_TOOLS,
        ),
    }


class SubagentRunner:
    """用主 Agent 的依赖克隆出子 Agent，跑完后只返回结论。"""

    def __init__(self, base: Agent, specs: dict[str, SubagentSpec] | None = None,
                 max_turns: int = MAX_SUBAGENT_TURNS) -> None:
        self.base = base
        self.specs = specs or default_specs()
        self.max_turns = max_turns
        self.records: list[dict] = []

    def catalog(self) -> str:
        lines = ["【可委派的子 Agent】（用 task 工具委派，只收回结论，不污染主上下文）"]
        lines += [f"- {s.name}：{s.description}" for s in self.specs.values()]
        return "\n".join(lines)

    def run(self, agent: str, goal: str) -> dict:
        """参数名必须与工具 schema 里的 `agent` / `goal` 一致。

        踩过的坑：方法签名原本是 `run(name, goal)`，
        而工具 schema 声明的是 `agent`，模型按 schema 传 `agent=...`，
        于是每次都报 unexpected keyword argument——
        **工具名对得上、参数名对不上，是最隐蔽的一类接线错误。**
        """
        spec = self.specs.get(agent)
        if spec is None:
            return {"error": f"未知子 Agent：{agent}", "available": list(self.specs)}

        child_state = RunState(
            goal=goal, depth=self.base.state.depth + 1, parent_goal=self.base.state.goal
        )
        sub = self.base.clone(
            system_prompt=spec.system_prompt,
            allow=spec.allow,
            max_turns=self.max_turns,
            state=child_state,
            name=spec.name,
        )

        if self.base.trace is not None:
            with self.base.trace.span("subagent", spec.name, goal=goal[:120]) as span:
                result = sub.run(goal)
                span["fields"]["turns"] = result.turns
                span["fields"]["chars"] = len(result.answer)
        else:
            result = sub.run(goal)

        rec = {
            "agent": spec.name,
            "goal": goal,
            "conclusion": result.answer,
            "turns": result.turns,
            "stop_reason": result.stop_reason,
            "usage": result.usage,
        }
        self.records.append(rec)
        return rec


def register_task_tool(registry: ToolRegistry, runner: SubagentRunner) -> None:
    registry.register(Tool(
        name="task",
        description=(
            "把一个可独立完成的子任务委派给子 Agent，只收回它的结论（不返回中间过程）。"
            "适用于：查一批政策的时效、算一组方案的税负、独立复核已有结论。"
            "委派时必须把背景写进 goal——子 Agent 看不到当前对话历史。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "子 Agent 名称，见 system prompt 的子 Agent 目录",
                },
                "goal": {
                    "type": "string",
                    "description": "子任务的完整描述，包含必要背景与期望输出格式",
                },
            },
            "required": ["agent", "goal"],
        },
        handler=lambda agent, goal: runner.run(agent, goal),
        read_only=True,
        category="delegate",
    ))
