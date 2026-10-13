"""多轮追问的测试。

核心断言只有一条：**带了历史就得接上，没带就得从零开始。**
接不上，追问就退化成独立任务（"那换成 2025 年买的呢？" 根本没法问）；
接错了（把历史当新任务重跑），上下文会串味。
"""

from taxharness.runtime.loop import Agent
from taxharness.runtime.tools import Tool, ToolRegistry


class _Recorder:
    """记录每次模型调用看到的消息角色序列。"""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []
        self.prompts: list[str] = []
        self.turn = 0
        self.usage: dict = {}

    def chat(self, messages, tools=None, **kw):
        self.turn += 1
        self.seen.append([m.get("role") for m in messages])
        self.prompts.append(
            " | ".join(str(m.get("content", ""))[:40] for m in messages)
        )
        return {"content": "结论", "tool_calls": [], "finish_reason": "stop"}


def _agent(llm) -> Agent:
    return Agent(ToolRegistry(), llm=llm, system_prompt="SYS", max_turns=3)


def test_no_history_starts_fresh():
    llm = _Recorder()
    _agent(llm).run("第一个问题")
    assert llm.seen[0] == ["system", "user"]


def test_history_is_prepended():
    """带历史时，消息序列应当是 system + 历史 + 本轮目标。"""
    llm = _Recorder()
    history = [
        {"role": "user", "content": "第一个问题"},
        {"role": "assistant", "content": "第一个答案"},
    ]
    _agent(llm).run("接着追问", history=history)
    assert llm.seen[0] == ["system", "user", "assistant", "user"]


def test_history_without_system_gets_one_prepended():
    """history 里不带 system 也要能跑——Hook 会补上最新的 system。"""
    llm = _Recorder()
    _agent(llm).run("追问", history=[{"role": "user", "content": "上一轮"}])
    assert llm.seen[0][0] == "system"
    assert llm.seen[0].count("system") == 1


def test_previous_content_reaches_the_model():
    """不只是条数对——上一轮的实际内容必须真的传下去了。"""
    llm = _Recorder()
    history = [{"role": "assistant", "content": "上一轮结论：节税 66.7 万元"}]
    _agent(llm).run("那再多呢", history=history)
    assert any("66.7" in p for p in llm.prompts), "上一轮结论没进上下文"


def test_result_carries_messages_for_next_turn():
    """RunResult 必须把消息带出来，且**最终结论也要在里面**。

    少了结论这条，追问时模型只看到上一轮的问题、看不到它自己的回答，
    上下文就断了——这是加多轮会话时才暴露出来的一个真 bug。
    """
    llm = _Recorder()
    r = _agent(llm).run("第一轮")
    assert r.messages[-1]["role"] == "assistant", "最终结论没进消息列表"
    assert r.messages[-1]["content"] == "结论"
    assert r.messages[0]["role"] == "system"


def test_multi_turn_accumulates():
    """连着问两轮，第二轮看到的应当是三轮对话。"""
    llm = _Recorder()
    agent = _agent(llm)
    r1 = agent.run("第一轮")
    r2 = agent.run("第二轮", history=r1.messages)
    # 第一轮：system + user
    assert llm.seen[0] == ["system", "user"]
    # 第二轮：system + (user, assistant) + user
    assert llm.seen[1] == ["system", "user", "assistant", "user"]
    assert len(r2.messages) > len(r1.messages)


def test_history_is_not_mutated_in_place():
    """不能就地改调用方传进来的列表——那会让上一轮的记录被污染。"""
    llm = _Recorder()
    history = [{"role": "user", "content": "上一轮"}]
    snapshot = list(history)
    _agent(llm).run("本轮", history=history)
    assert history == snapshot, "history 被就地修改了"


def test_can_reset_to_fresh_conversation():
    """不带 history 再跑，就应当回到全新对话（对应界面的「新对话」）。"""
    llm = _Recorder()
    agent = _agent(llm)
    r1 = agent.run("第一轮")
    agent.run("新对话的第一个问题")          # 刻意不传 history
    assert llm.seen[1] == ["system", "user"], "没传历史就该从零开始"
    assert len(r1.messages) > 2


def test_tools_still_work_with_history():
    """多轮下工具循环照常工作。"""
    reg = ToolRegistry()
    reg.register(Tool(name="noop", description="d", parameters={"type": "object"},
                      handler=lambda **k: {"ok": True}))

    class _ToolLLM(_Recorder):
        def chat(self, messages, tools=None, **kw):
            self.turn += 1
            self.seen.append([m.get("role") for m in messages])
            if self.turn == 1:
                return {"content": "", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "noop", "arguments": "{}"}}],
                    "finish_reason": "tool_calls"}
            return {"content": "完成", "tool_calls": [], "finish_reason": "stop"}

    llm = _ToolLLM()
    agent = Agent(reg, llm=llm, system_prompt="SYS", max_turns=4)
    hist = [{"role": "user", "content": "上一轮"}, {"role": "assistant", "content": "答"}]
    r = agent.run("用工具", history=hist)
    assert r.stop_reason == "completed"
    assert llm.seen[0] == ["system", "user", "assistant", "user"]
    assert "tool" in llm.seen[1]
