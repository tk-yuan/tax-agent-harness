"""Hook 装配与主循环的边界测试。

这里锁住两个实测踩过的坑：

1. **子 Agent 不能继承主 Agent 的目标闭环**。子 Agent 克隆时共享 HookManager，
   于是 `before_finish` 拿主清单的未完成项去卡子 Agent —— 子 Agent 永远收不了工，
   跑满 max_turns 后返回空结论。
2. **跑满轮数时不能返回空字符串**。模型每轮可能同时输出文字和工具调用，
   最后一轮说的话就是它当时的结论，应当作为兜底返回。
"""

import pytest

from taxharness.context.memory import LayeredMemory
from taxharness.runtime.hooks_impl import build_hooks
from taxharness.runtime.loop import Agent, RunState
from taxharness.runtime.tools import Tool, ToolRegistry
from taxharness.tasks.todo import TaskBoard


def _hooks(tmp_path, board):
    return build_hooks(base_prompt="基础提示", memory=LayeredMemory(tmp_path / "m.jsonl"), board=board)


# —— 目标闭环的作用域 ——
def test_goal_hook_blocks_main_agent_when_tasks_open(tmp_path):
    board = TaskBoard()
    board.add("甲")
    h = _hooks(tmp_path, board)
    out = h.run(
        "before_finish",
        {"answer": "做完了", "state": RunState(depth=0), "messages": []},
    )
    assert out.get("continue") is True


def test_goal_hook_does_not_block_subagent(tmp_path):
    """子 Agent 收工条件就是「模型不再调工具」，不能被主清单拦住。"""
    board = TaskBoard()
    board.add("甲")
    h = _hooks(tmp_path, board)
    out = h.run(
        "before_finish",
        {"answer": "子 Agent 的结论", "state": RunState(depth=1), "messages": []},
    )
    assert not out.get("continue")


def test_goal_hook_passes_when_board_clear(tmp_path):
    board = TaskBoard()
    h = _hooks(tmp_path, board)
    out = h.run(
        "before_finish",
        {"answer": "做完了", "state": RunState(depth=0), "messages": []},
    )
    assert not out.get("continue")


def test_goal_hook_stops_after_max_rounds(tmp_path):
    """兜底：不能无限续跑。"""
    board = TaskBoard()
    board.add("永远做不完的任务")
    h = _hooks(tmp_path, board)
    state = RunState(depth=0)
    seen_continue = 0
    for _ in range(6):
        out = h.run("before_finish", {"answer": "x", "state": state, "messages": []})
        if out.get("continue"):
            seen_continue += 1
    assert seen_continue <= 3, "目标闭环没有上限，会无限续跑"


# —— 任务清单的注入作用域 ——
def test_board_injected_into_main_prompt(tmp_path):
    board = TaskBoard()
    board.add("甲")
    h = _hooks(tmp_path, board)
    out = h.run(
        "before_model",
        {"messages": [{"role": "system", "content": "S"}], "state": RunState(depth=0), "goal": "g"},
    )
    assert "任务清单" in out["messages"][0]["content"]


def test_board_not_injected_into_subagent_prompt(tmp_path):
    board = TaskBoard()
    board.add("甲")
    h = _hooks(tmp_path, board)
    out = h.run(
        "before_model",
        {"messages": [{"role": "system", "content": "S"}], "state": RunState(depth=1), "goal": "g"},
    )
    assert "任务清单" not in out["messages"][0]["content"]


def test_subagent_catalog_not_injected_into_subagent(tmp_path):
    """不能告诉子 Agent「你可以委派」——它的工具集里没有 task。

    实测踩过：子 Agent 收到子 Agent 目录后，反复尝试调用一个它没有的工具，
    白白烧掉好几轮，最后跑满 max_turns。
    """
    board = TaskBoard()
    h = build_hooks(
        base_prompt="P",
        memory=LayeredMemory(tmp_path / "m.jsonl"),
        board=board,
        subagent_catalog="【可委派的子 Agent】\n- policy_researcher：查政策",
    )
    out = h.run(
        "before_model",
        {"messages": [{"role": "system", "content": "S"}], "state": RunState(depth=1), "goal": "g"},
    )
    assert "可委派的子 Agent" not in out["messages"][0]["content"]


def test_subagent_catalog_injected_into_main(tmp_path):
    board = TaskBoard()
    h = build_hooks(
        base_prompt="P",
        memory=LayeredMemory(tmp_path / "m.jsonl"),
        board=board,
        subagent_catalog="【可委派的子 Agent】\n- policy_researcher：查政策",
    )
    out = h.run(
        "before_model",
        {"messages": [{"role": "system", "content": "S"}], "state": RunState(depth=0), "goal": "g"},
    )
    assert "可委派的子 Agent" in out["messages"][0]["content"]


# —— 主循环：跑满轮数时的结论兜底 ——
class _StubLLM:
    """按脚本返回响应，不发网络请求。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.usage = {"calls": 0}

    def chat(self, messages, tools=None, **kw):
        return self.responses.pop(0)


def _registry():
    r = ToolRegistry()
    r.register(Tool(name="noop", description="空操作", parameters={"type": "object"},
                    handler=lambda **kw: {"ok": True}))
    return r


def _tool_call(idx: int) -> dict:
    return {
        "content": "",
        "tool_calls": [{"id": f"c{idx}", "type": "function",
                        "function": {"name": "noop", "arguments": "{}"}}],
        "finish_reason": "tool_calls",
    }


def test_loop_returns_last_text_on_max_turns():
    """一直调工具、不讲结论时，最后一轮说过的文字要作为结论返回。"""
    responses = [_tool_call(1), _tool_call(2), _tool_call(3)]
    llm = _StubLLM(responses)

    # 让第二轮带上一句文字
    llm.responses[1]["content"] = "阶段结论：最优方案是全部加速折旧"

    agent = Agent(_registry(), llm=llm, system_prompt="S", max_turns=3)
    result = agent.run("目标")
    assert result.stop_reason == "max_turns"
    assert result.answer == "阶段结论：最优方案是全部加速折旧"


def test_loop_completes_when_model_stops_calling_tools():
    llm = _StubLLM([{"content": "最终结论", "tool_calls": [], "finish_reason": "stop"}])
    agent = Agent(_registry(), llm=llm, system_prompt="S", max_turns=3)
    result = agent.run("目标")
    assert result.stop_reason == "completed"
    assert result.answer == "最终结论"


def test_loop_handles_model_error():
    class _Boom(_StubLLM):
        def chat(self, *a, **kw):
            raise RuntimeError("网络炸了")

    agent = Agent(_registry(), llm=_Boom([]), system_prompt="S", max_turns=3)
    result = agent.run("目标")
    assert result.stop_reason == "model_error"
    assert "中止" in result.answer
