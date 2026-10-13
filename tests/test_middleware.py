"""补充中间件的测试：缓存、重试、PII 检测。

缓存这块重点测**边界**：有副作用的工具绝不能被缓存——
缓存它们等于把第二次调用静默吞掉，是最危险的一类 bug。
"""

import json
import time

import pytest

from taxharness.runtime.loop import Agent
from taxharness.runtime.middleware import (
    LatencyTracker,
    ToolCache,
    cache_hook,
    mask_text,
    pii_guard_hook,
    retry_policy_hook,
)
from taxharness.runtime.tools import Tool, ToolRegistry


def _tool(name, category="policy", read_only=True, requires_approval=False, fn=None):
    return Tool(name=name, description="d", parameters={},
                handler=fn or (lambda **k: {"ok": True}),
                read_only=read_only, category=category,
                requires_approval=requires_approval)


# —— 缓存：可用性判定 ——
def test_readonly_allowlisted_tool_is_cacheable():
    c = ToolCache()
    assert c.eligible(_tool("policy_search"), {"query": "x"})


def test_write_tool_never_cached():
    """写操作绝不能缓存。"""
    c = ToolCache()
    assert not c.eligible(_tool("write_file", category="fs", read_only=False), {})


def test_fs_category_never_cached():
    c = ToolCache()
    assert not c.eligible(_tool("read_file", category="fs"), {})


def test_submit_category_never_cached():
    c = ToolCache()
    assert not c.eligible(_tool("submit_plan", category="submit", read_only=False), {})


def test_approval_required_tool_never_cached():
    """要人工审批的工具也不能缓存——否则审批被绕过。"""
    c = ToolCache()
    assert not c.eligible(_tool("policy_search", requires_approval=True), {})


def test_unknown_tool_not_cached():
    c = ToolCache()
    assert not c.eligible(_tool("some_new_tool"), {})


# —— 缓存：行为 ——
def test_cache_key_is_argument_order_independent():
    a = ToolCache.key("t", {"x": 1, "y": 2})
    b = ToolCache.key("t", {"y": 2, "x": 1})
    assert a == b


def test_cache_key_differs_by_args():
    assert ToolCache.key("t", {"x": 1}) != ToolCache.key("t", {"x": 2})


def test_cache_hit_sets_short_circuit():
    c = ToolCache()
    before, after = cache_hook(c)
    tool = _tool("policy_search")
    args = {"query": "小微"}

    assert before({"tool": tool, "args": args}) is None      # 首次未命中
    after({"tool": tool, "args": args, "result": {"hits": 1}})
    hit = before({"tool": tool, "args": args})
    assert hit and hit["short_circuit"] == {"hits": 1}
    assert c.stats()["hits"] == 1


def test_cache_evicts_when_full():
    c = ToolCache(maxsize=3)
    for i in range(5):
        c.put(f"k{i}", i)
    assert c.stats()["size"] == 3


def test_cache_stats_hit_rate():
    c = ToolCache()
    c.get("a")
    c.put("a", 1)
    c.get("a")
    assert c.stats()["hit_rate"] == 0.5


# —— 缓存：真的能挡住执行 ——
class _RepeatLLM:
    """连续两轮发出完全相同的调用，第二轮应命中缓存。"""

    def __init__(self):
        self.turn = 0
        self.usage = {"calls": 0}

    def chat(self, messages, tools=None, **kw):
        self.turn += 1
        if self.turn == 1:
            return {"content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "counted", "arguments": '{"k": 1}'}}],
                "finish_reason": "tool_calls"}
        if self.turn == 2:
            return {"content": "", "tool_calls": [
                {"id": "c2", "type": "function",
                 "function": {"name": "counted", "arguments": '{"k": 1}'}}],
                "finish_reason": "tool_calls"}
        return {"content": "done", "tool_calls": [], "finish_reason": "stop"}


def test_cache_prevents_second_execution(tmp_path):
    calls: list = []
    reg = ToolRegistry()
    reg.register(Tool(name="counted", description="d", parameters={"type": "object"},
                      handler=lambda k: (calls.append(k), {"v": k})[1],
                      read_only=True, category="policy"))

    cache = ToolCache(allow=("counted",))
    before, after = cache_hook(cache)
    from taxharness.runtime.hooks import HookManager

    hooks = HookManager()
    hooks.register("before_tool", before)
    hooks.register("after_tool", after)

    agent = Agent(reg, hooks=hooks, llm=_RepeatLLM(), system_prompt="s", max_turns=4)
    agent.run("go")
    assert len(calls) == 1, "第二次相同调用应当命中缓存、不再执行"
    assert cache.stats()["hits"] == 1


# —— 重试 ——
def test_retry_policy_hook_declares_budget():
    ctx = retry_policy_hook(max_retry=3, backoff=0.1)({})
    assert ctx["max_retry"] == 3


class _Flaky:
    """前两次抛异常，第三次成功。"""

    def __init__(self):
        self.n = 0
        self.turn = 0
        self.usage = {"calls": 0}

    def chat(self, messages, tools=None, **kw):
        self.turn += 1
        if self.turn == 1:
            return {"content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "flaky", "arguments": "{}"}}],
                "finish_reason": "tool_calls"}
        return {"content": "done", "tool_calls": [], "finish_reason": "stop"}


def test_exception_is_retried_up_to_budget():
    state = {"n": 0}

    def handler():
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("网络抖动")
        return {"ok": True}

    reg = ToolRegistry()
    reg.register(Tool(name="flaky", description="d", parameters={},
                      handler=handler, read_only=True, category="policy"))

    from taxharness.runtime.hooks import HookManager

    hooks = HookManager()
    hooks.register("before_tool", retry_policy_hook(max_retry=3, backoff=0.01))

    agent = Agent(reg, hooks=hooks, llm=_Flaky(), system_prompt="s", max_turns=3)
    agent.run("go")
    assert state["n"] == 3, "应当重试到成功"


def test_retry_exhausted_returns_error_not_crash():
    def always_fail():
        raise RuntimeError("一直失败")

    reg = ToolRegistry()
    reg.register(Tool(name="flaky", description="d", parameters={},
                      handler=always_fail, read_only=True, category="policy"))

    from taxharness.runtime.hooks import HookManager

    hooks = HookManager()
    hooks.register("before_tool", retry_policy_hook(max_retry=2, backoff=0.01))

    agent = Agent(reg, hooks=hooks, llm=_Flaky(), system_prompt="s", max_turns=3)
    r = agent.run("go")
    assert r.stop_reason == "completed"      # 任务不因工具失败而崩


# —— PII ——
@pytest.mark.parametrize("text,kind", [
    ("客户手机 13812345678", "phone"),
    ("身份证 11010519491231002X", "id_card"),
    ("邮箱 zhangsan@example.com", "email"),
])
def test_mask_detects_common_pii(text, kind):
    masked, hits = mask_text(text)
    assert any(kind in h for h in hits), f"未识别出 {kind}"
    assert masked != text


def test_mask_keeps_head_and_tail():
    masked, _ = mask_text("13812345678")
    assert masked.startswith("13") and masked.endswith("78")
    assert "*" in masked


def test_mask_leaves_normal_text_alone():
    text = "我们公司买了一台 400 万的设备"
    masked, hits = mask_text(text)
    assert masked == text and hits == []


def test_case_id_not_mistaken_for_pii():
    """算例编号里有数字，但不是 PII，不能被打码。"""
    masked, hits = mask_text("size_large_104_920008")
    assert masked == "size_large_104_920008"


def test_pii_guard_masks_args():
    before, _ = pii_guard_hook()
    out = before({"args": {"contact": "13812345678"}})
    assert out and "pii_hits" in out
    assert "*" in out["args"]["contact"]


def test_pii_guard_masks_result():
    _, after = pii_guard_hook()
    out = after({"result": "联系人手机 13812345678"})
    assert out and "*" in out["result"]


def test_pii_guard_disabled_changes_nothing():
    before, after = pii_guard_hook(enabled=False)
    assert before({"args": {"x": "13812345678"}}) is None
    assert after({"result": "13812345678"}) is None


# —— 耗时统计 ——
def test_latency_tracker_aggregates():
    lt = LatencyTracker()
    for ms in (10.0, 30.0):
        lt.after({"name": "policy_search", "duration_ms": ms})
    s = lt.stats()["policy_search"]
    assert s["calls"] == 2 and s["avg_ms"] == 20.0 and s["max_ms"] == 30.0


def test_latency_tracker_ignores_missing_duration():
    lt = LatencyTracker()
    lt.after({"name": "x"})
    assert lt.stats() == {}
