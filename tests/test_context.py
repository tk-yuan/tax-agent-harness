"""上下文层测试：分层记忆、上下文压缩。"""

import time
from pathlib import Path

from taxharness.context.compaction import compact_messages, spill_tool_result
from taxharness.context.memory import HALF_LIFE_DAYS, LayeredMemory, MemoryEntry


def _mem(tmp_path: Path) -> LayeredMemory:
    return LayeredMemory(tmp_path / "mem.jsonl")


# —— 分层记忆 ——
def test_remember_and_recall(tmp_path):
    m = _mem(tmp_path)
    m.remember("pattern", "设备一次性扣除要放在边际税率高的年度")
    hits = m.recall("一次性扣除 边际税率")
    assert hits and "一次性扣除" in hits[0].content


def test_recall_returns_empty_when_unrelated(tmp_path):
    m = _mem(tmp_path)
    m.remember("pattern", "设备一次性扣除")
    assert m.recall("完全不相关的查询词汇") == []


def test_recall_reinforces_hits(tmp_path):
    """被召回过的经验应该强化——这是「越用越准」的机制。"""
    m = _mem(tmp_path)
    m.remember("pattern", "研发费用加计扣除")
    assert m._entries[0].hits == 0
    m.recall("研发费用加计扣除")
    assert m._entries[0].hits == 1


def test_decay_reduces_weight_over_time(tmp_path):
    """衰减：久未使用的经验权重下降。否则陈旧经验会一直占位。"""
    now = time.time()
    fresh = MemoryEntry(id="a", kind="p", content="x", last_used=now)
    old = MemoryEntry(id="b", kind="p", content="x", last_used=now - HALF_LIFE_DAYS * 86400)
    assert old.decayed_weight(now) < fresh.decayed_weight(now)
    # 半衰期处应约为一半
    assert abs(old.decayed_weight(now) - 0.5) < 0.01


def test_reinforcement_offsets_decay(tmp_path):
    now = time.time()
    used = MemoryEntry(id="a", kind="p", content="x", hits=5, last_used=now)
    unused = MemoryEntry(id="b", kind="p", content="x", hits=0, last_used=now)
    assert used.decayed_weight(now) > unused.decayed_weight(now)


def test_prune_removes_never_used_stale(tmp_path):
    m = _mem(tmp_path)
    m.remember("pattern", "老经验")
    m._entries[0].last_used = time.time() - HALF_LIFE_DAYS * 86400 * 6
    removed = m.prune()
    assert removed == 1
    assert m.stats()["long_term_entries"] == 0


def test_prune_keeps_used_entries(tmp_path):
    m = _mem(tmp_path)
    m.remember("pattern", "常用的经验")
    m._entries[0].last_used = time.time() - HALF_LIFE_DAYS * 86400 * 6
    m._entries[0].hits = 3          # 用过就不淘汰
    assert m.prune() == 0


def test_persistence_across_instances(tmp_path):
    m1 = _mem(tmp_path)
    m1.remember("pattern", "持久化测试内容")
    m2 = _mem(tmp_path)
    assert m2.stats()["long_term_entries"] == 1


def test_working_memory_is_not_persisted(tmp_path):
    """工作记忆只作用于本次任务，不该落盘。"""
    m1 = _mem(tmp_path)
    m1.note("临时笔记")
    m2 = _mem(tmp_path)
    assert m2.working == []


def test_context_block_includes_both_layers(tmp_path):
    m = _mem(tmp_path)
    m.note("本次笔记")
    m.remember("pattern", "历史经验：加计扣除口径")
    block = m.context_block("加计扣除")
    assert "本次任务" in block and "历史经验" in block


# —— 上下文压缩 ——
def test_spill_small_result_untouched(tmp_path):
    text, path = spill_tool_result("很短", "t", tmp_path)
    assert text == "很短" and path is None


def test_spill_large_result_writes_file(tmp_path):
    big = "x" * 5000
    text, path = spill_tool_result(big, "policy_search", tmp_path)
    assert path is not None
    assert Path(path).exists()
    assert len(text) < len(big)
    assert Path(path).read_text(encoding="utf-8") == big


def test_compact_noop_when_under_budget():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    out, info = compact_messages(msgs, budget_chars=10_000)
    assert out == msgs and not info["compacted"]


def test_compact_reduces_size():
    msgs = [{"role": "system", "content": "s"}]
    for i in range(40):
        msgs.append({"role": "user", "content": f"问题{i}" + "x" * 200})
        msgs.append({"role": "assistant", "content": "答" + "y" * 200})
    out, info = compact_messages(msgs, budget_chars=1000, keep_recent=6)
    assert info["compacted"]
    assert info["chars_after"] < info["chars_before"]
    assert len(out) < len(msgs)


def test_compact_never_drops_system_message():
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(40):
        msgs.append({"role": "user", "content": "x" * 500})
    out, _ = compact_messages(msgs, budget_chars=500, keep_recent=4)
    assert out[0]["role"] == "system" and out[0]["content"] == "SYS"


def test_compact_keeps_recent_messages_intact():
    msgs = [{"role": "system", "content": "s"}]
    for i in range(30):
        msgs.append({"role": "user", "content": f"Q{i}" + "x" * 300})
    out, _ = compact_messages(msgs, budget_chars=500, keep_recent=6)
    assert out[-1]["content"] == msgs[-1]["content"]


def test_compact_does_not_split_tool_call_group():
    """裁剪点必须落在组边界上（user 或 assistant+tool_calls）。
    从别处切会把 tool_calls 和 tool 结果拆散，下一次请求 API 直接报错。"""
    msgs = [{"role": "system", "content": "s"}]
    msgs.append({"role": "user", "content": "目标" + "x" * 400})
    for i in range(20):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "t", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "r" * 300})
    out, info = compact_messages(msgs, budget_chars=800, keep_recent=5)
    assert info["compacted"]

    # 压缩摘要之后的第一条业务消息，必须是组边界
    summary_at = next(
        i for i, m in enumerate(out)
        if m["role"] == "user" and "压缩" in (m.get("content") or "")
    )
    first = out[summary_at + 1]
    assert first["role"] == "user" or (
        first["role"] == "assistant" and first.get("tool_calls")
    ), f"裁剪点落在组内部：{first['role']}"


def test_compact_works_when_only_one_user_turn():
    """整段对话只有一个 user（长工具链任务）时，压缩也必须能生效。"""
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "目标"}]
    for i in range(30):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "t", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "r" * 300})
    out, info = compact_messages(msgs, budget_chars=500, keep_recent=4)
    assert info["compacted"], "单 user 的长工具链应当仍可压缩"
    assert info["chars_after"] < info["chars_before"]
