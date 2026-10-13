"""上下文压缩（对应教程 s08）。

两条策略，对付两种膨胀：

  1. **单条工具结果过大** → 落盘到虚拟文件系统，上下文里只留「摘要 + 文件路径」。
     这是 JD 里「将大工具结果、长日志沉淀到虚拟文件系统，仅把阶段性摘要和
     文件链接保留在模型上下文中」的原话。

  2. **整段对话过长** → 把最早的一批消息蒸馏成一条摘要，保留最近若干轮原文。

为什么必须做：Agent 跑长任务时，工具结果线性堆积。不压缩，二十来轮就会撞上
模型上下文上限；即使没撞上，"中间遗忘"（lost in the middle）也会让模型
忽略夹在中间的关键信息。压缩不是省钱技巧，是长任务能否跑完的前提。

**压缩的边界必须安全**：assistant 的 tool_calls 和随后的 tool 结果是一组，
切开就报错。所以裁剪点要往前推到下一个 user 消息，绝不从中间切。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..runtime import messages as M

# 触发落盘的单条结果长度
SPILL_THRESHOLD = 2000
# 压缩时保留的最近消息条数
KEEP_RECENT = 8


def spill_tool_result(
    content: str, tool_name: str, out_dir: Path, keep_head: int = 400
) -> tuple[str, str | None]:
    """把超长工具结果落盘，返回（上下文里保留的摘要, 落盘路径或 None）。

    落盘的文件是**可用资产**：后续 Agent 可以 read_file 再取细节，
    也可以由子 Agent 处理。压缩不等于丢失。
    """
    if len(content) <= SPILL_THRESHOLD:
        return content, None

    out_dir.mkdir(parents=True, exist_ok=True)
    import time

    fname = f"{tool_name}_{time.strftime('%H%M%S')}_{abs(hash(content)) % 10000:04d}.json"
    path = out_dir / fname
    path.write_text(content, encoding="utf-8")

    head = content[:keep_head]
    summary = (
        f"{head}\n"
        f"…[完整结果 {len(content)} 字符已落盘至 {path}，"
        f"需要细节请用 read_file 读取或委派子 Agent 处理]"
    )
    return summary, str(path)


def _cut_index(messages: list[dict], keep_recent: int) -> int:
    """找一个安全的裁剪点。

    安全的边界有两类：
      · user 消息 —— 一轮新输入的开始
      · 带 tool_calls 的 assistant 消息 —— 一次工具调用组的开头

    只有这两处能切：从别处切会把 assistant 的 tool_calls 和随后的 tool 结果
    拆散，下一次请求会被 API 直接拒绝（tool_call_id 找不到对应的调用）。

    注意不能只找 user：像「一次提问触发二十轮工具调用」这种长任务，
    整段对话只有一个 user，只认 user 就永远压不动。
    """
    if len(messages) <= keep_recent + 1:
        return 0
    idx = len(messages) - keep_recent
    while idx > 1:
        m = messages[idx]
        role = m.get("role")
        if role == "user" or (role == "assistant" and m.get("tool_calls")):
            break
        idx -= 1
    return idx


def _rule_summary(old: list[dict]) -> str:
    """不调模型时的兜底摘要：保留调了哪些工具、结果有多长。"""
    lines = []
    for m in old:
        role = m.get("role")
        if role == "tool":
            lines.append(f"- 工具结果（{len(M.text_of(m))} 字符）")
        elif role == "assistant" and m.get("tool_calls"):
            names = [c["function"]["name"] for c in m["tool_calls"]]
            lines.append(f"- 调用了工具：{', '.join(names)}")
        elif role == "user":
            lines.append(f"- 用户输入：{M.text_of(m)[:80]}")
    return "【已压缩的历史】\n" + "\n".join(lines[:40])


def _llm_summary(old: list[dict], llm) -> str:
    """让模型把历史蒸馏成摘要。失败则退回规则摘要。"""
    from ..llm import LLMError

    transcript = []
    for m in old:
        role = m.get("role")
        if role == "tool":
            continue  # 工具原文太占地方，摘要时不重复喂
        if role == "assistant" and m.get("tool_calls"):
            names = [c["function"]["name"] for c in m["tool_calls"]]
            transcript.append(f"[助手] 调用工具 {', '.join(names)}")
        else:
            transcript.append(f"[{role}] {M.text_of(m)[:500]}")

    prompt = (
        "把下面这段 Agent 执行历史压缩成要点，供后续推理使用。\n"
        "必须保留：已经查明的结论与数字、已经调用过的工具及其关键返回、"
        "尚未完成的待办、以及任何结论所依据的政策文号。\n"
        "不要保留：寒暄、重复的中间过程。控制在 300 字以内。\n\n"
        + "\n".join(transcript[-60:])
    )
    try:
        text = llm.complete(prompt, system="你是对话压缩器，只输出压缩后的要点。", max_tokens=600)
        return "【已压缩的历史（模型蒸馏）】\n" + text.strip()
    except LLMError:
        return _rule_summary(old)


def compact_messages(
    messages: list[dict],
    budget_chars: int,
    llm=None,
    keep_recent: int = KEEP_RECENT,
) -> tuple[list[dict], dict]:
    """超预算就压缩。返回（新消息列表, 压缩信息）。"""
    used = M.approx_chars(messages)
    if used <= budget_chars:
        return messages, {"compacted": False, "chars": used}

    cut = _cut_index(messages, keep_recent)
    if cut <= 1:  # 只有 system + 第一条输入，压不动
        return messages, {"compacted": False, "chars": used, "reason": "无可压缩区间"}

    head = messages[:1]           # system 永不压缩
    old = messages[1:cut]
    tail = messages[cut:]

    summary = _llm_summary(old, llm) if llm is not None else _rule_summary(old)
    new = head + [{"role": "user", "content": summary}] + tail

    return new, {
        "compacted": True,
        "chars_before": used,
        "chars_after": M.approx_chars(new),
        "removed_messages": len(old),
        "strategy": "llm" if llm is not None else "rule",
    }
