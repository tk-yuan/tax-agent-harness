"""消息结构helper。

统一构造 OpenAI 兼容格式的消息，避免各处手写 dict 写错字段。
"""

from __future__ import annotations


def system(text: str) -> dict:
    return {"role": "system", "content": text}


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(content: str = "", tool_calls: list[dict] | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": content or ""}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def tool_result(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def text_of(message: dict) -> str:
    """取消息的文本内容（兼容多段 content 的情况）。"""
    c = message.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(
            part.get("text", "") for part in c if isinstance(part, dict)
        )
    return ""


def is_tool_message(message: dict) -> bool:
    return message.get("role") == "tool"


def approx_chars(messages: list[dict]) -> int:
    """粗略估算上下文长度，用于压缩触发判断。"""
    n = 0
    for m in messages:
        n += len(text_of(m))
        for tc in m.get("tool_calls") or []:
            n += len(str(tc.get("function", {}).get("arguments", "")))
            n += len(tc.get("function", {}).get("name", ""))
    return n
