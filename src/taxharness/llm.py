"""LLM 客户端：直接调 OpenAI 兼容接口，支持 function calling。

为什么不用 LangChain / LangGraph 的 Agent 抽象：
  本项目要交付的能力就是「运行时」本身——主循环、上下文压缩的触发时机、
  Hook 的注入点、中断恢复。这些恰恰是框架替你做掉的部分。
  用框架搭出来的多智能体，面试官问一句「你怎么控制压缩时机」就答不上来了。

  所以：模型调用层自己写（约 200 行），换来对全链路的完全控制。
"""

from __future__ import annotations

import json
import time

import httpx

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL

TIMEOUT = 120.0
MAX_RETRY = 3


class LLMError(RuntimeError):
    pass


class LLMClient:
    """OpenAI 兼容的对话客户端，带用量统计。"""

    def __init__(self, model: str = MODEL) -> None:
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.elapsed = 0.0

    @property
    def usage(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "elapsed_sec": round(self.elapsed, 2),
        }

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8000,
    ) -> dict:
        """一次模型调用，返回 {'content', 'tool_calls', 'finish_reason'}。

        max_tokens 默认给到 8000，是因为**工具调用的参数也算输出的 token**。
        模型写一份报告时，整篇正文是作为 write_file 的 arguments 输出的——
        token 上限太小会把 JSON 拦腰截断，参数直接解析失败。
        （本项目实测：2048 时报告类工具调用连续失败，调到 8000 后正常。）
        """
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = DEEPSEEK_BASE_URL.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        last: Exception | None = None
        for attempt in range(MAX_RETRY):
            try:
                t0 = time.time()
                r = httpx.post(url, headers=headers, json=payload, timeout=TIMEOUT)
                self.elapsed += time.time() - t0
                r.raise_for_status()
                data = r.json()
                u = data.get("usage") or {}
                self.prompt_tokens += u.get("prompt_tokens", 0)
                self.completion_tokens += u.get("completion_tokens", 0)
                self.calls += 1

                msg = data["choices"][0]["message"]
                return {
                    "content": msg.get("content") or "",
                    "tool_calls": msg.get("tool_calls") or [],
                    "finish_reason": data["choices"][0].get("finish_reason", ""),
                }
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.5 ** attempt)
        raise LLMError(f"模型调用失败：{last}")

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 2048) -> str:
        """纯文本补全（不挂工具）。"""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, max_tokens=max_tokens)["content"]


def parse_tool_arguments(raw: str) -> dict:
    """解析工具调用参数。模型偶尔会给出非法 JSON，这里做容错。"""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 容错：模型有时会在 JSON 外面包一层解释文字
    import re

    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"__parse_error__": raw}
