"""工具注册与分发（对应教程 s02）。

工具 = 名字 + 描述 + JSON Schema + 处理函数。

关键设计：**工具池是按上下文动态组装的，不是写死的**。
主循环每轮向模型提交的 tools 列表，由「注册表 + 当前权限 + 当前阶段」共同决定。
后面 Skills 渐进披露、子 Agent 权限收窄，都靠这一层实现——不需要改主循环一行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Any]
    read_only: bool = True          # 供权限层判断：只读工具放宽，写操作收紧
    category: str = "general"
    tags: tuple[str, ...] = field(default_factory=tuple)
    # 高风险标记：置 True 时该工具每次调用都要过人工审批门
    requires_approval: bool = False

    def spec(self) -> dict:
        """OpenAI function-calling 格式的工具描述。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重名：{tool.name}")
        self._tools[tool.name] = tool

    def tool(self, **kwargs):
        """装饰器写法：@registry.tool(name=..., description=..., parameters=...)"""

        def deco(fn):
            self.register(Tool(handler=fn, **kwargs))
            return fn

        return deco

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, allow: set[str] | None = None) -> list[dict]:
        """组装当前可用工具池的 schema 列表。allow=None 表示全部。"""
        return [
            t.spec()
            for n, t in sorted(self._tools.items())
            if allow is None or n in allow
        ]

    def call(self, name: str, args: dict) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"未知工具：{name}（可用：{self.names()}）")
        return tool.handler(**args)
