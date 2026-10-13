"""MCP 工具接入（对应教程 s14）：动态工具池。

MCP 把两件事解耦了：
    「工具有哪些」  —— 由 server 提供
    「工具怎么被用」—— 由 Harness 负责连接、命名、权限检查

于是新增一个数据源不必改 Harness 代码：连上 server，把它的工具并进工具池即可。

**本实现是进程内模拟 server**（与教程一致），演示 `tools/list` 与 `tools/call`
两个核心动作，以及工具池的动态组装。真实 MCP 的 stdio / HTTP transport
不在本模块范围内——那部分是协议实现，不是 Agent 能力。

安全要点：外部工具默认按**非只读**处理。
来自外部的工具不能默认可信——这正是 MCP 部署里最容易出事的地方。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from ..runtime.tools import Tool, ToolRegistry


@dataclass
class MCPServerInfo:
    name: str
    version: str = "1.0"
    description: str = ""


class InProcessMCPServer:
    """一个进程内的模拟 MCP server。"""

    def __init__(self, info: MCPServerInfo) -> None:
        self.info = info
        self._tools: dict[str, dict] = {}

    def add_tool(self, name: str, description: str, parameters: dict, fn: Callable) -> None:
        self._tools[name] = {
            "spec": {
                "name": name,
                "description": description,
                "inputSchema": parameters,
            },
            "fn": fn,
        }

    # —— MCP 协议的两个核心动作 ——
    def list_tools(self) -> list[dict]:
        return [t["spec"] for t in self._tools.values()]

    def call_tool(self, name: str, arguments: dict) -> dict:
        if name not in self._tools:
            return {"isError": True, "content": f"server 上没有工具 {name}"}
        try:
            out = self._tools[name]["fn"](**arguments)
            text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
            return {"isError": False, "content": text}
        except Exception as e:  # noqa: BLE001
            return {"isError": True, "content": f"{type(e).__name__}: {e}"}


@dataclass
class MCPClient:
    """连接一个 server，把它的工具转成 Harness 能用的 Tool。"""

    server: InProcessMCPServer
    namespace: str = ""

    def discover(self) -> list[Tool]:
        tools = []
        for spec in self.server.list_tools():
            name = f"{self.namespace}{spec['name']}" if self.namespace else spec["name"]
            tools.append(
                Tool(
                    name=name,
                    description=f"[来自 {self.server.info.name}] {spec['description']}",
                    parameters=spec.get("inputSchema", {"type": "object", "properties": {}}),
                    handler=self._wrap(spec["name"]),
                    # 外部工具默认不可信：按非只读处理，走权限检查
                    read_only=False,
                    category="mcp",
                )
            )
        return tools

    def _wrap(self, remote_name: str) -> Callable:
        def _call(**kwargs):
            result = self.server.call_tool(remote_name, kwargs)
            if result.get("isError"):
                return {"error": result["content"]}
            return result["content"]

        _call.__name__ = f"mcp_{remote_name}"
        return _call

    def connect(self, registry: ToolRegistry) -> list[str]:
        """连上并把工具并进工具池，返回新加入的工具名。"""
        added = []
        for tool in self.discover():
            if registry.get(tool.name) is not None:
                continue
            registry.register(tool)
            added.append(tool.name)
        return added


# —— 演示用的两个 server ——

def build_demo_servers(data_dir) -> list[MCPClient]:
    """构造两个演示 server：一个查政策库统计，一个做财务指标换算。

    真实场景里这两个 server 会由别的团队维护、跑在别的进程；
    这里进程内模拟，只为演示「工具池可动态扩展」这件事本身。
    """
    import json as _json
    from pathlib import Path

    data_dir = Path(data_dir)

    policy_server = InProcessMCPServer(
        MCPServerInfo(name="policy-registry", description="政策库只读查询服务")
    )

    def policy_stats() -> dict:
        lib = _json.loads((data_dir / "policy_library.json").read_text(encoding="utf-8"))
        by_cat: dict[str, int] = {}
        for p in lib:
            k = str(p.get("category", "未分类"))
            by_cat[k] = by_cat.get(k, 0) + 1
        return {"total": len(lib), "by_category": by_cat}

    def policy_detail(policy_id: str) -> dict:
        lib = _json.loads((data_dir / "policy_library.json").read_text(encoding="utf-8"))
        for p in lib:
            if p.get("id") == policy_id:
                return p
        return {"error": f"未找到政策 {policy_id}"}

    policy_server.add_tool(
        "policy_stats", "统计政策库的规模与分类分布",
        {"type": "object", "properties": {}}, policy_stats,
    )
    policy_server.add_tool(
        "policy_detail", "按 id 取一条政策的完整字段",
        {"type": "object", "properties": {"policy_id": {"type": "string"}},
         "required": ["policy_id"]},
        policy_detail,
    )

    calc_server = InProcessMCPServer(
        MCPServerInfo(name="unit-converter", description="财务单位换算服务")
    )
    calc_server.add_tool(
        "wan_to_yuan", "万元转元",
        {"type": "object", "properties": {"wan": {"type": "number"}}, "required": ["wan"]},
        lambda wan: {"yuan": round(float(wan) * 10000, 2)},
    )

    return [
        MCPClient(policy_server, namespace="mcp_"),
        MCPClient(calc_server, namespace="mcp_"),
    ]
