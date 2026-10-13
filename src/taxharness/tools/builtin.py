"""虚拟文件系统工具（对应 JD 的「虚拟文件系统」）。

Agent 把中间产物落成文件——核算明细、方案清单、对比报告。
关键收益：**大对象不占上下文**，上下文里只留路径和摘要。

实现用真实本地目录 + 路径白名单，不上真沙箱：
双层隔离是基础设施活，不是 Agent 能力活，硬做出来也讲不出深度。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..runtime.permissions import PermissionDenied
from ..runtime.tools import Tool, ToolRegistry


def _fmt(value, limit: int = 900) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + f"\n…[截断，原长 {len(text)}]"


def register_fs_tools(registry: ToolRegistry, policy) -> None:
    """注册文件工具。每个写操作都过权限层。"""

    def write_file(path: str, content: str, append: bool = False) -> dict:
        """写文件。append=True 时追加——长报告分节写，避免单次调用超 token 上限。

        模型输出工具参数也消耗 token：一次写几千字的报告很容易被截断，
        参数 JSON 直接解析失败。分节追加是更稳的写法。
        """
        try:
            p = policy.check_path(path, write=True)
        except PermissionDenied as e:
            return {"error": str(e)}
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a" if append else "w", encoding="utf-8") as f:
            f.write(content)
        size = p.stat().st_size
        return {"written": str(p), "mode": "append" if append else "overwrite",
                "chars_added": len(content), "total_bytes": size}

    def read_file(path: str, limit: int = 4000) -> dict:
        try:
            p = policy.check_path(path)
        except PermissionDenied as e:
            return {"error": str(e)}
        if not p.exists():
            return {"error": f"文件不存在：{path}"}
        text = p.read_text(encoding="utf-8", errors="replace")
        return {"path": str(p), "chars": len(text), "content": _fmt(text, limit)}

    def list_dir(path: str = ".") -> dict:
        try:
            p = policy.check_path(path)
        except PermissionDenied as e:
            return {"error": str(e)}
        if not p.exists():
            return {"error": f"目录不存在：{path}"}
        items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        return {
            "path": str(p),
            "entries": [
                {"name": x.name, "type": "file" if x.is_file() else "dir",
                 "bytes": x.stat().st_size if x.is_file() else None}
                for x in items[:60]
            ],
        }

    def search_files(pattern: str, path: str = ".") -> dict:
        try:
            p = policy.check_path(path)
        except PermissionDenied as e:
            return {"error": str(e)}
        hits = [str(f.relative_to(p)) for f in p.rglob(pattern)][:40]
        return {"pattern": pattern, "matches": hits, "count": len(hits)}

    registry.register(Tool(
        name="write_file",
        description=(
            "把内容写入工作区文件。用于保存核算明细、方案清单、对比报告，避免大段内容占用上下文。"
            "写长报告时用 append=true 分节追加（先写第一节，再逐节 append），"
            "这样单次调用不会因为内容太长被截断。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对 workspace/ 的路径，如 reports/plan_A.md"},
                "content": {"type": "string", "description": "文件内容"},
                "append": {"type": "boolean", "description": "true 表示追加到文件末尾，默认 false（覆盖）",
                           "default": False},
            },
            "required": ["path", "content"],
        },
        handler=write_file, read_only=False, category="fs",
    ))
    registry.register(Tool(
        name="read_file",
        description="读取工作区文件内容。",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer", "default": 4000}},
            "required": ["path"],
        },
        handler=read_file, read_only=True, category="fs",
    ))
    registry.register(Tool(
        name="list_dir",
        description="列出目录内容。",
        parameters={"type": "object", "properties": {"path": {"type": "string", "default": "."}}},
        handler=list_dir, read_only=True, category="fs",
    ))
    registry.register(Tool(
        name="search_files",
        description="按通配符搜索工作区文件。",
        parameters={
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "default": "."}},
            "required": ["pattern"],
        },
        handler=search_files, read_only=True, category="fs",
    ))
