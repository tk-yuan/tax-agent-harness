"""权限边界与审计（对应教程 s03）。

三件事：

1. **路径白名单**。Agent 只能读写 `workspace/` 与 `data/` 下的文件。
   所有路径先 `resolve()` 再判断归属，防止 `../../` 这类越界。

2. **工具分级**。标了 `read_only=True` 的工具可以随意调用；
   写操作要过额外检查（本项目中写操作仅限 workspace 内）。

3. **审计留痕**。每一次拒绝都写进 `.cache/audit.jsonl`。
   对应 JD 里的「操作拦截和审计上报」——这里落到本地文件，
   真实企业环境里是上送到链路追踪平台。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import AUDIT_LOG, FS_ALLOWED_ROOTS


class PermissionDenied(Exception):
    """越权操作。被主循环捕获后作为工具错误返回给模型。"""


def audit(event: str, **fields) -> None:
    """追加一条审计记录。"""
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class PermissionPolicy:
    def __init__(self, allowed_roots=None, allow_write: bool = True) -> None:
        self.allowed_roots = tuple(Path(p).resolve() for p in (allowed_roots or FS_ALLOWED_ROOTS))
        self.allow_write = allow_write

    def check_path(self, raw: str, write: bool = False) -> Path:
        """把相对路径解析到工作区，并校验它没有越界。"""
        p = Path(raw)
        if not p.is_absolute():
            p = self.allowed_roots[0] / p
        p = p.resolve()

        if not any(p == r or r in p.parents for r in self.allowed_roots):
            audit("path_denied", path=str(p), write=write)
            raise PermissionDenied(
                f"路径越界：{raw}。只允许访问 {[str(r) for r in self.allowed_roots]} 下的文件。"
            )
        if write and not self.allow_write:
            audit("write_denied", path=str(p))
            raise PermissionDenied("当前会话为只读模式，禁止写入。")
        return p

    def check_tool(self, tool, args: dict) -> None:
        """工具级检查：只读工具直接放行，写操作需通过路径校验。"""
        if getattr(tool, "read_only", True):
            return
        target = args.get("path") or args.get("file") or ""
        if target:
            self.check_path(str(target), write=True)
        audit("tool_allowed", tool=tool.name, write=True, args=target)


def default_policy() -> PermissionPolicy:
    return PermissionPolicy()
