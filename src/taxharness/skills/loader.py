"""Skills 技能系统（对应教程 s07）：渐进式披露。

一个 Skill 是一个目录，含 `SKILL.md`（YAML frontmatter + 正文）。

**渐进披露三层**，这是本机制的全部价值：

  第一层  只把 name + description（约 20 字）放进 system prompt
  第二层  模型判断需要时，调 load_skill(name) 取回正文
  第三层  正文里引用的参考文件，再按需读

于是可以挂几十个技能，而每个只花一行 prompt 的成本；全量塞进去早就撑爆了。

对比「把所有流程写进 system prompt」：那样每个任务都被无关 SOP 干扰，
且改一条要动整个提示词。Skills 让流程变成**按需加载的模块**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..runtime.tools import Tool, ToolRegistry

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: Path
    resources: list[str]

    def summary_line(self) -> str:
        return f"- {self.name}：{self.description}"


def _parse(text: str) -> dict:
    """解析 frontmatter 里的简单 key: value。"""
    m = FRONTMATTER.match(text)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip("\"'")
    return out


class SkillLibrary:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.skills: dict[str, Skill] = {}
        self._scan()

    def _scan(self) -> None:
        if not self.root.exists():
            return
        for d in sorted(self.root.iterdir()):
            f = d / "SKILL.md"
            if not (d.is_dir() and f.exists()):
                continue
            raw = f.read_text(encoding="utf-8")
            meta = _parse(raw)
            body = FRONTMATTER.sub("", raw).strip()
            resources = [
                str(p.relative_to(d)) for p in d.rglob("*")
                if p.is_file() and p.name != "SKILL.md"
            ]
            name = meta.get("name") or d.name
            self.skills[name] = Skill(
                name=name,
                description=meta.get("description", "（无描述）"),
                body=body,
                path=d,
                resources=resources,
            )

    def catalog(self) -> str:
        """第一层：只给名字和一句话描述。"""
        if not self.skills:
            return ""
        lines = ["【可用技能】（需要时用 load_skill 加载详细流程）"]
        lines += [s.summary_line() for s in self.skills.values()]
        return "\n".join(lines)

    def load(self, name: str) -> dict:
        """第二层：取回正文。"""
        s = self.skills.get(name)
        if s is None:
            return {"error": f"未知技能：{name}", "available": list(self.skills)}
        return {
            "name": s.name,
            "description": s.description,
            "instructions": s.body,
            "resources": s.resources,
            "resources_hint": (
                "该技能的参考文件在同一目录下，可用 read_file 读取："
                + "、".join(s.resources)
                if s.resources else "（无附带文件）"
            ),
        }


def register_skill_tools(registry: ToolRegistry, library: SkillLibrary) -> None:
    registry.register(Tool(
        name="load_skill",
        description=(
            "加载某个技能的详细操作流程。可用技能见 system prompt 里的技能目录。"
            "遇到需要遵循固定 SOP 的任务（如政策时效核查、方案设计、报告撰写）时先加载它。"
        ),
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "技能名称"}},
            "required": ["name"],
        },
        handler=library.load,
        read_only=True,
        category="skill",
    ))
