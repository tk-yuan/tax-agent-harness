"""分层记忆（对应教程 s09）。

三层，各司其职：

  短期记忆   当前对话消息        —— 由主循环维护，压缩由 compaction 负责
  工作记忆   本次任务的笔记      —— 注入 system prompt，随任务结束清理
  长期记忆   跨任务沉淀的经验    —— 落盘持久化，下次任务可选注入

长期记忆**不是"做个向量检索就叫记忆"**。这里实现了三个机制，
因为少了它们，"记忆"就只是个文件：

  衰减（decay）    越久没用到的经验，权重越低——避免陈旧经验一直占位
  强化（reinforce）被召回并实际用上的，计一次命中，权重上升
  淘汰（prune）    权重低于阈值的经验直接清掉，记忆库不会无限膨胀

检索用「关键词重合度 × 衰减 × 强化」打分，不是向量检索：
经验条目短、数量少（几十到几百条），词面信号足够，不值得为它多付一次
embedding API 调用。规模上千条之后换成向量检索更合适——这是规模决定方案。
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import CACHE_DIR

MEMORY_PATH = CACHE_DIR / "memory.jsonl"

# 衰减半衰期（天）：14 天没用到的经验，权重减半
HALF_LIFE_DAYS = 14.0
PRUNE_THRESHOLD = 0.05

_STOP = set("的 了 和 与 是 在 对 及 或 有 无 为 以 该 本 等 中 上 下 之 其 也 就 而".split())


def _tokens(text: str) -> set[str]:
    """粗分词：连续汉字取二元组 + 英文/数字词。不引分词依赖。"""
    text = re.sub(r"\s+", " ", text)
    words = set(re.findall(r"[A-Za-z0-9]{2,}", text))
    han = re.findall(r"[一-龥]+", text)
    for seg in han:
        for i in range(len(seg) - 1):
            words.add(seg[i : i + 2])
    return {w for w in words if w not in _STOP}


@dataclass
class MemoryEntry:
    id: str
    kind: str            # pattern | policy_note | failure | preference
    content: str
    tags: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    hits: int = 0
    last_used: float = field(default_factory=time.time)

    def decayed_weight(self, now: float | None = None) -> float:
        now = now or time.time()
        age_days = (now - self.last_used) / 86400.0
        decay = 0.5 ** (age_days / HALF_LIFE_DAYS)
        reinforce = 1.0 + math.log1p(self.hits)
        return decay * reinforce


class LayeredMemory:
    def __init__(self, path: Path = MEMORY_PATH) -> None:
        self.path = Path(path)
        self._entries: list[MemoryEntry] = []
        self.working: list[str] = []      # 本次任务的工作记忆
        self._load()

    # —— 持久化 ——
    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                self._entries.append(MemoryEntry(**json.loads(line)))
            except TypeError:
                continue

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "\n".join(json.dumps(asdict(e), ensure_ascii=False) for e in self._entries),
            encoding="utf-8",
        )

    # —— 写入 ——
    def remember(self, kind: str, content: str, tags: list[str] | None = None) -> str:
        eid = f"m{int(time.time() * 1000) % 10**10}"
        self._entries.append(MemoryEntry(id=eid, kind=kind, content=content, tags=tags or []))
        self._save()
        return eid

    def note(self, text: str) -> None:
        """写工作记忆：只作用于本次任务。"""
        if text not in self.working:
            self.working.append(text)

    # —— 读取 ——
    def recall(self, query: str, k: int = 3) -> list[MemoryEntry]:
        """按「关键词重合 × 衰减 × 强化」召回，命中即强化。"""
        if not self._entries:
            return []
        q = _tokens(query)
        now = time.time()
        scored = []
        for e in self._entries:
            t = _tokens(e.content + " " + " ".join(e.tags))
            overlap = len(q & t) / max(len(q), 1)
            if overlap <= 0:
                continue
            scored.append((overlap * e.decayed_weight(now), e))
        scored.sort(key=lambda x: -x[0])
        top = [e for _, e in scored[:k]]
        for e in top:
            e.hits += 1
            e.last_used = now
        if top:
            self._save()
        return top

    def prune(self) -> int:
        """淘汰衰减到阈值以下且从未被用过的经验。"""
        now = time.time()
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if e.hits > 0 or e.decayed_weight(now) >= PRUNE_THRESHOLD
        ]
        if len(self._entries) != before:
            self._save()
        return before - len(self._entries)

    # —— 上下文注入 ——
    def context_block(self, query: str = "", k: int = 3) -> str:
        """组装要注入 system prompt 的记忆块。"""
        parts = []
        if self.working:
            parts.append("【本次任务的笔记】\n" + "\n".join(f"- {n}" for n in self.working))
        recalled = self.recall(query, k=k) if query else []
        if recalled:
            parts.append(
                "【历史经验】（可能相关，请自行判断是否适用）\n"
                + "\n".join(f"- [{e.kind}] {e.content}" for e in recalled)
            )
        return "\n\n".join(parts)

    def stats(self) -> dict:
        now = time.time()
        by_kind: dict[str, int] = {}
        for e in self._entries:
            by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
        return {
            "long_term_entries": len(self._entries),
            "by_kind": by_kind,
            "working_notes": len(self.working),
            "avg_weight": round(
                sum(e.decayed_weight(now) for e in self._entries) / max(len(self._entries), 1), 3
            ),
        }
