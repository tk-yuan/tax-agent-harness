"""分层存储：短期层用本地 sqlite，长期层复用记忆模块。

对应 JD 的「Redis 保存短期对话历史、任务计划、审批断点和临时数据；
Milvus 保存长期规则经验、常用策略模板、用户偏好和历史排查结论，
支持跨会话恢复复杂任务」。

**本项目为什么不用真的 Redis / Milvus**：
  让一个作品集项目硬依赖两个外部服务，克隆下来就跑不起来。
  所以按接口分层，后端可换：
    短期层  ShortTermStore   —— 本地 sqlite（生产换 Redis：多实例共享 + TTL）
    长期层  LayeredMemory    —— 本地 JSONL + 词面检索（生产换 Milvus：向量召回）

**短期层解决什么具体问题**：任务跑到一半断了（审批超时、进程被杀、限额），
重启后要能接着来——而不是从头再烧一遍 token。所以每次关键节点落一份快照。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from ..config import CACHE_DIR

DB_PATH = CACHE_DIR / "short_term.db"


class ShortTermStore:
    """带 TTL 的键值存储。线程安全（服务模式下多个请求线程会并发写）。"""

    def __init__(self, path: Path = DB_PATH, default_ttl: int = 7 * 86400) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, exp REAL)"
        )
        self._conn.commit()

    def put(self, key: str, value: Any, ttl: int | None = None) -> None:
        exp = time.time() + (ttl if ttl is not None else self.default_ttl)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (k, v, exp) VALUES (?,?,?)",
                (key, json.dumps(value, ensure_ascii=False, default=str), exp),
            )
            self._conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT v, exp FROM kv WHERE k=?", (key,)).fetchone()
        if row is None:
            return default
        v, exp = row
        if exp and exp < time.time():
            self.delete(key)
            return default
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return default

    def delete(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM kv WHERE k=?", (key,))
            self._conn.commit()

    def keys(self, prefix: str = "") -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT k FROM kv WHERE k LIKE ? AND (exp IS NULL OR exp > ?) ORDER BY k",
                (f"{prefix}%", time.time()),
            ).fetchall()
        return [r[0] for r in rows]

    def purge_expired(self) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM kv WHERE exp IS NOT NULL AND exp < ?",
                                     (time.time(),))
            self._conn.commit()
        return cur.rowcount

    def stats(self) -> dict:
        return {"keys": len(self.keys()), "path": str(DB_PATH)}


# —— 任务快照：断点恢复用 ——
CHECKPOINT_PREFIX = "checkpoint:"


def save_checkpoint(store: ShortTermStore, run_id: str, payload: dict) -> None:
    """落一份任务快照。断点信息包含任务清单、已落盘产出、审批记录、当前结论。"""
    store.put(f"{CHECKPOINT_PREFIX}{run_id}", {**payload, "saved_at": time.time()})


def load_checkpoint(store: ShortTermStore, run_id: str) -> dict | None:
    return store.get(f"{CHECKPOINT_PREFIX}{run_id}")


def list_checkpoints(store: ShortTermStore) -> list[dict]:
    out = []
    for k in store.keys(CHECKPOINT_PREFIX):
        cp = store.get(k)
        if cp:
            out.append({"run_id": k[len(CHECKPOINT_PREFIX):], **cp})
    return out


_shared: ShortTermStore | None = None


def shared_store() -> ShortTermStore:
    """进程内共享一个连接，避免每次调用都开库。"""
    global _shared
    if _shared is None:
        _shared = ShortTermStore()
    return _shared
