"""分层存储的测试：短期 KV 与任务断点。"""

import time

from taxharness.storage.store import (
    ShortTermStore,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)


def _store(tmp_path) -> ShortTermStore:
    return ShortTermStore(path=tmp_path / "kv.db")


def test_put_and_get(tmp_path):
    s = _store(tmp_path)
    s.put("k", {"a": 1})
    assert s.get("k") == {"a": 1}


def test_get_missing_returns_default(tmp_path):
    assert _store(tmp_path).get("nope", "dflt") == "dflt"


def test_overwrite(tmp_path):
    s = _store(tmp_path)
    s.put("k", 1)
    s.put("k", 2)
    assert s.get("k") == 2


def test_ttl_expires(tmp_path):
    s = _store(tmp_path)
    s.put("k", "v", ttl=0)          # 立即过期
    time.sleep(0.02)
    assert s.get("k") is None


def test_delete(tmp_path):
    s = _store(tmp_path)
    s.put("k", "v")
    s.delete("k")
    assert s.get("k") is None


def test_keys_by_prefix(tmp_path):
    s = _store(tmp_path)
    s.put("checkpoint:a", {})
    s.put("checkpoint:b", {})
    s.put("other", {})
    assert sorted(s.keys("checkpoint:")) == ["checkpoint:a", "checkpoint:b"]


def test_purge_expired_removes_only_expired(tmp_path):
    s = _store(tmp_path)
    s.put("old", 1, ttl=0)
    s.put("new", 2, ttl=100)
    time.sleep(0.02)
    assert s.purge_expired() == 1
    assert s.get("new") == 2


def test_persistence_across_instances(tmp_path):
    """服务重启后断点还在——这正是短期存储存在的理由。"""
    p = tmp_path / "kv.db"
    ShortTermStore(path=p).put("k", {"v": 42})
    assert ShortTermStore(path=p).get("k") == {"v": 42}


# —— 任务断点 ——
def test_checkpoint_roundtrip(tmp_path):
    s = _store(tmp_path)
    save_checkpoint(s, "run-1", {"goal": "做筹划", "board": [], "status": "running"})
    cp = load_checkpoint(s, "run-1")
    assert cp["goal"] == "做筹划"
    assert "saved_at" in cp


def test_load_unknown_checkpoint(tmp_path):
    assert load_checkpoint(_store(tmp_path), "run-nope") is None


def test_list_checkpoints(tmp_path):
    s = _store(tmp_path)
    save_checkpoint(s, "run-1", {"goal": "甲"})
    save_checkpoint(s, "run-2", {"goal": "乙"})
    items = list_checkpoints(s)
    assert {c["run_id"] for c in items} == {"run-1", "run-2"}
    assert all("goal" in c for c in items)
