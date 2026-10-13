"""运行时组件测试：工具注册、Hook、权限、任务状态机。"""

import pytest

from taxharness.runtime.hooks import HookManager
from taxharness.runtime.permissions import PermissionDenied, PermissionPolicy
from taxharness.runtime.tools import Tool, ToolRegistry
from taxharness.tasks.todo import TaskBoard


# —— 工具注册与分发 ——
def _echo(**kw):
    return kw


def test_registry_register_and_call():
    r = ToolRegistry()
    r.register(Tool(name="echo", description="d", parameters={"type": "object"}, handler=_echo))
    assert r.call("echo", {"a": 1}) == {"a": 1}


def test_registry_rejects_duplicate():
    r = ToolRegistry()
    r.register(Tool(name="echo", description="d", parameters={}, handler=_echo))
    with pytest.raises(ValueError):
        r.register(Tool(name="echo", description="d", parameters={}, handler=_echo))


def test_registry_unknown_tool_raises():
    r = ToolRegistry()
    with pytest.raises(KeyError):
        r.call("nope", {})


def test_specs_filtered_by_allow():
    """工具池要能按权限收窄——子 Agent 的最小权限就靠这个。"""
    r = ToolRegistry()
    for n in ("a", "b", "c"):
        r.register(Tool(name=n, description="d", parameters={}, handler=_echo))
    assert {s["function"]["name"] for s in r.specs({"a", "b"})} == {"a", "b"}
    assert len(r.specs()) == 3


def test_spec_shape_is_openai_compatible():
    r = ToolRegistry()
    r.register(Tool(name="x", description="desc", parameters={"type": "object"}, handler=_echo))
    spec = r.specs()[0]
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "x"


# —— Hook ——
def test_hook_merges_context():
    h = HookManager()
    h.register("before_tool", lambda ctx: {"args": {"v": 2}})
    out = h.run("before_tool", {"args": {"v": 1}})
    assert out["args"] == {"v": 2}


def test_hook_deny_stops_pipeline():
    h = HookManager()
    calls = []
    h.register("before_tool", lambda ctx: {"deny": "no"}, name="first")
    h.register("before_tool", lambda ctx: calls.append(1), name="second")
    out = h.run("before_tool", {})
    assert out["deny"] == "no"
    assert calls == []      # 后面的 Hook 不该执行


def test_hook_error_on_before_denies():
    """before_* 的 Hook 抛异常必须拒绝，不能悄悄放行。"""
    h = HookManager()

    def boom(ctx):
        raise RuntimeError("x")

    h.register("before_tool", boom)
    assert "deny" in h.run("before_tool", {})


def test_hook_error_on_after_does_not_deny():
    h = HookManager()

    def boom(ctx):
        raise RuntimeError("x")

    h.register("after_tool", boom)
    assert "deny" not in h.run("after_tool", {})


def test_unknown_hook_point_rejected():
    h = HookManager()
    with pytest.raises(ValueError):
        h.register("nope", lambda ctx: None)


# —— 权限 ——
def test_permission_allows_inside_root(tmp_path):
    p = PermissionPolicy(allowed_roots=[tmp_path])
    assert p.check_path("a/b.txt").name == "b.txt"


def test_permission_blocks_traversal(tmp_path):
    """`../` 越界必须被挡住，且要 resolve 之后再判断。"""
    p = PermissionPolicy(allowed_roots=[tmp_path / "ws"])
    (tmp_path / "ws").mkdir(exist_ok=True)
    with pytest.raises(PermissionDenied):
        p.check_path("../secret.txt")


def test_permission_blocks_absolute_outside(tmp_path):
    p = PermissionPolicy(allowed_roots=[tmp_path / "ws"])
    with pytest.raises(PermissionDenied):
        p.check_path("D:/Windows/system32/cmd.exe")


def test_readonly_policy_blocks_write(tmp_path):
    p = PermissionPolicy(allowed_roots=[tmp_path], allow_write=False)
    with pytest.raises(PermissionDenied):
        p.check_path("x.txt", write=True)


# —— 任务状态机 ——
def test_board_add_and_progress():
    b = TaskBoard()
    b.add("甲")
    b.add("乙")
    assert b.progress()["counts"]["pending"] == 2
    assert b.progress()["total"] == 2


def test_board_dependency_blocks_start():
    """依赖未完成时不许开工——这是把流程约束编码进调度层。"""
    b = TaskBoard()
    t1 = b.add("先做")
    t2 = b.add("后做", blocked_by=[t1.id])
    assert "error" in b.start(t2.id)
    assert t2.id in b.progress()["blocked"]

    b.start(t1.id)
    b.complete(t1.id)
    assert b.start(t2.id).get("ok") is True


def test_board_ready_only_returns_unblocked():
    b = TaskBoard()
    t1 = b.add("甲")
    b.add("乙", blocked_by=[t1.id])
    assert [t.id for t in b.ready()] == [t1.id]


def test_board_complete_then_open_items_shrinks():
    b = TaskBoard()
    t = b.add("甲")
    assert len(b.open_items()) == 1
    b.complete(t.id, "done")
    assert b.open_items() == []


def test_board_fail_counts_as_done():
    b = TaskBoard()
    t = b.add("甲")
    b.fail(t.id, "做不了")
    assert t.done
    assert b.progress()["counts"]["failed"] == 1


def test_board_render_marks_status():
    b = TaskBoard()
    t = b.add("甲")
    b.complete(t.id)
    assert "[x]" in b.render()
