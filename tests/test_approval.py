"""人工审批门的测试。

重点测**阻塞语义**：Agent 必须真的等人批复，超时必须自动拒绝（不能挂死）。
"""

import threading
import time

from taxharness.runtime.approval import ApprovalGate, default_rules
from taxharness.runtime.tools import Tool


def _tool(name, read_only=True, requires_approval=False):
    return Tool(name=name, description="d", parameters={}, handler=lambda **k: None,
                read_only=read_only, requires_approval=requires_approval)


# —— 规则 ——
def test_flagged_tool_requires_approval():
    need, reason = default_rules(_tool("submit_plan", requires_approval=True), {})
    assert need and "高风险" in reason


def test_semantic_name_requires_approval():
    """名字带 submit/publish/deploy 语义的，即使没打标记也要审批。"""
    for n in ("publish_policy", "deploy_rule", "archive_report"):
        need, _ = default_rules(_tool(n), {})
        assert need, n


def test_write_to_controlled_path_requires_approval():
    need, reason = default_rules(_tool("write_file", read_only=False),
                                 {"path": "reports/final/plan.md"})
    assert need and "受控路径" in reason


def test_ordinary_readonly_tool_needs_no_approval():
    need, _ = default_rules(_tool("policy_search"), {"query": "小微"})
    assert not need


def test_ordinary_write_needs_no_approval():
    need, _ = default_rules(_tool("write_file", read_only=False),
                            {"path": "reports/draft.md"})
    assert not need


# —— 阻塞语义 ——
def test_approve_unblocks_the_waiter():
    gate = ApprovalGate(rules=lambda t, a: (True, "测试"), timeout=5)
    got = {}

    def waiter():
        got["req"] = gate.request("submit_plan", {}, "测试")

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.15)
    assert gate.pending(), "发起后应处于待审批状态"
    pid = gate.pending()[0].id
    assert gate.resolve(pid, approved=True, decider="tester")
    t.join(timeout=3)
    assert got["req"].status == "approved"
    assert got["req"].decider == "tester"


def test_reject_unblocks_and_marks_rejected():
    gate = ApprovalGate(rules=lambda t, a: (True, "测试"), timeout=5)

    def waiter():
        gate.request("submit_plan", {}, "测试")

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.15)
    pid = gate.pending()[0].id
    gate.resolve(pid, approved=False, note="证据不足")
    t.join(timeout=3)
    hist = gate.history()
    assert hist[0]["status"] == "rejected"
    assert hist[0]["note"] == "证据不足"


def test_timeout_auto_rejects():
    """超时必须自动拒绝——挂死比拒绝糟得多，拒绝是可审计的终态。"""
    gate = ApprovalGate(rules=lambda t, a: (True, "测试"), timeout=0.3)
    req = gate.request("submit_plan", {}, "测试")   # 无人批复
    assert req.status == "timeout"
    assert "未获批复" in req.note


def test_resolve_unknown_request_returns_false():
    gate = ApprovalGate()
    assert gate.resolve("ap-nope", approved=True) is False


def test_double_resolve_is_ignored():
    """已处理的审批不能被二次改写——否则审计链就不可信了。"""
    gate = ApprovalGate(rules=lambda t, a: (True, "x"), timeout=5)

    def waiter():
        gate.request("t", {}, "x")

    th = threading.Thread(target=waiter)
    th.start()
    time.sleep(0.15)
    pid = gate.pending()[0].id
    assert gate.resolve(pid, approved=True) is True
    assert gate.resolve(pid, approved=False) is False
    th.join(timeout=3)


def test_notifier_is_called():
    seen = []
    gate = ApprovalGate(rules=lambda t, a: (True, "x"), timeout=0.2,
                        notifier=lambda req: seen.append(req.tool))
    gate.request("submit_plan", {}, "x")
    assert seen == ["submit_plan"]


def test_notifier_failure_does_not_break_flow():
    """通知发不出去（网络/前端挂了），不能把整个审批流程带崩，转超时即可。"""

    def boom(req):
        raise RuntimeError("前端挂了")

    gate = ApprovalGate(rules=lambda t, a: (True, "x"), timeout=0.2, notifier=boom)
    req = gate.request("submit_plan", {}, "x")
    assert req.status == "timeout"


def test_stats_counts_outcomes():
    gate = ApprovalGate(rules=lambda t, a: (True, "x"), timeout=0.2)
    gate.request("t", {}, "x")            # 超时
    assert gate.stats()["by_status"].get("timeout") == 1
