"""回归测试：把执行链路变成可断言的用例。

用法：
    python scripts/run_regression.py              # 跑全部场景
    python scripts/run_regression.py tax-planning-basic   # 只跑一个

## 为什么 Agent 也需要回归测试

Agent 是概率系统，"能跑"和"跑得对"是两件事。改了提示词、
调了 Hook、换了模型——单元测试全绿，但**Agent 可能已经不再调用某个关键工具**，
或者开始忽略审批拒绝。这些退化单测抓不到，只有在完整链路上断言才看得见。

## 断言什么

  tools_required / tools_forbidden   链路里必须/不能出现哪些工具
  stop_reason_in                     退出原因（completed / max_turns / …）
  answer_regex / answer_must_not_regex  结论里必须/不能出现的内容
  min_spans                          各类 span 的最低数量
  no_tool_errors                     是否允许出现工具错误
  approval                           none / approve / reject —— 审批门怎么响应

## 代价

每个场景是一次**真实完整运行**（几十轮模型调用）。所以场景要少而精，
不要拿它当单元测试天天跑。
"""

from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxharness.assemble import build_harness  # noqa: E402
from taxharness.observability.trace import Trace  # noqa: E402
from taxharness.runtime.approval import ApprovalGate  # noqa: E402

SCENARIOS = ROOT / "regression" / "scenarios.jsonl"


def make_gate(mode: str):
    """按场景需要造审批门：none 不装，approve/reject 自动决策。

    自动决策是回归测试的关键——否则场景会卡在等人批复（默认 300 秒超时）。
    """
    if mode == "none":
        return None
    gate = ApprovalGate(timeout=15)

    def notifier(req) -> None:
        def decide() -> None:
            gate.resolve(req.id, approved=(mode == "approve"),
                         decider="regression", note=f"自动{mode}")

        threading.Timer(0.4, decide).start()

    gate.notifier = notifier
    return gate


def check(asserts: dict, trace: Trace, answer: str, stop_reason: str) -> list[str]:
    """返回失败原因列表，空列表表示全部通过。"""
    fails: list[str] = []
    tool_spans = [s for s in trace.spans if s["kind"] == "tool"]
    called = [s["name"] for s in tool_spans]

    for t in asserts.get("tools_required", []):
        if t not in called:
            fails.append(f"缺少必需的工具调用：{t}（实际调了 {sorted(set(called))}）")
    for t in asserts.get("tools_forbidden", []):
        if t in called:
            fails.append(f"出现了不该调用的工具：{t}")

    if "stop_reason_in" in asserts and stop_reason not in asserts["stop_reason_in"]:
        fails.append(f"退出原因 {stop_reason} 不在 {asserts['stop_reason_in']}")

    for pat in asserts.get("answer_regex", []):
        if not re.search(pat, answer):
            fails.append(f"结论缺少模式 /{pat}/")
    for pat in asserts.get("answer_must_not_regex", []):
        if re.search(pat, answer):
            fails.append(f"结论出现了不该有的内容 /{pat}/")

    counts: dict[str, int] = {}
    for s in trace.spans:
        counts[s["kind"]] = counts.get(s["kind"], 0) + 1
    for kind, least in (asserts.get("min_spans") or {}).items():
        if counts.get(kind, 0) < least:
            fails.append(f"span[{kind}] 只有 {counts.get(kind, 0)} 个，少于要求的 {least}")

    if asserts.get("no_tool_errors"):
        bad = [s for s in tool_spans if s["fields"].get("ok") is False
               and not str(s["fields"].get("outcome", "")).startswith("[被拒绝]")]
        if bad:
            fails.append("出现工具错误：" + "；".join(
                f"{s['name']} → {s['fields'].get('outcome', '')[:50]}" for s in bad[:3]))

    return fails


def run_one(sc: dict) -> dict:
    trace = Trace(sc["goal"])
    gate = make_gate(sc["assert"].get("approval", "none"))
    bundle = build_harness(trace=trace, approval_gate=gate)
    result = bundle["agent"].run(sc["goal"], max_turns=sc.get("max_turns", 40))
    fails = check(sc["assert"], trace, result.answer or "", result.stop_reason)
    _, html = trace.save({"scenario": sc["id"], "answer": result.answer})
    return {
        "id": sc["id"], "note": sc.get("note", ""),
        "passed": not fails, "fails": fails,
        "turns": result.turns, "stop_reason": result.stop_reason,
        "tools": sorted({s["name"] for s in trace.spans if s["kind"] == "tool"}),
        "trace": html.name,
    }


def main() -> None:
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    scenarios = [json.loads(l) for l in SCENARIOS.read_text(encoding="utf-8").splitlines() if l.strip()]
    if only:
        scenarios = [s for s in scenarios if s["id"] in only]
    if not scenarios:
        raise SystemExit("没有匹配的场景")

    print(f"共 {len(scenarios)} 个场景（每个是一次真实完整运行）\n")
    results = []
    for sc in scenarios:
        print(f"▶ {sc['id']}  {sc.get('note', '')}")
        print(f"  目标：{sc['goal'][:64]}")
        r = run_one(sc)
        results.append(r)
        mark = "✓ 通过" if r["passed"] else "✗ 失败"
        print(f"  {mark} | {r['turns']} 轮 {r['stop_reason']} | 调用了 {len(r['tools'])} 个工具")
        for f in r["fails"]:
            print(f"    · {f}")
        print(f"  链路：traces/{r['trace']}")
        print()

    passed = sum(1 for r in results if r["passed"])
    print(f"═══ {passed}/{len(results)} 通过 ═══")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
