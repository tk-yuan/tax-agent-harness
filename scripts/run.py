"""运行税筹决策 Agent。

用法：
    python scripts/run.py "帮 000638 这家企业做税负筹划"
    python scripts/run.py                      # 交互模式
    python scripts/run.py --no-trace "问题"     # 不记录链路

跑完会打印：结论、任务清单、子 Agent 调用记录、链路统计，
并把完整链路写进 traces/<trace_id>.json 与 .html（浏览器可打开）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxharness.assemble import build_harness  # noqa: E402
from taxharness.observability.trace import Trace  # noqa: E402


def report(bundle: dict, trace: Trace, result) -> None:
    print("\n" + "=" * 74)
    print("结论")
    print("-" * 74)
    print(result.answer or "（无输出）")

    board = bundle["board"]
    if board.items:
        print("\n" + "-" * 74)
        print("任务清单")
        print(board.render())

    if bundle["runner"].records:
        print("\n" + "-" * 74)
        print("子 Agent 委派记录")
        for r in bundle["runner"].records:
            print(f"  · {r['agent']}（{r['turns']} 轮，{r['stop_reason']}）")
            print(f"    目标：{r['goal'][:70]}")
            print(f"    结论：{r['conclusion'][:150]}")

    if result.state.artifacts:
        print("\n" + "-" * 74)
        print("落盘产出")
        for p in result.state.artifacts:
            print(f"  · {p}")

    print("\n" + "-" * 74)
    print(f"运行统计：{result.turns} 轮 | 停止原因 {result.stop_reason}")
    print(f"  token：{result.usage}")
    print(f"  链路：{trace.summary()['span_counts']}（共 {trace.summary()['total_spans']} 个 span）")
    print(f"  记忆：{bundle['memory'].stats()}")
    json_path, html_path = trace.save({"answer": result.answer, "board": board.to_dict()})
    print(f"  链路文件：{json_path}")
    print(f"            {html_path}  ← 浏览器打开可看时间线")
    print("=" * 74)


def main() -> None:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_trace = "--no-trace" not in sys.argv

    question = " ".join(argv).strip()
    print("正在装配 Agent …")
    trace = Trace(question or "（交互）") if use_trace else None
    bundle = build_harness(trace=trace)
    agent = bundle["agent"]
    print(f"就绪。模型 {bundle['model']}｜工具 {len(bundle['registry'].names())} 个｜"
          f"技能 {len(bundle['skills'].skills)} 个｜子 Agent {len(bundle['runner'].specs)} 个")

    if question:
        report(bundle, trace, agent.run(question))
        return

    print("\n输入目标回车执行，空行退出。示例：")
    print("  帮 000638 这家企业做税负筹划，给出方案和节税金额")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        report(bundle, trace, agent.run(q))


if __name__ == "__main__":
    main()
