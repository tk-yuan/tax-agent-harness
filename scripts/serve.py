"""启动 HTTP 服务。

用法：
    python scripts/serve.py                 # 默认 127.0.0.1:8000
    python scripts/serve.py 0.0.0.0 8080    # 指定地址与端口

启动后浏览器打开 http://127.0.0.1:8000 就是控制台：
左侧实时执行链路，右侧任务清单与审批面板。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402


def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    print(f"服务启动：http://{host}:{port}")
    print("  控制台      /                       实时链路 + 审批面板")
    print("  提交任务    POST /runs")
    print("  事件流      GET  /runs/{id}/events  (SSE)")
    print("  工具池      GET  /tools")
    print("  接口文档    /docs")
    uvicorn.run("taxharness.server.app:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
