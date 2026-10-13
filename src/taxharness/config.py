"""全局配置。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# 密钥只从项目根的 .env 读取（参考 .env.example）
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = ROOT / "data"
WORKSPACE_DIR = ROOT / "workspace"
SKILLS_DIR = ROOT / "skills"
CACHE_DIR = ROOT / ".cache"
TRACE_DIR = ROOT / "traces"

# —— 模型 ——
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

MODEL = os.getenv("HARNESS_MODEL", "deepseek-chat")
SUBAGENT_MODEL = os.getenv("HARNESS_SUBAGENT_MODEL", "deepseek-chat")  # 子 Agent 可换更便宜的
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# —— 运行时约束 ——
# 主循环轮数上限：一次「政策核查 → 组合测算 → 独立复核 → 写报告」的完整筹划，
# 实测要 20~35 轮（含子 Agent 委派）。给 24 会撞上限、拿不到完整结论，
# 这是踩过的坑，别往下调。
MAX_TURNS = int(os.getenv("HARNESS_MAX_TURNS", "40"))
CONTEXT_BUDGET_CHARS = int(os.getenv("HARNESS_CTX_BUDGET", "60000"))  # 上下文预算（字符）
TOOL_RESULT_INLINE_LIMIT = int(os.getenv("HARNESS_TOOL_INLINE", "2000"))  # 工具结果内联上限
MAX_SUBAGENT_TURNS = int(os.getenv("HARNESS_SUBAGENT_TURNS", "8"))

# —— 虚拟文件系统：允许 Agent 操作的目录白名单 ——
FS_ALLOWED_ROOTS = (WORKSPACE_DIR, DATA_DIR)

# —— 审计日志（对应 JD 里的「操作拦截和审计上报」，这里落到本地文件）——
AUDIT_LOG = CACHE_DIR / "audit.jsonl"

for _d in (WORKSPACE_DIR, CACHE_DIR, TRACE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
