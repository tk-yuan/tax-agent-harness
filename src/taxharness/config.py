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

# —— 虚拟文件系统：Agent 可持续操作的工作区布局 ——
# 对应 JD 的「抽象 /workspace、/reports、/logs、/policies、/memories 等目录，
# 支持文件读写、编辑、搜索和版本留痕；把工作产物作为 Agent 可持续操作的资产」。
#
# 为什么分这么多目录而不是一个大 workspace：
#   ① 权限可以分级——日志只追加、提交区只写新文件、报告区可改写
#   ② 清理策略不同——artifacts 可定期清，submissions 必须永久留存
#   ③ 人看得懂——出了事一眼知道该去哪找
VIRTUAL_FS = {
    "reports": "筹划报告、方案对比等交付物",
    "artifacts": "超长工具结果的落盘位置（上下文只留摘要与路径）",
    "logs": "执行日志快照与审计片段",
    "policies": "政策原文摘录、政策组合与时效核查材料",
    "memories": "跨会话经验沉淀的快照",
    "submissions": "已提交归档的方案（有对外后果，只增不改）",
}

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

# 建出虚拟文件系统的各分区，并在每个目录放一份说明，免得打开是空目录
for _name, _desc in VIRTUAL_FS.items():
    _p = WORKSPACE_DIR / _name
    _p.mkdir(parents=True, exist_ok=True)
    _readme = _p / "README.txt"
    if not _readme.exists():
        _readme.write_text(f"{_name}/\n{_desc}\n", encoding="utf-8")


def virtual_fs_layout() -> dict[str, str]:
    """返回虚拟文件系统布局，供文档与接口展示。"""
    return dict(VIRTUAL_FS)
