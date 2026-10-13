"""标准中间件的补充件：缓存、重试、PII 检测。

JD 原话：「在模型调用、工具调用、文件读写、任务状态变更等节点注入 Hook，
支持**缓存、重试、日志采集、PII 检测、权限校验、敏感操作拦截和审计上报**。」

七个能力在本项目里的归属：

| 能力 | 落在哪 |
|---|---|
| 权限校验 | `runtime/permissions.py` |
| 审计上报 | `runtime/permissions.py::audit` + `storage` |
| 敏感操作拦截 | `runtime/approval.py`（阻塞式人工审批） |
| **缓存** | 本模块 `cache_hook` |
| **重试** | 本模块 `retry_policy_hook` + 主循环的执行步骤 |
| **PII 检测** | 本模块 `pii_guard_hook` |
| 日志采集 | `observability/trace.py` |

## 为什么缓存和重试要做成 Hook 而不是写死在循环里

写死的话，"缓存哪些工具""重试几次"就变成了循环的硬编码知识。
做成 Hook：循环只认 `ctx["short_circuit"]` 和 `ctx["max_retry"]` 两个约定，
策略完全外置——换策略不改循环。

**缓存的边界要小心**：只缓存**只读且参数确定**的工具。
写文件、提交归档这类有副作用的绝不能缓存，否则第二次调用会被静默跳过。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time

# —— 缓存 ——
_DEFAULT_CACHEABLE = ("policy_search", "list_policies", "case_lookup", "get_case",
                      "feasible_options", "tax_calc", "optimize_plan")
SKIP_CATEGORIES = ("fs", "submit", "workflow", "delegate", "approval")


class ToolCache:
    """工具结果缓存。

    只缓存只读、参数确定的调用。有副作用的工具（写文件、提交）一律跳过——
    缓存它们等于把第二次调用静默吞掉，是最危险的一类 bug。
    """

    def __init__(self, allow: tuple[str, ...] = _DEFAULT_CACHEABLE, maxsize: int = 256) -> None:
        self.allow = set(allow)
        self.maxsize = maxsize
        self._data: dict[str, object] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(name: str, args: dict) -> str:
        raw = json.dumps({"n": name, "a": args}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def eligible(self, tool, args: dict) -> bool:
        if getattr(tool, "category", "") in SKIP_CATEGORIES:
            return False
        if not getattr(tool, "read_only", True):
            return False
        if getattr(tool, "requires_approval", False):
            return False
        return tool.name in self.allow

    def get(self, key: str):
        with self._lock:
            hit = key in self._data
            if hit:
                self.hits += 1
            else:
                self.misses += 1
            return self._data.get(key), hit

    def put(self, key: str, value) -> None:
        with self._lock:
            if len(self._data) >= self.maxsize:
                self._data.pop(next(iter(self._data)))
            self._data[key] = value

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits, "misses": self.misses, "size": len(self._data),
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


def cache_hook(cache: ToolCache):
    """生成一个挂到 before_tool / after_tool 的缓存 Hook 对。"""

    def before(ctx: dict) -> dict | None:
        tool, args = ctx["tool"], ctx["args"]
        if not cache.eligible(tool, args):
            return None
        value, hit = cache.get(ToolCache.key(tool.name, args))
        if hit:
            return {"short_circuit": value, "cache_hit": True}
        return None

    def after(ctx: dict) -> dict | None:
        tool, args = ctx["tool"], ctx["args"]
        if not cache.eligible(tool, args):
            return None
        cache.put(ToolCache.key(tool.name, args), ctx.get("result"))
        return None

    return before, after


# —— 重试 ——
def retry_policy_hook(max_retry: int = 2, backoff: float = 0.4):
    """为工具调用声明重试策略。

    只在工具**抛异常**时重试（网络抖动、限流）；返回错误字符串的不重试——
    那种是业务层面的失败，重试多少次都一样。
    """

    def before(ctx: dict) -> dict | None:
        return {"max_retry": max_retry, "retry_backoff": backoff}

    return before


# —— PII 检测 ——
PII_PATTERNS = (
    ("id_card", re.compile(r"\b\d{17}[\dXx]\b")),
    ("bank_card", re.compile(r"\b\d{16,19}\b")),
    ("phone", re.compile(r"\b1[3-9]\d{9}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
)


def mask_text(text: str) -> tuple[str, list[str]]:
    """把敏感信息打码，返回（脱敏后文本, 命中的类型）。

    保留首尾各若干字符——既能让人认出"这里有个手机号"，
    又不泄露完整值。全打掉的话，排查问题时连类型都看不出来。
    """
    hits: list[str] = []
    for name, pat in PII_PATTERNS:
        def _sub(m: re.Match) -> str:
            s = m.group(0)
            if len(s) <= 4:
                return "*" * len(s)
            keep = 2
            return s[:keep] + "*" * (len(s) - keep * 2) + s[-keep:]

        new, n = pat.subn(_sub, text)
        if n:
            hits.append(f"{name}×{n}")
            text = new
    return text, hits


def pii_guard_hook(enabled: bool = True, scan_result: bool = True):
    """检测并脱敏 PII：入参（用户填的）和工具结果（可能含客户信息）都扫。

    **默认只脱敏不拦截**：拦截会让任务直接失败，而多数场景下打码已经够用。
    命中记录会写进 trace 与审计，需要追责时查得到。
    """

    def before(ctx: dict) -> dict | None:
        if not enabled:
            return None
        args = ctx.get("args") or {}
        blob = json.dumps(args, ensure_ascii=False, default=str)
        masked, hits = mask_text(blob)
        if not hits:
            return None
        try:
            new_args = json.loads(masked)
        except json.JSONDecodeError:
            return {"pii_hits": hits}      # 脱敏后不再是合法 JSON，就只记录不改
        return {"args": new_args, "pii_hits": hits}

    def after(ctx: dict) -> dict | None:
        if not enabled or not scan_result:
            return None
        value = ctx.get("result")
        if not isinstance(value, str):
            return None
        masked, hits = mask_text(value)
        if not hits:
            return None
        return {"result": masked, "pii_hits": hits}

    return before, after


class LatencyTracker:
    """日志采集的轻量版：按工具统计调用次数与耗时。"""

    def __init__(self) -> None:
        self._rows: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def after(self, ctx: dict) -> None:
        ms = ctx.get("duration_ms")
        if ms is None:
            return
        with self._lock:
            self._rows.setdefault(ctx.get("name", "?"), []).append(float(ms))
        return None

    def stats(self) -> dict:
        with self._lock:
            return {
                name: {"calls": len(v), "avg_ms": round(sum(v) / len(v), 1),
                       "max_ms": round(max(v), 1)}
                for name, v in sorted(self._rows.items())
            }
