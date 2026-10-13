"""接线检查：工具 schema 与处理函数的参数名必须一致。

为什么单独写一个测试文件：本项目踩过这个坑——
`task` 工具在 schema 里声明参数叫 `agent`，处理函数签名却是 `run(name, goal)`。
结果工具名对得上、模型按 schema 传参，每次都报 unexpected keyword argument。
**Agent 自己发现了这个错误并在报告里写"该工具在本环境不可用"**，
但它本该是跑一遍测试就能拦住的问题。

这类 bug 的特点是：不报语法错、不报导入错，只有在模型真正调用时才暴露。
所以用反射做一次全量接线检查。
"""

import inspect

from taxharness.assemble import build_harness


def test_every_tool_handler_accepts_its_schema_params():
    bundle = build_harness()
    problems = []
    for name in bundle["registry"].names():
        tool = bundle["registry"].get(name)
        declared = set((tool.parameters.get("properties") or {}).keys())
        sig = inspect.signature(tool.handler)
        params = sig.parameters
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            continue  # 收 **kwargs 的一律安全
        missing = declared - set(params)
        if missing:
            problems.append(f"{name}: schema 声明了 {sorted(missing)}，函数不接受")
    assert not problems, "工具接线错误：\n" + "\n".join(problems)


def test_every_tool_has_valid_json_schema():
    bundle = build_harness()
    for name in bundle["registry"].names():
        tool = bundle["registry"].get(name)
        spec = tool.spec()
        assert spec["type"] == "function"
        assert spec["function"]["description"], f"{name} 缺少描述"
        schema = spec["function"]["parameters"]
        assert schema.get("type") == "object", f"{name} 的参数 schema 不是 object"
        req = schema.get("required", [])
        props = schema.get("properties", {})
        assert set(req) <= set(props), f"{name} 的 required 里有未声明的字段"


def test_required_params_have_no_defaults_in_handler():
    """必填参数在处理函数里不该有默认值——否则模型不传也能过，
    错误会被静默吞掉，表现为"工具返回了奇怪结果"而不是"参数缺了"。"""
    bundle = build_harness()
    problems = []
    for name in bundle["registry"].names():
        tool = bundle["registry"].get(name)
        req = set((tool.parameters or {}).get("required", []))
        sig = inspect.signature(tool.handler)
        for pname, p in sig.parameters.items():
            if pname in req and p.default is not inspect.Parameter.empty:
                problems.append(f"{name}.{pname} 是必填但给了默认值")
    assert not problems, "\n".join(problems)


def test_registry_has_expected_core_tools():
    bundle = build_harness()
    names = set(bundle["registry"].names())
    for expected in (
        "policy_search", "list_policies", "case_lookup", "get_case",
        "tax_calc", "optimize_plan", "write_file", "read_file",
        "task", "task_board", "load_skill", "run_workflow",
    ):
        assert expected in names, f"缺少核心工具 {expected}"


def test_mcp_tools_are_merged_into_pool():
    bundle = build_harness()
    assert bundle["mcp_tools"], "MCP 工具没有被并入工具池"
    for n in bundle["mcp_tools"]:
        assert bundle["registry"].get(n) is not None
