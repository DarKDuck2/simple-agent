from __future__ import annotations

import ast
import io
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ToolFunc = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    func: ToolFunc


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def execute(self, name: str, params: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"工具不存在：{name}"

        try:
            return tool.func(params)
        except Exception as exc:
            return f"工具执行失败：{type(exc).__name__}: {exc}"


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool("search", "Search local demo knowledge.", search_tool))
    registry.register(Tool("calculator", "Evaluate a math expression.", calculator_tool))
    registry.register(Tool("code_run", "Run a small Python snippet.", code_run_tool))
    return registry


def search_tool(params: dict[str, Any]) -> str:
    query = str(params.get("query", "")).strip()
    if not query:
        return "search 需要参数 query"

    knowledge = {
        "openclaw": "OpenClaw 是一个强调工具调用、上下文管理和任务执行闭环的 Agent 项目。",
        "claude code": "Claude Code 是面向代码库理解、编辑、终端执行和多轮协作的编程 Agent。",
        "agent": "Agent 通常由 LLM、工具、记忆/上下文、规划与执行循环组成。",
    }

    lowered_query = query.lower()
    for keyword, answer in knowledge.items():
        if keyword in lowered_query:
            return answer

    return f"没有找到精确资料。这是一个模拟 search 结果，查询词：{query}"


def calculator_tool(params: dict[str, Any]) -> str:
    expression = str(params.get("expression", "")).strip()
    if not expression:
        return "calculator 需要参数 expression"

    node = ast.parse(expression, mode="eval")
    if not _is_safe_math_node(node):
        return "表达式包含不允许的内容"

    result = eval(compile(node, "<calculator>", "eval"), {"__builtins__": {}}, {})
    return str(result)


def _is_safe_math_node(node: ast.AST) -> bool:
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
    )
    return all(isinstance(child, allowed) for child in ast.walk(node))


def code_run_tool(params: dict[str, Any]) -> str:
    code = str(params.get("code", "")).strip()
    if not code:
        return "code_run 需要参数 code"

    blocked_terms = ["import os", "import socket", "subprocess", "open(", "__import__"]
    if any(term in code for term in blocked_terms):
        return "代码包含演示沙箱不允许的内容"

    return _run_python_in_subprocess(code)


def _run_python_in_subprocess(code: str) -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = Path(temp_dir) / "snippet.py"
        script_path.write_text(code, encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=temp_dir,
            check=False,
        )

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    if completed.returncode != 0:
        return f"代码退出码：{completed.returncode}\nstderr:\n{stderr}"

    return stdout or "代码执行成功，但没有输出"


def _run_python_in_process(code: str) -> str:
    """Kept for comparison: simpler but less isolated than subprocess execution."""
    output = io.StringIO()
    with redirect_stdout(output):
        exec(code, {"__builtins__": {"print": print, "range": range, "sum": sum}}, {})
    return output.getvalue().strip()

