from __future__ import annotations

import difflib
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from simple_agent.repo_context import render_tree


ToolFunc = Callable[[dict[str, Any]], "ToolResult"]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    metadata: dict[str, Any] | None = None


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

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        return "\n".join(f"- {tool.name}: {tool.description}" for tool in self._tools.values())

    def execute(self, name: str, params: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, f"工具不存在：{name}")
        try:
            return tool.func(params)
        except Exception as exc:
            return ToolResult(False, f"工具执行失败：{type(exc).__name__}: {exc}")


class RepoToolKit:
    def __init__(
        self,
        root: str | Path,
        *,
        dry_run: bool = False,
        command_allowlist: list[str] | None = None,
        timeout_seconds: int = 10,
    ) -> None:
        self.root = Path(root).resolve()
        self.dry_run = dry_run
        self.timeout_seconds = timeout_seconds
        self.command_allowlist = command_allowlist or [
            "python",
            "python3",
            "pytest",
            "ruff",
            "mypy",
            "npm",
            "pnpm",
            "yarn",
            "go",
            "cargo",
            "git",
        ]

    def build_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(Tool("repo_tree", "Parse and display the repository directory tree.", self.repo_tree))
        registry.register(Tool("search_code", "Search repository files by keyword.", self.search_code))
        registry.register(Tool("read_file", "Read a file with optional line limits.", self.read_file))
        registry.register(Tool("edit_file", "Create, overwrite, append, or replace text in a repository file.", self.edit_file))
        registry.register(Tool("run_shell", "Run an allowlisted shell command in the repository.", self.run_shell))
        registry.register(Tool("show_diff", "Display repository changes as a unified diff.", self.show_diff))
        return registry

    def repo_tree(self, params: dict[str, Any]) -> ToolResult:
        depth = int(params.get("depth", 3))
        return ToolResult(True, render_tree(self.root, max_depth=depth))

    def search_code(self, params: dict[str, Any]) -> ToolResult:
        query = str(params.get("query", "")).strip()
        if not query:
            return ToolResult(False, "search_code 需要参数 query")
        limit = int(params.get("limit", 20))
        matches: list[str] = []
        for path in self._iter_text_files():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(lines, 1):
                if query.lower() in line.lower():
                    matches.append(f"{self._rel(path)}:{number}: {line.strip()}")
                    if len(matches) >= limit:
                        return ToolResult(True, "\n".join(matches))
        return ToolResult(True, "未找到匹配内容")

    def read_file(self, params: dict[str, Any]) -> ToolResult:
        path = self._safe_path(str(params.get("path", "")))
        if path is None:
            return ToolResult(False, "read_file 需要仓库内 path")
        if not path.exists() or not path.is_file():
            return ToolResult(False, f"文件不存在：{self._rel(path)}")
        start = int(params.get("start", 1))
        max_lines = int(params.get("max_lines", 200))
        lines = path.read_text(encoding="utf-8").splitlines()
        selected = lines[max(start - 1, 0) : max(start - 1, 0) + max_lines]
        numbered = [f"{idx}: {line}" for idx, line in enumerate(selected, start)]
        return ToolResult(True, "\n".join(numbered) if numbered else "文件为空")

    def edit_file(self, params: dict[str, Any]) -> ToolResult:
        path = self._safe_path(str(params.get("path", "")))
        if path is None:
            return ToolResult(False, "edit_file 需要仓库内 path")
        mode = str(params.get("mode", "replace"))
        content = str(params.get("content", ""))
        old = str(params.get("old", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        before = path.read_text(encoding="utf-8") if path.exists() else ""

        if mode == "replace":
            if not old:
                return ToolResult(False, "replace 模式需要 old")
            if old not in before:
                return ToolResult(False, f"没有找到待替换文本：{old}")
            after = before.replace(old, content, 1)
        elif mode == "overwrite":
            after = content
        elif mode == "append":
            after = before + content
        else:
            return ToolResult(False, f"不支持的 edit_file mode：{mode}")

        diff = _unified_diff(before, after, self._rel(path))
        if self.dry_run:
            return ToolResult(True, "dry-run：未写入文件\n" + diff, {"dry_run": True})

        path.write_text(after, encoding="utf-8")
        return ToolResult(True, diff or f"文件已更新：{self._rel(path)}")

    def run_shell(self, params: dict[str, Any]) -> ToolResult:
        command = str(params.get("command", "")).strip()
        if not command:
            return ToolResult(False, "run_shell 需要参数 command")
        allowed, reason = self._command_allowed(command)
        if not allowed:
            return ToolResult(False, reason)
        if self.dry_run:
            return ToolResult(True, f"dry-run：未执行命令 {command}", {"dry_run": True})
        completed = subprocess.run(
            command,
            shell=True,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        output = "\n".join(
            part
            for part in [
                f"exit_code={completed.returncode}",
                completed.stdout.strip(),
                completed.stderr.strip(),
            ]
            if part
        )
        return ToolResult(completed.returncode == 0, output, {"exit_code": completed.returncode})

    def show_diff(self, params: dict[str, Any]) -> ToolResult:
        command = "git diff -- ."
        completed = subprocess.run(
            command,
            shell=True,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return ToolResult(True, completed.stdout.strip())
        if completed.returncode == 0:
            return ToolResult(True, "当前没有 git diff")
        return ToolResult(False, completed.stderr.strip() or "git diff 执行失败")

    def _iter_text_files(self) -> list[Path]:
        ignored_dirs = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}
        result: list[Path] = []
        for current, dirs, files in os.walk(self.root):
            dirs[:] = [name for name in dirs if name not in ignored_dirs]
            for name in files:
                path = Path(current) / name
                if path.suffix in {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".zip"}:
                    continue
                result.append(path)
        return result

    def _safe_path(self, raw_path: str) -> Path | None:
        if not raw_path:
            return None
        path = (self.root / raw_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            return None
        return path

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _command_allowed(self, command: str) -> tuple[bool, str]:
        blocked_tokens = ["rm", "sudo", "chmod", "chown", "curl", "wget", "scp", "ssh", "mkfs", "dd"]
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return False, f"命令解析失败：{exc}"
        if not parts:
            return False, "空命令"
        executable = Path(parts[0]).name
        if executable in blocked_tokens:
            return False, f"高风险命令被拒绝：{executable}"
        if executable not in self.command_allowlist:
            return False, f"命令不在白名单：{executable}"
        if any(token in parts for token in [">", ">>", "2>", "&&", "||", ";"]):
            return False, "演示版 run_shell 不允许重定向或命令串联"
        return True, "ok"


def _unified_diff(before: str, after: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
