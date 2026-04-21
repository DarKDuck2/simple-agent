from __future__ import annotations

import difflib
import fnmatch
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
        timeout_seconds: int = 30,
    ) -> None:
        self.root = Path(root).resolve()
        self.dry_run = dry_run
        self.timeout_seconds = timeout_seconds

    def build_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(Tool("repo_tree", "Parse and display the repository directory tree.", self.repo_tree))
        registry.register(Tool("search_code", "Search repository files by keyword (simple text search).", self.search_code))
        registry.register(Tool("grep_search", "Search repository files using ripgrep (regex supported).", self.grep_search))
        registry.register(Tool("glob_search", "Find files matching a glob pattern.", self.glob_search))
        registry.register(Tool("read_file", "Read a file with optional line limits.", self.read_file))
        registry.register(Tool("edit_file", "Replace text in a file using old_string/new_string (must uniquely exist).", self.edit_file))
        registry.register(Tool("file_create", "Create a new file with the given content.", self.file_create))
        registry.register(Tool("file_delete", "Delete a file.", self.file_delete))
        registry.register(Tool("run_shell", "Run a shell command in the repository root.", self.run_shell))
        registry.register(Tool("show_diff", "Display repository changes as a unified diff.", self.show_diff))
        registry.register(Tool("git_status", "Show git status of the repository.", self.git_status))
        registry.register(Tool("git_commit", "Commit all changes with a message.", self.git_commit))
        return registry

    # --- Core tools ---

    def repo_tree(self, params: dict[str, Any]) -> ToolResult:
        depth = int(params.get("depth", 4))
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
        return ToolResult(True, "\n".join(matches) if matches else "未找到匹配内容")

    def grep_search(self, params: dict[str, Any]) -> ToolResult:
        pattern = str(params.get("pattern", "")).strip()
        if not pattern:
            return ToolResult(False, "grep_search 需要参数 pattern")
        limit = int(params.get("limit", 30))
        glob_filter = str(params.get("glob", "")).strip()

        # Check if ripgrep is available
        rg_check = subprocess.run(["rg", "--version"], capture_output=True, text=True)
        if rg_check.returncode != 0:
            # Fallback to simple search
            return self.search_code({"query": pattern, "limit": limit})

        cmd = ["rg", "--line-number", "--no-heading", "--color=never", "-m", str(limit), pattern]
        if glob_filter:
            cmd.extend(["-g", glob_filter])

        completed = subprocess.run(
            cmd,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return ToolResult(True, completed.stdout.strip())
        if completed.returncode == 1:
            return ToolResult(True, "未找到匹配内容")
        return ToolResult(False, completed.stderr.strip() or "ripgrep 执行失败")

    def glob_search(self, params: dict[str, Any]) -> ToolResult:
        pattern = str(params.get("pattern", "")).strip()
        if not pattern:
            return ToolResult(False, "glob_search 需要参数 pattern")

        matches: list[str] = []
        # Use pathlib glob
        for path in self.root.rglob(pattern):
            if path.is_file() and not self._ignored(path):
                matches.append(self._rel(path))

        if not matches:
            return ToolResult(True, "未找到匹配文件")
        return ToolResult(True, "\n".join(sorted(matches)))

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

        # New-style: old_string + new_string (Claude Code style)
        old_string = params.get("old_string")
        new_string = params.get("new_string")

        if old_string is not None and new_string is not None:
            return self._edit_file_new_style(path, str(old_string), str(new_string))

        # Legacy style: mode + old + content
        mode = str(params.get("mode", "replace"))
        content = str(params.get("content", ""))
        old = str(params.get("old", ""))
        return self._edit_file_legacy(path, mode, old, content)

    def _edit_file_new_style(self, path: Path, old_string: str, new_string: str) -> ToolResult:
        if not path.exists():
            return ToolResult(False, f"文件不存在：{self._rel(path)}")

        before = path.read_text(encoding="utf-8")

        # Check uniqueness
        count = before.count(old_string)
        if count == 0:
            return ToolResult(False, f"old_string 在文件中未找到：{old_string[:80]}...")
        if count > 1:
            # Show lines where it appears
            lines = []
            for i, line in enumerate(before.splitlines(), 1):
                if old_string in line:
                    lines.append(f"  Line {i}: {line.strip()[:100]}")
            return ToolResult(
                False,
                f"old_string 在文件中出现 {count} 次，必须唯一。出现位置：\n" + "\n".join(lines[:5]),
            )

        after = before.replace(old_string, new_string, 1)
        diff = _unified_diff(before, after, self._rel(path))

        if self.dry_run:
            return ToolResult(True, "dry-run：未写入文件\n" + diff, {"dry_run": True})

        path.write_text(after, encoding="utf-8")
        return ToolResult(True, diff or f"文件已更新：{self._rel(path)}")

    def _edit_file_legacy(self, path: Path, mode: str, old: str, content: str) -> ToolResult:
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

    def file_create(self, params: dict[str, Any]) -> ToolResult:
        path = self._safe_path(str(params.get("path", "")))
        if path is None:
            return ToolResult(False, "file_create 需要仓库内 path")
        content = str(params.get("content", ""))

        if path.exists():
            return ToolResult(False, f"文件已存在：{self._rel(path)}")

        path.parent.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            return ToolResult(True, f"dry-run：未创建文件 {self._rel(path)}\n内容预览（前 200 字符）：{content[:200]}", {"dry_run": True})

        path.write_text(content, encoding="utf-8")
        return ToolResult(True, f"文件已创建：{self._rel(path)}")

    def file_delete(self, params: dict[str, Any]) -> ToolResult:
        path = self._safe_path(str(params.get("path", "")))
        if path is None:
            return ToolResult(False, "file_delete 需要仓库内 path")
        if not path.exists():
            return ToolResult(False, f"文件不存在：{self._rel(path)}")

        if self.dry_run:
            return ToolResult(True, f"dry-run：未删除文件 {self._rel(path)}", {"dry_run": True})

        path.unlink()
        return ToolResult(True, f"文件已删除：{self._rel(path)}")

    def run_shell(self, params: dict[str, Any]) -> ToolResult:
        command = str(params.get("command", "")).strip()
        if not command:
            return ToolResult(False, "run_shell 需要参数 command")

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

        # Truncate very long outputs
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        max_output = 10000
        if len(stdout) > max_output:
            stdout = stdout[:max_output] + f"\n... (truncated {len(completed.stdout.strip()) - max_output} chars)"
        if len(stderr) > max_output:
            stderr = stderr[:max_output] + f"\n... (truncated {len(completed.stderr.strip()) - max_output} chars)"

        output = "\n".join(
            part
            for part in [
                f"exit_code={completed.returncode}",
                stdout,
                stderr,
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

    def git_status(self, params: dict[str, Any]) -> ToolResult:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode == 0:
            text = completed.stdout.strip() or "工作区干净，无未提交变更"
            return ToolResult(True, text)
        return ToolResult(False, completed.stderr.strip() or "git status 执行失败")

    def git_commit(self, params: dict[str, Any]) -> ToolResult:
        message = str(params.get("message", "")).strip()
        if not message:
            return ToolResult(False, "git_commit 需要参数 message")

        if self.dry_run:
            return ToolResult(True, f"dry-run：未执行 git commit -m '{message}'", {"dry_run": True})

        completed = subprocess.run(
            ["git", "add", "-A"],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            return ToolResult(False, f"git add 失败：{completed.stderr.strip()}")

        completed = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode == 0:
            return ToolResult(True, completed.stdout.strip())
        return ToolResult(False, completed.stderr.strip() or "git commit 执行失败")

    # --- Helpers ---

    def _iter_text_files(self) -> list[Path]:
        ignored_dirs = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}
        result: list[Path] = []
        for current, dirs, files in os.walk(self.root):
            dirs[:] = [name for name in dirs if name not in ignored_dirs]
            for name in files:
                path = Path(current) / name
                if self._ignored(path):
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

    def _ignored(self, path: Path) -> bool:
        return path.name in {".git", ".venv", "__pycache__", "node_modules"} or path.name.endswith((".pyc", ".pyo"))


def _unified_diff(before: str, after: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
