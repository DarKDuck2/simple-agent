from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from simple_agent.llm import ClaudePlanner, OpenAIPlanner, UsageStats


@dataclass(frozen=True)
class PlanStep:
    tool: str
    params: dict[str, Any]
    reason: str


@dataclass(frozen=True)
class Plan:
    goal: str
    steps: list[PlanStep]

    def describe(self) -> str:
        lines = [f"Goal: {self.goal}"]
        for index, step in enumerate(self.steps, 1):
            lines.append(f"{index}. {step.tool} {step.params} - {step.reason}")
        return "\n".join(lines)


# Anthropic-style tool schemas for LLM planners
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "repo_tree",
        "description": "Parse and display the repository directory tree.",
        "input_schema": {
            "type": "object",
            "properties": {
                "depth": {"type": "integer", "description": "Maximum depth to traverse (default: 4)"},
            },
            "required": [],
        },
    },
    {
        "name": "search_code",
        "description": "Search repository files by keyword (simple text search).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keyword"},
                "limit": {"type": "integer", "description": "Maximum results (default: 20)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "grep_search",
        "description": "Search repository files using ripgrep (regex supported, faster for large repos).",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search"},
                "glob": {"type": "string", "description": "File glob filter, e.g. '*.py'"},
                "limit": {"type": "integer", "description": "Maximum results (default: 30)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "glob_search",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file with optional line limits.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "start": {"type": "integer", "description": "Starting line number (1-indexed, default: 1)"},
                "max_lines": {"type": "integer", "description": "Maximum lines to read (default: 200)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace text in a file. The old_string must uniquely exist in the file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "old_string": {"type": "string", "description": "Exact text to replace (must uniquely exist)"},
                "new_string": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "file_create",
        "description": "Create a new file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the new file"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "file_delete",
        "description": "Delete a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file to delete"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_shell",
        "description": "Run a shell command in the repository root. Dangerous commands require confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "show_diff",
        "description": "Display repository changes as a unified diff.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "git_status",
        "description": "Show git status of the repository.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


class HeuristicPlanner:
    """A deterministic planner that keeps the prototype runnable without an API key."""

    def plan(self, task: str, repo_summary: str) -> Plan:
        text = task.strip()
        lowered = text.lower()
        steps: list[PlanStep] = []

        if any(word in lowered for word in ["目录", "tree", "结构"]):
            steps.append(PlanStep("repo_tree", {"depth": 4}, "理解仓库结构"))

        search_match = _match_after(text, ["搜索", "查找", "search"])
        if search_match:
            steps.append(PlanStep("search_code", {"query": search_match, "limit": 20}, "召回相关代码"))

        read_path = _extract_path_after(text, ["阅读", "读取", "查看文件", "read"])
        if read_path:
            steps.append(PlanStep("read_file", {"path": read_path, "max_lines": 200}, "读取目标文件"))

        replace_step = self._build_replace_step(text)
        if replace_step:
            steps.append(replace_step)

        create_step = self._build_create_step(text)
        if create_step:
            steps.append(create_step)

        if any(word in lowered for word in ["测试", "单测", "pytest", "run test", "run tests"]):
            command = "pytest -q" if "pytest" in lowered else "python -m unittest discover -s tests"
            steps.append(PlanStep("run_shell", {"command": command}, "运行测试验证修改"))

        if any(word in lowered for word in ["diff", "差异", "变更"]):
            steps.append(PlanStep("show_diff", {}, "展示修改 diff"))

        if not steps:
            steps = [
                PlanStep("repo_tree", {"depth": 3}, "先理解仓库结构"),
                PlanStep("search_code", {"query": _fallback_keyword(text), "limit": 20}, "按任务关键词检索代码"),
            ]

        return Plan(goal=text, steps=steps)

    def reflect(self, failed_step: PlanStep, output: str) -> list[PlanStep]:
        if failed_step.tool == "run_shell" and ("No module named pytest" in output or "pytest: command not found" in output):
            return [PlanStep("run_shell", {"command": "python -m unittest discover -s tests"}, "pytest 不可用，回退到 unittest")]
        if failed_step.tool == "read_file" and "文件不存在" in output:
            return [PlanStep("repo_tree", {"depth": 4}, "读取失败，重新查看目录")]
        return []

    def _build_replace_step(self, text: str) -> PlanStep | None:
        patterns = [
            r"(?:把|将)\s+(?P<path>\S+)\s+(?:中|里的)?\s*[\"'`](?P<old>.+?)[\"'`]\s*(?:替换为|改成)\s*[\"'`](?P<new>.*?)[\"']",
            r"replace\s+(?P<old>\S+)\s+with\s+(?P<new>\S+)\s+in\s+(?P<path>\S+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return PlanStep(
                    "edit_file",
                    {
                        "path": match.group("path").strip(),
                        "mode": "replace",
                        "old": match.group("old"),
                        "content": match.group("new"),
                    },
                    "按需求执行文件级替换",
                )
        return None

    def _build_create_step(self, text: str) -> PlanStep | None:
        # Pattern: 创建 <path> [文件] 内容为 <content>
        # Supports: "创建 foo.py 内容为 xxx", "创建 foo.py 文件内容为 xxx"
        pattern = r"(?:创建|写入|新增)\s+(?P<path>\S+?)\s*(?:文件\s*)?(?:内容为|with content)\s*[\"'](?P<content>.*?)[\"']"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            # Try simpler pattern without quotes
            pattern2 = r"(?:创建|写入|新增)\s+(?P<path>\S+?)\s*(?:文件\s*)?(?:内容为|with content)\s*(?P<content>\S.*)"
            match = re.search(pattern2, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        return PlanStep(
            "edit_file",
            {
                "path": match.group("path").strip(),
                "mode": "overwrite",
                "content": match.group("content"),
            },
            "创建或覆盖目标文件",
        )


class LLMPlanner:
    """Planner that delegates to an LLM backend (Claude/OpenAI) with HeuristicPlanner as fallback."""

    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._fallback = HeuristicPlanner()
        self._backend = self._create_backend(backend, model, api_key, base_url)
        self._history: list[dict[str, Any]] = []

    def _create_backend(
        self,
        backend: str | None,
        model: str | None,
        api_key: str | None,
        base_url: str | None,
    ) -> ClaudePlanner | OpenAIPlanner | None:
        if backend == "claude" or (backend is None and os.environ.get("ANTHROPIC_API_KEY")):
            kwargs: dict[str, Any] = {"api_key": api_key}
            if model:
                kwargs["model"] = model
            if base_url:
                kwargs["base_url"] = base_url
            try:
                return ClaudePlanner(**kwargs)
            except Exception:
                return None

        if backend == "openai" or (backend is None and os.environ.get("OPENAI_API_KEY")):
            kwargs = {"api_key": api_key}
            if model:
                kwargs["model"] = model
            if base_url:
                kwargs["base_url"] = base_url
            try:
                return OpenAIPlanner(**kwargs)
            except Exception:
                return None

        return None

    def plan(self, task: str, repo_summary: str) -> Plan:
        if self._backend is None:
            return self._fallback.plan(task, repo_summary)

        try:
            plan = self._backend.generate_plan(task, repo_summary, self._history, TOOL_SCHEMAS)
            self._history.append({"role": "assistant", "content": plan.describe()})
            return plan
        except Exception as exc:
            print(f"[LLMPlanner] LLM 规划失败，回退到启发式规划器：{exc}")
            return self._fallback.plan(task, repo_summary)

    def reflect(self, failed_step: PlanStep, output: str) -> list[PlanStep]:
        if self._backend is None:
            return self._fallback.reflect(failed_step, output)

        try:
            self._history.append({
                "role": "user",
                "content": f"工具 {failed_step.tool} 执行失败：{output[:1000]}",
            })
            steps = self._backend.reflect(failed_step, output, self._history, TOOL_SCHEMAS)
            if steps:
                self._history.append({"role": "assistant", "content": f"反思后生成 {len(steps)} 个后续步骤"})
            return steps
        except Exception:
            return self._fallback.reflect(failed_step, output)

    def record_tool_result(self, step: PlanStep, result_ok: bool, output: str) -> None:
        """Record tool execution result into conversation history."""
        self._history.append({
            "role": "user",
            "content": (
                f"工具执行结果：\n"
                f"工具：{step.tool}\n"
                f"参数：{json.dumps(step.params, ensure_ascii=False)}\n"
                f"状态：{'成功' if result_ok else '失败'}\n"
                f"输出：{output[:2000]}"
            ),
        })

    def total_usage(self) -> UsageStats:
        if self._backend is None:
            return UsageStats()
        return self._backend.total_usage()

    @property
    def using_llm(self) -> bool:
        return self._backend is not None


def _match_after(text: str, markers: list[str]) -> str | None:
    for marker in markers:
        if marker in text:
            tail = text.split(marker, 1)[1].strip()
            return _clean_phrase(tail)
    return None


def _extract_path_after(text: str, markers: list[str]) -> str | None:
    for marker in markers:
        if marker in text:
            tail = text.split(marker, 1)[1].strip()
            if tail:
                return tail.split()[0].strip("，,。. ")
    return None


def _clean_phrase(text: str) -> str:
    text = text.strip().strip("：:，,。. ")
    # Remove common noise words that aren't search terms
    noise = ["所有", "包含", "的", "文件", "代码", "查找", "搜索", "在哪里", "什么"]
    for word in noise:
        text = text.replace(word, " ")
    # Keep only meaningful tokens (letters, numbers, dots, underscores)
    tokens = re.findall(r"[\w.]+", text)
    return " ".join(tokens).strip() if tokens else text.strip()


def _fallback_keyword(text: str) -> str:
    words = re.findall(r"[\w./-]+", text)
    return max(words, key=len) if words else text[:20]
