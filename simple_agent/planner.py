from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


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
            r"(?:把|将)\s+(?P<path>\S+)\s+(?:中|里的)?\s*[\"'`“](?P<old>.+?)[\"'`”]\s*(?:替换为|改成)\s*[\"'`“](?P<new>.*?)[\"'`”]",
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
        pattern = r"(?:创建|写入|新增)\s+(?P<path>\S+)\s+(?:内容为|with content)\s*[\"'`“](?P<content>.*?)[\"'`”]"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
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
                return tail.split()[0].strip("，,。.")
    return None


def _clean_phrase(text: str) -> str:
    text = text.strip().strip("：:，,。.")
    for stop in ["并", "然后", "，", ",", "。"]:
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


def _fallback_keyword(text: str) -> str:
    words = re.findall(r"[\w./-]+", text)
    return max(words, key=len) if words else text[:20]
