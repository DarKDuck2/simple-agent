from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from simple_agent.planner import Plan, PlanStep


class LLM(Protocol):
    def infer(self, messages: list[dict[str, str]]) -> str:
        """Return the next assistant message."""


class LLMPlanner(Protocol):
    def generate_plan(self, task: str, repo_context: str, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> "Plan":
        """Generate a plan for the given task using available tools."""

    def reflect(self, failed_step: "PlanStep", output: str, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list["PlanStep"]:
        """Reflect on a failed step and generate follow-up actions."""


@dataclass(frozen=True)
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0


def _get_planner_classes():
    from simple_agent.planner import Plan, PlanStep
    return Plan, PlanStep


@dataclass
class ClaudePlanner:
    """Anthropic Claude-based planner with native tool use support."""

    model: str = "claude-sonnet-4-6"
    max_retries: int = 3
    retry_delay: float = 1.0
    api_key: str | None = None
    base_url: str | None = None
    _usage: list[UsageStats] = field(default_factory=list, repr=False)

    def _client(self):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("请先安装 anthropic：pip install anthropic") from exc

        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("请先设置 ANTHROPIC_API_KEY 环境变量或在构造函数中传入 api_key")

        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return anthropic.Anthropic(**kwargs)

    def _system_prompt(self) -> str:
        return (
            "你是一个面向代码仓库的 AI 编程助手。你的任务是根据用户需求，"
            "选择合适的工具来完成代码理解、搜索、阅读和编辑等操作。\n\n"
            "规则：\n"
            "1. 分析用户任务，选择合适的工具步骤\n"
            "2. 每一步只选择一个工具，并提供精确的参数\n"
            "3. 如果需要多步操作，按顺序列出所有步骤\n"
            "4. 对于文件编辑操作，必须确保 old_string 在目标文件中唯一存在\n"
            "5. 如果任务需要验证（如修改后运行测试），请在最后添加验证步骤\n"
        )

    def generate_plan(self, task: str, repo_context: str, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Plan:
        Plan, PlanStep = _get_planner_classes()
        client = self._client()
        messages = self._build_messages(task, repo_context, history)

        for attempt in range(self.max_retries):
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self._system_prompt(),
                    messages=messages,  # type: ignore[arg-type]
                    tools=tools,  # type: ignore[arg-type]
                    tool_choice={"type": "any"},
                )

                if response.usage:
                    self._usage.append(UsageStats(
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                    ))

                steps = self._parse_tool_uses(response.content)
                if steps:
                    return Plan(goal=task, steps=steps)

                # No tool use found, fallback to text response as a single step
                text = " ".join(
                    block.text for block in response.content
                    if block.type == "text"
                )
                return Plan(goal=task, steps=[
                    PlanStep("repo_tree", {"depth": 3}, f"无法解析工具调用，先查看仓库结构。LLM 回复：{text[:200]}")
                ])

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(f"LLM 调用失败：{e}") from e

        return Plan(goal=task, steps=[])

    def reflect(self, failed_step: PlanStep, output: str, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[PlanStep]:
        Plan, PlanStep = _get_planner_classes()
        prompt = self._build_reflect_prompt(failed_step, output, history)
        messages = [{"role": "user", "content": prompt}]

        try:
            client = self._client()
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self._system_prompt(),
                messages=messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                tool_choice={"type": "any"},
            )

            steps = self._parse_tool_uses(response.content)
            return steps

        except Exception:
            return []

    def _build_messages(self, task: str, repo_context: str, history: list[dict[str, Any]]) -> list[dict[str, str]]:
        system = (
            f"仓库上下文：\n{repo_context}\n\n"
            "可用工具：\n"
            "- repo_tree(depth): 列出目录结构\n"
            "- search_code(query, limit): 搜索代码内容\n"
            "- read_file(path, start, max_lines): 读取文件\n"
            "- edit_file(path, old_string, new_string): 替换文件内容\n"
            "- file_create(path, content): 创建新文件\n"
            "- file_delete(path): 删除文件\n"
            "- run_shell(command): 执行 Shell 命令\n"
            "- show_diff(): 显示差异\n"
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]

        for entry in history[-5:]:
            if entry.get("role") in ("user", "assistant"):
                messages.append({"role": entry["role"], "content": str(entry.get("content", ""))})

        messages.append({"role": "user", "content": task})
        return messages

    def _build_reflect_prompt(self, failed_step: PlanStep, output: str, history: list[dict[str, Any]]) -> str:
        return (
            f"上一步执行失败：\n"
            f"- 工具：{failed_step.tool}\n"
            f"- 参数：{failed_step.params}\n"
            f"- 输出：{output[:500]}\n\n"
            "请生成下一步修复方案。如果需要查看仓库结构，使用 repo_tree 工具。"
        )

    def _parse_tool_uses(self, content_blocks: list[Any]) -> list[PlanStep]:
        Plan, PlanStep = _get_planner_classes()
        steps: list[PlanStep] = []

        for block in content_blocks:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                if tool_name == "repo_tree":
                    steps.append(PlanStep(
                        "repo_tree",
                        {"depth": tool_input.get("depth", 3)},
                        "查看仓库结构",
                    ))
                elif tool_name == "search_code":
                    steps.append(PlanStep(
                        "search_code",
                        {"query": tool_input.get("query", ""), "limit": tool_input.get("limit", 20)},
                        "搜索代码",
                    ))
                elif tool_name == "read_file":
                    params: dict[str, Any] = {"path": tool_input.get("path", "")}
                    if "start" in tool_input:
                        params["start"] = tool_input["start"]
                    if "max_lines" in tool_input:
                        params["max_lines"] = tool_input["max_lines"]
                    steps.append(PlanStep("read_file", params, "读取文件"))
                elif tool_name == "edit_file":
                    steps.append(PlanStep(
                        "edit_file",
                        {
                            "path": tool_input.get("path", ""),
                            "old_string": tool_input.get("old_string", ""),
                            "new_string": tool_input.get("new_string", ""),
                        },
                        "编辑文件",
                    ))
                elif tool_name == "file_create":
                    steps.append(PlanStep(
                        "edit_file",
                        {"path": tool_input.get("path", ""), "mode": "overwrite", "content": tool_input.get("content", "")},
                        "创建文件",
                    ))
                elif tool_name == "file_delete":
                    steps.append(PlanStep(
                        "edit_file",
                        {"path": tool_input.get("path", ""), "mode": "delete"},
                        "删除文件",
                    ))
                elif tool_name == "run_shell":
                    steps.append(PlanStep(
                        "run_shell",
                        {"command": tool_input.get("command", "")},
                        "执行命令",
                    ))
                elif tool_name == "show_diff":
                    steps.append(PlanStep("show_diff", {}, "查看差异"))

        return steps


@dataclass
class OpenAIPlanner:
    """OpenAI-based planner with function calling support."""

    model: str = "gpt-4o"
    max_retries: int = 3
    retry_delay: float = 1.0
    api_key: str | None = None
    base_url: str | None = None
    _usage: list[UsageStats] = field(default_factory=list, repr=False)

    def _client(self):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("请先安装 openai：pip install openai") from exc

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("请先设置 OPENAI_API_KEY 环境变量或在构造函数中传入 api_key")

        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return openai.OpenAI(**kwargs)

    def _system_prompt(self) -> str:
        return (
            "你是一个面向代码仓库的 AI 编程助手。你的任务是根据用户需求，"
            "选择合适的工具来完成代码理解、搜索、阅读和编辑等操作。\n\n"
            "规则：\n"
            "1. 分析用户任务，选择合适的工具步骤\n"
            "2. 每一步只选择一个工具，并提供精确的参数\n"
            "3. 如果需要多步操作，按顺序列出所有步骤\n"
            "4. 对于文件编辑操作，必须确保 old_string 在目标文件中唯一存在\n"
            "5. 如果任务需要验证（如修改后运行测试），请在最后添加验证步骤\n"
        )

    def generate_plan(self, task: str, repo_context: str, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Plan:
        Plan, PlanStep = _get_planner_classes()
        client = self._client()
        messages = self._build_messages(task, repo_context, history)

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "automatic"},
                )

                message = response.choices[0].message
                if message.tool_calls:
                    steps = self._parse_tool_calls(message.tool_calls)
                    if steps:
                        return Plan(goal=task, steps=steps)

                text = message.content or ""
                return Plan(goal=task, steps=[
                    PlanStep("repo_tree", {"depth": 3}, f"无法解析工具调用，先查看仓库结构。LLM 回复：{text[:200]}")
                ])

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(f"LLM 调用失败：{e}") from e

        return Plan(goal=task, steps=[])

    def reflect(self, failed_step: PlanStep, output: str, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[PlanStep]:
        Plan, PlanStep = _get_planner_classes()
        prompt = self._build_reflect_prompt(failed_step, output, history)
        messages = [{"role": "system", "content": self._system_prompt()}, {"role": "user", "content": prompt}]

        try:
            client = self._client()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice={"type": "automatic"},
            )

            message = response.choices[0].message
            if message.tool_calls:
                return self._parse_tool_calls(message.tool_calls)

        except Exception:
            pass

        return []

    def _build_messages(self, task: str, repo_context: str, history: list[dict[str, Any]]) -> list[dict[str, str]]:
        system = (
            f"仓库上下文：\n{repo_context}\n\n"
            "可用工具：\n"
            "- repo_tree(depth): 列出目录结构\n"
            "- search_code(query, limit): 搜索代码内容\n"
            "- read_file(path, start, max_lines): 读取文件\n"
            "- edit_file(path, old_string, new_string): 替换文件内容\n"
            "- file_create(path, content): 创建新文件\n"
            "- file_delete(path): 删除文件\n"
            "- run_shell(command): 执行 Shell 命令\n"
            "- show_diff(): 显示差异\n"
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]

        for entry in history[-5:]:
            if entry.get("role") in ("user", "assistant"):
                messages.append({"role": entry["role"], "content": str(entry.get("content", ""))})

        messages.append({"role": "user", "content": task})
        return messages

    def _build_reflect_prompt(self, failed_step: PlanStep, output: str, history: list[dict[str, Any]]) -> str:
        return (
            f"上一步执行失败：\n"
            f"- 工具：{failed_step.tool}\n"
            f"- 参数：{failed_step.params}\n"
            f"- 输出：{output[:500]}\n\n"
            "请生成下一步修复方案。如果需要查看仓库结构，使用 repo_tree 工具。"
        )

    def _parse_tool_calls(self, tool_calls: Any) -> list[PlanStep]:
        Plan, PlanStep = _get_planner_classes()
        steps: list[PlanStep] = []

        for call in tool_calls:
            func = call.function
            name = func.name
            arguments = json.loads(func.arguments)

            if name == "repo_tree":
                steps.append(PlanStep(
                    "repo_tree",
                    {"depth": arguments.get("depth", 3)},
                    "查看仓库结构",
                ))
            elif name == "search_code":
                steps.append(PlanStep(
                    "search_code",
                    {"query": arguments.get("query", ""), "limit": arguments.get("limit", 20)},
                    "搜索代码",
                ))
            elif name == "read_file":
                params = {"path": arguments.get("path", "")}
                if "start" in arguments:
                    params["start"] = arguments["start"]
                if "max_lines" in arguments:
                    params["max_lines"] = arguments["max_lines"]
                steps.append(PlanStep("read_file", params, "读取文件"))
            elif name == "edit_file":
                steps.append(PlanStep(
                    "edit_file",
                    {
                        "path": arguments.get("path", ""),
                        "old_string": arguments.get("old_string", ""),
                        "new_string": arguments.get("new_string", ""),
                    },
                    "编辑文件",
                ))
            elif name == "file_create":
                steps.append(PlanStep(
                    "edit_file",
                    {"path": arguments.get("path", ""), "mode": "overwrite", "content": arguments.get("content", "")},
                    "创建文件",
                ))
            elif name == "file_delete":
                steps.append(PlanStep(
                    "edit_file",
                    {"path": arguments.get("path", ""), "mode": "delete"},
                    "删除文件",
                ))
            elif name == "run_shell":
                steps.append(PlanStep(
                    "run_shell",
                    {"command": arguments.get("command", "")},
                    "执行命令",
                ))
            elif name == "show_diff":
                steps.append(PlanStep("show_diff", {}, "查看差异"))

        return steps
