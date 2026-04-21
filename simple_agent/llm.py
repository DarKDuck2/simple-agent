from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from simple_agent.planner import Plan, PlanStep


class LLM(Protocol):
    def infer(self, messages: list[dict[str, str]]) -> str:
        """Return the next assistant message."""


class LLMPlanner(Protocol):
    def generate_plan(self, task: str, repo_context: str, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Plan:
        """Generate a plan for the given task using available tools."""

    def reflect(self, failed_step: PlanStep, output: str, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[PlanStep]:
        """Reflect on a failed step and generate follow-up actions."""


@dataclass(frozen=True)
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0


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

            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Claude API 调用失败（已重试 {self.max_retries} 次）：{exc}") from exc
                time.sleep(self.retry_delay * (attempt + 1))

        return Plan(goal=task, steps=[])

    def reflect(self, failed_step: PlanStep, output: str, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[PlanStep]:
        client = self._client()

        reflect_message = (
            f"上一步操作失败了：\n"
            f"工具：{failed_step.tool}\n"
            f"参数：{json.dumps(failed_step.params, ensure_ascii=False)}\n"
            f"失败输出：{output[:2000]}\n\n"
            f"请分析失败原因，并生成后续步骤来修复问题或采用替代方案。"
            f"如果无法修复，请直接返回空操作。"
        )

        messages = history + [{"role": "user", "content": reflect_message}]

        for attempt in range(self.max_retries):
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self._system_prompt(),
                    messages=messages,  # type: ignore[arg-type]
                    tools=tools,  # type: ignore[arg-type]
                    tool_choice={"type": "auto"},
                )

                if response.usage:
                    self._usage.append(UsageStats(
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                    ))

                return self._parse_tool_uses(response.content)

            except Exception as exc:
                if attempt == self.max_retries - 1:
                    return []
                time.sleep(self.retry_delay * (attempt + 1))

        return []

    def total_usage(self) -> UsageStats:
        if not self._usage:
            return UsageStats()
        return UsageStats(
            input_tokens=sum(u.input_tokens for u in self._usage),
            output_tokens=sum(u.output_tokens for u in self._usage),
        )

    def _build_messages(self, task: str, repo_context: str, history: list[dict[str, Any]]) -> list[dict[str, str]]:
        content = (
            f"## 任务\n{task}\n\n"
            f"## 仓库上下文\n{repo_context}\n\n"
            f"请生成完成此任务所需的工具调用步骤。"
        )
        messages = [{"role": "user", "content": content}]
        for msg in history:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            messages.append({"role": role, "content": text})
        return messages

    def _parse_tool_uses(self, content_blocks: list[Any]) -> list[PlanStep]:
        steps: list[PlanStep] = []
        for block in content_blocks:
            if getattr(block, "type", None) == "tool_use":
                name = getattr(block, "name", "")
                params = getattr(block, "input", {})
                if name:
                    steps.append(PlanStep(
                        tool=name,
                        params=dict(params) if params else {},
                        reason=f"LLM 选择工具 {name}",
                    ))
        return steps


@dataclass
class OpenAIPlanner:
    """OpenAI-compatible planner using function calling protocol."""

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

    def generate_plan(self, task: str, repo_context: str, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Plan:
        client = self._client()
        messages = self._build_messages(task, repo_context, history)
        openai_tools = self._convert_tools(tools)

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto" if openai_tools else None,
                    temperature=0,
                )

                if response.usage:
                    self._usage.append(UsageStats(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                    ))

                message = response.choices[0].message
                steps = self._parse_tool_calls(message)
                if steps:
                    return Plan(goal=task, steps=steps)

                text = message.content or ""
                return Plan(goal=task, steps=[
                    PlanStep("repo_tree", {"depth": 3}, f"无法解析工具调用，先查看仓库结构。LLM 回复：{text[:200]}")
                ])

            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"OpenAI API 调用失败（已重试 {self.max_retries} 次）：{exc}") from exc
                time.sleep(self.retry_delay * (attempt + 1))

        return Plan(goal=task, steps=[])

    def reflect(self, failed_step: PlanStep, output: str, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[PlanStep]:
        client = self._client()

        reflect_message = (
            f"上一步操作失败了：\n"
            f"工具：{failed_step.tool}\n"
            f"参数：{json.dumps(failed_step.params, ensure_ascii=False)}\n"
            f"失败输出：{output[:2000]}\n\n"
            f"请分析失败原因，并生成后续步骤来修复问题或采用替代方案。"
        )

        messages = history + [{"role": "user", "content": reflect_message}]
        openai_tools = self._convert_tools(tools)

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto" if openai_tools else None,
                    temperature=0,
                )

                if response.usage:
                    self._usage.append(UsageStats(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                    ))

                steps = self._parse_tool_calls(response.choices[0].message)
                return steps

            except Exception as exc:
                if attempt == self.max_retries - 1:
                    return []
                time.sleep(self.retry_delay * (attempt + 1))

        return []

    def total_usage(self) -> UsageStats:
        if not self._usage:
            return UsageStats()
        return UsageStats(
            input_tokens=sum(u.input_tokens for u in self._usage),
            output_tokens=sum(u.output_tokens for u in self._usage),
        )

    def _build_messages(self, task: str, repo_context: str, history: list[dict[str, Any]]) -> list[dict[str, str]]:
        system = (
            "你是一个面向代码仓库的 AI 编程助手。"
            "分析用户任务，选择合适的工具来完成代码理解、搜索、阅读和编辑等操作。"
        )
        content = (
            f"## 任务\n{task}\n\n"
            f"## 仓库上下文\n{repo_context}\n\n"
            f"请生成完成此任务所需的工具调用步骤。"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        for msg in history:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            messages.append({"role": role, "content": text})
        return messages

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic-style tools to OpenAI function calling format."""
        result: list[dict[str, Any]] = []
        for tool in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        return result

    def _parse_tool_calls(self, message: Any) -> list[PlanStep]:
        steps: list[PlanStep] = []
        for call in getattr(message, "tool_calls", []) or []:
            name = getattr(getattr(call, "function", None), "name", "")
            args_str = getattr(getattr(call, "function", None), "arguments", "{}")
            try:
                params = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                params = {}
            if name:
                steps.append(PlanStep(
                    tool=name,
                    params=params,
                    reason=f"LLM 选择工具 {name}",
                ))
        return steps


class MockLLM:
    """A deterministic LLM stub for learning the agent loop without an API key."""

    def infer(self, messages: list[dict[str, str]]) -> str:
        latest = messages[-1]

        if latest["role"] == "tool":
            return f"工具返回结果：{latest['content']}\n\n基于这个结果，我的最终回答如上。"

        user_text = messages[-1]["content"]
        lowered = user_text.lower()

        if any(word in user_text for word in ["运行代码", "执行代码", "code"]):
            code = user_text.replace("运行代码", "").replace("执行代码", "").strip()
            return _function_call("code_run", {"code": code})

        if any(word in user_text for word in ["搜索", "查找", "查询"]) or "search" in lowered:
            query = user_text.replace("搜索一下", "").replace("搜索", "").strip()
            return _function_call("search", {"query": query})

        if any(word in user_text for word in ["算", "计算", "+", "-", "*", "/", "(", ")"]):
            expression = (
                user_text.replace("帮我算一下", "")
                .replace("计算", "")
                .replace("算一下", "")
                .strip()
            )
            return _function_call("calculator", {"expression": expression})

        return f"这是一个无需调用工具的问题。你问的是：{user_text}"


class OpenAIChatLLM:
    """Minimal OpenAI adapter. Prefer official tool calling in real projects."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def infer(self, messages: list[dict[str, str]]) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("请先安装 openai：pip install openai") from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("请先设置环境变量 OPENAI_API_KEY")

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=_to_openai_messages(messages),  # type: ignore[arg-type]
            temperature=0,
        )
        return response.choices[0].message.content or ""


def _function_call(name: str, params: dict[str, str]) -> str:
    payload = json.dumps({"name": name, "params": params}, ensure_ascii=False)
    return f"<|FunctionCall|>{payload}<|End|>"


def _to_openai_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": f"工具返回结果：\n{message['content']}",
                }
            )
        else:
            converted.append(message)
    return converted
