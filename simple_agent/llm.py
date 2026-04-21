from __future__ import annotations

import json
import os
from typing import Protocol


class LLM(Protocol):
    def infer(self, messages: list[dict[str, str]]) -> str:
        """Return the next assistant message."""


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
