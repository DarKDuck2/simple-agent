from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


FUNCTION_CALL_RE = re.compile(r"<\|FunctionCall\|>(.*?)<\|End\|>", re.DOTALL)


@dataclass(frozen=True)
class FunctionCall:
    name: str
    params: dict[str, Any]


def parse_function_call(text: str) -> FunctionCall | None:
    match = FUNCTION_CALL_RE.search(text)
    if not match:
        return None

    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(f"FunctionCall JSON 解析失败：{exc}") from exc

    name = payload.get("name")
    params = payload.get("params", {})

    if not isinstance(name, str) or not name:
        raise ValueError("FunctionCall 缺少有效的 name")

    if not isinstance(params, dict):
        raise ValueError("FunctionCall params 必须是对象")

    return FunctionCall(name=name, params=params)

