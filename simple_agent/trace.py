from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    step: int
    phase: str
    tool: str | None
    params: dict[str, Any] | None
    ok: bool
    output: str
    elapsed_ms: int


class ActionTrace:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def add(
        self,
        *,
        step: int,
        phase: str,
        ok: bool,
        output: str,
        elapsed_ms: int = 0,
        tool: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            TraceEvent(
                step=step,
                phase=phase,
                tool=tool,
                params=params,
                ok=ok,
                output=_truncate(output),
                elapsed_ms=elapsed_ms,
            )
        )

    def write_jsonl(self, path: str | Path) -> None:
        trace_path = Path(path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def render(self) -> str:
        lines: list[str] = []
        for event in self.events:
            name = event.tool or event.phase
            status = "ok" if event.ok else "failed"
            lines.append(f"[{event.step}] {event.phase}:{name} {status} {event.elapsed_ms}ms")
            if event.output:
                lines.append(event.output)
        return "\n".join(lines)


def now_ms() -> int:
    return int(time.time() * 1000)


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... truncated {len(text) - limit} chars"
