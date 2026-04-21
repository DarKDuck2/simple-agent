from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from simple_agent.planner import HeuristicPlanner, Plan, PlanStep
from simple_agent.repo_context import RepoContext, build_repo_context
from simple_agent.repo_tools import RepoToolKit, ToolRegistry, ToolResult
from simple_agent.trace import ActionTrace, now_ms


MAX_AGENT_STEPS = 12


@dataclass
class AgentResult:
    ok: bool
    answer: str
    plan: Plan
    trace: ActionTrace


@dataclass
class RepoAgent:
    root: str | Path = "."
    max_steps: int = MAX_AGENT_STEPS
    dry_run: bool = False
    planner: HeuristicPlanner = field(default_factory=HeuristicPlanner)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.toolkit = RepoToolKit(self.root, dry_run=self.dry_run)
        self.tools: ToolRegistry = self.toolkit.build_registry()

    def run(self, task: str, *, trace_path: str | Path | None = None) -> AgentResult:
        trace = ActionTrace()
        repo_context = build_repo_context(self.root)
        plan = self.planner.plan(task, repo_context.compressed())
        trace.add(step=0, phase="plan", ok=True, output=plan.describe())

        queue = list(plan.steps)
        completed: list[tuple[PlanStep, ToolResult]] = []
        step_count = 0
        ok = True

        while queue and step_count < self.max_steps:
            step_count += 1
            step = queue.pop(0)
            start = now_ms()
            result = self.tools.execute(step.tool, step.params)
            elapsed = now_ms() - start
            completed.append((step, result))
            trace.add(
                step=step_count,
                phase="act",
                tool=step.tool,
                params=step.params,
                ok=result.ok,
                output=result.output,
                elapsed_ms=elapsed,
            )

            if not result.ok:
                follow_ups = self.planner.reflect(step, result.output)
                trace.add(
                    step=step_count,
                    phase="reflect",
                    ok=bool(follow_ups),
                    output=_reflection_text(step, result.output, follow_ups),
                )
                if follow_ups:
                    queue = follow_ups + queue
                else:
                    ok = False
                    break

        if queue and step_count >= self.max_steps:
            ok = False
            trace.add(
                step=step_count,
                phase="abort",
                ok=False,
                output=f"达到最大步数限制：{self.max_steps}",
            )

        answer = self._final_answer(task, repo_context, completed, ok)
        trace.add(step=step_count + 1, phase="final", ok=ok, output=answer)

        if trace_path:
            trace.write_jsonl(trace_path)

        return AgentResult(ok=ok, answer=answer, plan=plan, trace=trace)

    def _final_answer(
        self,
        task: str,
        repo_context: RepoContext,
        completed: list[tuple[PlanStep, ToolResult]],
        ok: bool,
    ) -> str:
        lines = [
            "RepoAgent 执行完成" if ok else "RepoAgent 执行失败",
            f"任务：{task}",
            f"仓库：{repo_context.root}",
            f"工具调用：{len(completed)} 次",
            "",
            "执行摘要：",
        ]
        for index, (step, result) in enumerate(completed, 1):
            status = "OK" if result.ok else "FAIL"
            first_line = result.output.splitlines()[0] if result.output else ""
            lines.append(f"{index}. [{status}] {step.tool} - {step.reason} - {first_line}")
        return "\n".join(lines)


def _reflection_text(step: PlanStep, output: str, follow_ups: list[PlanStep]) -> str:
    if not follow_ups:
        return f"{step.tool} 失败且没有安全回退动作，停止执行。\n{output}"
    return "根据失败输出追加后续动作：\n" + "\n".join(
        f"- {item.tool} {item.params}: {item.reason}" for item in follow_ups
    )


_default_agent: RepoAgent | None = None


def agent_run(user_query: str) -> str:
    global _default_agent
    if _default_agent is None:
        _default_agent = RepoAgent()
    return _default_agent.run(user_query).answer


def agent_reset() -> None:
    global _default_agent
    _default_agent = RepoAgent()


# Backward-compatible name for the original teaching demo imports.
Agent = RepoAgent
