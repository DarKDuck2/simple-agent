from __future__ import annotations

import argparse
import sys
from pathlib import Path

from simple_agent.agent import RepoAgent
from simple_agent.benchmark import run_benchmark
from simple_agent.planner import HeuristicPlanner


def main() -> None:
    # Compatibility shorthand: "python main.py 'task'" -> "python main.py run 'task'"
    if len(sys.argv) > 1 and sys.argv[1] not in {"run", "benchmark", "interactive", "-h", "--help"}:
        sys.argv.insert(1, "run")

    parser = argparse.ArgumentParser(description="RepoAgent: an AI-powered code agent for local repositories.")
    subparsers = parser.add_subparsers(dest="command")

    # run command
    run_parser = subparsers.add_parser("run", help="Run RepoAgent on a local repository task.")
    run_parser.add_argument("task", help="Natural language task for the agent.")
    run_parser.add_argument("--repo", default=".", help="Repository root. Defaults to current directory.")
    run_parser.add_argument("--trace", help="Write action trace as JSONL.")
    run_parser.add_argument("--dry-run", action="store_true", help="Plan and simulate edits/commands without side effects.")
    run_parser.add_argument("--max-steps", type=int, default=12, help="Maximum tool steps before aborting.")
    run_parser.add_argument("--show-trace", action="store_true", help="Print the action trace after the answer.")
    run_parser.add_argument("--model", help="LLM model to use (e.g., claude-sonnet-4-6, gpt-4o).")
    run_parser.add_argument("--backend", choices=["claude", "openai"], help="LLM backend to use.")
    run_parser.add_argument("--auto-confirm", action="store_true", help="Auto-approve all dangerous operations.")
    run_parser.add_argument("--no-llm", action="store_true", help="Force use HeuristicPlanner (no LLM API calls).")

    # interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive chat mode.")
    interactive_parser.add_argument("--repo", default=".", help="Repository root.")
    interactive_parser.add_argument("--dry-run", action="store_true", help="Simulate edits/commands without side effects.")
    interactive_parser.add_argument("--max-steps", type=int, default=12, help="Maximum tool steps per task.")
    interactive_parser.add_argument("--model", help="LLM model to use.")
    interactive_parser.add_argument("--backend", choices=["claude", "openai"], help="LLM backend to use.")
    interactive_parser.add_argument("--auto-confirm", action="store_true", help="Auto-approve all dangerous operations.")
    interactive_parser.add_argument("--no-llm", action="store_true", help="Force use HeuristicPlanner.")

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run the built-in task benchmark.")
    bench_parser.add_argument("--cases", type=int, default=22, help="Number of benchmark cases to run.")
    bench_parser.add_argument("--trace-dir", help="Directory for per-case JSONL traces.")
    bench_parser.add_argument("--no-llm", action="store_true", help="Force use HeuristicPlanner for benchmark.")

    args = parser.parse_args()

    if args.command == "benchmark":
        planner = HeuristicPlanner() if args.no_llm else None
        report = run_benchmark(cases=args.cases, trace_dir=args.trace_dir, planner=planner)
        print(report.render())
        return

    if args.command == "interactive":
        _run_interactive(args)
        return

    if args.command == "run" or args.command is None:
        if args.command is None:
            parser.print_help()
            return

        planner = _build_planner(args)
        agent = RepoAgent(
            root=Path(args.repo),
            dry_run=args.dry_run,
            max_steps=args.max_steps,
            planner=planner,
            auto_confirm=args.auto_confirm,
        )
        result = agent.run(args.task, trace_path=args.trace)
        print(result.answer)
        if args.trace:
            print(f"\naction trace 已写入：{args.trace}")
        if args.show_trace:
            print("\nAction Trace:")
            print(result.trace.render())
        return


def _build_planner(args: argparse.Namespace):
    if args.no_llm:
        return HeuristicPlanner()
    try:
        from simple_agent.planner import LLMPlanner
        kwargs: dict = {}
        if args.backend:
            kwargs["backend"] = args.backend
        if args.model:
            kwargs["model"] = args.model
        return LLMPlanner(**kwargs)
    except Exception:
        print("[警告] LLM Planner 初始化失败，回退到 HeuristicPlanner")
        return HeuristicPlanner()


def _run_interactive(args: argparse.Namespace) -> None:
    planner = _build_planner(args)
    agent = RepoAgent(
        root=Path(args.repo),
        dry_run=args.dry_run,
        max_steps=args.max_steps,
        planner=planner,
        auto_confirm=args.auto_confirm,
    )

    print(f"RepoAgent 交互模式（输入 exit 或 quit 退出）")
    print(f"仓库: {agent.root}")
    if planner and hasattr(planner, "using_llm") and planner.using_llm:
        print("模式: LLM 规划")
    else:
        print("模式: 启发式规划（无 LLM）")
    print()

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        print("\n" + "=" * 50)
        result = agent.run(user_input)
        print(result.answer)
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
