from __future__ import annotations

import argparse
import sys
from pathlib import Path

from simple_agent.agent import RepoAgent
from simple_agent.benchmark import run_benchmark


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] not in {"run", "benchmark", "-h", "--help"}:
        sys.argv.insert(1, "run")

    parser = argparse.ArgumentParser(description="RepoAgent: a tiny LLM-style code agent for local repositories.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run RepoAgent on a local repository task.")
    run_parser.add_argument("task", help="Natural language task for the agent.")
    run_parser.add_argument("--repo", default=".", help="Repository root. Defaults to current directory.")
    run_parser.add_argument("--trace", help="Write action trace as JSONL.")
    run_parser.add_argument("--dry-run", action="store_true", help="Plan and simulate edits/commands without side effects.")
    run_parser.add_argument("--max-steps", type=int, default=12, help="Maximum tool steps before aborting.")
    run_parser.add_argument("--show-trace", action="store_true", help="Print the action trace after the answer.")

    bench_parser = subparsers.add_parser("benchmark", help="Run the built-in 20+ task benchmark.")
    bench_parser.add_argument("--cases", type=int, default=22, help="Number of benchmark cases to run.")
    bench_parser.add_argument("--trace-dir", help="Directory for per-case JSONL traces.")

    args = parser.parse_args()

    if args.command == "benchmark":
        report = run_benchmark(cases=args.cases, trace_dir=args.trace_dir)
        print(report.render())
        return

    if args.command is None:
        parser.print_help()
        return

    agent = RepoAgent(root=Path(args.repo), dry_run=args.dry_run, max_steps=args.max_steps)
    result = agent.run(args.task, trace_path=args.trace)
    print(result.answer)
    if args.trace:
        print(f"\naction trace 已写入：{args.trace}")
    if args.show_trace:
        print("\nAction Trace:")
        print(result.trace.render())


if __name__ == "__main__":
    main()
