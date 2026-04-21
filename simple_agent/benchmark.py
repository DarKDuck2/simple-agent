from __future__ import annotations

import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path

from simple_agent.agent import RepoAgent
from simple_agent.planner import HeuristicPlanner


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    task: str
    expected_tool: str
    expected_file: str | None = None
    expected_text: str | None = None


@dataclass(frozen=True)
class CaseResult:
    name: str
    ok: bool
    steps: int
    tool_success_rate: float


@dataclass(frozen=True)
class BenchmarkReport:
    results: list[CaseResult]

    def render(self) -> str:
        total = len(self.results)
        passed = sum(1 for item in self.results if item.ok)
        avg_steps = sum(item.steps for item in self.results) / total if total else 0
        avg_tool_success = sum(item.tool_success_rate for item in self.results) / total if total else 0
        lines = [
            "RepoAgent Benchmark",
            f"cases={total}",
            f"task_success_rate={passed / total:.2%}" if total else "task_success_rate=0.00%",
            f"test_pass_rate={passed / total:.2%}" if total else "test_pass_rate=0.00%",
            f"avg_steps={avg_steps:.2f}",
            f"tool_success_rate={avg_tool_success:.2%}",
            "",
            "Cases:",
        ]
        for item in self.results:
            status = "PASS" if item.ok else "FAIL"
            lines.append(f"- {status} {item.name}: steps={item.steps}, tool_success={item.tool_success_rate:.2%}")
        return "\n".join(lines)


def run_benchmark(cases: int = 22, trace_dir: str | None = None, planner=None) -> BenchmarkReport:
    selected = BENCHMARK_CASES[: max(1, min(cases, len(BENCHMARK_CASES)))]
    results: list[CaseResult] = []
    with tempfile.TemporaryDirectory(prefix="repoagent-bench-") as temp_dir:
        repo = Path(temp_dir)
        _seed_repo(repo)
        for index, case in enumerate(selected, 1):
            agent = RepoAgent(root=repo, max_steps=8, planner=planner)
            trace_path = Path(trace_dir) / f"{index:02d}-{case.name}.jsonl" if trace_dir else None
            result = agent.run(case.task, trace_path=trace_path)
            tool_events = [event for event in result.trace.events if event.phase == "act"]
            tool_ok = sum(1 for event in tool_events if event.ok)
            has_expected_tool = any(event.tool == case.expected_tool for event in tool_events)
            file_ok = True
            if case.expected_file and case.expected_text is not None:
                path = repo / case.expected_file
                file_ok = path.exists() and case.expected_text in path.read_text(encoding="utf-8")
            ok = result.ok and has_expected_tool and file_ok
            success_rate = tool_ok / len(tool_events) if tool_events else 0
            results.append(CaseResult(case.name, ok, len(tool_events), success_rate))
    return BenchmarkReport(results)


def _seed_repo(root: Path) -> None:
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n\n"
        "def label():\n    return 'alpha'\n",
        encoding="utf-8",
    )
    (root / "src" / "strings.py").write_text(
        "GREETING = 'hello'\n"
        "def greet(name):\n    return f'{GREETING}, {name}'\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Tiny Repo\n\nalpha project\n", encoding="utf-8")
    (root / "tests" / "test_math_utils.py").write_text(
        "import unittest\n"
        "from src.math_utils import add\n\n"
        "class MathUtilsTest(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(1, 2), 3)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=root, capture_output=True, text=True, check=False)
    subprocess.run(["git", "add", "."], cwd=root, capture_output=True, text=True, check=False)
    subprocess.run(
        ["git", "-c", "user.name=RepoAgent", "-c", "user.email=repoagent@example.local", "commit", "-m", "seed"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


BENCHMARK_CASES = [
    BenchmarkCase("tree-1", "查看仓库目录结构", "repo_tree"),
    BenchmarkCase("tree-2", "tree src layout", "repo_tree"),
    BenchmarkCase("search-alpha", "搜索 alpha", "search_code"),
    BenchmarkCase("search-greeting", "查找 GREETING", "search_code"),
    BenchmarkCase("search-add", "search add", "search_code"),
    BenchmarkCase("read-readme", "阅读 README.md", "read_file"),
    BenchmarkCase("read-math", "读取 src/math_utils.py", "read_file"),
    BenchmarkCase("read-tests", "read tests/test_math_utils.py", "read_file"),
    BenchmarkCase("replace-label", "把 src/math_utils.py 中 'alpha' 替换为 'beta'", "edit_file", "src/math_utils.py", "beta"),
    BenchmarkCase("replace-greeting", "把 src/strings.py 中 'hello' 替换为 'hi'", "edit_file", "src/strings.py", "hi"),
    BenchmarkCase("replace-title", "把 README.md 中 'Tiny Repo' 替换为 'RepoAgent Bench'", "edit_file", "README.md", "RepoAgent Bench"),
    BenchmarkCase("create-notes", "创建 NOTES.md 内容为 'benchmark notes'", "edit_file", "NOTES.md", "benchmark notes"),
    BenchmarkCase("create-config", "新增 config.txt 内容为 'enabled=true'", "edit_file", "config.txt", "enabled=true"),
    BenchmarkCase("test-pytest", "运行 pytest 测试", "run_shell"),
    BenchmarkCase("test-cn", "运行单测", "run_shell"),
    BenchmarkCase("diff-1", "展示 diff 差异", "show_diff"),
    BenchmarkCase("tree-diff", "查看目录并展示 diff", "repo_tree"),
    BenchmarkCase("search-read", "搜索 greet，然后查看文件 src/strings.py", "search_code"),
    BenchmarkCase("replace-and-test", "把 src/strings.py 中 'hi' 替换为 'hello'，然后运行测试", "edit_file", "src/strings.py", "hello"),
    BenchmarkCase("create-and-diff", "创建 TODO.md 内容为 'ship repoagent' 并展示 diff", "edit_file", "TODO.md", "ship repoagent"),
    BenchmarkCase("read-created", "阅读 TODO.md", "read_file"),
    BenchmarkCase("search-created", "搜索 ship repoagent", "search_code"),
]
