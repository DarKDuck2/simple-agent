# RepoAgent

RepoAgent 是一个面向本地代码仓库的轻量 Code Agent 原型，参考 Claude Code 的工作方式，形成：

```text
需求输入 -> 仓库感知 -> 计划生成 -> 工具执行 -> 测试反馈 -> 反思回退 -> action trace
```

项目默认使用确定性启发式 Planner，不依赖 API key，便于演示仓库理解、工具调用、安全控制和评测闭环。后续可以把 `HeuristicPlanner` 替换为真实 LLM planner。

## CLI

```bash
python3 main.py run "查看仓库目录结构" --repo .
python3 main.py run "把 README.md 中 'RepoAgent' 替换为 'RepoAgent CLI' 并展示 diff" --repo . --dry-run --show-trace
python3 main.py run "运行单测" --repo . --trace traces/run.jsonl
python3 main.py benchmark --cases 22
```

兼容简写：

```bash
python3 main.py "查看仓库目录结构"
```

## 6 个核心工具

RepoAgent 只暴露 6 个仓库工具，便于控制行为边界：

1. `repo_tree`: 解析目录树，快速建立仓库结构认知。
2. `search_code`: 在仓库文本文件中做关键词检索。
3. `read_file`: 读取目标文件，可限制起始行和行数。
4. `edit_file`: 支持 overwrite、append、replace 三种文件修改。
5. `run_shell`: 在仓库根目录执行白名单命令，用于测试验证等动作。
6. `show_diff`: 展示当前仓库的 unified diff。

## 仓库感知模块

`simple_agent/repo_context.py` 会：

- 解析目录树，忽略 `.git`、`node_modules`、缓存目录等噪声。
- 收集文件级摘要，包括路径、行数、大小和首行/函数类定义摘要。
- 输出压缩后的上下文，供 planner 做多文件任务召回。

## 计划执行闭环

`simple_agent/agent.py` 的主循环包含：

1. `plan`: 基于任务和仓库摘要生成工具步骤。
2. `act`: 按步骤调用工具。
3. `reflect`: 根据失败输出追加安全回退动作。
4. `abort`: 达到最大步数或不可恢复失败时停止。
5. `final`: 汇总执行结果。

典型例子：当 `pytest -q` 不可用时，Agent 会根据错误输出回退到：

```bash
python -m unittest discover -s tests
```

## 安全控制

`run_shell` 有基础安全边界：

- 命令白名单：`python`、`python3`、`pytest`、`ruff`、`mypy`、`npm`、`pnpm`、`yarn`、`go`、`cargo`、`git`。
- 阻断高风险命令：`rm`、`sudo`、`chmod`、`chown`、`curl`、`wget`、`scp`、`ssh`、`mkfs`、`dd`。
- 支持 `--dry-run`，模拟编辑和命令执行。
- 支持 `--max-steps`，避免无限循环。
- 工具失败且无反思回退时自动中止。

## Action Trace

所有阶段都会记录为 JSONL：

```bash
python3 main.py run "运行单测" --trace traces/test.jsonl
```

每行包含：

- `step`
- `phase`
- `tool`
- `params`
- `ok`
- `output`
- `elapsed_ms`

也可以使用 `--show-trace` 在终端直接查看。

## 小 Benchmark

内置 22 个任务，覆盖：

- 仓库目录理解
- 代码搜索
- 文件读取
- 文件替换
- 文件创建
- 测试验证
- diff 展示
- 简单多步任务

运行：

```bash
python3 main.py benchmark --cases 22
```

输出指标：

- `task_success_rate`
- `test_pass_rate`
- `avg_steps`
- `tool_success_rate`

在本地验证中，22 个任务全部通过；其中 pytest 不存在时会回退到 unittest，因此工具级成功率可能低于任务成功率。

## 目录结构

```text
simple-agent/
  main.py
  simple_agent/
    agent.py          # RepoAgent 主循环
    benchmark.py      # 20+ 代码任务评测集
    planner.py        # 启发式计划与反思
    repo_context.py   # 目录树解析与文件级上下文压缩
    repo_tools.py     # 6 个核心工具与安全控制
    trace.py          # action trace JSONL
    llm.py            # 保留的 LLM 示例适配器
    parser.py         # 保留的 FunctionCall 文本协议解析
    tools.py          # 保留的教学 demo 工具
```
