# mini-coding-agent

一个轻量、低依赖的 **编程智能体（coding agent）**：通过与 OpenAI 兼容的 LLM 对话，并**在本地执行工具**来自主完成编程任务——读写文件、列目录、跑命令，直到任务完成。

本项目为研究生推免考核从零手写。**未使用任何 agent 框架或 SDK**（没有 LangChain、LlamaIndex、Agents SDK、CrewAI……）。唯一的第三方运行依赖是 `requests`（HTTP 客户端）。所有 agent 机制——上下文管理、工具定义与执行、模型输出解析、循环终止、错误处理——全部手写实现。

```
┌─────────────┐   messages   ┌──────────────┐   tool call JSON   ┌──────────────┐
│  your task  │ ───────────▶ │  Agent loop  │ ──────────────────▶ │  local tools  │
│  (CLI/REPL) │ ◀─────────── │  (Python)    │ ◀────────────────── │ (sandboxed)   │
└─────────────┘   final text  └──────────────┘    tool results     └──────────────┘
```

## 快速开始

```bash
pip install requests                 # 唯一的依赖

cp .env.example .env                 # 然后把密钥填进 .env
#   LLM_API_KEY=sk-...
#   LLM_BASE_URL=https://api.deepseek.com/v1
#   LLM_MODEL=deepseek-chat

python run.py "写一个 fib.py 并用测试验证它"
python run.py --interactive          # 交互式 REPL 模式
python run.py --mock "..."           # 离线运行（脚本化假模型，无需密钥）
python tests/test_agent.py           # 单元测试（仅标准库）

# 每次运行都会自动保存会话，可查看或从断点继续：
python run.py --list-sessions
python run.py --resume logs/session-*.json "再让它快一点"
```

## 技能（Skill）

**技能**让你把可复用的操作指引打包成小 Markdown 文件，agent 按需加载。一个技能就是一个带 `SKILL.md` 的目录：

```
skills/
  python-testing/SKILL.md
  code-review/SKILL.md
```

`SKILL.md` 格式（简单的 frontmatter + 操作步骤）：

```markdown
---
name: python-testing
description: 用标准库 unittest 编写并运行 Python 单元测试，验证其通过。
---

1. 把测试放在 `<模块名>_test.py` 中，只使用标准库 `unittest`。
2. 覆盖正常路径、边界情况和错误情况。
3. 用 `python -m unittest <模块名>_test -v` 运行。
...
```

agent 通过 `list_skills` 工具发现技能、用 `load_skill` 加载；加载后会在当前任务的剩余部分遵循其中的指引。想增加新技能，只要在 `skills/`（或自定义的 `SKILLS_DIR`）下新建一个目录即可，无需改代码。

## 会话记录（Sessions）

每次运行都会把完整对话历史连同任务与最终答案，保存为 `logs/` 下的 JSON 文件（`session-<时间戳>.json`）。这样运行过程事后可审计，也能**从断点继续**而不必重头再来：

```
$ python run.py "写一个 fib.py"                     # → logs/session-20260902-...json
$ python run.py --list-sessions                     # 查看已保存的会话
$ python run.py --resume logs/session-...json "扩展它"   # 从断点继续
```

续跑时会载入之前的完整历史，把新请求作为一条新的用户消息追加，然后继续推理循环。`--resume` 也可以配合 `--interactive` 使用。保存目录可通过 `--logs-dir` 或 `LOGS_DIR` 指定。

## 工作原理

agent 是单一确定的循环（[src/coding_agent/agent.py](src/coding_agent/agent.py)）：

1. 把用户任务放入对话历史。
2. 让模型继续；它要么发出 **工具调用**，要么给出纯文本回答。
3. 有工具调用 → 逐个解析并**在本地执行**，结果作为 `tool` 消息追加，继续循环。
4. 纯文本回答 → 循环终止，该文本即为结果。

循环的终止条件（满足其一即结束）：(a) 模型无工具调用的直接回答；(b) 达到 `MAX_ITERATIONS` 步数上限；(c) 用户 Ctrl+C 中断。

### 模块

| 文件 | 职责 |
|------|------|
| `config.py`   | 环境变量 / `.env` 配置（手写 dotenv 解析器） |
| `llm.py`      | 基于 `requests` 的 OpenAI 兼容 Chat-Completions 客户端：工具调用、重试退避 |
| `tools.py`    | `read_file`、`write_file`、`edit_file`、`list_files`、`run_command`、`list_skills`、`load_skill` + 工作区沙箱 |
| `context.py`  | 对话历史；基于字符预算的自动压缩 |
| `prompts.py`  | 系统提示词（agent 的行为契约） |
| `agent.py`    | 推理循环、参数解析、循环终止、错误处理 |
| `sessions.py` | 会话的保存 / 读取 / 列出（`--resume` 续跑的基础） |
| `main.py`     | CLI（任务模式、交互 REPL、mock 模式） |

### 值得了解的设计决策

- **沙箱。** 每个文件工具都会把路径解析后校验，拒绝任何逃出配置的 `WORKSPACE` 的路径，挡住 `../` 穿越和绝对路径。`run_command` 是**刻意**放开到系统范围的（编译、安装需要它），但始终以 workspace 作为工作目录执行。
- **上下文管理。** 用*字符数*（一种廉价、零依赖的 token 近似）估算。当历史超过 `CONTEXT_CHAR_BUDGET`，最早的每条消息被替换成一行摘要，最近 `CONTEXT_KEEP_RECENT` 条消息原样保留，保证模型始终能看到它正在推理的那条工具结果。
- **健壮的解析。** 模型的工具参数先按严格 JSON 解析；失败则从周围噪声中提取最外层 JSON 对象；两者都失败时把错误作为工具结果回传给模型，而不是让循环崩溃。
- **错误处理。** 单个工具调用失败绝不会终止循环——错误字符串喂回模型让它自己调整。LLM 传输错误按指数退避重试。整次运行可用 Ctrl+C 中断。
- **工具执行环境。** 沙箱 shell 继承运行 agent 的解释器，因此即使该环境不在系统 PATH 上，`python …` 也能正确解析。

## 环境变量

| 变量 | 默认值 | 含义 |
|------|--------|------|
| `LLM_API_KEY` | — | API 密钥（绝不入库） |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容接口地址 |
| `LLM_MODEL` | `deepseek-chat` | 模型名称 |
| `WORKSPACE` | `./workspace` | agent 工作的沙箱根目录 |
| `SKILLS_DIR` | `./skills` | 可复用技能的存放目录 |
| `LOGS_DIR` | `./logs` | 会话记录的保存目录 |
| `MAX_ITERATIONS` | `40` | 推理循环步数上限 |
| `MAX_TOOL_OUTPUT_CHARS` | `8000` | 超大工具结果的截断阈值 |
| `CONTEXT_CHAR_BUDGET` | `36000` | 触发历史压缩的近似字符数 |
| `CONTEXT_KEEP_RECENT` | `8` | 压缩后原样保留的最新消息条数 |
| `COMMAND_TIMEOUT` | `120` | `run_command` 的超时秒数 |
| `REQUEST_TIMEOUT` | `300` | 单次 LLM 请求的超时秒数 |
| `MAX_RETRIES` | `3` | LLM 请求遇到瞬时错误的尝试次数 |

## 目录结构

```
src/coding_agent/   agent 核心包
tests/              单元测试（标准库 unittest）
skills/             可复用技能（<技能名>/SKILL.md）
workspace/          agent 工作的沙箱（git 忽略）
logs/               保存的会话记录（git 忽略）
run.py              便捷启动入口
```
