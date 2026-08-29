# mini-coding-agent

A small, dependency-light **coding agent** that autonomously completes
programming tasks by talking to an OpenAI-compatible LLM and using **local**
tools: it reads and writes files, lists directories, and runs shell commands —
until the task is done.

Built from scratch for a graduate-school admission assessment. **No agent
framework or SDK is used** (no LangChain, no LlamaIndex, no Agents SDK,
no CrewAI…). The only third-party runtime dependency is `requests`, an HTTP
client. All agent logic — context management, tool definitions & execution,
model-output parsing, loop termination, error handling — is hand-written.

```
┌─────────────┐   messages   ┌──────────────┐   tool call JSON   ┌──────────────┐
│  your task  │ ───────────▶ │  Agent loop  │ ──────────────────▶ │  local tools  │
│  (CLI/REPL) │ ◀─────────── │  (Python)    │ ◀────────────────── │ (sandboxed)   │
└─────────────┘   final text  └──────────────┘    tool results     └──────────────┘
```

## Quick start

```bash
pip install requests                 # the only dependency

cp .env.example .env                 # then put your key in .env
#   LLM_API_KEY=sk-...
#   LLM_BASE_URL=https://api.deepseek.com/v1
#   LLM_MODEL=deepseek-chat

python run.py "create a fib.py and run it with a test"
python run.py --interactive          # REPL mode
python run.py --mock "..."           # offline run with a scripted fake LLM
python tests/test_agent.py           # unit tests (stdlib only)
```

## How it works

The agent is a single deterministic loop (`src/coding_agent/agent.py`):

1. The user's task is placed into the conversation history.
2. The model is asked to continue; it either emits **tool calls** or a plain
   answer.
3. If tool calls → each is parsed and executed **locally**; results are
   appended as `tool` messages and the loop continues.
4. If a plain answer → the loop terminates and that text is the result.

Termination is triggered by (a) the model answering without tool calls,
(b) hitting the `MAX_ITERATIONS` step budget, or (c) a user interrupt.

### Modules

| File | Responsibility |
|------|----------------|
| `config.py`   | Env / `.env` configuration (hand-written dotenv loader) |
| `llm.py`      | OpenAI-compatible chat-completions client over `requests`, tool calling, retry/backoff |
| `tools.py`    | `read_file`, `write_file`, `edit_file`, `list_files`, `run_command` + workspace sandbox |
| `context.py`  | Conversation history; character-budget based compaction |
| `prompts.py`  | System prompt (the agent's behavioural contract) |
| `agent.py`    | The reasoning loop, argument parsing, termination, error handling |
| `main.py`     | CLI (task mode, interactive REPL, mock mode) |

### Design decisions worth knowing

- **Sandbox.** Every file tool resolves its path and rejects anything outside
  the configured `WORKSPACE`, blocking `../` traversal and absolute paths.
  `run_command` deliberately runs system-wide (compiling/installing needs
  it), but always with the workspace as its working directory.
- **Context management.** We count *characters* (a cheap, dependency-free
  proxy for tokens). When history exceeds `CONTEXT_CHAR_BUDGET`, the oldest
  messages are replaced by a one-line summary each and the most recent
  `CONTEXT_KEEP_RECENT` messages stay verbatim, so the model never loses the
  tool result it is currently reasoning about.
- **Resilient parsing.** Model tool arguments are parsed as strict JSON first;
  if that fails we extract the outermost JSON object from surrounding noise;
  if both fail, the error is returned to the model as a tool result instead of
  crashing the loop.
- **Error handling.** A failed tool call never kills the loop — its error
  string is fed back to the model so it can adapt. LLM transport errors are
  retried with exponential backoff. The whole run can be interrupted with
  Ctrl+C.
- **Tool execution environment.** The sandbox shell inherits the interpreter
  that runs the agent, so `python …` resolves correctly even when the
  environment is not on the machine-wide PATH.

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_API_KEY` | — | API key (never commit it) |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `deepseek-chat` | Model name |
| `WORKSPACE` | `./workspace` | Sandbox root the agent works in |
| `MAX_ITERATIONS` | `40` | Loop step budget |
| `MAX_TOOL_OUTPUT_CHARS` | `8000` | Truncation for huge tool results |
| `CONTEXT_CHAR_BUDGET` | `36000` | Approx. history size before compaction |
| `CONTEXT_KEEP_RECENT` | `8` | Messages kept verbatim after compaction |
| `COMMAND_TIMEOUT` | `120` | Seconds before `run_command` gives up |
| `REQUEST_TIMEOUT` | `300` | Seconds per LLM request |
| `MAX_RETRIES` | `3` | LLM retry count on transient errors |

## Layout

```
src/coding_agent/   the agent package
tests/              unit tests (stdlib unittest)
workspace/          sandbox where the agent works (git-ignored)
run.py              convenience launcher
```
