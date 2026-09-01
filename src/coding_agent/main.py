"""Command-line entry point.

Usage:
    python -m coding_agent "write a script that ..."
    python -m coding_agent --interactive
    python -m coding_agent --mock "create a fib.py and test it"   # offline test
    python -m coding_agent --resume logs/session-*.json "extend it"  # continue
    python -m coding_agent --list-sessions                        # show sessions

Environment (see .env.example):
    LLM_API_KEY      required for real runs
    LLM_BASE_URL     default https://api.deepseek.com/v1
    LLM_MODEL        default deepseek-chat
    WORKSPACE        default ./workspace
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import Agent, prepare_workspace
from .config import Config
from .sessions import list_sessions, load_session, save_session

# So `python main.py` also works when this file is run directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def _make_mock_llm() -> "LLMClient":
    """A scripted fake LLM so the whole loop can be exercised with no API key."""
    import json as _json
    from .llm import LLMClient

    calls = {"list_files": 0, "write_file": 0, "run_command": 0}

    def tool_msg(name, args):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "mock_call_1",
                "type": "function",
                "function": {"name": name, "arguments": _json.dumps(args)},
            }],
        }

    class FakeLLM(LLMClient):
        def __init__(self, cfg):
            self.cfg = cfg
            self.base_url = "mock"

        def chat(self, messages, tools):
            if calls["list_files"] == 0:
                calls["list_files"] += 1
                return tool_msg("list_files", {"path": "."})
            if calls["write_file"] == 0:
                calls["write_file"] += 1
                return tool_msg("write_file", {
                    "path": "fib.py",
                    "content": (
                        "def fib(n):\n"
                        "    a, b = 0, 1\n"
                        "    for _ in range(n):\n"
                        "        a, b = b, a + b\n"
                        "    return a\n\n"
                        "if __name__ == '__main__':\n"
                        "    print([fib(i) for i in range(10)])\n"
                    ),
                })
            if calls["run_command"] == 0:
                calls["run_command"] += 1
                return tool_msg("run_command", {"command": "python fib.py"})
            return {
                "role": "assistant",
                "content": (
                    "Task complete (mock). Created `fib.py` and ran it "
                    "successfully."
                ),
            }

    return FakeLLM


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coding_agent", description="A tiny coding agent.")
    parser.add_argument("task", nargs="*", help="the programming task to complete")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="run an interactive REPL session")
    parser.add_argument("-m", "--mock", action="store_true",
                        help="use a scripted fake LLM (no API key needed); for offline testing")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="do not stream step-by-step output")
    parser.add_argument("--resume", metavar="FILE",
                        help="continue from a saved session file")
    parser.add_argument("--list-sessions", action="store_true",
                        help="list saved conversation sessions")
    parser.add_argument("--logs-dir", metavar="DIR",
                        help="where sessions are saved (default: ./logs)")
    args = parser.parse_args(argv)

    cfg = Config.load().resolve(Path.cwd())
    if args.logs_dir:
        cfg.logs_dir = Path(args.logs_dir).resolve()
    prepare_workspace(cfg)

    if args.list_sessions:
        return _show_sessions(cfg.logs_dir)

    llm = None
    if args.mock:
        from .llm import LLMClient
        llm = _make_mock_llm()(cfg)  # an instance of FakeLLM

    agent = Agent(cfg, llm=llm)
    stream = not args.quiet

    # Restore a saved conversation, if requested, so the run continues from it.
    resumed = False
    if args.resume:
        try:
            data = load_session(args.resume)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[cannot resume '{args.resume}': {e}]", file=sys.stderr)
            return 1
        agent.load_history(data["history"])
        resumed = True
        print(f"[resumed {args.resume} — {len(data['history'])} messages]")

    try:
        if args.interactive:
            print(f"Interactive mode ({'resumed' if resumed else 'fresh'} "
                  "session). Type a task, or Ctrl+C / 'quit' to exit.")
            while True:
                try:
                    task = input("\n> ")
                except EOFError:
                    break
                task = task.strip()
                if not task:
                    continue
                if task.lower() in {"quit", "exit"}:
                    break
                print("─" * 60)
                if resumed:
                    answer = agent.continue_run(task, stream=stream)
                else:
                    answer = agent.run(task, stream=stream)
                saved = save_session(agent.context.history, task, answer,
                                     cfg.logs_dir)
                print(f"[session saved → {saved}]")
        else:
            task = " ".join(args.task).strip()
            if not task:
                parser.print_help()
                return 2
            if resumed:
                answer = agent.continue_run(task, stream=stream)
            else:
                answer = agent.run(task, stream=stream)
            saved = save_session(agent.context.history, task, answer, cfg.logs_dir)
            print(f"\n[session saved → {saved}]")
    except KeyboardInterrupt:
        print("\n[interrupted by user]")
        return 130
    return 0


def _show_sessions(logs_dir: Path) -> int:
    """Print every saved session: path, task, and a one-line answer preview."""
    sessions = list_sessions(logs_dir)
    if not sessions:
        print("(no sessions saved yet — run a task first)")
        return 0
    for path in sessions:
        try:
            data = load_session(path)
        except (OSError, json.JSONDecodeError) as e:
            print(f"{path}\n    (unreadable: {e})")
            continue
        task = (data.get("task") or "").strip().replace("\n", " ")[:80]
        answer = (data.get("final_answer") or "").strip().replace("\n", " ")[:80]
        print(f"{path}\n    task: {task}\n    answer: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
