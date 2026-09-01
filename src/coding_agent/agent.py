"""The agent reasoning loop.

Single loop, executed entirely locally:

    for step in range(max_iterations):
        msg = llm.chat(history, tools)          # 1. model proposes an action
        if msg has tool_calls:                   # 2. it wants tools
            execute each tool locally            # 3. run them on this machine
            append results, continue             # 4. feed results back
        else:                                    #    no tools -> done
            return msg["content"] as final answer

Termination conditions (any one ends the loop):
  * the model returns an answer with no tool calls;
  * the step budget `max_iterations` is exhausted;
  * the user interrupts (KeyboardInterrupt).

Error handling:
  * a single failed tool call does NOT kill the loop — its error string is
    handed back to the model so it can adapt;
  * an LLM transport error is retried by the client with backoff;
  * malformed tool arguments are reported back to the model as errors.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .context import Context
from .llm import LLMClient, LLMError
from .prompts import build_system_prompt
from .tools import ToolBox

# A `tool_calls` block the model emitted but we could not parse. Parsed below.
_ARG_RE = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}")


@dataclass
class StepRecord:
    """One loop iteration, used for logging and the final transcript."""
    step: int
    message: dict
    tool_results: list[dict] = field(default_factory=list)


class Agent:
    def __init__(self, cfg: Config, llm: LLMClient | None = None):
        self.cfg = cfg
        self.llm = llm or LLMClient(cfg)
        self.tools = ToolBox(cfg)
        self.context = Context(
            system=build_system_prompt(str(cfg.workspace)),
            char_budget=cfg.context_char_budget,
            keep_recent=cfg.context_keep_recent,
        )
        self.records: list[StepRecord] = []

    # ------------------------------------------------------------------ public
    def run(self, task: str, stream: bool = False) -> str:
        """Run the agent on `task` as a fresh conversation.

        Returns the final answer text. `stream` toggles a human-friendly
        printout of every step, which is what we show during the demo video.
        """
        self.context.reset()
        self.records.clear()
        self.context.append({"role": "user", "content": task})
        if stream:
            self._say(f"\n> Task: {task}\n")
        return self._loop(stream)

    def load_history(self, history: list[dict]) -> None:
        """Restore a saved conversation (see `sessions.load_session`)."""
        self.context.history = list(history)

    def continue_run(self, task: str | None = None, stream: bool = False) -> str:
        """Continue the conversation already in `self.context`.

        Used after `load_history` so a previous session can be resumed from its
        stopping point. Appends `task` as a new user message when one is given.
        """
        self.records.clear()
        if task:
            self.context.append({"role": "user", "content": task})
            if stream:
                self._say(f"\n> Task: {task}\n")
        return self._loop(stream)

    def _loop(self, stream: bool) -> str:
        """The reasoning/tool loop; shared by fresh runs and resumed ones."""
        for step in range(1, self.cfg.max_iterations + 1):
            try:
                message = self.llm.chat(self.context.messages(), self.tools.schemas())
            except LLMError as e:
                if stream:
                    self._say(f"\n[LLM error, giving up]: {e}\n", stream)
                raise

            tool_calls = message.get("tool_calls")
            record = StepRecord(step=step, message=message)

            if not tool_calls:
                # Model produced a plain answer -> loop terminates here.
                answer = (message.get("content") or "").strip()
                if stream:
                    self._say(f"\n[final answer]\n{answer}\n")
                self.context.append(message)
                self.records.append(record)
                return answer

            # Model wants tools -> execute locally, append results, continue.
            self.context.append(message)
            record.tool_results = self._execute_tool_calls(tool_calls, stream)
            for result in record.tool_results:
                self.context.append(result)
            self.records.append(record)

            if stream:
                self._say(f"\n[step {step}/{self.cfg.max_iterations} done — "
                          f"history ≈ {self.context.summary_stats()['approx_chars']} chars]\n")

        # Loop budget exhausted without a final answer.
        closing = ("I was stopped after reaching the maximum number of steps "
                   f"({self.cfg.max_iterations}) without a clear final answer.")
        if stream:
            self._say(f"\n[max steps reached]\n{closing}\n")
        return closing

    # --------------------------------------------------------------- internals
    def _execute_tool_calls(self, tool_calls: list, stream: bool) -> list[dict]:
        results: list[dict] = []
        for call in tool_calls:
            tool_call_id = call.get("id", "")
            fn = call.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "")

            args, parse_error = self._parse_args(name, raw_args)
            if parse_error:
                output = parse_error
            else:
                if stream:
                    arg_preview = json.dumps(args, ensure_ascii=False)
                    if len(arg_preview) > 400:
                        arg_preview = arg_preview[:400] + "…"
                    self._say(f"\n  → {name}({arg_preview})")
                try:
                    output = self.tools.execute(name, args)
                except Exception as e:  # defensive: tools must never crash the loop
                    output = f"tool crashed: {type(e).__name__}: {e}"
            if stream:
                self._say(self._clip_output(output))

            results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": output,
            })
        return results

    @staticmethod
    def _parse_args(name: str, raw: str) -> tuple[dict, str | None]:
        """Parse tool-call arguments. The model may emit JSON with leading
        text or trailing noise, so we first try strict JSON, then fall back to
        extracting the outermost JSON object."""
        raw = (raw or "").strip()
        if not raw:
            return {}, None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            match = _ARG_RE.search(raw)
            if not match:
                return {}, (f"could not parse arguments for {name}: {raw[:200]!r}")
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}, (f"could not parse arguments for {name}: {raw[:200]!r}")
        if not isinstance(parsed, dict):
            return {}, f"arguments for {name} are not a JSON object: {parsed!r}"
        return parsed, None

    @staticmethod
    def _clip_output(text: str, width: int = 600) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.splitlines()
        shown = []
        for line in lines:
            if len(line) > width:
                line = line[:width] + "…"
            shown.append("    | " + line)
        return "\n".join(shown)

    @staticmethod
    def _say(text: str, stream: bool = True) -> None:
        if stream:
            print(text, flush=True)


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_workspace(cfg: Config) -> None:
    _mkdir(cfg.workspace)
