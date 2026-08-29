"""Conversation history and context management.

The agent loop appends every model message and tool result to `history`.
Because context windows are finite, `Context` tracks the approximate size of
the conversation and compacts it once it grows too large.

Compaction strategy (kept deliberately simple and explainable):
  1. the system prompt is always kept verbatim;
  2. the most recent `keep_recent` messages are kept verbatim, so the model
     still sees the last tool output it is reasoning about;
  3. every older message is replaced by a one-line summary of what happened
     (we do NOT ask the LLM to summarize — cheap, deterministic, offline).

The size estimate is a character count, which is a good-enough proxy for
token count for most OpenAI-compatible models and requires no tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _size_chars(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += len(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            args = (tc.get("function") or {}).get("arguments") or ""
            total += len(args)
    return total


def _one_line(m: dict) -> str:
    """Condense a message into a single summary line."""
    role = m["role"]
    content = (m.get("content") or "").strip().replace("\n", " ")[:120]
    if m.get("tool_calls"):
        names = [tc.get("function", {}).get("name", "?") for tc in m["tool_calls"]]
        return f"[assistant called {', '.join(names)}]"
    if role == "tool":
        return f"[tool {m.get('name', '?')} returned: {content[:120]}]"
    return f"[{role}: {content}]"


@dataclass
class Context:
    system: str
    char_budget: int = 36000
    keep_recent: int = 8
    history: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        self.history = []

    def append(self, message: dict) -> None:
        self.history.append(message)
        self._compact_if_needed()

    def messages(self) -> list[dict]:
        return [{"role": "system", "content": self.system}, *self.history]

    def summary_stats(self) -> dict:
        return {"messages": len(self.history), "approx_chars": _size_chars(self.history)}

    # ------------------------------------------------------------ compaction
    def _compact_if_needed(self) -> None:
        if _size_chars(self.history) <= self.char_budget:
            return
        keep = self.history[-self.keep_recent:] if self.keep_recent > 0 else []
        keep_roles = {m["role"] for m in keep}
        prefix_len = len(self.history) - len(keep)

        # Drop the *final* partial turn so we don't cut a tool result
        # away from the assistant message that requested it.
        if keep and keep_roles == {"tool"}:
            keep = [m for m in self.history if m.get("role") != "tool"][-1:] + keep

        if prefix_len <= 0:
            return

        summaries = [_one_line(m) for m in self.history[:prefix_len]]
        condensed = {
            "role": "system",
            "content": (
                "[Earlier conversation compressed by the agent. It was:\n"
                + "\n".join("  - " + s for s in summaries)
                + "\n]"
            ),
        }
        self.history = [condensed, *keep]
