"""Local tools the agent can call.

Every file tool is confined to the configured workspace directory (sandbox).
Path arguments are validated so the agent cannot escape the workspace with
`../` tricks or absolute paths.  Commands run with the workspace as their
working directory.

Each tool returns a *string*, because that is exactly what gets appended to
the conversation as a `tool` role message.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .config import Config

# Sentinel used to signal "unknown tool" back to the model so it can recover.
UNKNOWN_TOOL = "unknown_tool"


def _safe_resolve(cfg: Config, raw: str) -> tuple[Path | None, str | None]:
    """Resolve `raw` to an absolute path inside the workspace.

    Returns (path, None) on success or (None, error_message) when the path
    is missing / escapes the sandbox.
    """
    ws = cfg.workspace
    p = Path(raw)
    if not p.is_absolute():
        p = ws / p
    p = p.resolve()
    try:
        p.relative_to(ws)
    except ValueError:
        return None, f"path '{raw}' is outside the workspace sandbox"
    return p, None


def _read(path: Path, max_chars: int) -> str:
    """Read a text file, truncating content that is too large."""
    data = path.read_text(encoding="utf-8", errors="replace")
    if len(data) > max_chars:
        cut = data[: max_chars // 2] + data[-(max_chars // 2):]
        return (f"[file too large: {len(data)} chars, showing first/last "
                f"{max_chars // 2} chars]\n") + cut
    return data


def _parse_skill_md(fallback_name: str, raw: str) -> tuple[str, str, str]:
    """Parse a skill file's simple frontmatter block.

    Format:
        ---
        name: python-testing
        description: one-line summary
        ---
        <instructions body>

    Returns (name, description, content) where `content` is the body with the
    frontmatter stripped (kept readable for the model). Falls back gracefully
    when the frontmatter is missing or malformed.
    """
    body = raw
    name = fallback_name
    description = "(no description)"
    if raw.lstrip().startswith("---"):
        # Split into (frontmatter, body) on the second '---' line.
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            fm = parts[1]
            body = parts[2].lstrip("\n")
            for line in fm.splitlines():
                line = line.strip()
                if ":" in line:
                    k, _, v = line.partition(":")
                    k, v = k.strip().lower(), v.strip()
                    if k == "name" and v:
                        name = v
                    elif k == "description" and v:
                        description = v
    return name, description, body.strip()


class ToolBox:
    """Collection of tools plus their JSON schemas for the LLM."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ------------------------------------------------------------------ schemas
    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Read a UTF-8 text file inside the workspace. "
                        "Use this to inspect existing code or data files."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string",
                                     "description": "path, relative to workspace or absolute"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": (
                        "Create or overwrite a UTF-8 text file inside the workspace "
                        "with the given content."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string",
                                     "description": "path, relative to workspace or absolute"},
                            "content": {"type": "string", "description": "full file content"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": (
                        "Replace the first exact occurrence of `old_text` with "
                        "`new_text` inside a file. Use for small, targeted edits "
                        "instead of rewriting the whole file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_text": {"type": "string",
                                         "description": "exact substring to replace"},
                            "new_text": {"type": "string",
                                         "description": "replacement substring"},
                        },
                        "required": ["path", "old_text", "new_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List the entries of a directory (recursive if "
                                   "`recursive` is true).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "directory path"},
                            "recursive": {"type": "boolean", "default": False},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_skills",
                    "description": (
                        "List the reusable skills available in the skills "
                        "directory, each with a one-line description. Call this "
                        "before load_skill to see what is available."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "load_skill",
                    "description": (
                        "Load a reusable skill by name and return its full "
                        "instructions. Once loaded, follow those instructions "
                        "for the rest of the current task."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string",
                                     "description": "skill name (from list_skills)"},
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": (
                        "Run a shell command with the workspace as working directory "
                        "and capture its output. Use to compile, run tests, install "
                        "packages, or execute any program. Avoid interactive commands."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "shell command to run"},
                        },
                        "required": ["command"],
                    },
                },
            },
        ]

    # ---------------------------------------------------------------- execution
    def execute(self, name: str, args: dict) -> str:
        handlers = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "list_files": self.list_files,
            "list_skills": self.list_skills,
            "load_skill": self.load_skill,
            "run_command": self.run_command,
        }
        handler = handlers.get(name)
        if handler is None:
            return json.dumps(
                {"error": f"unknown tool '{name}'", "known_tools": sorted(handlers)},
                ensure_ascii=False,
            )
        try:
            return handler(**args)
        except TypeError as e:
            return json.dumps(
                {"error": f"bad arguments for {name}: {e}", "arguments_received": args},
                ensure_ascii=False,
            )

    # ------------------------------------------------------------------- tools
    def read_file(self, path: str) -> str:
        p, err = _safe_resolve(self.cfg, path)
        if err:
            return err
        if not p.is_file():
            return f"error: '{path}' is not a file"
        try:
            return _read(p, self.cfg.max_tool_output_chars)
        except OSError as e:
            return f"error reading '{path}': {e}"

    def write_file(self, path: str, content: str) -> str:
        p, err = _safe_resolve(self.cfg, path)
        if err:
            return err
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            p.write_text(content, encoding="utf-8")
        except OSError as e:
            return f"error writing '{path}': {e}"
        return f"wrote {len(content)} chars to '{p.relative_to(self.cfg.workspace)}'"

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        p, err = _safe_resolve(self.cfg, path)
        if err:
            return err
        if not p.is_file():
            return f"error: '{path}' is not a file"
        if old_text == new_text:
            return "old_text and new_text are identical; nothing to do"
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"error reading '{path}': {e}"
        if old_text not in text:
            return "error: old_text not found in file (exact match required)"
        text = text.replace(old_text, new_text, 1)
        try:
            p.write_text(text, encoding="utf-8")
        except OSError as e:
            return f"error writing '{path}': {e}"
        return "edit applied successfully"

    def list_files(self, path: str, recursive: bool = False) -> str:
        p, err = _safe_resolve(self.cfg, path)
        if err:
            return err
        if not p.is_dir():
            return f"error: '{path}' is not a directory"
        try:
            if recursive:
                entries = sorted(str(x.relative_to(self.cfg.workspace))
                                 for x in p.rglob("*"))
            else:
                entries = sorted(str(x.relative_to(self.cfg.workspace))
                                 for x in p.iterdir())
        except OSError as e:
            return f"error listing '{path}': {e}"
        if not entries:
            return "(empty directory)"
        return "\n".join(entries)

    # ------------------------------------------------------------------ skills
    def list_skills(self) -> str:
        """Return "<name>: <description>" for every skill in the skills dir."""
        skills = self._scan_skills()
        if not skills:
            return "(no skills found)"
        lines = []
        for name, info in sorted(skills.items()):
            lines.append(f"{name}: {info['description']}")
        return "\n".join(lines)

    def load_skill(self, name: str) -> str:
        """Return the full SKILL.md content (frontmatter + instructions)."""
        skills = self._scan_skills()
        if name not in skills:
            known = ", ".join(sorted(skills)) or "(none)"
            return f"error: unknown skill '{name}'. available: {known}"
        return skills[name]["content"]

    def _scan_skills(self) -> dict[str, dict]:
        """Scan `<skills_dir>/*/SKILL.md`, parsing a tiny YAML-ish frontmatter."""
        root = self.cfg.skills_dir
        if not root.is_dir():
            return {}
        found: dict[str, dict] = {}
        for entry in sorted(root.iterdir()):
            skill_file = entry / "SKILL.md"
            if not entry.is_dir() or not skill_file.is_file():
                continue
            raw = skill_file.read_text(encoding="utf-8", errors="replace")
            name, description, content = _parse_skill_md(entry.name, raw)
            found[name] = {"description": description, "content": content}
        return found

    def run_command(self, command: str) -> str:
        # `run_command` is given freedom to reach outside the workspace on
        # purpose: compiling/installing often needs the system. The shell is
        # still launched inside the workspace cwd.
        if command.strip().startswith("!"):
            return "error: interactive commands are not allowed"
        # Inherit the interpreter that is running the agent, so `python` and
        # friends resolve to the same environment the agent itself uses even
        # when that environment is not on the machine-wide PATH.
        env = os.environ.copy()
        _self_bin = str(Path(sys.executable).resolve().parent)
        env["PATH"] = _self_bin + os.pathsep + env.get("PATH", "")
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.cfg.workspace),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=self.cfg.command_timeout,
            )
        except subprocess.TimeoutExpired:
            return (f"error: command timed out after "
                    f"{self.cfg.command_timeout}s: {command}")
        except OSError as e:
            return f"error running command: {e}"

        out = proc.stdout + (proc.stderr if proc.stderr else "")
        out = out.rstrip("\n")
        limit = self.cfg.max_tool_output_chars
        if len(out) > limit:
            out = out[: limit // 2] + "\n...[truncated]...\n" + out[-limit // 2:]
        status = f"[exit code {proc.returncode}]"
        if not out:
            out = "(no output)"
        return f"{status}\n{out}"
