"""Configuration: load all runtime settings from environment variables.

No secrets are ever stored in the repo. Credentials come from either:
  * real environment variables, or
  * a local `.env` file (git-ignored).

The `.env` file is parsed here with a small hand-written loader so the project
does not depend on a third-party `python-dotenv` package.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: str | Path) -> None:
    """Minimal `.env` loader: KEY=VALUE per line, '#' comments, quotes stripped."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Config:
    # --- LLM endpoint -------------------------------------------------------
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"

    # --- Agent behaviour ----------------------------------------------------
    max_iterations: int = 40            # hard cap on reasoning/tool loop steps
    max_tool_output_chars: int = 8000   # truncate huge tool results
    context_char_budget: int = 36000    # approx chars before history compaction
    context_keep_recent: int = 8        # messages kept verbatim after compaction
    command_timeout: int = 120          # seconds, for run_command
    request_timeout: int = 300          # seconds, for a single LLM request
    max_retries: int = 3                # LLM request retries on transient errors

    # --- Skills ----------------------------------------------------------------
    # Directory of reusable skills (each as <name>/SKILL.md). Unlike the
    # workspace sandbox, skills are read-only project resources, not sandboxed.
    skills_dir: Path = field(default_factory=lambda: Path("skills"))

    # --- Workspace (sandbox) -------------------------------------------------
    # All file tools are confined inside this directory.
    workspace: Path = field(default_factory=lambda: Path("workspace").resolve())

    def resolve(self, base: Path) -> "Config":
        # Make the workspace and skills dir absolute relative to the caller.
        self.workspace = (base / self.workspace).resolve()
        self.skills_dir = (base / self.skills_dir).resolve()
        return self

    @staticmethod
    def load(dotenv_path: str | Path | None = None) -> "Config":
        env_file = dotenv_path or os.environ.get("CODING_AGENT_ENV", ".env")
        _load_dotenv(env_file)

        cfg = Config(
            api_key=os.environ.get("LLM_API_KEY", ""),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.environ.get("LLM_MODEL", "deepseek-chat"),
            max_iterations=int(os.environ.get("MAX_ITERATIONS", "40")),
            max_tool_output_chars=int(os.environ.get("MAX_TOOL_OUTPUT_CHARS", "8000")),
            context_char_budget=int(os.environ.get("CONTEXT_CHAR_BUDGET", "36000")),
            context_keep_recent=int(os.environ.get("CONTEXT_KEEP_RECENT", "8")),
            command_timeout=int(os.environ.get("COMMAND_TIMEOUT", "120")),
            request_timeout=int(os.environ.get("REQUEST_TIMEOUT", "300")),
            max_retries=int(os.environ.get("MAX_RETRIES", "3")),
            workspace=Path(os.environ.get("WORKSPACE", "workspace")),
            skills_dir=Path(os.environ.get("SKILLS_DIR", "skills")),
        )
        return cfg
