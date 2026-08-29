"""Thin OpenAI-compatible chat-completions client.

We talk to the HTTP endpoint directly (via `requests`) instead of using an
SDK. This keeps the project dependency-light and makes the wire protocol
fully transparent — every message, tool definition and tool call is plain
JSON that you can inspect in the debug logs.

Only the parts of the API we need are implemented:
  * non-streaming `POST /chat/completions`
  * `tools` + `tool_choice="auto"` (native function calling)
"""
from __future__ import annotations

import json
import time

import requests

from .config import Config

# HTTP statuses considered transient (we retry with backoff).
_RETRYABLE = {408, 429, 500, 502, 503, 504}


class LLMError(Exception):
    """Raised when the LLM endpoint fails non-transiently or we give up."""


class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self.session = requests.Session()

    def chat(self, messages: list[dict], tools: list[dict] | None) -> dict:
        """One completion call. Returns the raw `choices[0].message` dict.

        Retries transient HTTP failures with exponential backoff. Raises
        LLMError after the retries are exhausted or on a hard error.
        """
        payload: dict = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False),
                    timeout=self.cfg.request_timeout,
                )
            except requests.RequestException as e:
                last_err = e
            else:
                if resp.status_code == 200:
                    return self._parse_message(resp)
                if resp.status_code in _RETRYABLE:
                    last_err = LLMError(
                        f"HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                else:
                    # Hard, non-retryable error (e.g. 401 bad key, 400 bad request).
                    raise LLMError(
                        f"LLM API error: HTTP {resp.status_code}: {resp.text[:500]}"
                    )

            # Exponential backoff: 1s, 2s, 4s, ...
            if attempt < self.cfg.max_retries - 1:
                time.sleep(2 ** attempt)

        raise LLMError(f"LLM request failed after {self.cfg.max_retries} retries: {last_err}")

    # ------------------------------------------------------------- helpers
    def _parse_message(self, resp: requests.Response) -> dict:
        try:
            body = resp.json()
        except ValueError as e:
            raise LLMError(f"invalid JSON from LLM: {e}") from e

        try:
            choice = body["choices"][0]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected response shape: {json.dumps(body)[:400]}") from e

        message = choice.get("message", {})
        if not isinstance(message, dict):
            raise LLMError("message field is not an object")

        # Normalise content that the API may return as None when tool_calls present.
        message.setdefault("content", "")
        if message.get("content") is None:
            message["content"] = ""
        return message
