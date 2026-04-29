"""
OpenAI-compatible chat completion as a :class:`StringSource`.

Implements the *left* side of the pipeline: a remote LLM returns JSON describing
a sequence of grammar terminals; those lists are fed into
:class:`~pre3.dpda.simulator.DPDASimulator` (or an LR parser) unchanged from
``MockLLMSource``.

**Configuration (environment variables)**

* ``OPENAI_API_KEY`` — required for live calls (unless passed explicitly)
* ``OPENAI_BASE_URL`` — default ``https://api.openai.com/v1``
* ``OPENAI_MODEL`` — default ``gpt-4o-mini`` (or any model your provider exposes)

The HTTP client uses the standard library only (no extra packages).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Sequence

from ..grammar.cfg import ContextFreeGrammar


# ---------------------------------------------------------------------------
# HTTP — OpenAI-compatible chat.completions
# ---------------------------------------------------------------------------


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    response_format: Optional[dict[str, str]] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: int = 120,
) -> str:
    """
    POST ``/chat/completions``; return the assistant *content* string.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    data = json.dumps(payload).encode("utf-8")
    # Groq (and similar) sit behind Cloudflare; bare urllib has no User-Agent
    # and often gets 403 / error 1010 without one.
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "pre3-llm/0.1 (Python-urllib; OpenAI-compatible client)",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec: URL from caller
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {e.code}: {err}") from e
    except OSError as e:
        raise RuntimeError(f"API request failed: {e}") from e

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected API response shape: {body!r}") from e


def _parse_token_payload(content: str) -> list[str]:
    """
    Parse assistant JSON.  Expected: ``{"tokens": ["(", ")", ...]}``.
    """
    data = json.loads(content.strip())
    if not isinstance(data, dict) or "tokens" not in data:
        raise ValueError("JSON must be an object with a 'tokens' array")
    toks = data["tokens"]
    if not isinstance(toks, list):
        raise ValueError("'tokens' must be a list")
    out: list[str] = []
    for x in toks:
        if not isinstance(x, str):
            raise ValueError(f"Each token must be a string, got {x!r}")
        out.append(x)
    return out


def _filter_vocabulary(seq: list[str], allowed: set[str]) -> list[str]:
    return [t for t in seq if t in allowed]


# ---------------------------------------------------------------------------
# StringSource
# ---------------------------------------------------------------------------


@dataclass
class OpenAILLMSource:
    """
    For each *user* message, call the chat API once and parse ``tokens`` JSON.

    Unknown terminals (not in the grammar) are dropped so the automaton
    only sees well-formed terminal symbols.
    """

    grammar: ContextFreeGrammar
    user_messages: Sequence[str]
    system_message: str = (
        "You are a token generator. Output only valid JSON. "
        'Schema: {"tokens": ["<terminal>", ...]} — each string is one token.'
    )
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    use_json_object_mode: bool = True

    def __post_init__(self) -> None:
        self._allowed: set[str] = set(self.grammar.terminals)
        if not self._allowed:
            raise ValueError("grammar has no terminals")
        # Resolve API key from env if omitted
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "Missing API key: pass api_key=... or set environment variable OPENAI_API_KEY"
            )
        self._api_key = key
        self._model = os.environ.get("OPENAI_MODEL", self.model)

    def emit(self) -> Iterator[List[str]]:
        fmt = ({"type": "json_object"} if self.use_json_object_mode else None)
        for user in self.user_messages:
            content = chat_completion(
                [
                    {"role": "system", "content": self._system_enriched()},
                    {"role": "user", "content": user},
                ],
                model=self._model,
                api_key=self._api_key,
                base_url=self.base_url,
                response_format=fmt,
            )
            toks = _filter_vocabulary(_parse_token_payload(content), self._allowed)
            yield toks

    def _system_enriched(self) -> str:
        vocab = ", ".join(f"{t!r}" for t in sorted(self._allowed))
        return f"{self.system_message}\nValid terminal symbols (only these): {vocab}."


