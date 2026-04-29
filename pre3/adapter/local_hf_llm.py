"""
Local HuggingFace-backed StringSource for offline inference.

Typical flow:
  1) First run with `local_files_only=False` to download model/tokenizer.
  2) Subsequent runs with `local_files_only=True` are fully offline.

The model is prompted to output JSON:
    {"tokens": ["...", "..."]}
which is then filtered to grammar terminals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

from ..grammar.cfg import ContextFreeGrammar
from .api_llm import _filter_vocabulary, _parse_token_payload


@dataclass
class LocalHFLLMSource:
    grammar: ContextFreeGrammar
    user_messages: Sequence[str]
    model_id_or_path: str
    local_files_only: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.2
    do_sample: bool = False
    hf_token: Optional[str] = None

    def __post_init__(self) -> None:
        self._allowed: set[str] = set(self.grammar.terminals)
        if not self._allowed:
            raise ValueError("grammar has no terminals")
        self._tok = None
        self._model = None

    def _load(self) -> None:
        if self._tok is not None and self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            self._tok = AutoTokenizer.from_pretrained(
                self.model_id_or_path,
                local_files_only=self.local_files_only,
                token=self.hf_token,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id_or_path,
                local_files_only=self.local_files_only,
                token=self.hf_token,
            )
        except OSError as e:
            msg = str(e)
            if "gated repo" in msg.lower() or "access to model" in msg.lower():
                raise RuntimeError(
                    "Cannot load gated model. For Gemma, accept the model license on "
                    "Hugging Face, then set HF_TOKEN (or pass --hf-token), and run once "
                    "with --local-download to cache it locally for offline use."
                ) from e
            raise

    def _prompt(self, user: str) -> str:
        vocab = ", ".join(f"{t!r}" for t in sorted(self._allowed))
        return (
            "You are a token generator. Return JSON only.\n"
            "Schema: {\"tokens\": [\"<terminal>\", ...]}.\n"
            f"Valid terminals (only these): {vocab}\n"
            f"User request: {user}\n"
        )

    def emit(self) -> Iterator[List[str]]:
        self._load()
        assert self._tok is not None and self._model is not None
        for user in self.user_messages:
            p = self._prompt(user)
            inputs = self._tok(p, return_tensors="pt")
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
            )
            text = self._tok.decode(outputs[0], skip_special_tokens=True)
            # Keep only generated suffix if prompt is echoed
            suffix = text[len(p):] if text.startswith(p) else text
            try:
                toks = _filter_vocabulary(
                    _parse_token_payload(_extract_json(suffix)),
                    self._allowed,
                )
            except Exception:
                # Non-instruction models may not follow JSON; salvage terminals.
                toks = _fallback_terminals_from_text(suffix, self._allowed)
            yield toks


def _extract_json(text: str) -> str:
    """
    Find first top-level JSON object in generated text.
    """
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"Incomplete JSON object in model output: {text!r}")


def _fallback_terminals_from_text(text: str, allowed: set[str]) -> list[str]:
    """
    Best-effort fallback when the model doesn't emit valid JSON.

    Greedy longest-match scan over raw text using grammar terminals.
    This lets small non-instruction local models still feed the validator.
    """
    if not text or not allowed:
        return []
    terms = sorted(allowed, key=len, reverse=True)
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        matched = False
        for t in terms:
            if text.startswith(t, i):
                out.append(t)
                i += len(t)
                matched = True
                break
        if not matched:
            i += 1
    return out
