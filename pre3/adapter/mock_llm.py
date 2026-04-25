"""
MockLLMSource — pretends to be an LLM by emitting a canned list of
terminal-string sequences.  Use this for unit tests and demos.

Example:

    src = MockLLMSource([["(", "(", ")", ")"], ["(", ")", "("]])
    for s in src.emit():
        result = lrk_sim.run(s)
        print(s, "→", "ok" if result.accepted else "rejected")

When you are ready for a real LLM, write a `RealLLMSource` that:
  • prompts the LLM with the grammar
  • parses the LLM's textual output into a list of terminal symbols
  • yields that list

Nothing in `pre3/dpda/` or `pre3/grammar/` should ever depend on the
concrete source — only on the `StringSource` protocol.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence

from .string_source import StringSource


class MockLLMSource:
    """A stub that yields pre-supplied terminal-token sequences."""

    def __init__(self, emissions: Sequence[Sequence[str]], *, label: str = "mock") -> None:
        self._emissions = [list(e) for e in emissions]
        self.label = label

    def emit(self) -> Iterator[List[str]]:
        for s in self._emissions:
            yield list(s)

    def __repr__(self) -> str:
        return f"MockLLMSource(label={self.label!r}, n={len(self._emissions)})"


# Sanity: the protocol check works at import time too.
assert isinstance(MockLLMSource([]), StringSource)
