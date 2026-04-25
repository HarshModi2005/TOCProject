"""
StringSource — the boundary between *something that produces tokens*
(LLM, file, network, hand-crafted list) and the *automata-validation*
pipeline.

A StringSource yields lists of grammar-terminal strings, one list per
"emission".  Downstream code feeds each emission into:
    LR(k)Simulator(grammar).run(emission)
or
    DPDASimulator(dpda).run(emission)

By abstracting the source, we can wire up a real LLM later (returning
its tokenized output) without touching automata code.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Protocol, runtime_checkable


@runtime_checkable
class StringSource(Protocol):
    """Anything that yields a sequence of lists-of-terminals."""

    def emit(self) -> Iterator[List[str]]: ...
