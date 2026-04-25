"""
Context-Free Grammar representation with FIRST / FOLLOW set computation.

A CFG is a 4-tuple (V, Σ, P, S) where:
  V  = non-terminals
  Σ  = terminals
  P  = production rules  (A -> α₁ | α₂ | …)
  S  = start symbol
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Sequence

EPSILON = "ε"
END_MARKER = "$"


@dataclass(frozen=True)
class Production:
    """A single production rule  A → body  where body is a tuple of symbols."""

    head: str
    body: tuple[str, ...]

    def __repr__(self) -> str:
        rhs = " ".join(self.body) if self.body else EPSILON
        return f"{self.head} → {rhs}"


class ContextFreeGrammar:
    """
    Mutable grammar builder that also computes FIRST / FOLLOW sets.

    Symbols are plain strings.  By convention:
      - Non-terminals start with an uppercase letter or are wrapped in angle brackets.
      - Terminals are everything else (single chars, quoted strings, etc.).
    The caller decides this via *terminals* and *non_terminals* sets.
    """

    def __init__(
        self,
        terminals: set[str],
        non_terminals: set[str],
        productions: list[Production],
        start: str,
    ) -> None:
        assert start in non_terminals, f"Start symbol {start!r} must be a non-terminal"
        self.terminals = frozenset(terminals)
        self.non_terminals = frozenset(non_terminals)
        self.productions = list(productions)
        self.start = start

        self._prod_index: dict[str, list[Production]] = {}
        for p in self.productions:
            self._prod_index.setdefault(p.head, []).append(p)

        self._first: Optional[dict[str, set[str]]] = None
        self._follow: Optional[dict[str, set[str]]] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def productions_for(self, non_terminal: str) -> list[Production]:
        return self._prod_index.get(non_terminal, [])

    def is_terminal(self, sym: str) -> bool:
        return sym in self.terminals

    def is_non_terminal(self, sym: str) -> bool:
        return sym in self.non_terminals

    def augment(self) -> ContextFreeGrammar:
        """Return an augmented grammar with a new start  S' → S."""
        new_start = self.start + "'"
        while new_start in self.non_terminals:
            new_start += "'"
        new_prod = Production(new_start, (self.start,))
        return ContextFreeGrammar(
            terminals=set(self.terminals),
            non_terminals=set(self.non_terminals) | {new_start},
            productions=[new_prod] + self.productions,
            start=new_start,
        )

    # ------------------------------------------------------------------
    # FIRST sets
    # ------------------------------------------------------------------

    @property
    def first(self) -> dict[str, set[str]]:
        if self._first is None:
            self._first = self._compute_first()
        return self._first

    def first_of_sequence(self, symbols: Sequence[str]) -> set[str]:
        """Compute FIRST(α) for an arbitrary sequence of grammar symbols."""
        result: set[str] = set()
        for sym in symbols:
            f = self.first_of_symbol(sym)
            result |= f - {EPSILON}
            if EPSILON not in f:
                return result
        result.add(EPSILON)
        return result

    def first_of_symbol(self, sym: str) -> set[str]:
        if sym == EPSILON:
            return {EPSILON}
        if self.is_terminal(sym) or sym == END_MARKER:
            return {sym}
        return set(self.first.get(sym, set()))

    def _compute_first(self) -> dict[str, set[str]]:
        first: dict[str, set[str]] = {nt: set() for nt in self.non_terminals}

        def _sym_first(sym: str) -> set[str]:
            if sym == EPSILON:
                return {EPSILON}
            if self.is_terminal(sym) or sym == END_MARKER:
                return {sym}
            return set(first.get(sym, set()))

        def _seq_first(symbols: Sequence[str]) -> set[str]:
            result: set[str] = set()
            for sym in symbols:
                f = _sym_first(sym)
                result |= f - {EPSILON}
                if EPSILON not in f:
                    return result
            result.add(EPSILON)
            return result

        changed = True
        while changed:
            changed = False
            for prod in self.productions:
                before = len(first[prod.head])
                rhs_first = _seq_first(prod.body) if prod.body else {EPSILON}
                first[prod.head] |= rhs_first
                if len(first[prod.head]) > before:
                    changed = True
        return first

    # ------------------------------------------------------------------
    # FOLLOW sets
    # ------------------------------------------------------------------

    @property
    def follow(self) -> dict[str, set[str]]:
        if self._follow is None:
            self._follow = self._compute_follow()
        return self._follow

    def _compute_follow(self) -> dict[str, set[str]]:
        follow: dict[str, set[str]] = {nt: set() for nt in self.non_terminals}
        follow[self.start].add(END_MARKER)

        changed = True
        while changed:
            changed = False
            for prod in self.productions:
                trailer = set(follow[prod.head])
                for sym in reversed(prod.body):
                    if self.is_non_terminal(sym):
                        before = len(follow[sym])
                        follow[sym] |= trailer
                        if len(follow[sym]) > before:
                            changed = True
                        if EPSILON in self.first_of_symbol(sym):
                            trailer = (trailer | self.first_of_symbol(sym)) - {EPSILON}
                        else:
                            trailer = self.first_of_symbol(sym)
                    else:
                        trailer = self.first_of_symbol(sym)
        return follow

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = [f"CFG  start={self.start}"]
        for nt in sorted(self.non_terminals):
            for p in self.productions_for(nt):
                lines.append(f"  {p}")
        return "\n".join(lines)
