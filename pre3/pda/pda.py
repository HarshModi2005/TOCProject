"""
Generic Non-deterministic Pushdown Automaton (NPDA).

Definition (textbook):  M = (Q, Σ, Γ, δ, q₀, Z₀, F)

  Q   finite set of states
  Σ   input alphabet
  Γ   stack alphabet
  δ   transition function  δ : Q × (Σ ∪ {ε}) × Γ  →  𝒫(Q × Γ*)
  q₀  start state
  Z₀  initial stack symbol
  F   set of accepting states

This module supports BOTH common acceptance modes:
  • final-state acceptance     — input fully read, automaton in q ∈ F
  • empty-stack acceptance     — input fully read, stack is empty

Theorem (folklore).  The two modes recognize the same language class (CFL).

Transitions are stored as `PDATransition(state, input_symbol, stack_top,
next_state, stack_push)` where `input_symbol == ""` denotes ε.
The stack is a Python list with the **top on the right** (last element).
`stack_push` is a tuple of stack symbols pushed leftmost-first, so to
push `XY` (with X eventually on top) we use `stack_push=('Y','X')`.
For convenience, the conventional CFG → NPDA construction uses
`stack_push=tuple(reversed(α))` to push α left-to-right with leftmost on top.

We keep this representation simple and pedagogical; for performance we
defer to the LR(1)-derived DPDA in `pre3/dpda/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class PDATransition:
    """One transition rule  δ(q, a, X) ∋ (q', γ).

    `input_symbol == ""` denotes ε.
    `stack_push` is a tuple of stack-alphabet symbols, leftmost = new top.
    """

    state: str
    input_symbol: str             # "" = ε
    stack_top: str
    next_state: str
    stack_push: Tuple[str, ...]   # leftmost element ends up on top

    @property
    def is_epsilon(self) -> bool:
        return self.input_symbol == ""

    def __repr__(self) -> str:
        a = self.input_symbol if self.input_symbol else "ε"
        push = "".join(self.stack_push) if self.stack_push else "ε"
        return f"δ({self.state}, {a}, {self.stack_top}) → ({self.next_state}, {push})"


class NPDA:
    """A non-deterministic pushdown automaton."""

    def __init__(
        self,
        states: Iterable[str],
        input_alphabet: Iterable[str],
        stack_alphabet: Iterable[str],
        transitions: Iterable[PDATransition],
        start_state: str,
        start_stack: str,
        accept_states: Iterable[str] = (),
    ) -> None:
        self.states: FrozenSet[str] = frozenset(states)
        self.input_alphabet: FrozenSet[str] = frozenset(input_alphabet)
        self.stack_alphabet: FrozenSet[str] = frozenset(stack_alphabet)
        self.transitions: Tuple[PDATransition, ...] = tuple(transitions)
        self.start_state = start_state
        self.start_stack = start_stack
        self.accept_states: FrozenSet[str] = frozenset(accept_states)

        # index: (state, input_symbol_or_eps, stack_top) → list of transitions
        self._index: dict[Tuple[str, str, str], List[PDATransition]] = {}
        for t in self.transitions:
            self._index.setdefault((t.state, t.input_symbol, t.stack_top), []).append(t)

        self._validate()

    def _validate(self) -> None:
        if self.start_state not in self.states:
            raise ValueError(f"start_state {self.start_state!r} not in Q")
        if self.start_stack not in self.stack_alphabet:
            raise ValueError(f"start_stack {self.start_stack!r} not in Γ")
        if not self.accept_states <= self.states:
            raise ValueError("F ⊄ Q")
        for t in self.transitions:
            if t.state not in self.states or t.next_state not in self.states:
                raise ValueError(f"transition references unknown state: {t}")
            if t.stack_top not in self.stack_alphabet:
                raise ValueError(f"transition references unknown stack symbol: {t}")
            if t.input_symbol and t.input_symbol not in self.input_alphabet:
                raise ValueError(f"transition references unknown input symbol: {t}")
            for s in t.stack_push:
                if s not in self.stack_alphabet:
                    raise ValueError(f"transition pushes unknown stack symbol: {t}")

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def transitions_from(
        self, state: str, input_symbol: str, stack_top: str
    ) -> List[PDATransition]:
        """All applicable transitions, including ε-transitions."""
        out: List[PDATransition] = []
        out.extend(self._index.get((state, input_symbol, stack_top), []))
        if input_symbol != "":
            out.extend(self._index.get((state, "", stack_top), []))
        return out

    def epsilon_transitions(self, state: str, stack_top: str) -> List[PDATransition]:
        return list(self._index.get((state, "", stack_top), []))

    # ------------------------------------------------------------------
    # Pretty
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NPDA(|Q|={len(self.states)}, |Σ|={len(self.input_alphabet)}, "
            f"|Γ|={len(self.stack_alphabet)}, |δ|={len(self.transitions)}, "
            f"q₀={self.start_state}, Z₀={self.start_stack}, F={set(self.accept_states)})"
        )

    def describe(self) -> str:
        lines = [repr(self), "Transitions:"]
        for t in self.transitions:
            lines.append(f"  {t}")
        return "\n".join(lines)
