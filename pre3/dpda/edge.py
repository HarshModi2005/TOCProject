"""
Prefix-conditioned edge types for the DPDA.

Each edge carries:
  - accepted_symbols : frozenset of terminal symbols that trigger the edge
  - stack_match      : tuple of state-IDs that must sit on top of the stack
                       (top of stack = last element) for the edge to fire
  - stack_ops        : sequence of StackOp (push / pop)
  - source / target  : state IDs in the DPDA
  - kind             : ACCEPTANCE or REDUCTION
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence


class EdgeKind(Enum):
    ACCEPTANCE = auto()   # corresponds to LR(1) shift
    REDUCTION = auto()    # encodes one or more LR(1) reductions


class StackOpType(Enum):
    PUSH = auto()
    POP = auto()


@dataclass(frozen=True)
class StackOp:
    kind: StackOpType
    value: int = 0   # state ID to push, or number of entries to pop

    @staticmethod
    def push(state_id: int) -> StackOp:
        return StackOp(StackOpType.PUSH, state_id)

    @staticmethod
    def pop(count: int = 1) -> StackOp:
        return StackOp(StackOpType.POP, count)

    def __repr__(self) -> str:
        if self.kind == StackOpType.PUSH:
            return f"push({self.value})"
        return f"pop({self.value})"


@dataclass
class PrefixConditionedEdge:
    """
    A single deterministic edge in the DPDA.

    Determinism guarantee: for any (source, accepted_symbol, stack_match)
    there is at most ONE edge in the automaton.
    """

    source: int
    target: int
    accepted_symbols: frozenset[str]
    stack_match: tuple[int, ...]
    stack_ops: tuple[StackOp, ...]
    kind: EdgeKind

    def matches_stack(self, stack: list[int]) -> bool:
        """Check whether the runtime stack satisfies stack_match."""
        if len(self.stack_match) == 0:
            return True
        if len(stack) < len(self.stack_match):
            return False
        top_slice = tuple(stack[-len(self.stack_match) :])
        return top_slice == self.stack_match

    def apply_stack_ops(self, stack: list[int]) -> list[int]:
        """Return a new stack after applying this edge's operations."""
        s = list(stack)
        for op in self.stack_ops:
            if op.kind == StackOpType.POP:
                for _ in range(op.value):
                    if s:
                        s.pop()
            else:
                s.append(op.value)
        return s

    def __repr__(self) -> str:
        syms = ",".join(sorted(self.accepted_symbols)) if self.accepted_symbols else "ε"
        match_str = ",".join(str(x) for x in self.stack_match) if self.stack_match else "*"
        ops_str = " ".join(repr(o) for o in self.stack_ops) if self.stack_ops else "-"
        return (
            f"Edge({self.source}→{self.target} "
            f"[{self.kind.name}] on={{{syms}}} "
            f"match=[{match_str}] ops=[{ops_str}])"
        )
