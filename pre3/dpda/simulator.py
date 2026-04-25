"""
DPDA simulator – accepts or rejects a string of terminal symbols.

Provides two modes:
  1. DPDASimulator  – walks the DPDA edges (for mask-generation testing).
  2. LR1Simulator   – classical LR(1) parse (gold-standard correctness check).

Both are used for testing; the LR1Simulator is the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .builder import DPDA
from .edge import PrefixConditionedEdge


@dataclass
class DPDAConfig:
    """Snapshot of the DPDA's runtime configuration."""

    state: int
    stack: list[int]
    consumed: int = 0

    def clone(self) -> DPDAConfig:
        return DPDAConfig(state=self.state, stack=list(self.stack), consumed=self.consumed)


class DPDASimulator:
    """
    Steps the DPDA one symbol at a time.

    Note: this simulator is designed for **mask generation** (one symbol at a
    time during LLM decode).  For full string acceptance testing, use
    :class:`LR1Simulator` which faithfully implements LR(1) parse logic.
    """

    def __init__(self, dpda: DPDA) -> None:
        self.dpda = dpda

    def step(
        self, config: DPDAConfig, symbol: str
    ) -> tuple[DPDAConfig, Optional[PrefixConditionedEdge]]:
        """Advance one symbol; returns (new_config, edge_used)."""
        edge = self.dpda.find_edge(config.state, symbol, config.stack)
        if edge is None:
            return config, None
        new = config.clone()
        new.stack = edge.apply_stack_ops(new.stack)
        new.state = edge.target
        new.consumed += 1
        return new, edge

    def valid_symbols(self, config: DPDAConfig) -> set[str]:
        """Return the set of terminal symbols that have a valid edge."""
        result: set[str] = set()
        for edge in self.dpda.edges:
            if edge.source != config.state:
                continue
            if edge.matches_stack(config.stack):
                result |= edge.accepted_symbols
        return result

    def initial_config(self) -> DPDAConfig:
        return DPDAConfig(
            state=self.dpda.start_state,
            stack=[self.dpda.start_state],
        )


# ======================================================================
# LR(1) Simulator  (gold-standard acceptance check)
# ======================================================================

class LR1Simulator:
    """
    Classical LR(1) shift-reduce parser used as ground truth for testing.

    Directly uses the ACTION / GOTO tables from the LR(1) automaton.
    """

    def __init__(self, lr1_automaton) -> None:
        from ..grammar.lr1 import LR1Automaton, ActionType
        self.lr1: LR1Automaton = lr1_automaton
        self._ActionType = ActionType

    def run(self, symbols: list[str]) -> SimResult:
        """Parse a list of terminal symbols.  Returns a SimResult."""
        from ..grammar.cfg import END_MARKER
        ActionType = self._ActionType

        stack: list[int] = [0]
        input_syms = list(symbols) + [END_MARKER]
        pos = 0
        trace: list[tuple[str, Optional[PrefixConditionedEdge]]] = []

        while True:
            state = stack[-1]
            sym = input_syms[pos]
            key = (state, sym)
            action = self.lr1.action_table.get(key)

            if action is None:
                return SimResult(
                    accepted=False,
                    config=DPDAConfig(state=state, stack=list(stack), consumed=pos),
                    trace=trace,
                    reason=f"No action for state={state}, symbol={sym!r}",
                )

            if action.kind == ActionType.SHIFT:
                stack.append(action.state)
                trace.append((sym, None))
                pos += 1

            elif action.kind == ActionType.REDUCE:
                prod = action.production
                pop_count = len(prod.body)
                for _ in range(pop_count):
                    stack.pop()
                exposed = stack[-1]
                goto_key = (exposed, prod.head)
                goto_target = self.lr1.goto_table.get(goto_key)
                if goto_target is None:
                    return SimResult(
                        accepted=False,
                        config=DPDAConfig(state=exposed, stack=list(stack), consumed=pos),
                        trace=trace,
                        reason=f"No GOTO for state={exposed}, NT={prod.head!r}",
                    )
                stack.append(goto_target)
                trace.append((f"reduce({prod})", None))

            elif action.kind == ActionType.ACCEPT:
                return SimResult(
                    accepted=True,
                    config=DPDAConfig(state=state, stack=list(stack), consumed=pos),
                    trace=trace,
                    reason=None,
                )

            else:
                return SimResult(
                    accepted=False,
                    config=DPDAConfig(state=state, stack=list(stack), consumed=pos),
                    trace=trace,
                    reason=f"Error action at state={state}, symbol={sym!r}",
                )


@dataclass
class SimResult:
    accepted: bool
    config: DPDAConfig
    trace: list[tuple[str, Optional[PrefixConditionedEdge]]]
    reason: Optional[str] = None

    def __repr__(self) -> str:
        status = "ACCEPTED" if self.accepted else "REJECTED"
        return f"SimResult({status}, consumed={self.config.consumed}, reason={self.reason})"
