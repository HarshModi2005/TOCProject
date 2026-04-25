"""
DPDA simulator – accepts or rejects a string of terminal symbols.

Edge semantics (matches LR(1) parsing operationally):

  • ACCEPTANCE edge     consumes the next input symbol and transitions.
  • REDUCTION  edge     fires when its `accepted_symbols` matches the
                        current lookahead, but does NOT consume the symbol;
                        the simulator immediately re-tries from the new state.
                        This is exactly how chain reductions work in LR(1).

Final acceptance:  after consuming the entire input we feed `END_MARKER` ($)
and chase REDUCTION edges; we accept iff we land in `dpda.accepting_states`.

This module also provides:
  - LR1Simulator:  a classical LR(1) shift/reduce parser using the ACTION/
                   GOTO tables.  Used as the language oracle in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .builder import DPDA
from .edge import EdgeKind, PrefixConditionedEdge


@dataclass
class DPDAConfig:
    """Snapshot of the DPDA's runtime configuration."""

    state: int
    stack: list[int]
    consumed: int = 0

    def clone(self) -> "DPDAConfig":
        return DPDAConfig(state=self.state, stack=list(self.stack), consumed=self.consumed)


@dataclass
class SimResult:
    accepted: bool
    config: DPDAConfig
    trace: list
    reason: Optional[str] = None

    def __repr__(self) -> str:
        status = "ACCEPTED" if self.accepted else "REJECTED"
        return f"SimResult({status}, consumed={self.config.consumed}, reason={self.reason})"


# ======================================================================
# DPDA Simulator
# ======================================================================


class DPDASimulator:
    """Operational interpreter for a DPDA built by `build_dpda`."""

    def __init__(self, dpda: DPDA) -> None:
        self.dpda = dpda

    # ------------------------------------------------------------------
    # Edge selection (deterministic by construction)
    # ------------------------------------------------------------------

    def _matching_edge(
        self, state: int, stack: list[int], symbol: str
    ) -> Optional[PrefixConditionedEdge]:
        """Find THE edge applicable at (state, stack, lookahead).

        LR(1) gives us at most one REDUCTION and at most one ACCEPTANCE per
        (state, lookahead) pair (after disambiguation), and they should not
        coexist on the same lookahead in a conflict-free grammar.  We try
        REDUCTION first.
        """
        red: Optional[PrefixConditionedEdge] = None
        acc: Optional[PrefixConditionedEdge] = None
        for e in self.dpda.lookup(state, symbol):
            if not e.matches_stack(stack):
                continue
            if e.kind == EdgeKind.REDUCTION and red is None:
                red = e
            elif e.kind == EdgeKind.ACCEPTANCE and acc is None:
                acc = e
        return red if red is not None else acc

    def step(
        self, config: DPDAConfig, symbol: str
    ) -> tuple[DPDAConfig, Optional[PrefixConditionedEdge]]:
        """Apply ONE edge for the given lookahead.  REDUCTION does not
        advance `consumed`; ACCEPTANCE does.  Returns (new_config, edge)."""
        edge = self._matching_edge(config.state, config.stack, symbol)
        if edge is None:
            return config, None
        new = config.clone()
        new.stack = edge.apply_stack_ops(new.stack)
        new.state = edge.target
        if edge.kind == EdgeKind.ACCEPTANCE:
            new.consumed += 1
        return new, edge

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def valid_symbols(self, config: DPDAConfig) -> set[str]:
        """Set of terminal symbols that have a valid edge from this config."""
        result: set[str] = set()
        for edge in self.dpda.edges:
            if edge.source != config.state:
                continue
            if edge.matches_stack(config.stack):
                result |= edge.accepted_symbols
        return result

    def initial_config(self) -> DPDAConfig:
        return DPDAConfig(state=self.dpda.start_state, stack=[self.dpda.start_state])

    # ------------------------------------------------------------------
    # Full-string acceptance
    # ------------------------------------------------------------------

    def run(self, symbols: list[str], *, max_steps: int = 100_000) -> SimResult:
        from ..grammar.cfg import END_MARKER

        config = self.initial_config()
        trace: list[tuple[str, PrefixConditionedEdge]] = []
        budget = [max_steps]

        def fire_until_consume_or_dead(sym: str, must_consume: bool) -> tuple[DPDAConfig, bool]:
            """Fire reductions on `sym`, then optionally one acceptance.

            Returns (new_config, ok).  ok=False means we needed to consume
            but no acceptance edge was applicable.
            """
            cur = config
            chain_cap = self.dpda.num_states + 8
            for _ in range(chain_cap):
                if budget[0] <= 0:
                    return cur, False
                edge = self._matching_edge(cur.state, cur.stack, sym)
                if edge is None:
                    return cur, (not must_consume)
                if edge.kind == EdgeKind.REDUCTION:
                    new = cur.clone()
                    new.stack = edge.apply_stack_ops(new.stack)
                    new.state = edge.target
                    cur = new
                    trace.append((sym, edge))
                    budget[0] -= 1
                    continue
                # ACCEPTANCE
                new = cur.clone()
                new.stack = edge.apply_stack_ops(new.stack)
                new.state = edge.target
                new.consumed += 1
                cur = new
                trace.append((sym, edge))
                budget[0] -= 1
                return cur, True
            return cur, False  # chain too long: reject defensively

        for sym in symbols:
            config, ok = fire_until_consume_or_dead(sym, must_consume=True)
            if not ok:
                return SimResult(
                    accepted=False, config=config, trace=trace,
                    reason=f"no edge from state={config.state} on symbol={sym!r}",
                )

        # End of input: chase reductions on $.
        config, _ = fire_until_consume_or_dead(END_MARKER, must_consume=False)

        if config.state in self.dpda.accepting_states:
            return SimResult(accepted=True, config=config, trace=trace, reason=None)
        return SimResult(
            accepted=False, config=config, trace=trace,
            reason=f"end of input but state={config.state} is not accepting",
        )

    def accepts(self, symbols: list[str]) -> bool:
        return self.run(symbols).accepted


# ======================================================================
# LR(1) Simulator  (gold-standard acceptance oracle)
# ======================================================================


class LR1Simulator:
    """Classical LR(1) shift-reduce parser using the ACTION/GOTO tables."""

    def __init__(self, lr1_automaton) -> None:
        from ..grammar.lr1 import LR1Automaton, ActionType  # noqa: F401
        self.lr1 = lr1_automaton
        self._ActionType = ActionType

    def run(self, symbols: list[str]) -> SimResult:
        from ..grammar.cfg import END_MARKER
        ActionType = self._ActionType

        stack: list[int] = [0]
        input_syms = list(symbols) + [END_MARKER]
        pos = 0
        trace: list[tuple[str, Optional[PrefixConditionedEdge]]] = []

        while True:
            state = stack[-1]
            sym = input_syms[pos]
            action = self.lr1.action_table.get((state, sym))

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
                goto_target = self.lr1.goto_table.get((exposed, prod.head))
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
                    trace=trace, reason=None,
                )
            else:
                return SimResult(
                    accepted=False,
                    config=DPDAConfig(state=state, stack=list(stack), consumed=pos),
                    trace=trace,
                    reason=f"Error action at state={state}, symbol={sym!r}",
                )

    def accepts(self, symbols: list[str]) -> bool:
        return self.run(symbols).accepted
