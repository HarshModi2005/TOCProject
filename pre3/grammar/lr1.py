"""
Canonical LR(1) automaton construction.

Builds the LR(1) item-set family (states), the state-transition graph,
and the ACTION / GOTO parsing tables from a ContextFreeGrammar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .cfg import END_MARKER, EPSILON, ContextFreeGrammar, Production


# ======================================================================
# LR(1) Item
# ======================================================================

@dataclass(frozen=True)
class LR1Item:
    """
    An LR(1) item  [A → α · β, a]

    Stored as (production, dot_pos, lookahead).
    """

    production: Production
    dot: int
    lookahead: str

    @property
    def at_dot(self) -> Optional[str]:
        """Symbol immediately after the dot, or None if dot is at the end."""
        if self.dot < len(self.production.body):
            return self.production.body[self.dot]
        return None

    @property
    def is_reduce(self) -> bool:
        return self.dot >= len(self.production.body)

    def advance(self) -> LR1Item:
        assert not self.is_reduce, "Cannot advance past the end"
        return LR1Item(self.production, self.dot + 1, self.lookahead)

    @property
    def body_after_dot(self) -> tuple[str, ...]:
        """β in [A → α · Bβ, a]  (everything after the symbol at the dot)."""
        return self.production.body[self.dot + 1 :]

    def __repr__(self) -> str:
        body = list(self.production.body)
        body.insert(self.dot, "·")
        rhs = " ".join(body) if body else "·"
        return f"[{self.production.head} → {rhs}, {self.lookahead}]"


# ======================================================================
# LR(1) Item Set (= one state in the automaton)
# ======================================================================

class LR1State:
    """A named set of LR(1) items, representing one node in the automaton."""

    _counter = 0

    def __init__(self, items: FrozenSet[LR1Item], state_id: Optional[int] = None):
        self.items = items
        if state_id is not None:
            self.id = state_id
        else:
            self.id = LR1State._counter
            LR1State._counter += 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LR1State):
            return NotImplemented
        return self.items == other.items

    def __hash__(self) -> int:
        return hash(self.items)

    def __repr__(self) -> str:
        inner = ", ".join(repr(it) for it in sorted(self.items, key=str))
        return f"State({self.id}: {inner})"


# ======================================================================
# ACTION table entry types
# ======================================================================

class ActionType(Enum):
    SHIFT = auto()
    REDUCE = auto()
    ACCEPT = auto()
    ERROR = auto()


@dataclass(frozen=True)
class Action:
    kind: ActionType
    state: Optional[int] = None        # target state for SHIFT
    production: Optional[Production] = None  # rule for REDUCE

    def __repr__(self) -> str:
        if self.kind == ActionType.SHIFT:
            return f"s{self.state}"
        elif self.kind == ActionType.REDUCE:
            return f"r({self.production})"
        elif self.kind == ActionType.ACCEPT:
            return "acc"
        return "err"


# ======================================================================
# LR(1) Automaton
# ======================================================================

class LR1Automaton:
    """
    Full canonical LR(1) automaton: states, transition graph,
    ACTION table, GOTO table.
    """

    def __init__(self, grammar: ContextFreeGrammar) -> None:
        self.grammar = grammar.augment()
        self.states: list[LR1State] = []
        self.transitions: dict[int, dict[str, int]] = {}  # state_id -> {symbol: state_id}
        self.action_table: dict[tuple[int, str], Action] = {}
        self.goto_table: dict[tuple[int, str], int] = {}

        LR1State._counter = 0
        self._item_set_to_id: dict[FrozenSet[LR1Item], int] = {}
        self._build()

    # ------------------------------------------------------------------
    # CLOSURE
    # ------------------------------------------------------------------

    def closure(self, items: set[LR1Item]) -> FrozenSet[LR1Item]:
        closure_set = set(items)
        worklist = list(items)

        while worklist:
            item = worklist.pop()
            B = item.at_dot
            if B is None or not self.grammar.is_non_terminal(B):
                continue

            beta_a = item.body_after_dot + (item.lookahead,)
            lookaheads = self.grammar.first_of_sequence(beta_a) - {EPSILON}

            for prod in self.grammar.productions_for(B):
                for la in lookaheads:
                    new_item = LR1Item(prod, 0, la)
                    if new_item not in closure_set:
                        closure_set.add(new_item)
                        worklist.append(new_item)

        return frozenset(closure_set)

    # ------------------------------------------------------------------
    # GOTO
    # ------------------------------------------------------------------

    def goto(self, items: FrozenSet[LR1Item], symbol: str) -> FrozenSet[LR1Item]:
        moved: set[LR1Item] = set()
        for item in items:
            if item.at_dot == symbol:
                moved.add(item.advance())
        if not moved:
            return frozenset()
        return self.closure(moved)

    # ------------------------------------------------------------------
    # Build automaton
    # ------------------------------------------------------------------

    def _build(self) -> None:
        start_prod = self.grammar.productions[0]
        start_item = LR1Item(start_prod, 0, END_MARKER)
        start_items = self.closure({start_item})

        state0 = LR1State(start_items)
        self.states.append(state0)
        self._item_set_to_id[start_items] = state0.id

        worklist = [state0]
        all_symbols = self.grammar.terminals | self.grammar.non_terminals

        while worklist:
            state = worklist.pop()
            self.transitions.setdefault(state.id, {})

            for sym in all_symbols:
                target_items = self.goto(state.items, sym)
                if not target_items:
                    continue

                if target_items in self._item_set_to_id:
                    target_id = self._item_set_to_id[target_items]
                else:
                    new_state = LR1State(target_items)
                    self.states.append(new_state)
                    self._item_set_to_id[target_items] = new_state.id
                    worklist.append(new_state)
                    target_id = new_state.id

                self.transitions[state.id][sym] = target_id

            self._fill_actions(state)

    def _fill_actions(self, state: LR1State) -> None:
        aug_start = self.grammar.start
        for item in state.items:
            if item.is_reduce:
                if (
                    item.production.head == aug_start
                    and item.lookahead == END_MARKER
                ):
                    self.action_table[(state.id, END_MARKER)] = Action(
                        ActionType.ACCEPT
                    )
                else:
                    key = (state.id, item.lookahead)
                    action = Action(ActionType.REDUCE, production=item.production)
                    if key in self.action_table and self.action_table[key] != action:
                        existing = self.action_table[key]
                        if existing.kind == ActionType.SHIFT:
                            # Shift/reduce conflict: same resolution as in the
                            # shift branch — prefer shift (e.g. dangling-else:
                            # bind 'else' to the inner 'if').
                            continue
                        raise ValueError(
                            f"Reduce/reduce conflict at state {state.id}, "
                            f"lookahead {item.lookahead!r}: "
                            f"{existing} vs {action}"
                        )
                    self.action_table[key] = action
            else:
                sym = item.at_dot
                assert sym is not None
                if self.grammar.is_terminal(sym):
                    target = self.transitions.get(state.id, {}).get(sym)
                    if target is not None:
                        key = (state.id, sym)
                        action = Action(ActionType.SHIFT, state=target)
                        if key in self.action_table:
                            existing = self.action_table[key]
                            if existing.kind == ActionType.REDUCE:
                                # Shift/reduce conflict — default: prefer shift
                                pass
                            elif existing != action:
                                raise ValueError(
                                    f"Shift/shift conflict at state {state.id}, "
                                    f"symbol {sym!r}"
                                )
                        self.action_table[key] = action

        for sym in self.grammar.non_terminals:
            target = self.transitions.get(state.id, {}).get(sym)
            if target is not None:
                self.goto_table[(state.id, sym)] = target

    # ------------------------------------------------------------------
    # Accessors for the DPDA builder
    # ------------------------------------------------------------------

    def shift_edges(self, state_id: int) -> dict[str, int]:
        """Terminal-symbol shift transitions from a state."""
        result: dict[str, int] = {}
        for sym, target in self.transitions.get(state_id, {}).items():
            if self.grammar.is_terminal(sym):
                result[sym] = target
        return result

    def goto_edges(self, state_id: int) -> dict[str, int]:
        """Non-terminal GOTO transitions from a state."""
        result: dict[str, int] = {}
        for sym, target in self.transitions.get(state_id, {}).items():
            if self.grammar.is_non_terminal(sym):
                result[sym] = target
        return result

    def reduce_items(self, state_id: int) -> list[LR1Item]:
        """Items with the dot at the end (= reductions) in a state."""
        state = self.states[state_id]
        return [it for it in state.items if it.is_reduce]

    @property
    def state_count(self) -> int:
        return len(self.states)

    def __repr__(self) -> str:
        lines = [f"LR(1) Automaton  ({self.state_count} states)"]
        for s in self.states:
            lines.append(f"\n  State {s.id}:")
            for it in sorted(s.items, key=str):
                lines.append(f"    {it}")
            for sym, tgt in sorted(self.transitions.get(s.id, {}).items()):
                lines.append(f"    -- {sym} --> {tgt}")
        return "\n".join(lines)
