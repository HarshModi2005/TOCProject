"""
Generic LR(k) automaton construction for any k ≥ 0.

This is the theoretical generalization of LR(1):
  - LR(0): no lookahead.  Very restrictive (most grammars fail).
  - LR(1): one-symbol lookahead.  Equivalent to deterministic context-free
            languages with the prefix property (Knuth, 1965).
  - LR(k>1): k-symbol lookahead.  Strictly more expressive than LR(1) on
            grammar form, but recognizes the same DCFL class.

A lookahead is a *fixed-length-k tuple* of terminal symbols, padded with
the END_MARKER (`$`) when the derivation can terminate within ≤ k symbols.

Provides:
  - LRkItem            — items  [A → α · β, w] with |w| = k
  - LRkAutomaton(g, k) — full canonical LR(k) state machine
  - LRkSimulator       — gold-standard shift/reduce parser for any k

The classical theorems this code constructively realizes:
  T1.  L is DCFL with prefix property  iff  L = L(G) for some LR(1) grammar G.
  T2.  Every LR(k) grammar is unambiguous.
  T3.  L(LR(k)) = L(LR(1)) as language classes (for k ≥ 1).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .cfg import END_MARKER, EPSILON, ContextFreeGrammar, Production


# ======================================================================
# Conflict reporting
# ======================================================================


class GrammarConflictError(ValueError):
    """Raised when a grammar is not LR(k) for the requested k.

    Carries enough info to explain *why* the grammar isn't LR(k):
    the offending state, the lookahead tuple, and the conflicting actions.
    """

    def __init__(self, kind: str, state: int, lookahead: Tuple[str, ...], detail: str):
        super().__init__(f"{kind} conflict in state {state} on lookahead {lookahead}: {detail}")
        self.kind = kind
        self.state = state
        self.lookahead = lookahead
        self.detail = detail


# ======================================================================
# LR(k) Item
# ======================================================================


@dataclass(frozen=True)
class LRkItem:
    """An LR(k) item  [A → α · β, w]  where |w| = k (padded with `$`)."""

    production: Production
    dot: int
    lookahead: Tuple[str, ...]  # length exactly k

    @property
    def at_dot(self) -> Optional[str]:
        if self.dot < len(self.production.body):
            return self.production.body[self.dot]
        return None

    @property
    def is_reduce(self) -> bool:
        return self.dot >= len(self.production.body)

    def advance(self) -> "LRkItem":
        assert not self.is_reduce, "Cannot advance past the end"
        return LRkItem(self.production, self.dot + 1, self.lookahead)

    @property
    def body_after_dot(self) -> Tuple[str, ...]:
        return self.production.body[self.dot + 1 :]

    def __repr__(self) -> str:
        body = list(self.production.body)
        body.insert(self.dot, "·")
        rhs = " ".join(body) if body else "·"
        la = " ".join(self.lookahead) if self.lookahead else "ε"
        return f"[{self.production.head} → {rhs}, {la}]"


# ======================================================================
# LR(k) State
# ======================================================================


class LRkState:
    """A named set of LR(k) items — one node in the state-transition graph."""

    def __init__(self, items: FrozenSet[LRkItem], state_id: int):
        self.items = items
        self.id = state_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LRkState):
            return NotImplemented
        return self.items == other.items

    def __hash__(self) -> int:
        return hash(self.items)

    def __repr__(self) -> str:
        inner = ", ".join(repr(it) for it in sorted(self.items, key=str))
        return f"State({self.id}: {inner})"


# ======================================================================
# Action
# ======================================================================


class ActionType(Enum):
    SHIFT = auto()
    REDUCE = auto()
    ACCEPT = auto()
    ERROR = auto()


@dataclass(frozen=True)
class Action:
    kind: ActionType
    state: Optional[int] = None
    production: Optional[Production] = None

    def __repr__(self) -> str:
        if self.kind == ActionType.SHIFT:
            return f"s{self.state}"
        if self.kind == ActionType.REDUCE:
            return f"r({self.production})"
        if self.kind == ActionType.ACCEPT:
            return "acc"
        return "err"


# ======================================================================
# LR(k) Automaton
# ======================================================================


class LRkAutomaton:
    """Canonical LR(k) automaton for any k ≥ 0.

    Public attributes:
      states           list of LRkState
      transitions      state_id → {symbol: state_id}
      shift_table      state_id → {terminal: target_state}        (for SHIFTs)
      reduce_table     state_id → {lookahead-tuple: Production}   (for REDUCEs)
      goto_table       (state_id, non_terminal) → state_id
      accept_states    set of state_ids where ACCEPT fires on $$$
    """

    def __init__(
        self,
        grammar: ContextFreeGrammar,
        k: int = 1,
        *,
        strict: bool = True,
    ) -> None:
        if k < 0:
            raise ValueError("k must be ≥ 0")
        self.k = k
        self.grammar = grammar.augment()
        self.strict = strict

        self.states: List[LRkState] = []
        self.transitions: Dict[int, Dict[str, int]] = {}
        self.shift_table: Dict[int, Dict[str, int]] = {}
        self.reduce_table: Dict[int, Dict[Tuple[str, ...], Production]] = {}
        self.goto_table: Dict[Tuple[int, str], int] = {}
        self.accept_states: Set[int] = set()
        self.conflicts: List[GrammarConflictError] = []

        self._first_k: Dict[str, Set[Tuple[str, ...]]] = self._compute_first_k()
        self._item_set_to_id: Dict[FrozenSet[LRkItem], int] = {}
        self._build()

    # ------------------------------------------------------------------
    # FIRST_k
    # ------------------------------------------------------------------

    def _first_of_symbol(self, sym: str) -> Set[Tuple[str, ...]]:
        if sym == EPSILON:
            return {()}
        if self.grammar.is_terminal(sym) or sym == END_MARKER:
            return {(sym,)[: self.k]}
        return self._first_k.get(sym, set())

    def _truncated_concat(
        self, set1: Set[Tuple[str, ...]], set2: Set[Tuple[str, ...]]
    ) -> Set[Tuple[str, ...]]:
        """{(a + b)[:k] : a ∈ set1, b ∈ set2}, but stop extending once length ≥ k."""
        k = self.k
        result: Set[Tuple[str, ...]] = set()
        if not set1 or not set2:
            return result
        for a in set1:
            if len(a) >= k:
                result.add(a[:k])
            else:
                for b in set2:
                    result.add((a + b)[:k])
        return result

    def first_k_of_seq(self, symbols: Sequence[str]) -> Set[Tuple[str, ...]]:
        """FIRST_k(α): set of length-≤-k tuples that can begin some derivation of α."""
        k = self.k
        if k == 0:
            return {()}
        current: Set[Tuple[str, ...]] = {()}
        for sym in symbols:
            sym_first = self._first_of_symbol(sym)
            current = self._truncated_concat(current, sym_first)
            if not current:
                break
        return current

    def _compute_first_k(self) -> Dict[str, Set[Tuple[str, ...]]]:
        first_k: Dict[str, Set[Tuple[str, ...]]] = {nt: set() for nt in self.grammar.non_terminals}
        k = self.k

        def sym_first_lookup(sym: str) -> Set[Tuple[str, ...]]:
            if sym == EPSILON:
                return {()}
            if self.grammar.is_terminal(sym) or sym == END_MARKER:
                return {(sym,)[:k]}
            return first_k.get(sym, set())

        def seq_first(symbols: Sequence[str]) -> Set[Tuple[str, ...]]:
            if k == 0:
                return {()}
            current: Set[Tuple[str, ...]] = {()}
            for sym in symbols:
                sf = sym_first_lookup(sym)
                if not sf:
                    return set()
                next_set: Set[Tuple[str, ...]] = set()
                for prefix in current:
                    if len(prefix) >= k:
                        next_set.add(prefix[:k])
                        continue
                    for s in sf:
                        next_set.add((prefix + s)[:k])
                current = next_set
                if not current:
                    return set()
            return current

        changed = True
        while changed:
            changed = False
            for prod in self.grammar.productions:
                rhs = seq_first(prod.body) if prod.body else {()}
                before = len(first_k[prod.head])
                first_k[prod.head] |= rhs
                if len(first_k[prod.head]) > before:
                    changed = True
        return first_k

    # ------------------------------------------------------------------
    # Lookahead utilities
    # ------------------------------------------------------------------

    def _pad_lookahead(self, w: Tuple[str, ...]) -> Tuple[str, ...]:
        """Pad/truncate w to length exactly k, padding with END_MARKER on the right."""
        k = self.k
        if len(w) >= k:
            return w[:k]
        return w + (END_MARKER,) * (k - len(w))

    @property
    def end_lookahead(self) -> Tuple[str, ...]:
        """The lookahead tuple representing 'pure end of input'  =  $^k."""
        return (END_MARKER,) * self.k

    # ------------------------------------------------------------------
    # CLOSURE / GOTO
    # ------------------------------------------------------------------

    def closure(self, items: Set[LRkItem]) -> FrozenSet[LRkItem]:
        closure_set: Set[LRkItem] = set(items)
        worklist: List[LRkItem] = list(items)

        while worklist:
            item = worklist.pop()
            B = item.at_dot
            if B is None or not self.grammar.is_non_terminal(B):
                continue

            beta = item.body_after_dot
            beta_first = self.first_k_of_seq(beta)
            # Concat with the item's own lookahead, truncated to k.
            new_las = self._truncated_concat(beta_first, {item.lookahead})
            # Pad anything shorter than k (only possible when k = 0)
            new_las = {self._pad_lookahead(la) for la in new_las}

            for prod in self.grammar.productions_for(B):
                for la in new_las:
                    new_item = LRkItem(prod, 0, la)
                    if new_item not in closure_set:
                        closure_set.add(new_item)
                        worklist.append(new_item)

        return frozenset(closure_set)

    def goto(self, items: FrozenSet[LRkItem], symbol: str) -> FrozenSet[LRkItem]:
        moved: Set[LRkItem] = set()
        for item in items:
            if item.at_dot == symbol:
                moved.add(item.advance())
        if not moved:
            return frozenset()
        return self.closure(moved)

    # ------------------------------------------------------------------
    # Build the automaton
    # ------------------------------------------------------------------

    def _build(self) -> None:
        start_prod = self.grammar.productions[0]
        start_item = LRkItem(start_prod, 0, self.end_lookahead)
        start_items = self.closure({start_item})

        state0 = LRkState(start_items, state_id=0)
        self.states.append(state0)
        self._item_set_to_id[start_items] = 0

        worklist: List[LRkState] = [state0]
        all_symbols = list(self.grammar.terminals) + list(self.grammar.non_terminals)

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
                    target_id = len(self.states)
                    new_state = LRkState(target_items, state_id=target_id)
                    self.states.append(new_state)
                    self._item_set_to_id[target_items] = target_id
                    worklist.append(new_state)
                self.transitions[state.id][sym] = target_id

        for state in self.states:
            self._fill_actions(state)

    def _fill_actions(self, state: LRkState) -> None:
        aug_start = self.grammar.start
        shifts: Dict[str, int] = {}
        reduces: Dict[Tuple[str, ...], Production] = {}
        accept = False

        for item in state.items:
            if item.is_reduce:
                if item.production.head == aug_start and item.lookahead == self.end_lookahead:
                    accept = True
                else:
                    w = item.lookahead
                    if w in reduces and reduces[w] != item.production:
                        err = GrammarConflictError(
                            "reduce/reduce", state.id, w,
                            f"{reduces[w]}  vs  {item.production}",
                        )
                        self.conflicts.append(err)
                        if self.strict:
                            raise err
                    reduces[w] = item.production
            else:
                sym = item.at_dot
                if sym is not None and self.grammar.is_terminal(sym):
                    target = self.transitions.get(state.id, {}).get(sym)
                    if target is not None:
                        if sym in shifts and shifts[sym] != target:
                            err = GrammarConflictError(
                                "shift/shift", state.id, (sym,) + (END_MARKER,) * (self.k - 1),
                                f"shift→{shifts[sym]}  vs  shift→{target}",
                            )
                            self.conflicts.append(err)
                            if self.strict:
                                raise err
                        shifts[sym] = target

        # Detect shift/reduce conflicts.  In LR(k≥1), the lookahead w
        # determines whether to shift or reduce: shift is keyed by w[0];
        # reduce by full w.  In LR(0) (k=0), the lookahead is the empty
        # tuple and ANY shift conflicts with ANY reduce in the same state.
        if self.k == 0:
            if reduces and shifts:
                for sym, target in shifts.items():
                    for w, prod in reduces.items():
                        err = GrammarConflictError(
                            "shift/reduce", state.id, w,
                            f"shift {sym}→{target}  vs  reduce {prod}",
                        )
                        self.conflicts.append(err)
                        if self.strict:
                            raise err
        else:
            for w, prod in reduces.items():
                head = w[0] if w else END_MARKER
                if head in shifts:
                    err = GrammarConflictError(
                        "shift/reduce", state.id, w,
                        f"shift→{shifts[head]}  vs  reduce {prod}",
                    )
                    self.conflicts.append(err)
                    if self.strict:
                        raise err

        self.shift_table[state.id] = shifts
        self.reduce_table[state.id] = reduces
        if accept:
            self.accept_states.add(state.id)

        for sym in self.grammar.non_terminals:
            target = self.transitions.get(state.id, {}).get(sym)
            if target is not None:
                self.goto_table[(state.id, sym)] = target

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def shift_edges(self, state_id: int) -> Dict[str, int]:
        return dict(self.shift_table.get(state_id, {}))

    def goto_edges(self, state_id: int) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for sym, target in self.transitions.get(state_id, {}).items():
            if self.grammar.is_non_terminal(sym):
                result[sym] = target
        return result

    def reduce_items(self, state_id: int) -> List[LRkItem]:
        return [it for it in self.states[state_id].items if it.is_reduce]

    @property
    def state_count(self) -> int:
        return len(self.states)

    def __repr__(self) -> str:
        kind = f"LR({self.k})"
        return f"{kind} Automaton  (states={self.state_count}, conflicts={len(self.conflicts)})"


# ======================================================================
# LR(k) Simulator  (gold-standard parser)
# ======================================================================


@dataclass
class ParseResult:
    accepted: bool
    consumed: int
    stack: List[int]
    reason: Optional[str] = None
    trace: List[str] = None  # type: ignore

    def __repr__(self) -> str:
        status = "ACCEPTED" if self.accepted else "REJECTED"
        return f"ParseResult({status}, consumed={self.consumed}, reason={self.reason})"


class LRkSimulator:
    """Classical LR(k) shift/reduce parser, used as the language oracle."""

    def __init__(self, lrk: LRkAutomaton):
        self.lrk = lrk

    def run(self, symbols: Sequence[str], *, max_steps: int = 100_000) -> ParseResult:
        k = self.lrk.k
        # Pad input with k+1 end markers so we can always fetch a length-k window.
        extended: List[str] = list(symbols) + [END_MARKER] * (k + 1)
        stack: List[int] = [0]
        pos = 0
        trace: List[str] = []

        for _ in range(max_steps):
            state = stack[-1]
            window: Tuple[str, ...] = tuple(extended[pos : pos + k]) if k > 0 else ()
            head = extended[pos] if pos < len(extended) else END_MARKER

            # 1. ACCEPT
            if state in self.lrk.accept_states and head == END_MARKER:
                # All non-end input must be consumed.
                if pos >= len(symbols):
                    return ParseResult(
                        accepted=True, consumed=pos, stack=list(stack),
                        reason=None, trace=trace,
                    )

            # 2. REDUCE  (precedence determined by table — conflicts resolved at build)
            prod = self.lrk.reduce_table.get(state, {}).get(window)
            if prod is not None:
                pop_count = len(prod.body)
                for _ in range(pop_count):
                    if not stack:
                        return ParseResult(
                            accepted=False, consumed=pos, stack=list(stack),
                            reason=f"stack underflow on reduce {prod}", trace=trace,
                        )
                    stack.pop()
                exposed = stack[-1]
                goto = self.lrk.goto_table.get((exposed, prod.head))
                if goto is None:
                    return ParseResult(
                        accepted=False, consumed=pos, stack=list(stack),
                        reason=f"no GOTO[{exposed}, {prod.head}]", trace=trace,
                    )
                stack.append(goto)
                trace.append(f"reduce {prod}")
                continue

            # 3. SHIFT
            shift_target = self.lrk.shift_table.get(state, {}).get(head)
            if shift_target is not None and pos < len(symbols):
                stack.append(shift_target)
                trace.append(f"shift {head}")
                pos += 1
                continue

            # 4. ERROR
            return ParseResult(
                accepted=False, consumed=pos, stack=list(stack),
                reason=f"no action at state={state}, lookahead={window}",
                trace=trace,
            )

        return ParseResult(
            accepted=False, consumed=pos, stack=list(stack),
            reason=f"exceeded max_steps={max_steps}", trace=trace,
        )

    # Convenience
    def accepts(self, symbols: Sequence[str]) -> bool:
        return self.run(symbols).accepted
