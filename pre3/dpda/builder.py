"""
Cycle-aware DPDA construction from an LR(1) state-transition graph.

Implements Algorithm 1 from the Pre³ paper:
  Step 1  – Detect cycles and insert back-edges with stack-match guards.
  Step 2  – Add acceptance (shift) edges with push operations.
  Step 3  – Recursively generate reduction edges, merging ε-reductions
            with acceptance edges to maintain prefix-conditioned determinism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from ..grammar.cfg import END_MARKER, Production
from ..grammar.lr1 import ActionType, LR1Automaton, LR1Item
from .edge import EdgeKind, PrefixConditionedEdge, StackOp, StackOpType


# ======================================================================
# DPDA  –  the final automaton produced by the builder
# ======================================================================

class DPDA:
    """
    Deterministic Pushdown Automaton built from an LR(1) grammar.

    Nodes are inherited from the LR(1) state-transition graph.
    Edges are prefix-conditioned (acceptance + reduction).
    """

    def __init__(self, start_state: int, num_states: int) -> None:
        self.start_state = start_state
        self.num_states = num_states
        self.accepting_states: set[int] = set()
        self.edges: list[PrefixConditionedEdge] = []

        # Index: (source, symbol) → list of edges  (fast lookup at runtime)
        self._index: dict[tuple[int, str], list[PrefixConditionedEdge]] = {}

    def add_edge(self, edge: PrefixConditionedEdge) -> None:
        self.edges.append(edge)
        for sym in edge.accepted_symbols:
            key = (edge.source, sym)
            self._index.setdefault(key, []).append(edge)

    def lookup(self, state: int, symbol: str) -> list[PrefixConditionedEdge]:
        return self._index.get((state, symbol), [])

    def find_edge(
        self, state: int, symbol: str, stack: list[int]
    ) -> Optional[PrefixConditionedEdge]:
        """
        Deterministic lookup: given (state, symbol, stack), return the
        unique matching edge or None.
        """
        for edge in self.lookup(state, symbol):
            if edge.matches_stack(stack):
                return edge
        return None

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def __repr__(self) -> str:
        lines = [
            f"DPDA(states={self.num_states}, edges={self.edge_count}, "
            f"start={self.start_state})"
        ]
        for e in self.edges:
            lines.append(f"  {e}")
        return "\n".join(lines)


# ======================================================================
# Builder
# ======================================================================

class DPDABuilder:
    """
    Constructs a DPDA from an LR(1) automaton following the Pre³ algorithm.
    """

    def __init__(self, lr1: LR1Automaton, *, max_reduction_depth: int = 200) -> None:
        self.lr1 = lr1
        self.max_depth = max_reduction_depth
        self.dpda = DPDA(start_state=0, num_states=lr1.state_count)
        self._cycle_back_edges: dict[tuple[int, int], tuple[int, ...]] = {}
        self._visited_reductions: set[tuple[int, tuple[int, ...]]] = set()

    def build(self) -> DPDA:
        self._step1_handle_cycles()
        self._step2_add_acceptance_edges()
        self._step3_add_reduction_edges()
        self._mark_accepting_states()
        return self.dpda

    # ------------------------------------------------------------------
    # Step 1: Cycle detection  (Alg 1, lines 1-5)
    # ------------------------------------------------------------------

    def _step1_handle_cycles(self) -> None:
        """
        Detect cycles in the state-transition graph that would cause
        infinite reduction-edge generation.  For each cycle, modify the
        back-edge to carry a stack-match for the full cycle and pop
        the cycle's states on traversal.
        """
        visited: set[int] = set()
        on_stack: set[int] = set()
        stack_path: list[int] = []

        def dfs(state_id: int) -> None:
            visited.add(state_id)
            on_stack.add(state_id)
            stack_path.append(state_id)

            for sym, target in self.lr1.transitions.get(state_id, {}).items():
                if not self.lr1.grammar.is_non_terminal(sym):
                    continue
                if target in on_stack:
                    # Found a cycle on non-terminal edges (reduction-relevant)
                    idx = stack_path.index(target)
                    cycle = tuple(stack_path[idx:])
                    if len(cycle) >= 2:
                        back_src = cycle[-1]
                        back_dst = cycle[0]
                        self._cycle_back_edges[(back_src, back_dst)] = cycle
                elif target not in visited:
                    dfs(target)

            stack_path.pop()
            on_stack.discard(state_id)

        for sid in range(self.lr1.state_count):
            if sid not in visited:
                dfs(sid)

    # ------------------------------------------------------------------
    # Step 2: Acceptance (shift) edges  (Alg 1, lines 6-8)
    # ------------------------------------------------------------------

    def _step2_add_acceptance_edges(self) -> None:
        """
        For every shift transition in the LR(1) graph, create an
        acceptance edge that pushes the target state onto the stack.
        """
        for state_id in range(self.lr1.state_count):
            for sym, target in self.lr1.shift_edges(state_id).items():
                edge = PrefixConditionedEdge(
                    source=state_id,
                    target=target,
                    accepted_symbols=frozenset({sym}),
                    stack_match=(),  # acceptance edges match any stack
                    stack_ops=(StackOp.push(target),),
                    kind=EdgeKind.ACCEPTANCE,
                )
                self.dpda.add_edge(edge)

    # ------------------------------------------------------------------
    # Step 3: Reduction edges  (Alg 1, lines 9-18)
    # ------------------------------------------------------------------

    def _step3_add_reduction_edges(self) -> None:
        """Generate all reduction edges starting from every state."""
        for state_id in range(self.lr1.state_count):
            self._generate_reduction_edges(state_id, depth=0)

    def _generate_reduction_edges(self, state_id: int, depth: int) -> None:
        if depth > self.max_depth:
            return

        for item in self.lr1.reduce_items(state_id):
            prod = item.production
            lookahead = item.lookahead

            # Skip the augmented start reduction (that's acceptance)
            if prod.head == self.lr1.grammar.start:
                continue

            pop_count = len(prod.body)

            # We need to figure out which states could be exposed after
            # popping |body| items.  We trace backwards through the graph.
            self._trace_reduction(
                origin=state_id,
                production=prod,
                lookahead=lookahead,
                pop_count=pop_count,
                depth=depth,
            )

    def _trace_reduction(
        self,
        origin: int,
        production: Production,
        lookahead: str,
        pop_count: int,
        depth: int,
    ) -> None:
        """
        For a reduction  A → β  with |β| = pop_count, figure out all
        possible stacks that could lead here, compute the exposed state,
        do the GOTO on A, and create a reduction edge.

        We enumerate backward paths of length pop_count through the
        shift-transition graph to find valid stack prefixes.
        """
        # Must agree with LR(1) ACTION: if the table says SHIFT on
        # (origin, lookahead), the parser never reduces on that symbol
        # (e.g. dangling-else: prefer shifting 'else').
        act = self.lr1.action_table.get((origin, lookahead))
        if act is not None and act.kind == ActionType.SHIFT:
            return

        backward_paths = self._enumerate_backward_paths(origin, pop_count)

        for path in backward_paths:
            # path = [origin, prev_1, prev_2, ..., prev_pop_count]
            # where prev_pop_count is the EXPOSED state (the state revealed
            # AFTER popping pop_count items).
            # Runtime stack just before the reduction:
            #   [..., exposed, prev_{pc-1}, ..., prev_1, origin]   (top on right)
            # The top-anchored stack_match must include the EXPOSED state
            # too, because the GOTO target depends on it — different exposed
            # states require different reduction edges.
            exposed_state = path[-1]
            goto_target = self.lr1.goto_edges(exposed_state).get(production.head)
            if goto_target is None:
                continue

            # Full path reversed = (exposed, prev_{pc-1}, ..., prev_1, origin)
            stack_match = tuple(reversed(path))

            visit_key = (origin, stack_match, production, lookahead)
            if visit_key in self._visited_reductions:
                continue
            self._visited_reductions.add(visit_key)

            # Check if this involves a cycle back-edge
            cycle_seq = self._cycle_back_edges.get((origin, goto_target))
            if cycle_seq is not None:
                # Add cycle-handling pop: match full cycle, pop it
                cycle_match = cycle_seq
                cycle_ops = (
                    StackOp.pop(len(cycle_seq)),
                    StackOp.push(cycle_seq[0]),
                )
                cycle_edge = PrefixConditionedEdge(
                    source=origin,
                    target=cycle_seq[0],
                    accepted_symbols=frozenset(),  # ε-transition
                    stack_match=cycle_match,
                    stack_ops=cycle_ops,
                    kind=EdgeKind.REDUCTION,
                )
                self.dpda.add_edge(cycle_edge)

            # Build the reduction edge:  pop |body| states, push goto_target
            ops: list[StackOp] = []
            if pop_count > 0:
                ops.append(StackOp.pop(pop_count))
            ops.append(StackOp.push(goto_target))

            # Always emit a "lookahead-triggered" ε-reduction edge:
            #   accepted_symbols = {lookahead}     (the *trigger*, not consumed)
            #   stack_match      = stack_match     (top-anchored, deep-to-top)
            #   stack_ops        = pop(|β|), push(goto_target)
            #   target           = goto_target
            #
            # The simulator interprets REDUCTION edges as "fire iff next input
            # symbol matches accepted_symbols, but DON'T consume the symbol".
            # Chain reductions fall out naturally: after firing this edge the
            # simulator re-tries the same input symbol from goto_target.
            # ACCEPTANCE edges (Step 2) consume the symbol.
            self.dpda.add_edge(PrefixConditionedEdge(
                source=origin,
                target=goto_target,
                accepted_symbols=frozenset({lookahead}),
                stack_match=stack_match,
                stack_ops=tuple(ops),
                kind=EdgeKind.REDUCTION,
            ))

    def _enumerate_backward_paths(
        self, state_id: int, length: int
    ) -> list[list[int]]:
        """
        Find all paths of exactly *length* transitions that END at *state_id*
        through shift (terminal) edges.  Returns paths as lists
        [state_id, prev_1, prev_2, ..., prev_length].
        """
        if length == 0:
            return [[state_id]]

        # Build reverse adjacency for shift edges
        reverse: dict[int, list[int]] = {}
        for sid in range(self.lr1.state_count):
            for sym, tgt in self.lr1.shift_edges(sid).items():
                reverse.setdefault(tgt, []).append(sid)
            for sym, tgt in self.lr1.goto_edges(sid).items():
                reverse.setdefault(tgt, []).append(sid)

        paths: list[list[int]] = [[state_id]]
        for _ in range(length):
            new_paths: list[list[int]] = []
            for p in paths:
                tip = p[-1]
                for prev in reverse.get(tip, []):
                    new_paths.append(p + [prev])
            paths = new_paths
            if not paths:
                break
        return paths

    # ------------------------------------------------------------------
    # Mark accepting states
    # ------------------------------------------------------------------

    def _mark_accepting_states(self) -> None:
        """States whose items include  [S' → S ·, $]  are accepting."""
        aug_start = self.lr1.grammar.start
        for state in self.lr1.states:
            for item in state.items:
                if (
                    item.production.head == aug_start
                    and item.is_reduce
                    and item.lookahead == END_MARKER
                ):
                    self.dpda.accepting_states.add(state.id)


# ======================================================================
# Convenience function
# ======================================================================

def build_dpda(lr1: LR1Automaton) -> DPDA:
    """One-liner: LR(1) automaton → DPDA."""
    return DPDABuilder(lr1).build()
