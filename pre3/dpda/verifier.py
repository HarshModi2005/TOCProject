"""
Determinism verifier for DPDAs.

A pushdown automaton is *deterministic* iff for every configuration
(state, input-symbol, stack-top-prefix) at most one transition applies.

In the prefix-conditioned-edge representation used here, this means:
for every (source, symbol) pair, all edges' `stack_match` patterns must
be **pairwise non-overlapping** (no two patterns describe a common stack).

`verify_determinism` raises `NondeterminismError` on the first conflict
(or with `collect_all=True`, raises after collecting them all).

Two stack-match tuples `m1` and `m2` overlap iff there exists some real
runtime stack matching both:
  - either is empty       → matches every stack       → overlap
  - one is a strict suffix of the other  → overlap
  - otherwise no common prefix in suffixes → no overlap
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple

from .builder import DPDA
from .edge import EdgeKind, PrefixConditionedEdge


class NondeterminismError(Exception):
    """Raised when a DPDA contains overlapping edges that violate determinism."""

    def __init__(self, conflicts: List["EdgeConflict"]):
        msg = f"DPDA is non-deterministic: {len(conflicts)} conflict(s) detected"
        if conflicts:
            msg += "\nFirst conflict:\n  " + str(conflicts[0])
        super().__init__(msg)
        self.conflicts = conflicts


@dataclass
class EdgeConflict:
    state: int
    symbol: str
    edge_a: PrefixConditionedEdge
    edge_b: PrefixConditionedEdge
    reason: str

    def __str__(self) -> str:
        return (
            f"state={self.state} sym={self.symbol!r}\n"
            f"     A:  {self.edge_a}\n"
            f"     B:  {self.edge_b}\n"
            f"     →  {self.reason}"
        )


def stack_match_overlap(m1: Tuple[int, ...], m2: Tuple[int, ...]) -> bool:
    """Two stack-match patterns overlap iff some runtime stack satisfies both.

    A pattern matches a stack `S` iff `S[-len(m):] == m`.  Therefore
    two patterns overlap iff the shorter is a (top-anchored) suffix of
    the longer, i.e. `long[-len(short):] == short`.
    The empty pattern `()` matches every stack and overlaps with all.
    """
    if not m1 or not m2:
        return True
    short, long = (m1, m2) if len(m1) <= len(m2) else (m2, m1)
    return long[-len(short):] == short


def verify_determinism(
    dpda: DPDA,
    *,
    collect_all: bool = False,
    ignore_kinds: Tuple[EdgeKind, ...] = (),
) -> List[EdgeConflict]:
    """Check determinism.  Raises NondeterminismError on the first conflict
    (or after collecting all if collect_all=True).

    Returns list of conflicts (empty when DPDA is deterministic and we run
    in `collect_all=True` mode without raising).
    """
    by_key: dict[tuple[int, str], list[PrefixConditionedEdge]] = defaultdict(list)
    for edge in dpda.edges:
        if edge.kind in ignore_kinds:
            continue
        for sym in edge.accepted_symbols:
            by_key[(edge.source, sym)].append(edge)

    conflicts: List[EdgeConflict] = []
    for (state, sym), edges in by_key.items():
        n = len(edges)
        for i in range(n):
            for j in range(i + 1, n):
                e1, e2 = edges[i], edges[j]
                if stack_match_overlap(e1.stack_match, e2.stack_match):
                    if e1.target == e2.target and e1.stack_ops == e2.stack_ops:
                        # Same edge effectively: this is a duplicate, not a true conflict.
                        continue
                    conflicts.append(
                        EdgeConflict(
                            state=state, symbol=sym,
                            edge_a=e1, edge_b=e2,
                            reason="overlapping stack_match with different effects",
                        )
                    )
                    if not collect_all:
                        raise NondeterminismError(conflicts)

    if conflicts and not collect_all:
        raise NondeterminismError(conflicts)
    return conflicts


def determinism_certificate(dpda: DPDA) -> str:
    """Human-readable summary of the determinism check.

    Useful as a 'proof artifact' to print at the end of build_dpda.
    Does not raise — returns a string verdict.
    """
    conflicts = verify_determinism(dpda, collect_all=True)
    n_edges = dpda.edge_count
    n_states = dpda.num_states
    if not conflicts:
        return (
            f"✓ DETERMINISM VERIFIED   states={n_states}  edges={n_edges}\n"
            f"  Every (state, symbol) pair has pairwise-disjoint stack_match patterns."
        )
    return (
        f"✗ NON-DETERMINISTIC        states={n_states}  edges={n_edges}\n"
        f"  {len(conflicts)} overlapping edge pair(s); first:\n  {conflicts[0]}"
    )
