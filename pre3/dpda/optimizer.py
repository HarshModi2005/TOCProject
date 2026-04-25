"""
DPDA edge optimizations (Section 3.3 of the Pre³ paper).

Two main optimizations run during preprocessing:
  1. Edge Aggregation  – merge edges that differ only in accepted_symbols.
  2. Edge Merging      – shortcut sequential edge pairs into single edges.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .builder import DPDA
from .edge import EdgeKind, PrefixConditionedEdge, StackOp


def optimize(dpda: DPDA) -> DPDA:
    """Apply all optimizations and return a new DPDA."""
    optimized = DPDA(
        start_state=dpda.start_state,
        num_states=dpda.num_states,
    )
    optimized.accepting_states = set(dpda.accepting_states)

    edges = list(dpda.edges)
    edges = aggregate_edges(edges)
    edges = merge_edges(edges)

    for e in edges:
        optimized.add_edge(e)

    return optimized


# ======================================================================
# Edge Aggregation
# ======================================================================

def aggregate_edges(edges: list[PrefixConditionedEdge]) -> list[PrefixConditionedEdge]:
    """
    Combine edges that share (source, target, stack_match, stack_ops, kind)
    but differ only in accepted_symbols.

    Example: ten digit edges → one edge accepting {'0','1',…,'9'}.
    """
    Key = Tuple[int, int, tuple, tuple, EdgeKind]

    groups: dict[Key, list[PrefixConditionedEdge]] = defaultdict(list)
    for e in edges:
        key: Key = (e.source, e.target, e.stack_match, e.stack_ops, e.kind)
        groups[key].append(e)

    result: list[PrefixConditionedEdge] = []
    for key, group in groups.items():
        if len(group) == 1:
            result.append(group[0])
            continue

        merged_symbols: set[str] = set()
        for e in group:
            merged_symbols |= e.accepted_symbols

        result.append(
            PrefixConditionedEdge(
                source=key[0],
                target=key[1],
                accepted_symbols=frozenset(merged_symbols),
                stack_match=key[2],
                stack_ops=key[3],
                kind=key[4],
            )
        )
    return result


# ======================================================================
# Edge Merging  (path shortcutting)
# ======================================================================

def merge_edges(edges: list[PrefixConditionedEdge]) -> list[PrefixConditionedEdge]:
    """
    Find sequential edge pairs  e1: A→B, e2: B→C  where:
      - e1 and e2 stack_match conditions are compatible
      - stack_ops can be composed
    and create a shortcut  A→C  with combined ops.

    This reduces the number of transitions per multi-character token.
    """
    by_source: dict[int, list[PrefixConditionedEdge]] = defaultdict(list)
    for e in edges:
        by_source[e.source].append(e)

    merged_set: set[int] = set()
    new_edges: list[PrefixConditionedEdge] = []

    for e1 in edges:
        for e2 in by_source.get(e1.target, []):
            if not _stack_compatible(e1, e2):
                continue

            combined_ops = _compose_ops(e1.stack_ops, e2.stack_ops)
            if combined_ops is None:
                continue

            combined_match = _combine_match(e1.stack_match, e2.stack_match, e1.stack_ops)
            if combined_match is None:
                continue

            shortcut = PrefixConditionedEdge(
                source=e1.source,
                target=e2.target,
                accepted_symbols=e1.accepted_symbols,
                stack_match=combined_match,
                stack_ops=combined_ops,
                kind=e1.kind,
            )
            new_edges.append(shortcut)

    # Keep originals and add shortcuts
    result = list(edges) + new_edges
    return result


def _stack_compatible(
    e1: PrefixConditionedEdge, e2: PrefixConditionedEdge
) -> bool:
    """
    Check whether e2's stack_match is satisfiable after e1's stack_ops
    have been applied.  Conservative: only merge when e2.stack_match is
    a suffix of the stack produced by e1.
    """
    if not e2.stack_match:
        return True
    # Simulate: if e1 pushes state X and e2 requires X on top, compatible
    last_push = _last_pushed(e1.stack_ops)
    if last_push is not None and e2.stack_match and e2.stack_match[-1] == last_push:
        return True
    return False


def _last_pushed(ops: tuple[StackOp, ...]) -> Optional[int]:
    for op in reversed(ops):
        if op.kind.name == "PUSH":
            return op.value
    return None


def _compose_ops(
    ops1: tuple[StackOp, ...], ops2: tuple[StackOp, ...]
) -> Optional[tuple[StackOp, ...]]:
    """Concatenate and simplify stack operations."""
    combined = list(ops1) + list(ops2)
    return tuple(combined)


def _combine_match(
    match1: tuple[int, ...],
    match2: tuple[int, ...],
    ops1: tuple[StackOp, ...],
) -> Optional[tuple[int, ...]]:
    """
    The combined edge's stack_match is e1's match (since e1 runs first
    and e2's match is satisfied by e1's ops).
    """
    return match1
