"""Tests for the DPDA determinism verifier."""

import pytest

from pre3.dpda.builder import build_dpda
from pre3.dpda.edge import EdgeKind, PrefixConditionedEdge, StackOp
from pre3.dpda.verifier import (
    EdgeConflict, NondeterminismError,
    determinism_certificate, stack_match_overlap, verify_determinism,
)
from pre3.grammar.grammar_loader import arithmetic, balanced_parens, from_rules
from pre3.grammar.lr1 import LR1Automaton


class TestStackMatchOverlap:
    def test_empty_overlaps_with_anything(self):
        assert stack_match_overlap((), (1, 2, 3))
        assert stack_match_overlap((1, 2), ())
        assert stack_match_overlap((), ())

    def test_suffix_overlap(self):
        # (4, 5, 6) is a top-anchored suffix of (1, 2, 3, 4, 5, 6)?  YES.
        assert stack_match_overlap((4, 5, 6), (1, 2, 3, 4, 5, 6))
        assert stack_match_overlap((1, 2, 3, 4, 5, 6), (4, 5, 6))

    def test_no_overlap_different_top(self):
        # Different top-of-stack → no overlap.
        assert not stack_match_overlap((1, 2), (1, 3))


class TestDeterminismVerifier:
    @pytest.mark.parametrize("grammar_fn", [
        balanced_parens, arithmetic,
        lambda: from_rules({"S": ["a S b", ""]}, start="S"),
        lambda: from_rules({"L": ["L , a", "a"]}, start="L"),
        lambda: from_rules({"S": ["a S a", "b S b", "c"]}, start="S"),
    ])
    def test_lr1_dpdas_are_deterministic(self, grammar_fn):
        g = grammar_fn()
        dpda = build_dpda(LR1Automaton(g))
        # Should not raise.
        conflicts = verify_determinism(dpda, collect_all=True)
        assert conflicts == []

    def test_certificate_format(self):
        dpda = build_dpda(LR1Automaton(balanced_parens()))
        cert = determinism_certificate(dpda)
        assert "DETERMINISM VERIFIED" in cert
        assert "states=" in cert
        assert "edges=" in cert

    def test_artificial_conflict_is_detected(self):
        # Manually construct a tiny non-deterministic DPDA and verify we catch it.
        from pre3.dpda.builder import DPDA
        d = DPDA(start_state=0, num_states=2)
        # Two overlapping edges on (0, "a")
        d.add_edge(PrefixConditionedEdge(
            source=0, target=1, accepted_symbols=frozenset({"a"}),
            stack_match=(0,), stack_ops=(StackOp.push(1),),
            kind=EdgeKind.ACCEPTANCE,
        ))
        d.add_edge(PrefixConditionedEdge(
            source=0, target=1, accepted_symbols=frozenset({"a"}),
            stack_match=(),  # empty → matches everything → overlaps
            stack_ops=(StackOp.push(0),),
            kind=EdgeKind.ACCEPTANCE,
        ))
        with pytest.raises(NondeterminismError) as exc:
            verify_determinism(d)
        assert exc.value.conflicts
        assert isinstance(exc.value.conflicts[0], EdgeConflict)
