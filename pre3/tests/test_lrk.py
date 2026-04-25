"""Tests for the generic LR(k) automaton and parser, k = 0, 1, 2, 3."""

import pytest

from pre3.grammar.cfg import END_MARKER
from pre3.grammar.grammar_loader import balanced_parens, arithmetic, from_rules
from pre3.grammar.lrk import GrammarConflictError, LRkAutomaton, LRkSimulator


# ----------------------------------------------------------------------
# FIRST_k correctness
# ----------------------------------------------------------------------


class TestFirstK:
    def test_terminal_first1(self):
        g = from_rules({"S": ["a"]}, start="S")
        a = LRkAutomaton(g, k=1)
        assert a.first_k_of_seq(("a",)) == {("a",)}

    def test_epsilon_in_first1(self):
        g = from_rules({"S": ["a", ""]}, start="S")
        a = LRkAutomaton(g, k=1)
        # FIRST_1(S) should contain (a,) and ()
        first_S = a._first_k["S"]
        assert ("a",) in first_S
        assert () in first_S

    def test_first2_two_symbols(self):
        g = from_rules({"S": ["a b"]}, start="S")
        a = LRkAutomaton(g, k=2)
        assert ("a", "b") in a.first_k_of_seq(("a", "b"))

    def test_first2_with_nullable_prefix(self):
        # A → ε | x ;   B → A b
        g = from_rules({"A": ["x", ""], "B": ["A b"]}, start="B")
        a = LRkAutomaton(g, k=2)
        first_B = a._first_k["B"]
        # Both (x, b) and (b,) should be in FIRST_2(B) since A is nullable.
        assert ("x", "b") in first_B
        # (b,) padded would be (b, $) via lookahead padding; raw FIRST_2 may
        # contain (b,) of length 1 indicating early termination.
        assert ("b",) in first_B or ("b", END_MARKER) in first_B


# ----------------------------------------------------------------------
# LR(k) parser ≡ LR(1) parser on LR(1) grammars (for k ≥ 1)
# ----------------------------------------------------------------------


class TestLRkParser:
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_balanced_parens(self, k):
        sim = LRkSimulator(LRkAutomaton(balanced_parens(), k=k))
        assert sim.accepts([])
        assert sim.accepts(["(", ")"])
        assert sim.accepts(["(", "(", ")", ")"])
        assert not sim.accepts(["("])
        assert not sim.accepts([")"])
        assert not sim.accepts(["(", "("])

    @pytest.mark.parametrize("k", [1, 2])
    def test_arithmetic(self, k):
        sim = LRkSimulator(LRkAutomaton(arithmetic(), k=k))
        assert sim.accepts(["id"])
        assert sim.accepts(["id", "+", "id"])
        assert sim.accepts(["(", "id", "+", "id", ")", "*", "id"])
        assert not sim.accepts(["+"])
        assert not sim.accepts(["id", "+"])

    def test_anbn(self):
        sim = LRkSimulator(LRkAutomaton(from_rules({"S": ["a S b", ""]}, start="S"), k=1))
        for n in range(5):
            assert sim.accepts(["a"] * n + ["b"] * n)
            if n > 0:
                assert not sim.accepts(["a"] * n + ["b"] * (n + 1))


# ----------------------------------------------------------------------
# LR(0) is strictly less expressive than LR(1)
# ----------------------------------------------------------------------


class TestLRkHierarchy:
    def test_lr0_rejects_balanced_parens(self):
        # balanced_parens isn't LR(0): the ε-reduction conflicts with shift on '('.
        # Non-strict mode collects conflicts; strict mode raises.
        nonstrict = LRkAutomaton(balanced_parens(), k=0, strict=False)
        assert nonstrict.conflicts, "expected at least one LR(0) conflict"
        with pytest.raises(GrammarConflictError):
            LRkAutomaton(balanced_parens(), k=0, strict=True)

    def test_lr1_succeeds_where_lr0_fails(self):
        # Same grammar, LR(1) succeeds.
        a = LRkAutomaton(balanced_parens(), k=1, strict=True)
        assert a.state_count > 0
        assert not a.conflicts

    def test_wwR_is_not_lrk(self):
        # wwR is CFL but NOT DCFL: every k must produce conflicts.
        g = from_rules({"S": ["a S a", "b S b", ""]}, start="S")
        for k in [1, 2, 3]:
            with pytest.raises(GrammarConflictError):
                LRkAutomaton(g, k=k, strict=True)


# ----------------------------------------------------------------------
# Conflict detection
# ----------------------------------------------------------------------


class TestConflictDetection:
    def test_reduce_reduce_conflict(self):
        # A → x ; B → x — same string, two productions.
        g = from_rules({"S": ["A", "B"], "A": ["x"], "B": ["x"]}, start="S")
        with pytest.raises(GrammarConflictError):
            LRkAutomaton(g, k=1, strict=True)

    def test_strict_false_collects_conflicts(self):
        g = from_rules({"S": ["A", "B"], "A": ["x"], "B": ["x"]}, start="S")
        a = LRkAutomaton(g, k=1, strict=False)
        assert len(a.conflicts) >= 1
