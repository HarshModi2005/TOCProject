"""
End-to-end tests: Grammar → LR(1) → acceptance via LR(1) simulator,
plus DPDA construction and mask-generation smoke tests.
"""

from __future__ import annotations

import pytest

from pre3.grammar.grammar_loader import (
    arithmetic,
    balanced_parens,
    from_rules,
    simple_json,
)
from pre3.grammar.lr1 import LR1Automaton
from pre3.dpda.builder import build_dpda
from pre3.dpda.optimizer import optimize
from pre3.dpda.simulator import DPDASimulator, LR1Simulator


def _lr1_sim(grammar):
    """Grammar → LR(1) → LR1Simulator."""
    lr1 = LR1Automaton(grammar)
    return LR1Simulator(lr1)


def _dpda_sim(grammar):
    """Grammar → LR(1) → DPDA → optimized DPDA → DPDASimulator."""
    lr1 = LR1Automaton(grammar)
    dpda = build_dpda(lr1)
    dpda = optimize(dpda)
    return DPDASimulator(dpda)


# ======================================================================
# Acceptance tests  (LR(1) simulator – ground truth)
# ======================================================================


class TestBalancedParens:
    @pytest.fixture
    def sim(self):
        return _lr1_sim(balanced_parens())

    @pytest.mark.parametrize(
        "inp",
        [
            [],
            ["(", ")"],
            ["(", "(", ")", ")"],
            ["(", "(", "(", ")", ")", ")"],
        ],
    )
    def test_valid(self, sim, inp):
        result = sim.run(inp)
        assert result.accepted, f"Should accept {inp}: {result}"

    @pytest.mark.parametrize(
        "inp",
        [
            ["("],
            [")"],
            ["(", "(", ")"],
            [")", "("],
        ],
    )
    def test_invalid(self, sim, inp):
        result = sim.run(inp)
        assert not result.accepted, f"Should reject {inp}: {result}"


class TestSimpleGrammar:
    def test_single_symbol(self):
        sim = _lr1_sim(from_rules({"S": ["x"]}, start="S"))
        assert sim.run(["x"]).accepted
        assert not sim.run(["y"]).accepted
        assert not sim.run([]).accepted
        assert not sim.run(["x", "x"]).accepted

    def test_two_symbols(self):
        sim = _lr1_sim(from_rules({"S": ["a b"]}, start="S"))
        assert sim.run(["a", "b"]).accepted
        assert not sim.run(["a"]).accepted
        assert not sim.run(["b"]).accepted
        assert not sim.run(["a", "a"]).accepted

    def test_alternatives(self):
        sim = _lr1_sim(from_rules({"S": ["a", "b", "c"]}, start="S"))
        assert sim.run(["a"]).accepted
        assert sim.run(["b"]).accepted
        assert sim.run(["c"]).accepted
        assert not sim.run(["d"]).accepted


class TestArithmetic:
    @pytest.fixture
    def sim(self):
        return _lr1_sim(arithmetic())

    @pytest.mark.parametrize(
        "inp",
        [
            ["id"],
            ["id", "+", "id"],
            ["id", "*", "id"],
            ["id", "+", "id", "*", "id"],
            ["(", "id", "+", "id", ")", "*", "id"],
        ],
    )
    def test_valid(self, sim, inp):
        result = sim.run(inp)
        assert result.accepted, f"Should accept {inp}: {result}"

    @pytest.mark.parametrize(
        "inp",
        [
            ["+"],
            ["id", "+"],
            ["(", "id"],
            ["id", "id"],
        ],
    )
    def test_invalid(self, sim, inp):
        result = sim.run(inp)
        assert not result.accepted, f"Should reject {inp}: {result}"


class TestRecursiveGrammar:
    """Test a grammar with recursion to exercise cycle handling."""

    def test_nested_lists(self):
        g = from_rules(
            {
                "S": ["[ Items ]", "[ ]"],
                "Items": ["Item , Items", "Item"],
                "Item": ["a", "S"],
            },
            start="S",
        )
        sim = _lr1_sim(g)
        assert sim.run(["[", "]"]).accepted
        assert sim.run(["[", "a", "]"]).accepted
        assert sim.run(["[", "a", ",", "a", "]"]).accepted
        assert sim.run(["[", "[", "a", "]", "]"]).accepted
        assert not sim.run(["[", ","]).accepted
        assert not sim.run(["a"]).accepted


# ======================================================================
# DPDA construction smoke tests
# ======================================================================


class TestDPDABuilds:
    """Verify DPDA construction completes for various grammars."""

    def test_balanced_parens_dpda(self):
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        assert dpda.edge_count > 0

    def test_arithmetic_dpda(self):
        g = arithmetic()
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        assert dpda.edge_count > 0

    def test_recursive_dpda(self):
        g = from_rules(
            {
                "S": ["[ Items ]", "[ ]"],
                "Items": ["Item , Items", "Item"],
                "Item": ["a", "S"],
            },
            start="S",
        )
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        assert dpda.edge_count > 0


class TestMaskSmoke:
    """DPDA mask: check valid_symbols returns something sensible."""

    def test_parens_initial_valid(self):
        sim = _dpda_sim(balanced_parens())
        config = sim.initial_config()
        valid = sim.valid_symbols(config)
        assert "(" in valid

    def test_simple_initial_valid(self):
        sim = _dpda_sim(from_rules({"S": ["a"]}, start="S"))
        config = sim.initial_config()
        valid = sim.valid_symbols(config)
        assert "a" in valid


class TestEdgeCounts:
    """Sanity-check that optimization doesn't break the DPDA."""

    def test_optimize_preserves_states(self):
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        opt = optimize(dpda)
        assert opt.num_states == dpda.num_states
        assert opt.edge_count >= dpda.edge_count
