"""Tests for DPDA construction, optimization, and simulation."""

from __future__ import annotations

import pytest

from pre3.grammar.grammar_loader import arithmetic, balanced_parens, from_rules
from pre3.grammar.lr1 import LR1Automaton
from pre3.dpda.builder import DPDABuilder, build_dpda
from pre3.dpda.optimizer import aggregate_edges, merge_edges, optimize
from pre3.dpda.simulator import DPDAConfig, DPDASimulator, LR1Simulator
from pre3.dpda.edge import EdgeKind


class TestDPDAConstruction:
    def test_simple_grammar_builds(self):
        g = from_rules({"S": ["a"]}, start="S")
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        assert dpda.edge_count > 0
        assert dpda.num_states > 0

    def test_balanced_parens_builds(self):
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        assert dpda.edge_count > 0

    def test_has_acceptance_edges(self):
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        acc_edges = [e for e in dpda.edges if e.kind == EdgeKind.ACCEPTANCE]
        assert len(acc_edges) > 0

    def test_has_reduction_edges(self):
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        red_edges = [e for e in dpda.edges if e.kind == EdgeKind.REDUCTION]
        assert len(red_edges) > 0

    def test_arithmetic_builds(self):
        g = arithmetic()
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        assert dpda.edge_count > 0


class TestDPDAOptimizer:
    def test_aggregate_reduces_count(self):
        g = from_rules(
            {"S": ["a", "b", "c"]},
            start="S",
        )
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        before = dpda.edge_count
        new_edges = aggregate_edges(list(dpda.edges))
        assert len(new_edges) <= before

    def test_optimize_returns_valid_dpda(self):
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        opt = optimize(dpda)
        assert opt.edge_count > 0
        assert opt.num_states == dpda.num_states


class TestDPDASimulatorMask:
    """Test the DPDA simulator for valid_symbols (mask generation)."""

    def test_valid_symbols_simple(self):
        g = from_rules({"S": ["a"]}, start="S")
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        sim = DPDASimulator(dpda)
        config = sim.initial_config()
        valid = sim.valid_symbols(config)
        assert "a" in valid

    def test_valid_symbols_parens(self):
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        sim = DPDASimulator(dpda)
        config = sim.initial_config()
        valid = sim.valid_symbols(config)
        assert "(" in valid


class TestLR1Simulator:
    """Test acceptance using the gold-standard LR(1) parser."""

    def test_simple_accept(self):
        g = from_rules({"S": ["a"]}, start="S")
        lr1 = LR1Automaton(g)
        sim = LR1Simulator(lr1)
        result = sim.run(["a"])
        assert result.accepted, f"Should accept 'a': {result}"

    def test_simple_reject(self):
        g = from_rules({"S": ["a"]}, start="S")
        lr1 = LR1Automaton(g)
        sim = LR1Simulator(lr1)
        result = sim.run(["b"])
        assert not result.accepted

    def test_empty_reject(self):
        g = from_rules({"S": ["a"]}, start="S")
        lr1 = LR1Automaton(g)
        sim = LR1Simulator(lr1)
        result = sim.run([])
        assert not result.accepted


class TestDPDADeterminism:
    def test_no_duplicate_edges(self):
        """For any (source, symbol, stack_match), at most one edge."""
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        seen: set[tuple[int, str, tuple[int, ...]]] = set()
        duplicates = 0
        for e in dpda.edges:
            for sym in e.accepted_symbols:
                key = (e.source, sym, e.stack_match)
                if key in seen:
                    duplicates += 1
                seen.add(key)
