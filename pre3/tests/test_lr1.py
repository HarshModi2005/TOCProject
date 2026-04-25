"""Tests for LR(1) automaton construction."""

import pytest

from pre3.grammar.grammar_loader import arithmetic, balanced_parens, from_rules
from pre3.grammar.lr1 import LR1Automaton, ActionType


class TestLR1Construction:
    def test_balanced_parens_states(self):
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        assert lr1.state_count > 0
        assert len(lr1.transitions) > 0

    def test_arithmetic_states(self):
        g = arithmetic()
        lr1 = LR1Automaton(g)
        assert lr1.state_count > 0

    def test_simple_grammar(self):
        g = from_rules({"S": ["a"]}, start="S")
        lr1 = LR1Automaton(g)
        # S' -> S ; S -> a
        # Should have at least 3 states: start, after 'a', after 'S'
        assert lr1.state_count >= 3


class TestActionTable:
    def test_has_accept(self):
        g = from_rules({"S": ["a"]}, start="S")
        lr1 = LR1Automaton(g)
        accept_found = any(
            a.kind == ActionType.ACCEPT
            for a in lr1.action_table.values()
        )
        assert accept_found, "ACTION table must contain an ACCEPT entry"

    def test_has_shift(self):
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        shift_found = any(
            a.kind == ActionType.SHIFT
            for a in lr1.action_table.values()
        )
        assert shift_found

    def test_has_reduce(self):
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        reduce_found = any(
            a.kind == ActionType.REDUCE
            for a in lr1.action_table.values()
        )
        assert reduce_found


class TestGotoTable:
    def test_goto_present(self):
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        assert len(lr1.goto_table) > 0

    def test_goto_targets_valid_states(self):
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        valid_ids = {s.id for s in lr1.states}
        for (sid, sym), target in lr1.goto_table.items():
            assert target in valid_ids


class TestTransitionGraph:
    def test_shift_edges(self):
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        all_shifts = {}
        for sid in range(lr1.state_count):
            se = lr1.shift_edges(sid)
            all_shifts.update(se)
        assert len(all_shifts) > 0

    def test_reduce_items(self):
        g = balanced_parens()
        lr1 = LR1Automaton(g)
        found_reduce = False
        for sid in range(lr1.state_count):
            if lr1.reduce_items(sid):
                found_reduce = True
                break
        assert found_reduce
