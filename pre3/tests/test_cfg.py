"""Tests for CFG construction, FIRST and FOLLOW sets."""

import pytest

from pre3.grammar.cfg import EPSILON, END_MARKER, ContextFreeGrammar, Production
from pre3.grammar.grammar_loader import (
    arithmetic,
    balanced_parens,
    from_ebnf,
    from_rules,
    simple_json,
)


class TestProduction:
    def test_repr(self):
        p = Production("S", ("a", "B"))
        assert "S → a B" in repr(p)

    def test_epsilon_repr(self):
        p = Production("S", ())
        assert EPSILON in repr(p)


class TestGrammarLoader:
    def test_from_rules_basic(self):
        g = from_rules({"S": ["a S b", ""]}, start="S")
        assert "S" in g.non_terminals
        assert "a" in g.terminals
        assert "b" in g.terminals
        assert len(g.productions) == 2

    def test_from_ebnf(self):
        text = """
        S ::= a S b | epsilon
        """
        g = from_ebnf(text)
        assert g.start == "S"
        assert len(g.productions) == 2

    def test_balanced_parens(self):
        g = balanced_parens()
        assert g.start == "S"
        assert "(" in g.terminals
        assert ")" in g.terminals

    def test_simple_json(self):
        g = simple_json()
        assert g.start == "Value"
        assert "Object" in g.non_terminals

    def test_arithmetic(self):
        g = arithmetic()
        assert g.start == "E"
        assert "+" in g.terminals


class TestFirstSets:
    def test_balanced_parens(self):
        g = balanced_parens()
        first_S = g.first["S"]
        assert "(" in first_S
        assert EPSILON in first_S

    def test_arithmetic(self):
        g = arithmetic()
        first_E = g.first["E"]
        assert "(" in first_E
        assert "id" in first_E
        assert EPSILON not in first_E

    def test_first_of_sequence(self):
        g = balanced_parens()
        seq_first = g.first_of_sequence(("S", ")"))
        assert "(" in seq_first
        assert ")" in seq_first


class TestFollowSets:
    def test_balanced_parens(self):
        g = balanced_parens()
        follow_S = g.follow["S"]
        assert ")" in follow_S
        assert END_MARKER in follow_S

    def test_arithmetic(self):
        g = arithmetic()
        assert END_MARKER in g.follow["E"]
        assert "+" in g.follow["E"]
        assert ")" in g.follow["E"]


class TestAugment:
    def test_augment_adds_new_start(self):
        g = balanced_parens()
        g2 = g.augment()
        assert g2.start != g.start
        assert g2.start.startswith(g.start)
        assert g.start in g2.non_terminals
