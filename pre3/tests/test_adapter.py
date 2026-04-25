"""Smoke tests for the StringSource / MockLLMSource adapter."""

from pre3.adapter.mock_llm import MockLLMSource
from pre3.adapter.string_source import StringSource
from pre3.dpda.builder import build_dpda
from pre3.dpda.simulator import DPDASimulator
from pre3.grammar.grammar_loader import balanced_parens
from pre3.grammar.lr1 import LR1Automaton


def test_mock_llm_implements_protocol():
    src = MockLLMSource([["(", ")"]])
    assert isinstance(src, StringSource)


def test_pipeline_with_mock_llm():
    """End-to-end: MockLLMSource → DPDA validator."""
    src = MockLLMSource([
        ["(", ")"],                  # ✓
        ["(", "(", ")", ")"],        # ✓
        ["(", ")", ")"],             # ✗
        ["("],                       # ✗
    ])
    dpda = build_dpda(LR1Automaton(balanced_parens()))
    sim = DPDASimulator(dpda)

    results = [(s, sim.accepts(s)) for s in src.emit()]
    assert results[0] == (["(", ")"], True)
    assert results[1] == (["(", "(", ")", ")"], True)
    assert results[2] == (["(", ")", ")"], False)
    assert results[3] == (["("], False)
