"""Tests for OpenAI adapter (no network: mock ``chat_completion``)."""

import json
from unittest.mock import patch

from pre3.grammar.grammar_loader import from_rules, balanced_parens
from pre3.grammar.lr1 import LR1Automaton
from pre3.dpda.builder import build_dpda
from pre3.dpda.simulator import DPDASimulator
from pre3.adapter.api_llm import (
    OpenAILLMSource,
    _filter_vocabulary,
    _parse_token_payload,
)

from pre3.adapter.string_source import StringSource


def test_parse_token_payload():
    s = json.dumps({"tokens": ["(", ")"]})
    assert _parse_token_payload(s) == ["(", ")"]


def test_filter_vocabulary():
    assert _filter_vocabulary(["a", "x", "b"], {"a", "b"}) == ["a", "b"]


def test_openai_llm_source_emit_mocked():
    g = balanced_parens()
    src = OpenAILLMSource(
        grammar=g,
        user_messages=["p1", "p2"],
        api_key="sk-test",
    )

    fake1 = json.dumps({"tokens": ["(", ")"]})
    fake2 = json.dumps({"tokens": [")", "("]})

    with patch("pre3.adapter.api_llm.chat_completion", side_effect=[fake1, fake2]) as m:
        emissions = list(src.emit())
    assert m.call_count == 2
    assert emissions[0] == ["(", ")"]
    assert emissions[1] == [")", "("]


def test_openai_protocol_runtime_checkable():
    g = from_rules({"S": ["a"]}, start="S")
    s = OpenAILLMSource(grammar=g, user_messages=["x"], api_key="k")
    assert isinstance(s, StringSource)


def test_e2e_llm_source_to_dpda_mocked():
    g = balanced_parens()
    dpda = build_dpda(LR1Automaton(g))
    sim = DPDASimulator(dpda)
    src = OpenAILLMSource(
        grammar=g,
        user_messages=["one shot"],
        api_key="sk-test",
    )
    fake = json.dumps({"tokens": ["(", "(", ")", ")"]})
    with patch("pre3.adapter.api_llm.chat_completion", return_value=fake):
        toks = next(src.emit())
    assert sim.accepts(toks) is True
