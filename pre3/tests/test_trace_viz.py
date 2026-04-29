"""Tests for trace_viz JSON export."""

import pytest

from pre3.tools.trace_viz import build_trace, build_trace_from_tokens


def test_build_trace_schema_balanced_parens():
    t = build_trace("balanced_parens", "(())")
    assert t["format_version"] == 2
    assert t["example"] == "balanced_parens"
    assert t["tokens"] == ["(", "(", ")", ")"]
    assert "grammar" in t and "productions" in t["grammar"]
    assert "npda" in t and isinstance(t["npda"]["steps"], list)
    assert "lr" in t and isinstance(t["lr"]["steps"], list)
    assert "dpda" in t and isinstance(t["dpda"]["steps"], list)
    assert t["npda"]["accepted"] is True
    assert t["lr"]["accepted"] is True
    assert t["dpda"]["accepted"] is True


def test_build_trace_contains_machine_visualization_payloads():
    t = build_trace("balanced_parens", "()")

    assert t["npda"]["automaton"]["states"]
    assert t["npda"]["automaton"]["edges"]
    assert t["npda"]["steps"][0]["phase"] == "epsilon_closure"
    assert "frontier_after" in t["npda"]["steps"][0]

    assert t["lr"]["automaton"]["states"]
    assert t["lr"]["automaton"]["edges"]
    assert "stack_symbols_after" in t["lr"]["steps"][0]

    assert t["dpda"]["automaton"]["states"]
    assert t["dpda"]["automaton"]["edges"]
    assert "valid_next_terminals" in t["dpda"]["steps"][0]


def test_build_trace_reject_case():
    t = build_trace("balanced_parens", "(()")
    assert t["npda"]["accepted"] is False
    assert t["lr"]["accepted"] is False
    assert t["dpda"]["accepted"] is False


def test_build_trace_from_exact_tokens_carries_source():
    t = build_trace_from_tokens(
        "balanced_parens",
        ["(", ")"],
        source={"kind": "llm", "backend": "mock", "prompt_index": 1},
    )
    assert t["source"]["kind"] == "llm"
    assert t["tokens"] == ["(", ")"]
    assert t["lr"]["accepted"] is True
    step = t["dpda"]["steps"][0]
    assert "accepted_symbols" in step
    assert "stack_ops" in step


def test_build_trace_from_exact_tokens_rejects_unknown_terminal():
    with pytest.raises(ValueError):
        build_trace_from_tokens("balanced_parens", ["not-a-terminal"])
