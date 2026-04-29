"""
Difficult / long / nested cases:  mocked LLM JSON  →  filter  →  DPDA vs LR(1) oracle.

No network.  Every string is also checked with ``LR1Simulator`` (ground truth).
"""

from __future__ import annotations

import json
from typing import List

from unittest.mock import patch

from pre3.adapter.api_llm import OpenAILLMSource
from pre3.grammar.grammar_loader import (
    arithmetic,
    balanced_parens,
    from_rules,
)
from pre3.grammar.lr1 import LR1Automaton
from pre3.dpda.builder import build_dpda
from pre3.dpda.simulator import DPDASimulator, LR1Simulator
from pre3.dpda.verifier import verify_determinism


def _oracle_agrees(g, tokens: List[str]) -> bool:
    lr1 = LR1Automaton(g)
    lsim = LR1Simulator(lr1)
    dpda = build_dpda(lr1)
    verify_determinism(dpda)
    dsim = DPDASimulator(dpda)
    a, b = lsim.accepts(tokens), dsim.accepts(tokens)
    assert a is b, f"LR1={a} DPDA={b} on {tokens!r}"
    return a


def _emit_mocked(grammar, payloads: list[list[str]]):
    """Build JSON responses and yield tokens from OpenAILLMSource with mocked HTTP."""
    fakes = [json.dumps({"tokens": p}) for p in payloads]
    src = OpenAILLMSource(
        grammar=grammar,
        user_messages=[f"p{i}" for i in range(len(payloads))],
        api_key="sk-test",
    )
    with patch("pre3.adapter.api_llm.chat_completion", side_effect=fakes):
        return list(src.emit())


def test_deeply_nested_parens_40_pairs():
    n = 40
    good = ["("] * n + [")"] * n
    bad = ["("] * n + [")"] * (n - 1)
    g = balanced_parens()
    assert _oracle_agrees(g, good) is True
    assert _oracle_agrees(g, bad) is False
    toks0, toks1 = _emit_mocked(g, [good, bad])
    assert toks0 == good
    assert toks1 == bad


def test_dyck2_valid_nested_and_unbalanced():
    g = from_rules(
        {"S": ["( S ) S", "[ S ] S", ""]},
        start="S",
    )
    good = list("([()])")  # mixed brackets, well-formed
    bad = list("([)]")  # bracket mismatch
    assert _oracle_agrees(g, good) is True
    assert _oracle_agrees(g, bad) is False
    t0, t1 = _emit_mocked(g, [good, bad])
    assert t0 == good
    assert t1 == bad


def test_arithmetic_long_mixed_precedence():
    g = arithmetic()
    # ( id + id * ( id + id ) ) * id
    good = [
        "(", "id", "+", "id", "*", "(", "id", "+", "id", ")", ")", "*", "id",
    ]
    bad = ["id", "+", "*", "id"]  # cannot reduce
    assert _oracle_agrees(g, good) is True
    assert _oracle_agrees(g, bad) is False
    t0, t1 = _emit_mocked(g, [good, bad])
    assert t0 == good
    assert t1 == bad


def test_dangling_else_style_nested_if():
    g = from_rules(
        {"S": ["if e then S else S", "if e then S", "x"]},
        start="S",
    )
    # inner else binds to inner if:  if e then if e then x else x
    good = "if e then if e then x else x".split()
    bad = "if e then x else x else x".split()  # extra else — invalid
    assert _oracle_agrees(g, good) is True
    assert _oracle_agrees(g, bad) is False
    t0, t1 = _emit_mocked(g, [good, bad])
    assert t0 == good
    assert t1 == bad


def test_wcwR_palindrome_center_marker():
    g = from_rules(
        {"S": ["a S a", "b S b", "c"]},
        start="S",
    )
    good = ["a", "b", "a", "c", "a", "b", "a"]
    # After c, must be w^R: w=ab ⇒ need b a, not a b
    bad = ["a", "b", "c", "a", "b"]
    assert _oracle_agrees(g, good) is True
    assert _oracle_agrees(g, bad) is False
    t0, t1 = _emit_mocked(g, [good, bad])
    assert t0 == good
    assert t1 == bad


def test_benign_middleware_noise_still_respects_oracle():
    """Garbage tokens dropped; remaining string checked vs LR1/DPDA."""
    g = balanced_parens()
    # Model sneaks in prose + spaces as fake tokens: only ( ) survive filter.
    dirty = [
        "noise", "(", "ALSO", "(", ")", "bad", ")", "x",
    ]
    # Filter keeps ( ( ) ) — valid
    t = _emit_mocked(g, [dirty])[0]
    assert t == ["(", "(", ")", ")"]
    assert _oracle_agrees(g, t) is True


def test_challenge_batch_all_emissions_match_oracle():
    """Eight per-grammar emissions: each payload goes through the mocked LLM + filter + oracle."""
    if_else = from_rules(
        {"S": ["if e then S else S", "if e then S", "x"]},
        start="S",
    )
    anbn = from_rules({"S": ["a S b", ""]}, start="S")
    dy = from_rules(
        {"S": ["( S ) S", "[ S ] S", ""]},
        start="S",
    )
    cases: list = [
        (balanced_parens(), ["(", "(", "(", ")", ")", ")", ")", "("]),
        (balanced_parens(), []),
        (anbn, ["a", "a", "a", "b", "b", "b"]),
        (anbn, ["a", "a", "b"]),
        (arithmetic(), ["id"]),
        (if_else, "if e then if e then x else x".split()),
        (dy, list("()[]")),
        (dy, list("([)]")),
    ]
    for gram, toks in cases:
        _oracle_agrees(gram, toks)
    fakes = [json.dumps({"tokens": p}) for _g, p in cases]
    for i, (gram, want) in enumerate(cases):
        src = OpenAILLMSource(
            grammar=gram,
            user_messages=[f"case{i}"],
            api_key="sk-test",
        )
        with patch("pre3.adapter.api_llm.chat_completion", return_value=fakes[i]):
            got = next(src.emit())
        assert got == want
        _oracle_agrees(gram, got)