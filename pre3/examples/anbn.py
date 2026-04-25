"""
Language:  L = { aⁿbⁿ : n ≥ 0 }
Class:     DCFL  (deterministic context-free)

Grammar (LR(1) — the canonical balanced-counter grammar):
    S → a S b | ε

Pumping lemma:  L is CF (witness pumping length p, decompose any string of
length ≥ p ...). DPDA recognizes by counting a's, then matching b's.
"""

from __future__ import annotations

from typing import List

from ..grammar.cfg import ContextFreeGrammar
from ..grammar.grammar_loader import from_rules
from ..pda.pda import NPDA, PDATransition

description = "L = {aⁿbⁿ : n ≥ 0}.  DCFL.  LR(1) grammar: S → a S b | ε."


def grammar() -> ContextFreeGrammar:
    return from_rules({"S": ["a S b", ""]}, start="S")


def npda() -> NPDA:
    """A direct DPDA-shaped NPDA: push 'A' on every 'a', pop on every 'b'."""
    Z, A = "Z", "A"
    transitions = [
        # On 'a' with anything on top, push A
        PDATransition("q0", "a", Z, "q0", (A, Z)),
        PDATransition("q0", "a", A, "q0", (A, A)),
        # ε-move from q0 to q1 to start matching b's (or finish on empty)
        PDATransition("q0", "", Z, "q_acc", (Z,)),
        PDATransition("q0", "", A, "q1", (A,)),
        # On 'b' with A on top, pop A
        PDATransition("q1", "b", A, "q1", ()),
        # When Z exposed in q1, accept
        PDATransition("q1", "", Z, "q_acc", (Z,)),
    ]
    return NPDA(
        states={"q0", "q1", "q_acc"},
        input_alphabet={"a", "b"},
        stack_alphabet={Z, A},
        transitions=transitions,
        start_state="q0",
        start_stack=Z,
        accept_states={"q_acc"},
    )


positive_examples: List[List[str]] = [
    [],
    ["a", "b"],
    ["a", "a", "b", "b"],
    ["a", "a", "a", "b", "b", "b"],
]

negative_examples: List[List[str]] = [
    ["a"],
    ["b"],
    ["a", "a", "b"],
    ["a", "b", "b"],
    ["b", "a"],
    ["a", "b", "a", "b"],
]
