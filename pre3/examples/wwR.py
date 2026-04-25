"""
Language:  L = { w wᴿ : w ∈ {a,b}* }   — even-length palindromes, NO marker
Class:     CFL,  but **NOT** DCFL.

Significance for the project:
  This is the classical example of a context-free language that is not
  deterministic.  Without the centre marker `c` of `wcwᴿ`, the automaton
  cannot tell where `w` ends and `wᴿ` starts deterministically.

  Therefore:
    • CFG → NPDA construction succeeds (NPDA accepts L).
    • LR(1) (and any LR(k)) construction **must** fail with conflicts.

This is a `negative result` we display in the project: our LR(k) builder
will report a reduce/reduce conflict, exactly as the theory predicts.

Grammar:
    S → a S a | b S b | ε
"""

from __future__ import annotations

from typing import List

from ..grammar.cfg import ContextFreeGrammar
from ..grammar.grammar_loader import from_rules
from ..pda.pda import NPDA, PDATransition

description = (
    "L = {w wᴿ : w ∈ {a,b}*}.  CFL but NOT DCFL.  "
    "LR(k) construction is expected to FAIL for all k ≥ 0."
)


def grammar() -> ContextFreeGrammar:
    return from_rules({"S": ["a S a", "b S b", ""]}, start="S")


def npda() -> NPDA:
    """Two-phase NPDA: non-deterministically guess where w ends.

    Phase q_push: push the input onto the stack.
    Phase q_pop:  pop and match against subsequent input.
    The non-determinism is the ε-transition q_push → q_pop, fired at any
    point — the automaton 'guesses' the centre.
    """
    Z, A, B = "Z", "A", "B"
    transitions = [
        PDATransition("q_push", "a", Z, "q_push", (A, Z)),
        PDATransition("q_push", "a", A, "q_push", (A, A)),
        PDATransition("q_push", "a", B, "q_push", (A, B)),
        PDATransition("q_push", "b", Z, "q_push", (B, Z)),
        PDATransition("q_push", "b", A, "q_push", (B, A)),
        PDATransition("q_push", "b", B, "q_push", (B, B)),
        # Non-deterministic: guess the midpoint.
        PDATransition("q_push", "", Z, "q_pop", (Z,)),
        PDATransition("q_push", "", A, "q_pop", (A,)),
        PDATransition("q_push", "", B, "q_pop", (B,)),
        # Pop phase
        PDATransition("q_pop", "a", A, "q_pop", ()),
        PDATransition("q_pop", "b", B, "q_pop", ()),
        # Accept on bottom marker
        PDATransition("q_pop", "", Z, "q_acc", (Z,)),
    ]
    return NPDA(
        states={"q_push", "q_pop", "q_acc"},
        input_alphabet={"a", "b"},
        stack_alphabet={Z, A, B},
        transitions=transitions,
        start_state="q_push",
        start_stack=Z,
        accept_states={"q_acc"},
    )


positive_examples: List[List[str]] = [
    [],
    ["a", "a"],
    ["b", "b"],
    ["a", "b", "b", "a"],
    ["a", "a", "b", "b", "a", "a"],   # ← would be 'aab|baa' midpoint guess
    ["b", "a", "b", "b", "a", "b"],
]

negative_examples: List[List[str]] = [
    ["a"],
    ["a", "b"],
    ["a", "b", "a"],
    ["a", "a", "b"],
    ["a", "b", "b", "b"],
]
