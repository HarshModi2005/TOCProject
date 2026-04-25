"""
Standard textbook construction:  CFG  →  one-state NPDA  (empty-stack accept).

Theorem (Chomsky 1962).  For every context-free grammar G there is a
non-deterministic pushdown automaton N with L(N) = L(G).

Construction (Sipser, Theorem 2.20):
  • States:           { q }                 (one state)
  • Input alphabet:   Σ_G  (terminals of G)
  • Stack alphabet:   V_G ∪ Σ_G ∪ { Z }     (V = non-terminals, Z = bottom marker)
  • Start state:      q
  • Start stack:      Z
  • Transitions:
      - For each production A → α in G:        δ(q, ε, A) ∋ (q, α)
      - For each terminal a:                   δ(q, a, a) ∋ (q, ε)
      - Initial:                               δ(q, ε, Z) ∋ (q, S Z)
  • Accept by EMPTY STACK after consuming all input; we encode this by
    popping the bottom marker Z via δ(q, ε, Z) ∋ (q, ε) at the very end.

This NPDA non-deterministically simulates a leftmost derivation of G.
The empty-stack acceptance is the natural mode for this construction.

For determinism we DO NOT use this NPDA — that's what the LR(1)→DPDA
pipeline in `pre3/dpda/` is for.  This module is for theoretical
completeness (CFL ≡ L(NPDA) shown by construction).
"""

from __future__ import annotations

from typing import List

from ..grammar.cfg import ContextFreeGrammar
from .pda import NPDA, PDATransition


# Sentinel stack-bottom marker (must not collide with grammar symbols).
BOTTOM = "⊥"


def cfg_to_npda(grammar: ContextFreeGrammar) -> NPDA:
    """Build a one-state NPDA recognizing L(grammar) by empty-stack acceptance.

    We allow the resulting NPDA to also be queried in `final_state` mode
    by adding a sole accepting state `q_acc` reached when the bottom
    marker is exposed.
    """
    q = "q"
    q_start = "q_start"
    q_acc = "q_acc"

    # Sanity: BOTTOM must be unique
    if BOTTOM in grammar.terminals or BOTTOM in grammar.non_terminals:
        raise ValueError(
            f"Reserved bottom symbol {BOTTOM!r} clashes with the grammar; "
            "rename a symbol or change BOTTOM."
        )

    states = {q_start, q, q_acc}
    input_alphabet = set(grammar.terminals)
    stack_alphabet = set(grammar.terminals) | set(grammar.non_terminals) | {BOTTOM}
    transitions: List[PDATransition] = []

    # 1.  Initial: push S onto the stack and move to the main state q.
    #     This is the ONLY way to leave q_start; we never re-enter q_start.
    transitions.append(PDATransition(
        state=q_start, input_symbol="", stack_top=BOTTOM,
        next_state=q, stack_push=(grammar.start, BOTTOM),
    ))

    # 2.  For each production A → α, allow expanding A on top of stack.
    for prod in grammar.productions:
        push = tuple(prod.body) if prod.body else ()
        transitions.append(PDATransition(
            state=q, input_symbol="", stack_top=prod.head,
            next_state=q, stack_push=push,
        ))

    # 3.  For each terminal a, match it by popping.
    for a in grammar.terminals:
        transitions.append(PDATransition(
            state=q, input_symbol=a, stack_top=a,
            next_state=q, stack_push=(),
        ))

    # 4.  When bottom is exposed at q (= the parse derived all of S), accept.
    #     Note: this only fires AFTER the initial expansion has completed,
    #     since at q_start the stack is [BOTTOM] but we only have transitions
    #     from q_start that push S.  Therefore the empty input is accepted iff
    #     S =>* ε.
    transitions.append(PDATransition(
        state=q, input_symbol="", stack_top=BOTTOM,
        next_state=q_acc, stack_push=(BOTTOM,),
    ))

    return NPDA(
        states=states,
        input_alphabet=input_alphabet,
        stack_alphabet=stack_alphabet,
        transitions=transitions,
        start_state=q_start,
        start_stack=BOTTOM,
        accept_states={q_acc},
    )
