"""
Language:  Dyck-2  =  the language of strings of properly nested matching
           pairs over { (, ), [, ] }.
Class:     DCFL.

Grammar (UNAMBIGUOUS form — the naive "S → S S" version is ambiguous):
    S → ( S ) S
    S → [ S ] S
    S → ε

This is the standard right-recursive disambiguation that yields LR(1).
"""

from __future__ import annotations

from typing import List

from ..grammar.cfg import ContextFreeGrammar
from ..grammar.grammar_loader import from_rules

description = "Dyck-2 (nested matching parens & brackets).  DCFL.  LR(1)."


def grammar() -> ContextFreeGrammar:
    return from_rules({"S": ["( S ) S", "[ S ] S", ""]}, start="S")


positive_examples: List[List[str]] = [
    [],
    ["(", ")"],
    ["[", "]"],
    ["(", ")", "[", "]"],
    ["(", "[", "]", ")"],
    ["(", "(", ")", ")"],
    ["[", "(", ")", "[", "]", "]"],
]

negative_examples: List[List[str]] = [
    ["("],
    [")"],
    ["(", "]"],
    ["[", ")"],
    ["(", "(", ")"],
    ["[", "[", "(", "]", ")", "]"],   # ← mismatched nesting
]
