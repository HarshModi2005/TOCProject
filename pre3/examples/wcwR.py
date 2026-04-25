"""
Language:  L = { w c wᴿ : w ∈ {a,b}* }
Class:     DCFL  (deterministic context-free) — the centre marker `c`
           tells us where to switch from "push" to "pop".

Grammar (LR(1)):
    S → a S a | b S b | c
"""

from __future__ import annotations

from typing import List

from ..grammar.cfg import ContextFreeGrammar
from ..grammar.grammar_loader import from_rules

description = "L = {w c wᴿ : w ∈ {a,b}*}.  DCFL.  LR(1) grammar: S → a S a | b S b | c."


def grammar() -> ContextFreeGrammar:
    return from_rules({"S": ["a S a", "b S b", "c"]}, start="S")


positive_examples: List[List[str]] = [
    ["c"],
    ["a", "c", "a"],
    ["b", "c", "b"],
    ["a", "b", "c", "b", "a"],
    ["b", "a", "a", "c", "a", "a", "b"],
]

negative_examples: List[List[str]] = [
    [],
    ["a"], ["b"],
    ["c", "c"],
    ["a", "c", "b"],
    ["a", "b", "c", "a", "b"],   # not a palindrome around c
]
