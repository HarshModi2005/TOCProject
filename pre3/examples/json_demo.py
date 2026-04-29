"""
Language:  Simplified JSON
Class:     DCFL.

Grammar for a subset of JSON (Object, Array, Primitives) using abstract tokens.
This demonstrates how structured data formats like JSON can be constrained using DPDA.
"""

from __future__ import annotations

from typing import List

from ..grammar.cfg import ContextFreeGrammar
from ..grammar.grammar_loader import from_rules

description = "Simplified JSON (Objects, Arrays, Primitives). DCFL. LR(1)."


def grammar() -> ContextFreeGrammar:
    # Note: Tokens like "str", "num", "true", "false", "null" are treated as abstract terminals.
    return from_rules({
        "VALUE": [
            "str",
            "num",
            "true",
            "false",
            "null",
            "OBJ",
            "ARR"
        ],
        "OBJ": [
            "{ }",
            "{ MEMBERS }"
        ],
        "MEMBERS": [
            "MEMBER",
            "MEMBER , MEMBERS"
        ],
        "MEMBER": [
            "str : VALUE"
        ],
        "ARR": [
            "[ ]",
            "[ ELEMENTS ]"
        ],
        "ELEMENTS": [
            "VALUE",
            "VALUE , ELEMENTS"
        ]
    }, start="VALUE")


positive_examples: List[List[str]] = [
    ["str"],
    ["{", "}"],
    ["[", "]"],
    ["{", "str", ":", "num", "}"],
    ["[", "true", ",", "false", "]"],
    ["{", "str", ":", "[", "num", ",", "null", "]", "}"],
    ["{", "str", ":", "{", "}", ",", "str", ":", "str", "}"],
]

negative_examples: List[List[str]] = [
    ["{"],
    ["{", "str", "}"],  # Missing colon and value
    ["[", "num", ","],  # Trailing comma
    ["{", "str", ":", "num", ",", "}"], # Trailing comma
    ["[", "]", "num"], # Multiple root values
]
