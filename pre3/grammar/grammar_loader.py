"""
Utilities to build a ContextFreeGrammar from convenient shorthand or from
simple EBNF-ish text.

Supports two entry points:
  1. from_rules()  – Python dict  { "S": ["a B", ""] }
  2. from_ebnf()   – multi-line text  S ::= a B | ε
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .cfg import EPSILON, ContextFreeGrammar, Production


def from_rules(
    rules: Dict[str, List[str]],
    start: str,
    terminals: Optional[set[str]] = None,
) -> ContextFreeGrammar:
    """
    Build a CFG from a dict mapping non-terminal names to lists of
    right-hand-side strings (space-separated symbols).

    Example::

        grammar = from_rules({
            "S": ["( S )", ""],
        }, start="S")

    An empty string ``""`` means an ε-production.
    If *terminals* is None it is inferred as every symbol that never appears
    on the left-hand side of a rule.
    """
    non_terminals: set[str] = set(rules.keys())
    productions: list[Production] = []
    all_symbols: set[str] = set()

    for head, alternatives in rules.items():
        for alt in alternatives:
            body = tuple(alt.split()) if alt.strip() else ()
            productions.append(Production(head, body))
            all_symbols.update(body)

    if terminals is None:
        terminals = (all_symbols - non_terminals) - {EPSILON}

    return ContextFreeGrammar(
        terminals=terminals,
        non_terminals=non_terminals,
        productions=productions,
        start=start,
    )


_RULE_RE = re.compile(r"^\s*(\S+)\s*::=\s*(.+)$")


def from_ebnf(text: str, start: Optional[str] = None) -> ContextFreeGrammar:
    """
    Parse a simple BNF/EBNF text where each line is::

        NonTerminal ::= alt1 | alt2 | ...

    Terminals are inferred.  ``ε`` or ``epsilon`` denotes the empty production.
    The first rule's head is used as start if *start* is not given.
    """
    rules: Dict[str, List[str]] = {}
    first_nt: Optional[str] = None

    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _RULE_RE.match(line)
        if not m:
            raise ValueError(f"Cannot parse grammar line: {line!r}")
        head = m.group(1)
        if first_nt is None:
            first_nt = head
        alternatives = [a.strip() for a in m.group(2).split("|")]
        cleaned: list[str] = []
        for alt in alternatives:
            alt = alt.replace("epsilon", EPSILON)
            if alt == EPSILON:
                cleaned.append("")
            else:
                cleaned.append(alt)
        rules.setdefault(head, []).extend(cleaned)

    if start is None:
        start = first_nt
    if start is None:
        raise ValueError("No rules found in grammar text")

    return from_rules(rules, start=start)


# ------------------------------------------------------------------
# Built-in grammars useful for testing
# ------------------------------------------------------------------


def balanced_parens() -> ContextFreeGrammar:
    """S → ( S ) | ε"""
    return from_rules({"S": ["( S )", ""]}, start="S")


def simple_json() -> ContextFreeGrammar:
    """
    A tiny JSON-like grammar (not full JSON, but exercises nested
    structures, lists, key-value pairs).

    value   ::= object | array | STRING | NUMBER | true | false | null
    object  ::= { members }  |  { }
    members ::= pair , members | pair
    pair    ::= STRING : value
    array   ::= [ elements ]  |  [ ]
    elements::= value , elements | value
    """
    return from_rules(
        {
            "Value": [
                "Object",
                "Array",
                "STRING",
                "NUMBER",
                "true",
                "false",
                "null",
            ],
            "Object": ["{ Members }", "{ }"],
            "Members": ["Pair , Members", "Pair"],
            "Pair": ["STRING : Value"],
            "Array": ["[ Elements ]", "[ ]"],
            "Elements": ["Value , Elements", "Value"],
        },
        start="Value",
    )


def arithmetic() -> ContextFreeGrammar:
    """E → E + T | T ;  T → T * F | F ;  F → ( E ) | id"""
    return from_rules(
        {
            "E": ["E + T", "T"],
            "T": ["T * F", "F"],
            "F": ["( E )", "id"],
        },
        start="E",
    )
