"""
A small CLI / library tool that prints a step-by-step shift/reduce
trace of any of the canonical examples through both the LR(1) parser
*and* the DPDA simulator, side-by-side.

This is the "viva-friendly" demo:  show the stack growing, reductions
firing, and the final accept/reject verdict in a clean table.

Usage:
    python -m pre3.tools.trace --example balanced_parens --input '(())'
    python -m pre3.tools.trace --example wcwR          --input 'abcba'
    python -m pre3.tools.trace --example anbn          --input 'aaabbb'

The same module exposes `trace_lr1(grammar, input)` and
`trace_dpda(grammar, input)` for programmatic use.
"""

from __future__ import annotations

import argparse
from typing import List, Sequence, Tuple

from ..dpda.builder import build_dpda
from ..dpda.edge import EdgeKind, PrefixConditionedEdge
from ..dpda.simulator import DPDASimulator
from ..grammar.cfg import END_MARKER, ContextFreeGrammar
from ..grammar.lr1 import LR1Automaton
from ..grammar.lrk import ActionType


# ======================================================================
# Tokenization helper
# ======================================================================


def tokenize(input_str: str, terminals: Sequence[str]) -> List[str]:
    """Greedy longest-match tokenizer using the grammar's own terminals."""
    sorted_terminals = sorted(terminals, key=len, reverse=True)
    tokens: List[str] = []
    i = 0
    s = input_str
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        for t in sorted_terminals:
            if s.startswith(t, i):
                tokens.append(t)
                i += len(t)
                break
        else:
            raise ValueError(f"Cannot tokenize at position {i}: {s[i:]!r}")
    return tokens


# ======================================================================
# LR(1) shift/reduce trace
# ======================================================================


def trace_lr1(grammar: ContextFreeGrammar, tokens: List[str]) -> Tuple[bool, List[List[str]]]:
    """Return (accepted, table_rows).  Each row: [step, stack, input, action]."""
    lr1 = LR1Automaton(grammar)
    stack: List[int] = [0]
    pos = 0
    extended = list(tokens) + [END_MARKER]
    rows: List[List[str]] = []
    step = 0

    while True:
        state = stack[-1]
        sym = extended[pos]
        action = lr1.action_table.get((state, sym))
        stack_str = " ".join(str(s) for s in stack)
        input_str = " ".join(extended[pos:])
        if action is None:
            rows.append([str(step), stack_str, input_str, "ERROR (no action)"])
            return False, rows
        if action.kind == ActionType.SHIFT:
            rows.append([str(step), stack_str, input_str, f"shift  {sym!r} → state {action.state}"])
            stack.append(action.state)
            pos += 1
        elif action.kind == ActionType.REDUCE:
            rows.append([str(step), stack_str, input_str, f"reduce {action.production}"])
            for _ in range(len(action.production.body)):
                stack.pop()
            exposed = stack[-1]
            goto = lr1.goto_table.get((exposed, action.production.head))
            if goto is None:
                rows.append([str(step + 1), stack_str, input_str, "ERROR (no goto)"])
                return False, rows
            stack.append(goto)
        elif action.kind == ActionType.ACCEPT:
            rows.append([str(step), stack_str, input_str, "ACCEPT ✓"])
            return True, rows
        else:
            rows.append([str(step), stack_str, input_str, "ERROR"])
            return False, rows
        step += 1


# ======================================================================
# DPDA edge-by-edge trace
# ======================================================================


def trace_dpda(grammar: ContextFreeGrammar, tokens: List[str]) -> Tuple[bool, List[List[str]]]:
    """Same shape of return as `trace_lr1`."""
    lr1 = LR1Automaton(grammar)
    dpda = build_dpda(lr1)
    sim = DPDASimulator(dpda)
    config = sim.initial_config()
    rows: List[List[str]] = []
    step = 0

    def fmt_edge(e: PrefixConditionedEdge) -> str:
        kind = "ACC" if e.kind == EdgeKind.ACCEPTANCE else "RED"
        syms = ",".join(sorted(e.accepted_symbols)) if e.accepted_symbols else "ε"
        return f"[{kind}] {syms}  →  state {e.target}"

    def add_row(sym: str, edge_or_msg, conf):
        stack_str = " ".join(str(s) for s in conf.stack)
        rows.append([str(step), stack_str, sym, edge_or_msg])

    extended = list(tokens) + [END_MARKER]
    consume_pos = 0

    chain_cap = dpda.num_states + 8
    while consume_pos <= len(tokens):
        sym = extended[consume_pos]
        # Chase reductions then optionally one acceptance.
        progressed = False
        for _ in range(chain_cap):
            edge = sim._matching_edge(config.state, config.stack, sym)
            if edge is None:
                break
            new = config.clone()
            new.stack = edge.apply_stack_ops(new.stack)
            new.state = edge.target
            if edge.kind == EdgeKind.ACCEPTANCE:
                new.consumed += 1
                add_row(sym, fmt_edge(edge), new)
                config = new
                consume_pos += 1
                step += 1
                progressed = True
                break
            else:
                add_row(sym + " (peek)", fmt_edge(edge), new)
                config = new
                step += 1
                progressed = True
        if not progressed:
            if consume_pos == len(tokens):
                if config.state in dpda.accepting_states:
                    add_row("$", "ACCEPT ✓", config)
                    return True, rows
                add_row("$", "REJECT (final state not accepting)", config)
                return False, rows
            add_row(sym, "REJECT (no applicable edge)", config)
            return False, rows
        if consume_pos == len(tokens) and config.state in dpda.accepting_states:
            add_row("$", "ACCEPT ✓", config)
            return True, rows

    return config.state in dpda.accepting_states, rows


# ======================================================================
# Pretty-print
# ======================================================================


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    """Plain-text aligned table."""
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    line = lambda r: "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |"
    out = [sep, line(headers), sep]
    for r in rows:
        out.append(line(r))
    out.append(sep)
    return "\n".join(out)


# ======================================================================
# CLI
# ======================================================================


EXAMPLES = {
    "balanced_parens": ("pre3.examples.anbn", "pre3.grammar.grammar_loader", "balanced_parens"),
}


def _load_example_grammar(name: str) -> ContextFreeGrammar:
    if name == "balanced_parens":
        from ..grammar.grammar_loader import balanced_parens
        return balanced_parens()
    if name == "arithmetic":
        from ..grammar.grammar_loader import arithmetic
        return arithmetic()
    if name == "anbn":
        from ..examples.anbn import grammar
        return grammar()
    if name == "wcwR":
        from ..examples.wcwR import grammar
        return grammar()
    if name == "dyck2":
        from ..examples.dyck2 import grammar
        return grammar()
    raise SystemExit(f"unknown example: {name}")


def main() -> None:
    p = argparse.ArgumentParser(description="Step-trace LR(1) parser and DPDA simulator")
    p.add_argument("--example", required=True,
                   choices=["balanced_parens", "arithmetic", "anbn", "wcwR", "dyck2"])
    p.add_argument("--input", required=True, help="raw string; whitespace-insensitive")
    p.add_argument("--mode", choices=["lr1", "dpda", "both"], default="both")
    args = p.parse_args()

    grammar = _load_example_grammar(args.example)
    tokens = tokenize(args.input, grammar.terminals)
    print(f"Tokens: {tokens}")

    if args.mode in ("lr1", "both"):
        ok, rows = trace_lr1(grammar, tokens)
        print(f"\n=== LR(1) trace  →  {'ACCEPTED' if ok else 'REJECTED'} ===")
        print(render_table(["#", "stack", "input", "action"], rows))
    if args.mode in ("dpda", "both"):
        ok, rows = trace_dpda(grammar, tokens)
        print(f"\n=== DPDA trace  →  {'ACCEPTED' if ok else 'REJECTED'} ===")
        print(render_table(["#", "stack", "lookahead", "edge"], rows))


if __name__ == "__main__":
    main()
