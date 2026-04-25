"""DOT rendering for DPDAs (LR(1)-derived)."""

from __future__ import annotations

from typing import Optional

from ..dpda.builder import DPDA
from ..dpda.edge import EdgeKind, PrefixConditionedEdge


def _edge_label(e: PrefixConditionedEdge) -> str:
    syms = ",".join(sorted(e.accepted_symbols)) if e.accepted_symbols else "ε"
    if e.stack_match:
        match = "/".join(str(s) for s in e.stack_match)
    else:
        match = "*"
    ops = []
    for op in e.stack_ops:
        ops.append(repr(op))
    ops_str = " ".join(ops) if ops else "—"
    kind_short = "A" if e.kind == EdgeKind.ACCEPTANCE else "R"
    return f"[{kind_short}] {syms} | {match}\\n{ops_str}"


def render_dpda(dpda: DPDA, *, title: Optional[str] = None) -> str:
    """Return Graphviz DOT source for the DPDA."""
    lines = ["digraph DPDA {", '  rankdir=LR;', '  node [fontname="monospace"];']
    if title:
        lines.append(f'  labelloc="t"; label="{title}";')

    # States
    for s in range(dpda.num_states):
        shape = "doublecircle" if s in dpda.accepting_states else "circle"
        lines.append(f'  s{s} [shape={shape}, label="q{s}"];')

    # Start arrow
    lines.append('  __start [shape=point, width=0.1];')
    lines.append(f'  __start -> s{dpda.start_state};')

    # Edges
    for e in dpda.edges:
        color = "blue" if e.kind == EdgeKind.ACCEPTANCE else "darkred"
        style = "solid" if e.kind == EdgeKind.ACCEPTANCE else "dashed"
        lines.append(
            f'  s{e.source} -> s{e.target} '
            f'[label="{_edge_label(e)}", color={color}, style={style}, fontsize=9];'
        )

    lines.append("}")
    return "\n".join(lines)


def write_dpda_dot(dpda: DPDA, path: str, *, title: Optional[str] = None) -> str:
    src = render_dpda(dpda, title=title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    return path
