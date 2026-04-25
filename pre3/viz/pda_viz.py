"""DOT rendering for generic NPDAs."""

from __future__ import annotations

from typing import Optional

from ..pda.pda import NPDA, PDATransition


def _label(t: PDATransition) -> str:
    a = t.input_symbol if t.input_symbol else "ε"
    push = "".join(t.stack_push) if t.stack_push else "ε"
    return f"{a}, {t.stack_top}/{push}"


def render_npda(npda: NPDA, *, title: Optional[str] = None) -> str:
    lines = ["digraph NPDA {", '  rankdir=LR;', '  node [fontname="monospace"];']
    if title:
        lines.append(f'  labelloc="t"; label="{title}";')
    for s in sorted(npda.states):
        shape = "doublecircle" if s in npda.accept_states else "circle"
        lines.append(f'  "{s}" [shape={shape}];')
    lines.append('  __start [shape=point, width=0.1];')
    lines.append(f'  __start -> "{npda.start_state}";')
    for t in npda.transitions:
        lines.append(
            f'  "{t.state}" -> "{t.next_state}" '
            f'[label="{_label(t)}", fontsize=9];'
        )
    lines.append("}")
    return "\n".join(lines)


def write_npda_dot(npda: NPDA, path: str, *, title: Optional[str] = None) -> str:
    src = render_npda(npda, title=title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    return path
