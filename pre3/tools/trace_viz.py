"""
Export rich step-by-step execution traces for the HTML viewer.

Schema v2 (this file emits ``format_version: 2``):

* Adds an ``npda`` block containing the CFG-derived NPDA's state machine,
  its bespoke transitions, and a *frontier* trace (set of live
  configurations after every input symbol — the visual proof of
  non-determinism).
* Adds ``automaton`` sub-blocks (states + edges + LR(1) item sets) so
  the viewer can draw real state-machine graphs, not just step lists.
* Adds ``valid_next_terminals`` to every DPDA step (the actual
  *Pre³ grammar mask* — the paper's contribution made visible).
* LR(1) steps now carry a parallel ``stack_symbols`` (a, b, S, …)
  alongside state IDs and a ``production_index`` for highlighting the
  rule being reduced.

Old (v1) consumers that only read ``tokens``, ``grammar``, ``lr.steps``
and ``dpda.steps`` keep working.

Usage::

    python -m pre3.tools.trace_viz --example balanced_parens \
        --input "(())" --out trace.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..dpda.builder import DPDA, build_dpda
from ..dpda.edge import EdgeKind, PrefixConditionedEdge
from ..dpda.simulator import DPDASimulator
from ..grammar.cfg import END_MARKER, ContextFreeGrammar, Production
from ..grammar.lr1 import ActionType, LR1Automaton, LR1State
from ..pda.cfg_to_pda import cfg_to_npda
from ..pda.pda import NPDA, PDATransition
from .trace import _load_example_grammar, tokenize


FORMAT_VERSION = 2


# ----------------------------------------------------------------------
# Step records (dataclasses are fine for backward compat)
# ----------------------------------------------------------------------


@dataclass
class LRStep:
    step: int
    state_before: int
    stack_before: List[int]
    stack_symbols_before: List[str]
    lookahead: str
    action_kind: str
    action_text: str
    production_index: Optional[int]
    valid_actions: List[str]
    state_after: int
    stack_after: List[int]
    stack_symbols_after: List[str]
    input_pos_before: int
    input_pos_after: int


@dataclass
class DPDAStep:
    step: int
    state_before: int
    stack_before: List[int]
    lookahead: str
    edge_kind: str
    edge_text: str
    state_after: int
    stack_after: List[int]
    consumed_before: int
    consumed_after: int
    accepted_symbols: List[str]
    stack_match: List[int]
    stack_ops: List[str]
    consumes_input: bool
    valid_next_terminals: List[str]


# ----------------------------------------------------------------------
# Grammar serialization
# ----------------------------------------------------------------------


def _grammar_meta(g: ContextFreeGrammar) -> Dict[str, Any]:
    productions = [
        {"index": i, "head": p.head, "body": list(p.body)}
        for i, p in enumerate(g.productions)
    ]
    return {
        "start": g.start,
        "terminals": sorted(g.terminals),
        "non_terminals": sorted(g.non_terminals),
        "productions": [repr(p) for p in g.productions],
        "production_data": productions,
    }


def _production_index(grammar: ContextFreeGrammar, prod: Production) -> Optional[int]:
    """
    Find the original (un-augmented) index of a production.  LR(1) uses an
    augmented grammar (S' → S at index 0); we want indices into the
    user-visible grammar, so we match by (head, body).
    """
    for i, p in enumerate(grammar.productions):
        if p.head == prod.head and p.body == prod.body:
            return i
    return None


# ----------------------------------------------------------------------
# LR(1) — automaton + execution trace
# ----------------------------------------------------------------------


def _lr_automaton_graph(lr1: LR1Automaton) -> Dict[str, Any]:
    """Serialize the LR(1) automaton as a node/edge graph plus item sets."""
    states: List[Dict[str, Any]] = []
    aug_start = lr1.grammar.start
    for s in lr1.states:
        items_repr = sorted([repr(it) for it in s.items])
        is_accepting = any(
            (it.production.head == aug_start and it.is_reduce and it.lookahead == END_MARKER)
            for it in s.items
        )
        states.append({
            "id": s.id,
            "items": items_repr,
            "is_start": s.id == 0,
            "is_accepting": is_accepting,
        })

    edges: List[Dict[str, Any]] = []
    for src, by_sym in lr1.transitions.items():
        for sym, dst in by_sym.items():
            edges.append({
                "src": src,
                "dst": dst,
                "label": sym,
                "kind": "shift" if lr1.grammar.is_terminal(sym) else "goto",
            })
    return {"states": states, "edges": edges}


def _lr_valid_actions(lr1: LR1Automaton, state: int) -> List[str]:
    """All terminal symbols that have a non-error ACTION in this state."""
    out: List[str] = []
    for (sid, sym), _ in lr1.action_table.items():
        if sid == state:
            out.append(sym)
    return sorted(set(out))


def _detect_lr_conflicts(lr1: LR1Automaton) -> List[Dict[str, Any]]:
    """
    Find shift/reduce and reduce/reduce conflicts that ``LR1Automaton``
    silently resolves (it prefers shift / first-rule).

    For every state, scan its items for the classic patterns:
      • SHIFT(a) coexists with [A → α ·, a]   ← shift/reduce on `a`
      • [A → α ·, a] coexists with [B → β ·, a]   ← reduce/reduce on `a`

    These are the demo's smoking gun for non-LR(1) grammars (wwᴿ etc.).
    """
    conflicts: List[Dict[str, Any]] = []
    for state in lr1.states:
        reduces_by_la: Dict[str, List[str]] = {}
        shifts: set = set()
        for it in state.items:
            if it.is_reduce:
                reduces_by_la.setdefault(it.lookahead, []).append(repr(it.production))
            else:
                d = it.at_dot
                if d is not None and lr1.grammar.is_terminal(d):
                    shifts.add(d)
        for la, prods in reduces_by_la.items():
            if la in shifts:
                conflicts.append({
                    "state": state.id,
                    "kind": "shift/reduce",
                    "lookahead": la,
                    "rules": prods,
                    "message": (
                        f"State {state.id}: shift/reduce on {la!r} — "
                        f"silently resolved as shift; reductions skipped: {prods}"
                    ),
                })
            if len(prods) > 1:
                conflicts.append({
                    "state": state.id,
                    "kind": "reduce/reduce",
                    "lookahead": la,
                    "rules": prods,
                    "message": (
                        f"State {state.id}: reduce/reduce on {la!r} between {prods}"
                    ),
                })
    return conflicts


def _trace_lr(
    lr1: LR1Automaton,
    user_grammar: ContextFreeGrammar,
    tokens: List[str],
) -> Tuple[List[LRStep], bool, str]:
    state_stack: List[int] = [0]
    sym_stack: List[str] = []
    pos = 0
    ext = list(tokens) + [END_MARKER]
    steps: List[LRStep] = []
    step_i = 0
    accepted = False
    reason = ""

    while True:
        state = state_stack[-1]
        lookahead = ext[pos]
        action = lr1.action_table.get((state, lookahead))
        valid_acts = _lr_valid_actions(lr1, state)

        if action is None:
            reason = f"No action for state={state}, symbol={lookahead!r}"
            steps.append(LRStep(
                step=step_i,
                state_before=state, stack_before=list(state_stack),
                stack_symbols_before=list(sym_stack),
                lookahead=lookahead,
                action_kind="ERROR", action_text=reason,
                production_index=None, valid_actions=valid_acts,
                state_after=state, stack_after=list(state_stack),
                stack_symbols_after=list(sym_stack),
                input_pos_before=pos, input_pos_after=pos,
            ))
            break

        before_states = list(state_stack)
        before_syms = list(sym_stack)
        before_pos = pos
        action_kind = action.kind.name
        action_text = ""
        prod_idx: Optional[int] = None

        if action.kind == ActionType.SHIFT:
            state_stack.append(action.state)
            sym_stack.append(lookahead)
            pos += 1
            action_text = f"shift {lookahead!r} → s{action.state}"
        elif action.kind == ActionType.REDUCE:
            prod = action.production
            prod_idx = _production_index(user_grammar, prod)
            action_text = f"reduce {prod}"
            for _ in range(len(prod.body)):
                state_stack.pop()
                if sym_stack:
                    sym_stack.pop()
            exposed = state_stack[-1]
            goto = lr1.goto_table.get((exposed, prod.head))
            if goto is None:
                reason = f"No goto for state={exposed}, non-terminal={prod.head!r}"
                steps.append(LRStep(
                    step=step_i,
                    state_before=state, stack_before=before_states,
                    stack_symbols_before=before_syms,
                    lookahead=lookahead,
                    action_kind=action_kind,
                    action_text=action_text + " (goto missing)",
                    production_index=prod_idx, valid_actions=valid_acts,
                    state_after=state_stack[-1], stack_after=list(state_stack),
                    stack_symbols_after=list(sym_stack),
                    input_pos_before=before_pos, input_pos_after=pos,
                ))
                break
            state_stack.append(goto)
            sym_stack.append(prod.head)
        elif action.kind == ActionType.ACCEPT:
            accepted = True
            action_text = "accept"
        else:
            reason = f"Error action at state={state}, symbol={lookahead!r}"
            break

        steps.append(LRStep(
            step=step_i,
            state_before=state, stack_before=before_states,
            stack_symbols_before=before_syms,
            lookahead=lookahead,
            action_kind=action_kind, action_text=action_text,
            production_index=prod_idx, valid_actions=valid_acts,
            state_after=state_stack[-1], stack_after=list(state_stack),
            stack_symbols_after=list(sym_stack),
            input_pos_before=before_pos, input_pos_after=pos,
        ))
        step_i += 1
        if accepted:
            reason = "Accepted"
            break

    return steps, accepted, reason


# ----------------------------------------------------------------------
# DPDA — automaton + execution trace
# ----------------------------------------------------------------------


def _dpda_automaton_graph(dpda: DPDA) -> Dict[str, Any]:
    states = [
        {
            "id": s,
            "is_start": s == dpda.start_state,
            "is_accepting": s in dpda.accepting_states,
        }
        for s in range(dpda.num_states)
    ]
    edges: List[Dict[str, Any]] = []
    for e in dpda.edges:
        edges.append({
            "src": e.source,
            "dst": e.target,
            "kind": "ACCEPTANCE" if e.kind == EdgeKind.ACCEPTANCE else "REDUCTION",
            "symbols": sorted(e.accepted_symbols),
            "stack_match": list(e.stack_match),
            "stack_ops": [repr(op) for op in e.stack_ops],
        })
    return {"states": states, "edges": edges}


def _dpda_edge_text(e: PrefixConditionedEdge) -> str:
    syms = ",".join(sorted(e.accepted_symbols)) if e.accepted_symbols else "ε"
    match = ",".join(str(x) for x in e.stack_match) if e.stack_match else "*"
    ops = " ".join(repr(op) for op in e.stack_ops) if e.stack_ops else "—"
    kind = "ACC" if e.kind == EdgeKind.ACCEPTANCE else "RED"
    return f"{kind} on={{{syms}}} match=[{match}] ops=[{ops}] → s{e.target}"


def _dpda_valid_next(sim: DPDASimulator, state: int, stack: List[int]) -> List[str]:
    from ..dpda.simulator import DPDAConfig
    cfg = DPDAConfig(state=state, stack=list(stack))
    return sorted(sim.valid_symbols(cfg) - {END_MARKER})


def _trace_dpda(
    dpda_sim: DPDASimulator, tokens: List[str]
) -> Tuple[List[DPDAStep], bool, str]:
    config = dpda_sim.initial_config()
    ext = list(tokens) + [END_MARKER]
    consume_pos = 0
    steps: List[DPDAStep] = []
    step_i = 0
    chain_cap = dpda_sim.dpda.num_states + 8

    while consume_pos <= len(tokens):
        lookahead = ext[consume_pos]
        progressed = False
        for _ in range(chain_cap):
            e = dpda_sim._matching_edge(config.state, config.stack, lookahead)
            if e is None:
                break
            before = config.clone()
            valid_now = _dpda_valid_next(dpda_sim, before.state, before.stack)
            nxt = config.clone()
            nxt.stack = e.apply_stack_ops(nxt.stack)
            nxt.state = e.target
            if e.kind == EdgeKind.ACCEPTANCE:
                nxt.consumed += 1
                consume_pos += 1
            steps.append(DPDAStep(
                step=step_i,
                state_before=before.state,
                stack_before=list(before.stack),
                lookahead=lookahead,
                edge_kind=e.kind.name,
                edge_text=_dpda_edge_text(e),
                state_after=nxt.state,
                stack_after=list(nxt.stack),
                consumed_before=before.consumed,
                consumed_after=nxt.consumed,
                accepted_symbols=sorted(e.accepted_symbols),
                stack_match=list(e.stack_match),
                stack_ops=[repr(op) for op in e.stack_ops],
                consumes_input=(e.kind == EdgeKind.ACCEPTANCE),
                valid_next_terminals=valid_now,
            ))
            step_i += 1
            config = nxt
            progressed = True
            if e.kind == EdgeKind.ACCEPTANCE:
                break

        if not progressed:
            if consume_pos == len(tokens):
                if config.state in dpda_sim.dpda.accepting_states:
                    return steps, True, "Accepted"
                return steps, False, f"End of input in non-accepting state s{config.state}"
            return steps, False, f"No edge at state s{config.state}, symbol {lookahead!r}"

        if consume_pos == len(tokens) and config.state in dpda_sim.dpda.accepting_states:
            return steps, True, "Accepted"

    return steps, config.state in dpda_sim.dpda.accepting_states, "Finished"


# ----------------------------------------------------------------------
# NPDA — automaton + frontier trace
# ----------------------------------------------------------------------


def _npda_automaton_graph(npda: NPDA) -> Dict[str, Any]:
    states = [
        {
            "id": s,
            "is_start": s == npda.start_state,
            "is_accepting": s in npda.accept_states,
        }
        for s in sorted(npda.states)
    ]
    edges: List[Dict[str, Any]] = []
    for t in npda.transitions:
        a = t.input_symbol if t.input_symbol else "ε"
        push = "".join(t.stack_push) if t.stack_push else "ε"
        edges.append({
            "src": t.state,
            "dst": t.next_state,
            "label": f"{a}, {t.stack_top}/{push}",
            "is_epsilon": t.input_symbol == "",
            "input": a,
            "stack_top": t.stack_top,
            "push": list(t.stack_push),
        })
    return {
        "states": states,
        "edges": edges,
        "alphabet": sorted(npda.input_alphabet),
        "stack_alphabet": sorted(npda.stack_alphabet),
        "start_state": npda.start_state,
        "start_stack": npda.start_stack,
    }


def _serialize_config(c: Any) -> Dict[str, Any]:
    return {
        "state": c.state,
        "stack": list(c.stack),
        "remaining": list(c.remaining),
    }


def _epsilon_closure(
    npda: NPDA, frontier: List[Any], *, cap: int, max_stack: int = 64
) -> Tuple[List[Any], bool]:
    """
    Saturate the frontier under ε-transitions.  Bounded to ``cap`` configs;
    deduplicates by (state, stack, remaining).  Returns (closed_frontier,
    truncated).
    """
    from ..pda.simulator import PDAConfig
    seen = set(frontier)
    queue: List[Any] = list(frontier)
    closed: List[Any] = list(frontier)
    truncated = False

    while queue:
        if len(closed) >= cap:
            truncated = True
            break
        cfg = queue.pop(0)
        if not cfg.stack:
            continue
        top = cfg.stack[-1]
        for t in npda.epsilon_transitions(cfg.state, top):
            new_stack = list(cfg.stack[:-1])
            for s in reversed(t.stack_push):
                new_stack.append(s)
            if len(new_stack) > max_stack:
                continue
            nxt = PDAConfig(t.next_state, cfg.remaining, tuple(new_stack))
            if nxt in seen:
                continue
            seen.add(nxt)
            closed.append(nxt)
            queue.append(nxt)

    return closed, truncated


def _consume_step(
    npda: NPDA, frontier: List[Any], symbol: str, *, cap: int, max_stack: int = 64
) -> Tuple[List[Any], bool]:
    """Apply one symbol-consuming transition to every frontier config."""
    from ..pda.simulator import PDAConfig
    seen = set()
    out: List[Any] = []
    truncated = False
    for cfg in frontier:
        if not cfg.stack or not cfg.remaining or cfg.remaining[0] != symbol:
            continue
        top = cfg.stack[-1]
        for t in npda._index.get((cfg.state, symbol, top), []):
            new_stack = list(cfg.stack[:-1])
            for s in reversed(t.stack_push):
                new_stack.append(s)
            if len(new_stack) > max_stack:
                continue
            nxt = PDAConfig(t.next_state, cfg.remaining[1:], tuple(new_stack))
            if nxt in seen:
                continue
            seen.add(nxt)
            out.append(nxt)
            if len(out) >= cap:
                truncated = True
                return out, truncated
    return out, truncated


def _frontier_is_accepting(npda: NPDA, frontier: List[Any]) -> bool:
    for cfg in frontier:
        if not cfg.remaining and cfg.state in npda.accept_states:
            return True
    return False


def _trace_npda_frontier(
    npda: NPDA, tokens: List[str], *, frontier_cap: int = 64
) -> Tuple[List[Dict[str, Any]], bool, str]:
    """
    Frontier-style NPDA trace: ε-closure, then consume one terminal at a
    time, recording the live configurations after each phase.
    """
    from ..pda.simulator import PDAConfig

    init = PDAConfig(
        state=npda.start_state,
        remaining=tuple(tokens),
        stack=(npda.start_stack,),
    )
    frontier: List[Any] = [init]

    steps: List[Dict[str, Any]] = []
    step_i = 0
    consumed = 0

    closed, trunc0 = _epsilon_closure(npda, frontier, cap=frontier_cap)
    steps.append({
        "step": step_i,
        "phase": "epsilon_closure",
        "input_pos_before": 0,
        "input_pos_after": 0,
        "consumed_before": 0,
        "consumed_after": 0,
        "lookahead": tokens[0] if tokens else "$",
        "frontier_before": [_serialize_config(c) for c in frontier],
        "frontier_after": [_serialize_config(c) for c in closed],
        "frontier_size_before": len(frontier),
        "frontier_size_after": len(closed),
        "truncated": trunc0,
        "summary": f"ε-closure: {len(frontier)} → {len(closed)} configs"
                   + (" (truncated)" if trunc0 else ""),
    })
    step_i += 1
    frontier = closed

    for tok in tokens:
        before_cnt = len(frontier)
        consume_out, t1 = _consume_step(npda, frontier, tok, cap=frontier_cap)
        steps.append({
            "step": step_i,
            "phase": "consume",
            "input_pos_before": consumed,
            "input_pos_after": consumed + 1,
            "consumed_before": consumed,
            "consumed_after": consumed + 1,
            "lookahead": tok,
            "frontier_before": [_serialize_config(c) for c in frontier],
            "frontier_after": [_serialize_config(c) for c in consume_out],
            "frontier_size_before": before_cnt,
            "frontier_size_after": len(consume_out),
            "truncated": t1,
            "summary": f"consume {tok!r}: {before_cnt} → {len(consume_out)} configs",
        })
        step_i += 1
        consumed += 1

        if not consume_out:
            return steps, False, f"Frontier empty after consuming {tok!r} at pos {consumed}"

        closed, t2 = _epsilon_closure(npda, consume_out, cap=frontier_cap)
        steps.append({
            "step": step_i,
            "phase": "epsilon_closure",
            "input_pos_before": consumed,
            "input_pos_after": consumed,
            "consumed_before": consumed,
            "consumed_after": consumed,
            "lookahead": tokens[consumed] if consumed < len(tokens) else "$",
            "frontier_before": [_serialize_config(c) for c in consume_out],
            "frontier_after": [_serialize_config(c) for c in closed],
            "frontier_size_before": len(consume_out),
            "frontier_size_after": len(closed),
            "truncated": t2,
            "summary": f"ε-closure: {len(consume_out)} → {len(closed)} configs"
                       + (" (truncated)" if t2 else ""),
        })
        step_i += 1
        frontier = closed

    accepted = _frontier_is_accepting(npda, frontier)
    reason = "Accepted" if accepted else "Input fully read but no accepting config in frontier"
    return steps, accepted, reason


# ----------------------------------------------------------------------
# Top-level builders
# ----------------------------------------------------------------------


def _safe_lr_block(
    grammar: ContextFreeGrammar, tokens: List[str]
) -> Dict[str, Any]:
    try:
        lr1 = LR1Automaton(grammar)
    except Exception as e:
        return {
            "available": False,
            "error": f"LR(1) construction failed: {e}",
            "accepted": False,
            "reason": str(e),
            "state_count": 0,
            "automaton": {"states": [], "edges": []},
            "steps": [],
        }
    steps, ok, reason = _trace_lr(lr1, grammar, tokens)
    conflicts = _detect_lr_conflicts(lr1)
    return {
        "available": True,
        "accepted": ok,
        "reason": reason,
        "state_count": lr1.state_count,
        "automaton": _lr_automaton_graph(lr1),
        "steps": [asdict(s) for s in steps],
        "conflicts": conflicts,
        "is_lr1": len(conflicts) == 0,
    }


def _safe_dpda_block(
    grammar: ContextFreeGrammar, tokens: List[str]
) -> Dict[str, Any]:
    try:
        lr1 = LR1Automaton(grammar)
        dpda = build_dpda(lr1)
        sim = DPDASimulator(dpda)
    except Exception as e:
        return {
            "available": False,
            "error": f"DPDA construction failed: {e}",
            "accepted": False,
            "reason": str(e),
            "state_count": 0,
            "edge_count": 0,
            "accepting_states": [],
            "automaton": {"states": [], "edges": []},
            "steps": [],
        }
    steps, ok, reason = _trace_dpda(sim, tokens)
    return {
        "available": True,
        "accepted": ok,
        "reason": reason,
        "state_count": dpda.num_states,
        "edge_count": dpda.edge_count,
        "accepting_states": sorted(dpda.accepting_states),
        "automaton": _dpda_automaton_graph(dpda),
        "steps": [asdict(s) for s in steps],
    }


def _safe_npda_block(
    grammar: ContextFreeGrammar, tokens: List[str], *, frontier_cap: int = 64
) -> Dict[str, Any]:
    try:
        npda = cfg_to_npda(grammar)
    except Exception as e:
        return {
            "available": False,
            "error": f"NPDA construction failed: {e}",
            "accepted": False,
            "reason": str(e),
            "state_count": 0,
            "automaton": {"states": [], "edges": []},
            "steps": [],
        }
    steps, ok, reason = _trace_npda_frontier(npda, tokens, frontier_cap=frontier_cap)
    return {
        "available": True,
        "accepted": ok,
        "reason": reason,
        "state_count": len(npda.states),
        "transition_count": len(npda.transitions),
        "automaton": _npda_automaton_graph(npda),
        "frontier_cap": frontier_cap,
        "steps": steps,
    }


def build_trace_for_grammar(
    example: str,
    grammar: ContextFreeGrammar,
    tokens: List[str],
    *,
    raw_input: Optional[str] = None,
    source: Optional[Dict[str, Any]] = None,
    frontier_cap: int = 64,
) -> Dict[str, Any]:
    unknown = [t for t in tokens if t not in grammar.terminals]
    if unknown:
        allowed = ", ".join(sorted(grammar.terminals))
        raise ValueError(
            f"Token(s) not in grammar terminals: {unknown!r}. Allowed: {allowed}"
        )
    return {
        "format_version": FORMAT_VERSION,
        "example": example,
        "raw_input": raw_input if raw_input is not None else " ".join(tokens),
        "tokens": tokens,
        "source": source or {"kind": "manual"},
        "grammar": _grammar_meta(grammar),
        "lr": _safe_lr_block(grammar, tokens),
        "dpda": _safe_dpda_block(grammar, tokens),
        "npda": _safe_npda_block(grammar, tokens, frontier_cap=frontier_cap),
    }


def build_trace_from_tokens(
    example: str,
    tokens: List[str],
    *,
    raw_input: Optional[str] = None,
    source: Optional[Dict[str, Any]] = None,
    frontier_cap: int = 64,
) -> Dict[str, Any]:
    grammar = _load_example_grammar(example)
    return build_trace_for_grammar(
        example, grammar, tokens,
        raw_input=raw_input,
        source=source,
        frontier_cap=frontier_cap,
    )


def build_trace(
    example: str,
    raw_input: str,
    *,
    source: Optional[Dict[str, Any]] = None,
    frontier_cap: int = 64,
) -> Dict[str, Any]:
    grammar = _load_example_grammar(example)
    tokens = tokenize(raw_input, grammar.terminals)
    return build_trace_from_tokens(
        example, tokens,
        raw_input=raw_input,
        source=source or {"kind": "manual", "input_mode": "raw"},
        frontier_cap=frontier_cap,
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _tokens_from_json(payload: str) -> List[str]:
    data = json.loads(payload)
    if isinstance(data, dict):
        data = data.get("tokens")
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(
            "--tokens-json must be a JSON string array or an object with a "
            "string-array 'tokens'"
        )
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Export JSON trace for HTML viewer (v2)")
    p.add_argument(
        "--example",
        required=True,
        choices=["balanced_parens", "arithmetic", "anbn", "wcwR", "dyck2", "json_demo"],
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Raw input string")
    group.add_argument(
        "--tokens-json",
        help='Exact token stream as JSON, e.g. \'{"tokens":["(",")"]}\' or \'["(",")"]\'.',
    )
    p.add_argument("--out", required=True, help="Output trace json path")
    p.add_argument(
        "--source-label", default=None,
        help="Optional source label shown in the viewer",
    )
    p.add_argument(
        "--frontier-cap", type=int, default=64,
        help="Cap on NPDA frontier configurations per step (default 64).",
    )
    p.add_argument("--indent", type=int, default=2)
    args = p.parse_args()

    if args.tokens_json is not None:
        trace = build_trace_from_tokens(
            args.example,
            _tokens_from_json(args.tokens_json),
            source={
                "kind": "manual", "input_mode": "tokens-json",
                "label": args.source_label,
            },
            frontier_cap=args.frontier_cap,
        )
    else:
        trace = build_trace(
            args.example,
            args.input,
            source={
                "kind": "manual", "input_mode": "raw",
                "label": args.source_label,
            },
            frontier_cap=args.frontier_cap,
        )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=args.indent)
    print(f"Wrote v{FORMAT_VERSION} trace JSON to {args.out}")


if __name__ == "__main__":
    main()
