"""
End-to-end validation simulation.

Runs every component of the project against curated AND random inputs and
prints a clean, human-readable validation report.  Use it as a sanity
check after refactors and as the "live demo" for a viva.

Run:
    python3 -m pre3.tools.validate
"""

from __future__ import annotations

import random
import sys
import time
from typing import Callable, List, Tuple

from ..adapter.mock_llm import MockLLMSource
from ..dpda.builder import build_dpda
from ..dpda.simulator import DPDASimulator, LR1Simulator
from ..dpda.verifier import determinism_certificate, verify_determinism
from ..examples.catalogue import REGISTRY
from ..grammar.grammar_loader import (
    arithmetic, balanced_parens, from_rules,
)
from ..grammar.lr1 import LR1Automaton
from ..grammar.lrk import GrammarConflictError, LRkAutomaton, LRkSimulator
from ..pda.cfg_to_pda import cfg_to_npda
from ..pda.simulator import PDASimulator


# ANSI helpers (no-op if not a tty)
def _g(s: str) -> str:
    return f"\033[32m{s}\033[0m" if sys.stdout.isatty() else s


def _r(s: str) -> str:
    return f"\033[31m{s}\033[0m" if sys.stdout.isatty() else s


def _b(s: str) -> str:
    return f"\033[1m{s}\033[0m" if sys.stdout.isatty() else s


def _hr(title: str) -> None:
    print()
    print(_b(f"━━━━━ {title} " + "━" * (60 - len(title))))


# ----------------------------------------------------------------------
# Section 1: Grammar → LR(k) pipeline
# ----------------------------------------------------------------------


def section_lrk_construction() -> Tuple[int, int]:
    _hr("1. LR(k) construction (k = 0, 1, 2, 3)")
    grammars = [
        ("balanced_parens", balanced_parens()),
        ("arithmetic", arithmetic()),
        ("a^n b^n", from_rules({"S": ["a S b", ""]}, start="S")),
        ("wcw^R", from_rules({"S": ["a S a", "b S b", "c"]}, start="S")),
    ]
    print(f"  {'grammar':18s}  k=0          k=1          k=2          k=3")
    passed = total = 0
    for name, g in grammars:
        cells = []
        for k in [0, 1, 2, 3]:
            total += 1
            try:
                a = LRkAutomaton(g, k=k, strict=True)
                cells.append(_g(f"OK ({a.state_count}st)"))
                passed += 1
            except GrammarConflictError:
                # Conflict is expected for some grammar/k combos; treat as
                # PASS only if it's THE expected behaviour (not for "test" but
                # for the simulation we mark it red but informative).
                cells.append(_r("conflict"))
        cell_w = 12
        print(f"  {name:18s}  " + "  ".join(c.ljust(cell_w + 9) for c in cells))
    return passed, total


# ----------------------------------------------------------------------
# Section 2: LR(k) parser correctness
# ----------------------------------------------------------------------


def section_lrk_parser() -> Tuple[int, int]:
    _hr("2. LR(k) parser sanity checks (k = 1, 2, 3 on balanced_parens)")
    g = balanced_parens()
    cases = [
        ([], True), (["(", ")"], True),
        (["(", "(", ")", ")"], True), (["(", "(", "(", ")", ")", ")"], True),
        (["("], False), ([")"], False),
        (["(", "(", ")"], False), (["(", ")", ")"], False),
    ]
    passed = total = 0
    for k in [1, 2, 3]:
        sim = LRkSimulator(LRkAutomaton(g, k=k))
        ok = 0
        for s, expect in cases:
            total += 1
            got = sim.accepts(s)
            if got == expect:
                ok += 1
                passed += 1
        status = _g("✓") if ok == len(cases) else _r("✗")
        print(f"  LR({k}):  {ok}/{len(cases)}  {status}")
    return passed, total


# ----------------------------------------------------------------------
# Section 3: CFG → NPDA equivalence with LR(1)
# ----------------------------------------------------------------------


def section_npda_equiv() -> Tuple[int, int]:
    _hr("3. CFG → NPDA  vs  LR(1) parser  (each must agree on every string)")
    grammars = [
        ("balanced_parens", balanced_parens(), ["(", ")"], 6),
        ("a^n b^n",         from_rules({"S": ["a S b", ""]}, start="S"), ["a", "b"], 6),
        ("chain A→B→C→x",   from_rules({"A": ["B"], "B": ["C"], "C": ["x"]}, start="A"), ["x", "y"], 4),
        ("wcw^R",           from_rules({"S": ["a S a", "b S b", "c"]}, start="S"), ["a", "b", "c"], 6),
    ]
    rng = random.Random(0xCAFEBABE)
    passed = total = 0
    for name, g, alphabet, maxlen in grammars:
        lsim = LR1Simulator(LR1Automaton(g))
        psim = PDASimulator(cfg_to_npda(g), max_configs=30000)
        agree = disagree = 0
        for _ in range(100):
            n = rng.randint(0, maxlen)
            s = [rng.choice(alphabet) for _ in range(n)]
            if lsim.accepts(s) == psim.accepts(s, mode="final_state"):
                agree += 1
            else:
                disagree += 1
        total += 1
        if disagree == 0:
            passed += 1
        status = _g(f"✓ {agree}/100") if disagree == 0 else _r(f"✗ {disagree}/100 disagree")
        print(f"  {name:22s}  {status}")
    return passed, total


# ----------------------------------------------------------------------
# Section 4: LR(1) ≡ DPDA on real inputs (the central theorem)
# ----------------------------------------------------------------------


def section_dpda_equiv() -> Tuple[int, int]:
    _hr("4. LR(1) ≡ DPDA  (random sampling, the central correctness theorem)")
    grammars = [
        ("balanced_parens", balanced_parens(), ["(", ")"], 12),
        ("arithmetic",      arithmetic(), ["id", "+", "*", "(", ")"], 8),
        ("a^n b^n",         from_rules({"S": ["a S b", ""]}, start="S"), ["a", "b"], 12),
        ("a^n b^m",         from_rules({"S": ["A B"], "A": ["a A", ""], "B": ["b B", ""]}, start="S"), ["a", "b"], 12),
        ("left-rec list",   from_rules({"L": ["L , a", "a"]}, start="L"), [",", "a"], 12),
        ("if-then-else",    from_rules({"S": ["if e then S else S", "if e then S", "x"]}, start="S"),
                            ["if", "e", "then", "else", "x"], 10),
        ("wcw^R",           from_rules({"S": ["a S a", "b S b", "c"]}, start="S"), ["a", "b", "c"], 11),
        ("dyck-2 right-rec",from_rules({"S": ["( S ) S", "[ S ] S", ""]}, start="S"), ["(", ")", "[", "]"], 8),
    ]
    rng = random.Random(0xFEEDFACE)
    passed = total = 0
    for name, g, alphabet, maxlen in grammars:
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        verify_determinism(dpda)  # raises on issue
        lsim, dsim = LR1Simulator(lr1), DPDASimulator(dpda)
        agree = disagree = 0
        for _ in range(500):
            n = rng.randint(0, maxlen)
            s = [rng.choice(alphabet) for _ in range(n)]
            if lsim.accepts(s) == dsim.accepts(s):
                agree += 1
            else:
                disagree += 1
        total += 1
        if disagree == 0:
            passed += 1
        status = _g(f"✓ {agree}/500") if disagree == 0 else _r(f"✗ {disagree}/500 disagree")
        print(f"  {name:22s}  states={dpda.num_states:3d} edges={dpda.edge_count:4d}  {status}")
    return passed, total


# ----------------------------------------------------------------------
# Section 5: stress — deep / long inputs
# ----------------------------------------------------------------------


def section_deep_inputs() -> Tuple[int, int]:
    _hr("5. Stress test:  deep recursion / long inputs")
    cases = [
        ("a^n b^n  (n=200)",     from_rules({"S": ["a S b", ""]}, start="S"),
         ["a"] * 200 + ["b"] * 200, True),
        ("a^n b^n  off-by-one (n=200)", from_rules({"S": ["a S b", ""]}, start="S"),
         ["a"] * 200 + ["b"] * 201, False),
        ("(((...)))  (n=200)",   balanced_parens(), ["("] * 200 + [")"] * 200, True),
        ("unbalanced (n=200,199)",balanced_parens(), ["("] * 200 + [")"] * 199, False),
        ("left-rec list len=300",from_rules({"L": ["L , a", "a"]}, start="L"),
         ["a"] + [",", "a"] * 150, True),
    ]
    passed = total = 0
    for name, g, s, expect in cases:
        lsim = LR1Simulator(LR1Automaton(g))
        dsim = DPDASimulator(build_dpda(LR1Automaton(g)))
        t0 = time.perf_counter()
        a = lsim.accepts(s); b = dsim.accepts(s)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total += 1
        ok = (a == b == expect)
        if ok:
            passed += 1
        status = _g("✓") if ok else _r("✗")
        print(f"  {name:30s}  LR1={a}  DPDA={b}  expected={expect}  {elapsed_ms:5.1f} ms  {status}")
    return passed, total


# ----------------------------------------------------------------------
# Section 6: Negative results (the heart of the ToC story)
# ----------------------------------------------------------------------


def section_negative_results() -> Tuple[int, int]:
    _hr("6. Negative results — language hierarchy demonstrated")
    passed = total = 0

    # 6a. wwR is CFL but not DCFL.
    print("  6a. wwR  (CFL ∖ DCFL):  LR(k) construction MUST fail.")
    g_wwR = from_rules({"S": ["a S a", "b S b", ""]}, start="S")
    for k in [1, 2, 3]:
        total += 1
        try:
            LRkAutomaton(g_wwR, k=k, strict=True)
            print(f"      LR({k}): {_r('built without conflict — theory violated!')}")
        except GrammarConflictError as e:
            passed += 1
            print(f"      LR({k}): {_g(f'{e.kind} conflict at state {e.state}')}")
    print("       … but the NPDA accepts wwR fine (it's still a CFL):")
    sim = PDASimulator(cfg_to_npda(g_wwR), max_configs=20000)
    cases = [(["a", "b", "b", "a"], True), (["a", "b", "a"], False),
             (["a", "a", "b", "b", "a", "a"], True), ([], True)]
    for s, expect in cases:
        total += 1
        got = sim.accepts(s, mode="final_state")
        if got == expect:
            passed += 1
            print(f"      NPDA({s!s:30s}) = {_g(str(got))}")
        else:
            print(f"      NPDA({s!s:30s}) = {_r(str(got))} (expected {expect})")

    # 6b. a^n b^n c^n is not CFL.
    print()
    print("  6b. aⁿbⁿcⁿ (not CFL):  no grammar exists; pumping lemma witness:")
    from ..examples.anbncn import pumping_lemma_witness
    s, explanation = pumping_lemma_witness(p=3)
    print(f"      witness s = {s}")
    print(f"      {explanation[:140]}…")
    total += 1
    if "pumping" in explanation.lower() and len(s) == 9:
        passed += 1
        print(f"      {_g('✓ pumping-lemma witness has the expected shape')}")

    # 6c. LR(0) ⊊ LR(1).
    print()
    print("  6c. LR(0) ⊊ LR(1):  balanced_parens fails LR(0).")
    total += 1
    try:
        LRkAutomaton(balanced_parens(), k=0, strict=True)
        print(f"      {_r('LR(0) clean — theory violated!')}")
    except GrammarConflictError as e:
        passed += 1
        print(f"      {_g(f'LR(0) {e.kind} conflict — as expected')}")
    total += 1
    a = LRkAutomaton(balanced_parens(), k=1, strict=True)
    if not a.conflicts:
        passed += 1
        print(f"      {_g('LR(1) clean — as expected')}")

    return passed, total


# ----------------------------------------------------------------------
# Section 7: End-to-end pipeline (the LLM-stub left of pipeline)
# ----------------------------------------------------------------------


def section_e2e_pipeline() -> Tuple[int, int]:
    _hr("7. End-to-end pipeline:  StringSource → DPDA validator")
    src = MockLLMSource([
        ["(", ")"],                  # ✓
        ["(", "(", ")", ")"],        # ✓
        ["(", "(", ")"],             # ✗
        [],                          # ✓
        ["(", ")", ")"],             # ✗
    ], label="paren-LLM-stub")
    expected = [True, True, False, True, False]
    dpda = build_dpda(LR1Automaton(balanced_parens()))
    sim = DPDASimulator(dpda)

    passed = total = 0
    for i, (tokens, want) in enumerate(zip(src.emit(), expected)):
        total += 1
        got = sim.accepts(tokens)
        ok = got == want
        if ok:
            passed += 1
        marker = _g("✓") if ok else _r("✗")
        print(f"  {marker}  {tokens!s:35s} → {got}  (expected {want})")
    return passed, total


# ----------------------------------------------------------------------
# Section 8: example-catalogue spot-check
# ----------------------------------------------------------------------


def section_catalogue() -> Tuple[int, int]:
    _hr("8. Canonical-example catalogue")
    print(f"  {'name':12s}  class            CFG?  LR(1) ok?")
    passed = total = 0
    for entry in REGISTRY:
        mod = entry["module"]
        has_grammar = hasattr(mod, "grammar")
        ok_lrk = "n/a"
        if has_grammar:
            try:
                a = LRkAutomaton(mod.grammar(), k=1, strict=False)
                ok_lrk = "yes (clean)" if not a.conflicts else "no (conflict)"
            except Exception as e:
                ok_lrk = f"err: {e.__class__.__name__}"
        expected_lrk = "yes" if entry["lrk_buildable"] else "no"
        agrees = (("yes" in ok_lrk) == ("yes" in expected_lrk))
        total += 1
        if agrees:
            passed += 1
        marker = _g("✓") if agrees else _r("✗")
        print(f"  {entry['name']:12s}  {entry['class']:14s}  {('yes' if has_grammar else 'no'):4s}  {ok_lrk:14s} {marker}")
    return passed, total


# ----------------------------------------------------------------------
# Section 9: trace one example to show the runtime
# ----------------------------------------------------------------------


def section_trace_demo() -> None:
    _hr("9. Live trace demo:  '(())' through balanced_parens DPDA")
    from .trace import tokenize, trace_dpda, trace_lr1, render_table
    g = balanced_parens()
    tokens = tokenize("(())", g.terminals)
    ok, rows = trace_dpda(g, tokens)
    print(render_table(["#", "stack", "lookahead", "edge"], rows))
    print(f"  result: {_g('ACCEPTED') if ok else _r('REJECTED')}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    print(_b("Pre³ — End-to-End Validation Simulation"))
    sections = [
        section_lrk_construction,
        section_lrk_parser,
        section_npda_equiv,
        section_dpda_equiv,
        section_deep_inputs,
        section_negative_results,
        section_e2e_pipeline,
        section_catalogue,
    ]
    grand_pass = grand_total = 0
    for sec in sections:
        p, t = sec()
        grand_pass += p
        grand_total += t

    section_trace_demo()  # informational

    _hr("SUMMARY")
    rate = (grand_pass / grand_total * 100.0) if grand_total else 0.0
    color = _g if grand_pass == grand_total else _r
    print(color(f"  {grand_pass}/{grand_total} validation checks passed   ({rate:.1f}%)"))
    return 0 if grand_pass == grand_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
