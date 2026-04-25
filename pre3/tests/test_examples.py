"""Tests over the canonical-language catalogue."""

import pytest

from pre3.dpda.builder import build_dpda
from pre3.dpda.simulator import DPDASimulator
from pre3.examples.catalogue import REGISTRY
from pre3.grammar.lr1 import LR1Automaton
from pre3.grammar.lrk import GrammarConflictError, LRkAutomaton
from pre3.pda.cfg_to_pda import cfg_to_npda
from pre3.pda.simulator import PDASimulator


@pytest.mark.parametrize("entry", REGISTRY, ids=[e["name"] for e in REGISTRY])
def test_lrk_buildable_matches_classification(entry):
    """Each entry's `lrk_buildable` flag must match what LRkAutomaton does."""
    mod = entry["module"]
    if not hasattr(mod, "grammar"):
        pytest.skip("no grammar (e.g. aⁿbⁿcⁿ)")
    g = mod.grammar()
    if entry["lrk_buildable"]:
        # Should build with LR(1) without conflicts.
        a = LRkAutomaton(g, k=1, strict=False)
        assert not a.conflicts, f"{entry['name']} expected LR(1)-clean but had {a.conflicts}"
    else:
        # Must produce conflicts for SOME k (we test k=1).
        a = LRkAutomaton(g, k=1, strict=False)
        assert a.conflicts, f"{entry['name']} expected LR(1)-conflict but had none"


@pytest.mark.parametrize("entry", REGISTRY, ids=[e["name"] for e in REGISTRY])
def test_dpda_accepts_positive_rejects_negative(entry):
    """LR(1)-buildable grammars: DPDA accepts positives, rejects negatives."""
    if not entry["lrk_buildable"]:
        pytest.skip("not LR(1) — DPDA pipeline doesn't apply")
    mod = entry["module"]
    g = mod.grammar()
    dpda = build_dpda(LR1Automaton(g))
    sim = DPDASimulator(dpda)
    for s in mod.positive_examples:
        assert sim.accepts(s), f"{entry['name']}: DPDA must accept {s}"
    for s in mod.negative_examples:
        assert not sim.accepts(s), f"{entry['name']}: DPDA must reject {s}"


@pytest.mark.parametrize("entry", REGISTRY, ids=[e["name"] for e in REGISTRY])
def test_npda_accepts_positive_rejects_negative(entry):
    """Where a CFG exists (CFL), the CFG → NPDA accepts/rejects correctly."""
    if not entry["is_cfl"]:
        pytest.skip("not a CFL — no CFG → NPDA")
    mod = entry["module"]
    g = mod.grammar()
    npda = cfg_to_npda(g)
    sim = PDASimulator(npda, max_configs=30000)
    for s in mod.positive_examples:
        assert sim.accepts(s, mode="final_state"), f"{entry['name']}: NPDA must accept {s}"
    for s in mod.negative_examples:
        assert not sim.accepts(s, mode="final_state"), f"{entry['name']}: NPDA must reject {s}"


def test_anbncn_is_witnessed_by_pumping_lemma():
    from pre3.examples.anbncn import pumping_lemma_witness
    s, explanation = pumping_lemma_witness(p=4)
    assert s == ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    assert "pumping" in explanation.lower()


def test_wwR_lrk_fails_for_all_small_k():
    """The non-DCFL property: NO k makes wwR LR(k)-clean."""
    from pre3.examples.wwR import grammar
    g = grammar()
    for k in [1, 2, 3]:
        with pytest.raises(GrammarConflictError):
            LRkAutomaton(g, k=k, strict=True)
