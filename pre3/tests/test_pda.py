"""Tests for the generic NPDA and the CFG → NPDA construction."""

import pytest

from pre3.grammar.grammar_loader import balanced_parens, from_rules
from pre3.grammar.lr1 import LR1Automaton
from pre3.dpda.simulator import LR1Simulator
from pre3.pda.cfg_to_pda import cfg_to_npda
from pre3.pda.pda import NPDA, PDATransition
from pre3.pda.simulator import PDASimulator


# ----------------------------------------------------------------------
# Hand-crafted NPDA: a^n b^n
# ----------------------------------------------------------------------


def _anbn_npda() -> NPDA:
    Z, A = "Z", "A"
    return NPDA(
        states={"q0", "q1", "qa"},
        input_alphabet={"a", "b"},
        stack_alphabet={Z, A},
        transitions=[
            PDATransition("q0", "a", Z, "q0", (A, Z)),
            PDATransition("q0", "a", A, "q0", (A, A)),
            PDATransition("q0", "", Z, "qa", (Z,)),
            PDATransition("q0", "", A, "q1", (A,)),
            PDATransition("q1", "b", A, "q1", ()),
            PDATransition("q1", "", Z, "qa", (Z,)),
        ],
        start_state="q0",
        start_stack=Z,
        accept_states={"qa"},
    )


class TestAnBnNPDA:
    @pytest.fixture
    def sim(self):
        return PDASimulator(_anbn_npda())

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 5])
    def test_accepts_balanced(self, sim, n):
        assert sim.accepts(["a"] * n + ["b"] * n, mode="final_state")

    @pytest.mark.parametrize("s", [["a"], ["b"], ["a", "a", "b"], ["b", "a"]])
    def test_rejects(self, sim, s):
        assert not sim.accepts(s, mode="final_state")


# ----------------------------------------------------------------------
# CFG → NPDA construction
# ----------------------------------------------------------------------


class TestCFGtoNPDAEquivalence:
    """For LR(1) grammars, the NPDA must agree with the LR(1) parser."""

    def _agree(self, grammar, strs, max_configs=10000):
        npda = cfg_to_npda(grammar)
        psim = PDASimulator(npda, max_configs=max_configs)
        lsim = LR1Simulator(LR1Automaton(grammar))
        for s in strs:
            a = psim.accepts(s, mode="final_state")
            b = lsim.accepts(s)
            assert a == b, f"disagree on {s}: NPDA={a}, LR1={b}"

    def test_balanced_parens(self):
        cases = [[], ["(", ")"], ["(", "(", ")", ")"], ["(", "(", "(", ")", ")", ")"],
                 ["("], [")"], ["(", "(", ")"], ["(", ")", ")"]]
        self._agree(balanced_parens(), cases)

    def test_anbn(self):
        g = from_rules({"S": ["a S b", ""]}, start="S")
        cases = [["a"] * n + ["b"] * n for n in range(5)] + \
                [["a"], ["b"], ["a", "a", "b"], ["a", "b", "a"]]
        self._agree(g, cases)

    def test_chain_rules(self):
        g = from_rules({"A": ["B"], "B": ["C"], "C": ["x"]}, start="A")
        self._agree(g, [["x"], [], ["y"], ["x", "x"]])


# ----------------------------------------------------------------------
# NPDA can recognize wwR (a CFL but NOT DCFL)
# ----------------------------------------------------------------------


class TestNPDAExceedsDPDA:
    """The whole point of NPDAs:  recognize CFLs that no DPDA can."""

    def test_wwR(self):
        from pre3.examples.wwR import npda, positive_examples, negative_examples
        sim = PDASimulator(npda(), max_configs=20000)
        for s in positive_examples:
            assert sim.accepts(s, mode="final_state"), f"NPDA must accept {s}"
        for s in negative_examples:
            assert not sim.accepts(s, mode="final_state"), f"NPDA must reject {s}"
