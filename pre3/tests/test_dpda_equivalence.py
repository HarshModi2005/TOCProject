"""
The CENTRAL correctness test:  for every LR(1) grammar G,
   L(LR1Simulator(G))  ==  L(DPDASimulator(build_dpda(G)))

We test this by random sampling over each grammar's terminal alphabet,
augmented with the hand-curated positive / negative examples per language.

A `hypothesis` property test is also included for stronger coverage.
"""

import random
from typing import Callable, List

import pytest

try:
    from hypothesis import given, settings, strategies as st
    HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover
    HAS_HYPOTHESIS = False

from pre3.dpda.builder import build_dpda
from pre3.dpda.simulator import DPDASimulator, LR1Simulator
from pre3.dpda.verifier import verify_determinism
from pre3.grammar.cfg import ContextFreeGrammar
from pre3.grammar.grammar_loader import (
    arithmetic, balanced_parens, from_rules,
)
from pre3.grammar.lr1 import LR1Automaton


def _build_pair(grammar: ContextFreeGrammar):
    lr1 = LR1Automaton(grammar)
    dpda = build_dpda(lr1)
    verify_determinism(dpda)   # must not raise
    return LR1Simulator(lr1), DPDASimulator(dpda)


# (name, grammar_fn, alphabet, max_len)
GRAMMARS: List = [
    ("balanced_parens", balanced_parens, ["(", ")"], 10),
    ("arithmetic", arithmetic, ["id", "+", "*", "(", ")"], 8),
    ("a^n b^n", lambda: from_rules({"S": ["a S b", ""]}, start="S"), ["a", "b"], 10),
    ("chains", lambda: from_rules({"A": ["B"], "B": ["C"], "C": ["x"]}, start="A"), ["x", "y"], 4),
    ("left-rec list", lambda: from_rules({"L": ["L , a", "a"]}, start="L"), [",", "a"], 10),
    ("w c w^R", lambda: from_rules({"S": ["a S a", "b S b", "c"]}, start="S"), ["a", "b", "c"], 9),
    ("dyck-2 (right-rec)",
     lambda: from_rules({"S": ["( S ) S", "[ S ] S", ""]}, start="S"),
     ["(", ")", "[", "]"], 8),
]


@pytest.mark.parametrize("name,grammar_fn,alphabet,max_len", GRAMMARS, ids=[g[0] for g in GRAMMARS])
def test_random_equivalence(name, grammar_fn, alphabet, max_len):
    rng = random.Random(0xC0FFEE ^ hash(name))
    grammar = grammar_fn()
    lsim, dsim = _build_pair(grammar)

    samples = 400
    fails: List[tuple] = []
    for _ in range(samples):
        n = rng.randint(0, max_len)
        s = [rng.choice(alphabet) for _ in range(n)]
        a = lsim.accepts(s)
        b = dsim.accepts(s)
        if a != b:
            fails.append((s, a, b))

    assert not fails, (
        f"{name}: {len(fails)}/{samples} disagreements.\n"
        + "\n".join(f"  {s} -> LR1={a}, DPDA={b}" for s, a, b in fails[:5])
    )


@pytest.mark.parametrize("grammar_fn,alphabet", [
    (balanced_parens, ["(", ")"]),
    (lambda: from_rules({"S": ["a S b", ""]}, start="S"), ["a", "b"]),
    (lambda: from_rules({"S": ["a S a", "b S b", "c"]}, start="S"), ["a", "b", "c"]),
])
def test_handcurated_strings(grammar_fn, alphabet):
    """Sanity-check on the simplest, most surprising strings."""
    grammar = grammar_fn()
    lsim, dsim = _build_pair(grammar)
    # Exhaustively try strings of length ≤ 5.
    from itertools import product
    for length in range(0, 6):
        for s in product(alphabet, repeat=length):
            s = list(s)
            assert lsim.accepts(s) == dsim.accepts(s), \
                f"disagree on {s}: LR1={lsim.accepts(s)}, DPDA={dsim.accepts(s)}"


if HAS_HYPOTHESIS:
    @given(st.lists(st.sampled_from(["(", ")"]), max_size=10))
    @settings(max_examples=200, deadline=None)
    def test_property_balanced_parens(s):
        lsim, dsim = _build_pair(balanced_parens())
        assert lsim.accepts(s) == dsim.accepts(s)

    @given(st.lists(st.sampled_from(["a", "b"]), max_size=10))
    @settings(max_examples=200, deadline=None)
    def test_property_anbn(s):
        g = from_rules({"S": ["a S b", ""]}, start="S")
        lsim, dsim = _build_pair(g)
        assert lsim.accepts(s) == dsim.accepts(s)
