"""
Stress tests:  equivalence and determinism over MANY grammars and LONG
random inputs.  Designed to expose edge cases the small fixture grammars
might miss.
"""

import random
from typing import Callable, List, Tuple

import pytest

from pre3.dpda.builder import build_dpda
from pre3.dpda.simulator import DPDASimulator, LR1Simulator
from pre3.dpda.verifier import verify_determinism
from pre3.grammar.cfg import ContextFreeGrammar
from pre3.grammar.grammar_loader import (
    arithmetic, balanced_parens, from_rules,
)
from pre3.grammar.lr1 import LR1Automaton


def _equiv_pair(grammar: ContextFreeGrammar):
    lr1 = LR1Automaton(grammar)
    dpda = build_dpda(lr1)
    verify_determinism(dpda)
    return LR1Simulator(lr1), DPDASimulator(dpda)


# ----------------------------------------------------------------------
# A wider catalogue of LR(1)-clean grammars to stress-test
# ----------------------------------------------------------------------

GRAMMARS: List[Tuple[str, Callable[[], ContextFreeGrammar], List[str], int]] = [
    ("balanced_parens",     balanced_parens, ["(", ")"], 14),
    ("arithmetic",          arithmetic, ["id", "+", "*", "(", ")"], 10),
    ("a^n b^n",             lambda: from_rules({"S": ["a S b", ""]}, start="S"), ["a", "b"], 14),
    ("a^n b^m  m,n>=0",     lambda: from_rules({"S": ["A B"], "A": ["a A", ""], "B": ["b B", ""]}, start="S"), ["a", "b"], 14),
    ("left-rec list (a)",   lambda: from_rules({"L": ["L , a", "a"]}, start="L"), [",", "a"], 14),
    ("right-rec list (a)",  lambda: from_rules({"L": ["a , L", "a"]}, start="L"), [",", "a"], 14),
    ("arith + unary minus", lambda: from_rules({
        "E": ["E + T", "E - T", "T"],
        "T": ["T * F", "T / F", "F"],
        "F": ["( E )", "- F", "id"],
    }, start="E"), ["id", "+", "-", "*", "/", "(", ")"], 10),
    ("if-then-else",        lambda: from_rules({
        "S": ["if e then S else S", "if e then S", "x"],
    }, start="S"), ["if", "e", "then", "else", "x"], 10),
    ("wcw^R (DCFL)",        lambda: from_rules({"S": ["a S a", "b S b", "c"]}, start="S"), ["a", "b", "c"], 11),
    ("dyck-2 right-rec",    lambda: from_rules({"S": ["( S ) S", "[ S ] S", ""]}, start="S"),
                            ["(", ")", "[", "]"], 10),
    ("chain A->B->C->x",    lambda: from_rules({"A": ["B"], "B": ["C"], "C": ["x"]}, start="A"), ["x", "y"], 4),
    ("nullable chain",      lambda: from_rules({"S": ["A B"], "A": ["a", ""], "B": ["b", ""]}, start="S"),
                            ["a", "b"], 5),
]


# ----------------------------------------------------------------------
# Random equivalence sweep
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name,grammar_fn,alphabet,max_len", GRAMMARS, ids=[g[0] for g in GRAMMARS])
def test_random_equivalence_long(name, grammar_fn, alphabet, max_len):
    """1500 random strings up to length ``max_len`` per grammar."""
    rng = random.Random(0xDADA ^ hash(name))
    g = grammar_fn()
    lsim, dsim = _equiv_pair(g)
    fails: List[tuple] = []
    for _ in range(1500):
        n = rng.randint(0, max_len)
        s = [rng.choice(alphabet) for _ in range(n)]
        if lsim.accepts(s) != dsim.accepts(s):
            fails.append(s)
    assert not fails, f"{name}: {len(fails)} disagreements; first: {fails[:3]}"


# ----------------------------------------------------------------------
# Deeply nested / long-run inputs
# ----------------------------------------------------------------------


def test_deep_anbn():
    """a^200 b^200 must round-trip through both simulators identically."""
    g = from_rules({"S": ["a S b", ""]}, start="S")
    lsim, dsim = _equiv_pair(g)
    for n in [50, 100, 200]:
        s = ["a"] * n + ["b"] * n
        assert lsim.accepts(s) is True
        assert dsim.accepts(s) is True
        # Off-by-one
        bad = ["a"] * n + ["b"] * (n + 1)
        assert lsim.accepts(bad) is False
        assert dsim.accepts(bad) is False


def test_deep_balanced_parens():
    g = balanced_parens()
    lsim, dsim = _equiv_pair(g)
    for n in [50, 100, 200]:
        s = ["("] * n + [")"] * n
        assert dsim.accepts(s) and lsim.accepts(s)
        s_bad = ["("] * n + [")"] * (n - 1)
        assert not dsim.accepts(s_bad) and not lsim.accepts(s_bad)


def test_deep_left_recursion():
    """L → L , a | a — a long left-recursive list."""
    g = from_rules({"L": ["L , a", "a"]}, start="L")
    lsim, dsim = _equiv_pair(g)
    for n in [10, 50, 100]:
        s = ["a"] + ([",", "a"] * n)  # 1 + 2n tokens
        assert lsim.accepts(s) and dsim.accepts(s)
        s_bad = s[:-1]   # trailing comma
        assert not lsim.accepts(s_bad) and not dsim.accepts(s_bad)


def test_deep_wcwR():
    """Center-marked palindromes up to length 41 (20 a's, c, 20 a's)."""
    g = from_rules({"S": ["a S a", "b S b", "c"]}, start="S")
    lsim, dsim = _equiv_pair(g)
    rng = random.Random(7)
    for _ in range(50):
        n = rng.randint(0, 20)
        w = [rng.choice(["a", "b"]) for _ in range(n)]
        good = w + ["c"] + list(reversed(w))
        bad = w + ["c"] + (list(reversed(w))[:-1] + [rng.choice(["a", "b"])]) if n else ["c", "a"]
        assert dsim.accepts(good) == lsim.accepts(good)
        assert dsim.accepts(bad) == lsim.accepts(bad)


# ----------------------------------------------------------------------
# Determinism under stress
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name,grammar_fn,alphabet,max_len", GRAMMARS, ids=[g[0] for g in GRAMMARS])
def test_no_nondeterminism_anywhere(name, grammar_fn, alphabet, max_len):
    g = grammar_fn()
    dpda = build_dpda(LR1Automaton(g))
    # If verify raises, the test fails.
    verify_determinism(dpda, collect_all=True)
    assert True


# ----------------------------------------------------------------------
# CFG → NPDA equivalence at scale
# ----------------------------------------------------------------------


def test_cfg_to_npda_equivalence_random():
    """For 4 LR(1) grammars × 100 random short strings, NPDA == LR(1)."""
    from pre3.pda.cfg_to_pda import cfg_to_npda
    from pre3.pda.simulator import PDASimulator

    grammars = [
        (balanced_parens(), ["(", ")"], 6),
        (from_rules({"S": ["a S b", ""]}, start="S"), ["a", "b"], 6),
        (from_rules({"L": ["L , a", "a"]}, start="L"), [",", "a"], 6),
        (from_rules({"S": ["a S a", "b S b", "c"]}, start="S"), ["a", "b", "c"], 7),
    ]
    rng = random.Random(99)
    for g, alphabet, max_len in grammars:
        lsim = LR1Simulator(LR1Automaton(g))
        psim = PDASimulator(cfg_to_npda(g), max_configs=50000)
        disagreements = 0
        for _ in range(100):
            n = rng.randint(0, max_len)
            s = [rng.choice(alphabet) for _ in range(n)]
            if lsim.accepts(s) != psim.accepts(s, mode="final_state"):
                disagreements += 1
        assert disagreements == 0


# ----------------------------------------------------------------------
# Empty-string handling (many corner cases live here)
# ----------------------------------------------------------------------


def test_epsilon_acceptance_consistency():
    """For each grammar, both simulators agree on []."""
    for name, gfn, _alphabet, _ml in GRAMMARS:
        g = gfn()
        lsim, dsim = _equiv_pair(g)
        assert lsim.accepts([]) == dsim.accepts([]), f"{name}: disagree on []"
