"""
Language:  L = { aⁿbⁿcⁿ : n ≥ 0 }
Class:     **NOT** context-free.  (Classical pumping-lemma example.)

This module exists to demonstrate a *limit of CFG/PDA expressiveness*.
There is no CFG generating L, hence no PDA recognises it, hence the
project's CFG → NPDA pipeline cannot produce one.

Pumping-lemma proof sketch (Sipser, Thm 2.34):
  Suppose L is CF with pumping length p.  Take s = aᵖbᵖcᵖ.  By the lemma
  s = uvxyz with |vy| ≥ 1, |vxy| ≤ p, and uvⁿxyⁿz ∈ L for all n ≥ 0.
  Case analysis on which letters appear in v and y all yield a string
  outside L (e.g. unequal counts of one letter).  Contradiction.

We provide :func:`pumping_lemma_witness` to generate the contradiction
string for a given pumping length, useful for a viva demonstration.
"""

from __future__ import annotations

from typing import List, Tuple

description = (
    "L = {aⁿbⁿcⁿ : n ≥ 0}.  NOT context-free.  "
    "Pumping lemma witness: s = aᵖbᵖcᵖ for any pumping length p."
)


def pumping_lemma_witness(p: int) -> Tuple[List[str], str]:
    """Produce the pumping-lemma witness string and the textbook argument.

    Returns (s, explanation) where s = aᵖbᵖcᵖ.
    """
    s = ["a"] * p + ["b"] * p + ["c"] * p
    explanation = (
        f"Take s = aᵖbᵖcᵖ with p={p}, |s|={3*p}.  By the pumping lemma "
        f"there should exist u v x y z = s with |vy| ≥ 1, |vxy| ≤ p, "
        f"and uvⁿxyⁿz ∈ L for all n ≥ 0.  Since |vxy| ≤ p, vxy lies in "
        f"at most two of the three blocks aᵖ, bᵖ, cᵖ.  Pumping (n=2) "
        f"therefore unbalances counts in at least one letter.  ∎"
    )
    return s, explanation


# We deliberately do not provide a grammar/NPDA — none exists.
positive_examples: List[List[str]] = [
    [],
    ["a", "b", "c"],
    ["a", "a", "b", "b", "c", "c"],
]

negative_examples: List[List[str]] = [
    ["a", "b", "b", "c"],
    ["a", "a", "b", "c", "c"],
    ["a", "b", "c", "a", "b", "c"],
]
