"""
Benchmark: per-step decode overhead  (mask generation time).

Simulates the per-step cost of constrained decoding by measuring how
long it takes to generate a vocabulary mask at each step.

Run with:  python -m pre3.benchmarks.bench_decode
"""

from __future__ import annotations

import time

from pre3.grammar.grammar_loader import balanced_parens, from_rules, simple_json
from pre3.grammar.lr1 import LR1Automaton
from pre3.dpda.builder import build_dpda
from pre3.dpda.optimizer import optimize
from pre3.dpda.simulator import DPDAConfig
from pre3.decoding.token_trie import TokenTrie
from pre3.decoding.mask_generator import CachedMaskGenerator, MaskGenerator


def _build_char_vocab(chars: str, extra: int = 50) -> dict[int, str]:
    """Build a toy character-level vocab + some multi-char tokens."""
    vocab: dict[int, str] = {}
    idx = 0
    for ch in chars:
        vocab[idx] = ch
        idx += 1
    # Add some multi-character tokens
    for i in range(extra):
        vocab[idx] = f"tok{i}"
        idx += 1
    return vocab


def bench_mask_generation(
    name: str,
    grammar,
    vocab: dict[int, str],
    steps: int = 100,
) -> None:
    lr1 = LR1Automaton(grammar)
    dpda = build_dpda(lr1)
    dpda = optimize(dpda)
    trie = TokenTrie.from_vocabulary(vocab)

    mg = MaskGenerator(dpda, trie)
    cmg = CachedMaskGenerator(dpda, trie)

    config = DPDAConfig(state=dpda.start_state, stack=[dpda.start_state])

    # Benchmark uncached
    t0 = time.perf_counter()
    for _ in range(steps):
        mg.generate_mask(config)
    t1 = time.perf_counter()
    uncached_us = (t1 - t0) / steps * 1e6

    # Benchmark cached
    t0 = time.perf_counter()
    for _ in range(steps):
        cmg.generate_mask(config)
    t1 = time.perf_counter()
    cached_us = (t1 - t0) / steps * 1e6

    print(
        f"  {name:30s}  vocab={len(vocab):5d}  "
        f"uncached={uncached_us:8.1f} µs/step  "
        f"cached={cached_us:8.1f} µs/step"
    )


def main() -> None:
    print("=== Pre³ Per-step Decode Overhead Benchmark ===\n")

    # Balanced parens with char-level vocab
    vocab_bp = _build_char_vocab("()")
    bench_mask_generation("balanced_parens (small vocab)", balanced_parens(), vocab_bp)

    # Balanced parens with bigger vocab
    vocab_bp_large = _build_char_vocab("()", extra=500)
    bench_mask_generation("balanced_parens (med vocab)", balanced_parens(), vocab_bp_large)

    # JSON-like
    json_chars = '{}[]:,"abcdefg0123456789'
    vocab_json = _build_char_vocab(json_chars, extra=200)
    bench_mask_generation("simple_json", simple_json(), vocab_json)

    print("\nDone.")


if __name__ == "__main__":
    main()
