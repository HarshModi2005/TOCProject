# Pre³ — Deterministic Pushdown Automata for Structured LLM Generation

Implementation of the paper *"Pre³: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation"* (arXiv:2506.03887).

## Quick Start

```bash
cd pre3/

# Run all tests (73 tests)
python3 -m pytest tests/ -v

# Run benchmarks
python3 -m pre3.benchmarks.bench_preprocess
python3 -m pre3.benchmarks.bench_decode
```

## Project Structure

```
pre3/
├── grammar/
│   ├── cfg.py              # CFG data model, FIRST/FOLLOW sets
│   ├── lr1.py              # Canonical LR(1) automaton (items, CLOSURE, GOTO, ACTION/GOTO tables)
│   └── grammar_loader.py   # Grammar builders (from_rules, from_ebnf, built-in grammars)
├── dpda/
│   ├── edge.py             # PrefixConditionedEdge, StackOp types
│   ├── builder.py          # Algorithm 1: LR(1) → DPDA (cycle handling, acceptance/reduction edges)
│   ├── optimizer.py        # Edge aggregation & merging (Section 3.3)
│   └── simulator.py        # DPDASimulator (mask generation) + LR1Simulator (acceptance)
├── decoding/
│   ├── token_trie.py       # Vocabulary token → character-sequence trie
│   ├── mask_generator.py   # DPDA + trie → vocabulary mask (cached & uncached)
│   └── logits_processor.py # HuggingFace-compatible logits processor
├── tests/                  # pytest suite (73 tests)
└── benchmarks/             # Preprocessing & per-step decode benchmarks
```

## How It Works

1. **Grammar → LR(1) automaton**: Builds the canonical item-set family, state-transition graph, and ACTION/GOTO parsing tables.
2. **LR(1) → DPDA**: Algorithm 1 from the paper — detects cycles, adds acceptance edges (shifts) and reduction edges with prefix-conditioned stack matching.
3. **DPDA optimization**: Aggregates edges sharing the same transitions and merges sequential edges into shortcuts.
4. **Constrained decoding**: At each LLM decode step, walks the token trie in tandem with the DPDA to produce a vocabulary mask. Invalid tokens get masked to -∞ before sampling.

## Requirements

- Python ≥ 3.9
- pytest (for tests)
- PyTorch (optional, only needed for the HuggingFace LogitsProcessor path)
