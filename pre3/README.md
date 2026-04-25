# Pre³ — A Theory-of-Computation Project on Pushdown Automata

A faithful, fully-tested Python implementation of the formal-language
hierarchy from CFG → NPDA → LR(k) → DPDA, designed as a **Theory of
Computation** project artifact and as the substrate for grammar-constrained
LLM decoding (the original Pre³ paper application).

## What this repository proves, in code

| Theorem (textbook)                                       | Realization in this repo                                  | Tested in                       |
|----------------------------------------------------------|-----------------------------------------------------------|---------------------------------|
| Every CFG has an equivalent NPDA  (Chomsky 1962)         | `pre3.pda.cfg_to_pda.cfg_to_npda`                         | `tests/test_pda.py`             |
| L(LR(1)) = DCFL with prefix property  (Knuth 1965)       | `pre3.grammar.lrk.LRkAutomaton(g, k=1)` + DPDA pipeline   | `tests/test_dpda_equivalence.py`|
| LR(k) is well-defined for any k ≥ 0                      | `pre3.grammar.lrk.LRkAutomaton(g, k=K)` for arbitrary K    | `tests/test_lrk.py`             |
| **DCFL ⊊ CFL**:  wwᴿ is CFL but not DCFL                 | LR(k) construction **fails** on `pre3.examples.wwR`        | `tests/test_examples.py`        |
| aⁿbⁿcⁿ is **not CFL**  (pumping lemma)                   | `pre3.examples.anbncn.pumping_lemma_witness`              | `tests/test_examples.py`        |
| LR(0) ⊊ LR(1) (LR(0) cannot do balanced parens)          | `LRkAutomaton(balanced_parens(), k=0, strict=True)` raises | `tests/test_lrk.py`             |

## Equivalence test — the core correctness story

For every LR(1) grammar `G`, the test suite asserts
`L(DPDA(G)) == L(LR(1)-parser(G))` via random sampling (400 strings/grammar)
**and** exhaustive enumeration up to length 5.  See
`tests/test_dpda_equivalence.py`.

## Architecture

```
pre3/
├── grammar/
│   ├── cfg.py              CFG, FIRST/FOLLOW
│   ├── grammar_loader.py   Grammar builders (from_rules, from_ebnf, …)
│   ├── lr1.py              Canonical LR(1) automaton (legacy / specialised path)
│   └── lrk.py              ★ Generic LR(k) for ANY k ≥ 0, with conflict reporter
├── pda/
│   ├── pda.py              ★ Generic NPDA (Q, Σ, Γ, δ, q₀, Z₀, F)
│   ├── simulator.py        ★ BFS NPDA simulator (final-state OR empty-stack)
│   └── cfg_to_pda.py       ★ Standard CFG → NPDA construction (Sipser 2.20)
├── dpda/
│   ├── edge.py             Prefix-conditioned edges
│   ├── builder.py          LR(1) → DPDA construction (Pre³ Algorithm 1)
│   ├── simulator.py        DPDA simulator with FULL-string `run()` & oracle
│   ├── verifier.py         ★ Determinism verifier  +  certificate
│   └── optimizer.py        Edge aggregation / merging  (preprocessing)
├── examples/               ★ Canonical languages (aⁿbⁿ, wcwᴿ, wwᴿ, aⁿbⁿcⁿ, dyck-2)
│   └── catalogue.py        Registry of examples with formal classification
├── viz/                    ★ Graphviz DOT rendering for DPDAs and NPDAs
├── adapter/                ★ Pluggable string source (mock LLM stub for now)
├── decoding/               LLM-side application: vocab masks, logits processor
├── benchmarks/             Preprocessing / per-step decode timing
├── tests/                  138 passing tests (pytest + hypothesis)
└── report/
    ├── REPORT.md           ★ Full theoretical writeup
    └── figures/            Auto-generated DOT diagrams (render with `dot`)
```

★ = **new in the ToC reframing.**

## Quick start

```bash
cd /app/pre3
pip install pytest hypothesis
python3 -m pytest tests/ -v               # 138 tests, ~1 s
python3 -m pre3.benchmarks.bench_preprocess
```

### A 30-second demo

```python
from pre3.grammar.grammar_loader      import balanced_parens
from pre3.grammar.lrk                 import LRkAutomaton, LRkSimulator
from pre3.grammar.lr1                 import LR1Automaton
from pre3.dpda.builder                import build_dpda
from pre3.dpda.simulator              import DPDASimulator
from pre3.dpda.verifier               import determinism_certificate
from pre3.pda.cfg_to_pda              import cfg_to_npda
from pre3.pda.simulator               import PDASimulator

g = balanced_parens()                            # S → ( S ) | ε

# 1) LR(k) parser, any k
lrk = LRkSimulator(LRkAutomaton(g, k=2))
print(lrk.accepts(['(', '(', ')', ')']))         # True

# 2) Generic NPDA (CFL ≡ NPDA)
npda = cfg_to_npda(g)
print(PDASimulator(npda).accepts(['(', ')']))    # True

# 3) DPDA built via the LR(1) → DPDA pipeline; verify determinism.
dpda = build_dpda(LR1Automaton(g))
print(determinism_certificate(dpda))             # ✓ DETERMINISM VERIFIED …
print(DPDASimulator(dpda).accepts(['(', ')']))   # True
```

### A "negative result" demo — the heart of the theory

```python
from pre3.examples.wwR     import grammar, npda
from pre3.grammar.lrk      import LRkAutomaton, GrammarConflictError
from pre3.pda.cfg_to_pda   import cfg_to_npda
from pre3.pda.simulator    import PDASimulator

g = grammar()                                     # S → aSa | bSb | ε  (= wwᴿ)

# DPDA pipeline FAILS — wwᴿ is CFL but not DCFL.
for k in [1, 2, 3, 4]:
    try:
        LRkAutomaton(g, k=k, strict=True)
        print(f'LR({k}): clean (would be a counterexample to theory!)')
    except GrammarConflictError as e:
        print(f'LR({k}): {e.kind} conflict — exactly as expected')

# ... but the NPDA, allowed to be non-deterministic, accepts wwᴿ fine:
sim = PDASimulator(cfg_to_npda(g), max_configs=20000)
assert sim.accepts(['a', 'b', 'b', 'a'])          # ✓
assert not sim.accepts(['a', 'b', 'a'])           # ✓
```

## How LLM-style usage plugs in (for later)

The left of the pipeline is intentionally a stub:

```python
from pre3.adapter.mock_llm import MockLLMSource   # ← swap for a real LLM later
from pre3.dpda.simulator   import DPDASimulator
from pre3.dpda.builder     import build_dpda
from pre3.grammar.lr1      import LR1Automaton
from pre3.grammar.grammar_loader import balanced_parens

src   = MockLLMSource([['(', ')'], ['(', ')', ')']])
dpda  = build_dpda(LR1Automaton(balanced_parens()))
check = DPDASimulator(dpda).accepts

for tokens in src.emit():
    print(tokens, '→', 'valid' if check(tokens) else 'invalid')
```

When you want a real LLM, write a `RealLLMSource` that prompts the model
and yields its tokenised output — **nothing in `pre3/grammar/`,
`pre3/dpda/`, or `pre3/pda/` needs to change**.

## Visualisation

```bash
python3 -c "from pre3.viz.dpda_viz import write_dpda_dot; \
            from pre3.dpda.builder import build_dpda; \
            from pre3.grammar.lr1 import LR1Automaton; \
            from pre3.grammar.grammar_loader import balanced_parens; \
            write_dpda_dot(build_dpda(LR1Automaton(balanced_parens())), 'parens.dot')"
dot -Tpng parens.dot > parens.png
```

## Reading order for graders

1. `report/REPORT.md` — the formal writeup.
2. `tests/test_dpda_equivalence.py` — the central correctness theorem in code.
3. `examples/wwR.py` + `tests/test_examples.py::test_wwR_lrk_fails_for_all_small_k` — the negative result.
4. `grammar/lrk.py` — LR(k) construction algorithm.
5. `dpda/verifier.py` — what makes the "D" in DPDA a checked invariant.
