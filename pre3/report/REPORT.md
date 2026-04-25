# Pre³ — Theory-of-Computation Report

## 1. Scope and intent

This project implements, end-to-end, the chain of constructions

> **CFG  →  NPDA  →  LR(k)  →  DPDA  →  (LLM-validation adapter)**

with the LLM piece intentionally stubbed out so that the body of the
project can be evaluated as a Theory-of-Computation artifact: the
formal-language hierarchy realized in code, with each construction
proved correct by exhaustive and randomized property tests.

The original motivation comes from the paper *"Pre³: Enabling
Deterministic Pushdown Automata for Faster Structured LLM Generation"*
(arXiv:2506.03887), but here we deliberately foreground the underlying
automata theory rather than the LLM application.

## 2. Formal background

### 2.1 Definitions

A **context-free grammar** is a 4-tuple `G = (V, Σ, P, S)` where `V` is a
finite set of non-terminals, `Σ` is a finite set of terminals (`V ∩ Σ = ∅`),
`P ⊆ V × (V ∪ Σ)*` is a finite set of productions, and `S ∈ V` is the start.

A **non-deterministic pushdown automaton** is a 7-tuple
`N = (Q, Σ, Γ, δ, q₀, Z₀, F)` where `δ : Q × (Σ ∪ {ε}) × Γ → 𝒫(Q × Γ*)`.
A **deterministic pushdown automaton** restricts `δ` to be deterministic:
for every `(q, X) ∈ Q × Γ`, *either* `δ(q, ε, X)` is a singleton and
`δ(q, a, X)` is empty for every `a ∈ Σ`, *or* `δ(q, ε, X)` is empty and
`|δ(q, a, X)| ≤ 1` for every `a`.

### 2.2 Theorems we depend on

* **(Chomsky 1962, Sipser 2.20)** A language `L` is context-free iff
  `L = L(N)` for some NPDA `N`.
* **(Knuth 1965)** A language `L` is **DCFL** with the *prefix property*
  iff `L = L(G)` for some LR(1) grammar `G`.  Equivalently, the class of
  languages recognised by DPDAs is exactly the deterministic context-free
  languages.
* **(Knuth 1965)** For all `k ≥ 1`, every LR(k) grammar can be transformed
  into an LR(1) grammar accepting the same language.  Hence
  `L(LR(k)) = L(LR(1))` as language classes for `k ≥ 1`.  LR(0) is
  strictly weaker.
* **(Pumping lemma for CFLs)** If `L` is context-free, there exists `p ≥ 1`
  such that every `s ∈ L` with `|s| ≥ p` decomposes as `s = uvxyz` with
  `|vy| ≥ 1`, `|vxy| ≤ p`, and `uvⁿxyⁿz ∈ L` for all `n ≥ 0`.

The last is used to *exclude* `aⁿbⁿcⁿ` from CFL.

### 2.3 The hierarchy at a glance

```
   regular  ⊊  DCFL  ⊊  CFL  ⊊  context-sensitive
              =                  ⊋
       L(LR(1))                 aⁿbⁿcⁿ ∉ CFL
       =                         (pumping lemma)
       L(DPDA)
                  ⊋
              wwᴿ ∈ CFL ∖ DCFL
              (no DPDA recognises it)
```

The repository contains a **realisation** of every line of this diagram.

## 3. Constructions implemented

### 3.1 CFG → NPDA  (`pre3/pda/cfg_to_pda.py`)

A standard one-state-plus-bookkeeping construction:

* `q_start` does the initial push of `S` onto the stack.
* `q` (the main state) has, for each production `A → α`, an ε-transition
  `δ(q, ε, A) ∋ (q, α)`, and for each terminal `a ∈ Σ`, the matching
  pop-on-input transition `δ(q, a, a) ∋ (q, ε)`.
* `q_acc` is reached via ε from `q` once the stack-bottom marker `⊥` is
  exposed (i.e. the start symbol has been fully derived).

The simulator (`pre3/pda/simulator.py`) is a bounded BFS over
configurations `(state, remaining_input, stack)` — sound and complete on
inputs that fit within the configurable `max_configs` and
`max_stack_depth` limits.

This realises Chomsky's `CFL ⊆ L(NPDA)` direction constructively.

### 3.2 LR(k) for arbitrary `k ≥ 0`  (`pre3/grammar/lrk.py`)

The implementation generalises the textbook LR(1) construction to
`k`-symbol lookahead:

* **Items** `[A → α · β, w]` carry a *length-`k` tuple* `w` of terminal
  symbols (padded with `$`).
* **FIRST_k** is computed by a fixed-point iteration with a
  *truncated-concatenation* operator: `S₁ ∘ₖ S₂ = { (a · b)[:k] : a ∈ S₁, b ∈ S₂ }`,
  with the rule that any `a` already at length `k` is not extended.
* **CLOSURE** for an item `[A → α · Bβ, w]` adds `[B → · γ, u]` for every
  production `B → γ` and every `u ∈ FIRST_k(βw)`.
* **GOTO** is identical to the LR(1) version on items.
* The **ACTION table** is keyed by `(state, length-k lookahead tuple)`.
  Conflicts are detected:
  - *shift/reduce*: lookahead `w` triggers reduce, but `w[0]` (or, for
    `k = 0`, any terminal) also triggers a shift.
  - *reduce/reduce*: two reduce items share the same lookahead `w`.

For each conflict, a `GrammarConflictError` carries the offending state,
the lookahead, and the conflicting actions, making the conflict
*explainable* — a "negative result" we can show.

### 3.3 LR(1) → DPDA  (`pre3/dpda/builder.py`)

Algorithm 1 from the Pre³ paper, with three corrections relative to the
draft we inherited:

1. **`stack_match` ordering bug**: the original code generated stack-match
   tuples in *reversed* order, missing the top-of-stack anchor.  We now
   build `stack_match = tuple(reversed(path))` so that `stack_match[-1]`
   is the runtime top.
2. **Exposed-state inclusion**: the original code dropped the exposed
   state from `stack_match`, which collapsed multiple distinct backward
   paths (with different GOTO targets) into one edge.  Including the
   exposed state preserves uniqueness.
3. **Reduction edges are now ε-fires**: an LR(1) reduction fires *when*
   the lookahead matches but does not consume the symbol.  This matches
   the operational semantics of LR parsing and lets *chain reductions*
   (e.g. `T → F`, `E → T`, …) compose naturally in the simulator.
   The previously-attempted "merge with shift" optimisation was incorrect
   for left-recursive grammars and has been removed in favour of clarity.

### 3.4 Determinism verifier  (`pre3/dpda/verifier.py`)

We make determinism a *checked invariant*.  Two stack-match patterns
`m₁`, `m₂` *overlap* iff some real stack matches both — equivalently,
the shorter one is a top-anchored suffix of the longer (or either is
empty).  The verifier groups edges by `(source_state, accepted_symbol)`
and asserts pairwise non-overlap of stack-match patterns.  On failure
it raises `NondeterminismError` carrying the offending edge pairs.

`determinism_certificate(dpda)` produces a printable verdict used at the
end of every build.

### 3.5 DPDA simulator  (`pre3/dpda/simulator.py`)

Edges have two kinds:

| kind          | semantics in `step()`                         |
|---------------|-----------------------------------------------|
| `ACCEPTANCE`  | consumes the next input symbol; advances pos. |
| `REDUCTION`   | fires when its `accepted_symbols` matches the lookahead, but does NOT consume; the simulator immediately re-tries from the new state. |

This is exactly the operational semantics of LR parsing.  Final
acceptance is signalled when, after consuming all input and chasing
reductions on `END_MARKER`, the simulator lands in a state of
`dpda.accepting_states`.

The same module also provides `LR1Simulator` — a classical shift-reduce
parser using the ACTION/GOTO tables — used as the *language oracle* in
the equivalence tests below.

## 4. Equivalence theorem (verified empirically)

> **Claim.**  For every LR(1) grammar `G`,
> `L(LR1Simulator(G)) == L(DPDASimulator(build_dpda(G)))`.

`tests/test_dpda_equivalence.py` verifies this on:

* 7 distinct grammars (balanced parens, arithmetic, aⁿbⁿ, chain rules,
  left-recursive list, wcwᴿ, dyck-2 right-recursive form);
* 400 random terminal strings each;
* exhaustive enumeration of all strings up to length 5 over each
  grammar's alphabet for the most-surprising grammars;
* property-based testing via `hypothesis` for balanced parens and aⁿbⁿ.

The DPDA's `verify_determinism` is also asserted clean for each grammar.

## 5. Negative results — the heart of the theory

### 5.1 wwᴿ is in CFL ∖ DCFL  (`pre3/examples/wwR.py`)

The grammar `S → a S a | b S b | ε` is *unambiguous* but generates
`{w wᴿ : w ∈ {a,b}*}` — a CFL that is *not* DCFL because no
deterministic automaton can decide where `w` ends without a centre marker.

* `cfg_to_npda(g)` produces a perfectly-working NPDA (the simulator
  non-deterministically guesses the midpoint).
* `LRkAutomaton(g, k, strict=True)` raises `GrammarConflictError` for
  every `k ∈ {1, 2, 3}`, **as the theory predicts**.

This is verified by `tests/test_examples.py::test_wwR_lrk_fails_for_all_small_k`.

### 5.2 aⁿbⁿcⁿ is not context-free  (`pre3/examples/anbncn.py`)

We do not provide a CFG.  Instead, `pumping_lemma_witness(p)` returns
`s = aᵖbᵖcᵖ` and the canonical pumping-lemma argument as text.  The test
`test_anbncn_is_witnessed_by_pumping_lemma` checks the witness shape and
explanation.

### 5.3 LR(0) is strictly weaker than LR(1)  (`pre3/grammar/lrk.py`, `tests/test_lrk.py`)

Even balanced-parens fails LR(0) — the ε-reduction `S → ε` has no
lookahead to disambiguate it from the shift on `(`.  In strict mode
the LR(0) automaton raises immediately; in non-strict mode it surfaces
the full conflict list.

## 6. Complexity sketch

* **Grammar → LR(k) automaton**: number of items per state is bounded by
  `|productions| · (|Σ| + 1)^k`; the number of states is at most `2^|items|`
  but is exponentially smaller in practice for unambiguous grammars.
  Construction runs in `O(|items|² · |Σ|)` time.
* **LR(1) → DPDA**: each backward-path enumeration is `O(|states|^pop_count)`
  in the worst case, but pop counts are bounded by the longest production
  body and almost all grammars have very small bodies (≤ 5).
* **Determinism check**: `O(|edges|² / |state|)` worst case; trivially
  parallelisable.
* **Per-symbol DPDA step**: `O(1)` lookup + `O(stack_match_length)`
  comparison on the indexed `(state, symbol)` map.

Empirical numbers from `pre3/benchmarks/bench_preprocess.py` and
`bench_decode.py` are reported in `report/benchmarks.md` (run them to
populate; see the `make bench` target in the README for a future
addition).

## 7. Limitations and honest caveats

* The DPDA construction is implemented for **LR(1)** specifically.  The
  generic LR(k) machinery is exercised at the *grammar/parser* level, not
  yet at the *DPDA-construction* level — extending Algorithm 1 to k ≥ 2
  requires k-symbol look-ahead in the trie/input pipeline; this is left
  as future work and is documented in the report.
* The optimiser (`pre3/dpda/optimizer.py`) is *not* part of the verified
  pipeline; the equivalence tests run on the *unoptimised* DPDA.  The
  optimiser is included for the LLM-decoding application but is
  intentionally not relied upon for correctness.
* The LLM adapter is a stub (`MockLLMSource`).  Real-LLM integration is
  out of scope for the ToC submission and is sketched only as a future
  pluggable interface.

## 8. How to read the code

| Theme                     | Start file                                |
|---------------------------|-------------------------------------------|
| CFG basics                | `pre3/grammar/cfg.py`                     |
| LR(k) construction        | `pre3/grammar/lrk.py`                     |
| Generic NPDA              | `pre3/pda/pda.py`                         |
| CFG → NPDA                | `pre3/pda/cfg_to_pda.py`                  |
| Pre³ DPDA construction    | `pre3/dpda/builder.py`                    |
| Determinism invariant     | `pre3/dpda/verifier.py`                   |
| Operational equivalence   | `tests/test_dpda_equivalence.py`          |
| Examples + classification | `pre3/examples/catalogue.py`              |

## 9. Bibliography

* D. Knuth, *On the translation of languages from left to right*,
  Information and Control 8(6), 1965.
* N. Chomsky, *Context-free grammars and pushdown storage*,
  MIT Quarterly Progress Reports 65, 1962.
* M. Sipser, *Introduction to the Theory of Computation*, 3rd ed., 2012.
* Pre³ paper, *Enabling Deterministic Pushdown Automata for Faster
  Structured LLM Generation*, arXiv:2506.03887.
