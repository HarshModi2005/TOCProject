# Paper and visualization tool understanding
_Exported on 27/04/2026 at 19:44:43 GMT+5:30 from Cursor (3.2.11)_

---

**User**

understand the paper and visualization tool

---

**Cursor**

I’ll quickly scan the project to locate the paper and the visualization tool, then I’ll explain both in a concise, connected way (what the paper says and how the tool maps to it).

I found likely project code under `pre3`; next I’ll focus there (excluding `.venv`) to identify the exact paper file and the visualization entrypoint.

The project is centered on the **Pre³ paper idea** and gives you a practical way to inspect it.

- **Paper core idea**: convert grammar constraints into a **deterministic pushdown automaton (DPDA)** so structured generation/validation is faster and predictable.
- **Theory chain implemented**: `CFG -> NPDA -> LR(k) -> DPDA -> token validation`.
- **What this repo emphasizes**: theory-first correctness (formal-language hierarchy + tests), with LLM integration as a pluggable adapter.
- **Key claim verified in code**: for LR(1) grammars, the LR parser and generated DPDA accept/reject the same language (tested via random + exhaustive cases).
- **Important negative results included**: e.g., `wwR` is CFL but not DCFL, so deterministic construction fails as expected; `a^n b^n c^n` is shown non-CFL via pumping-lemma witness.

For the writeup, start with `pre3/report/REPORT.md` and `pre3/README.md`.

For the visualization tool, there are **two parts**:

- `pre3/tools/trace_viz.py`: generates a JSON execution trace with side-by-side LR and DPDA steps (state, stack, lookahead, consume/reduce behavior).
- `pre3/tools/trace_viewer.html`: interactive viewer that loads that JSON and lets you scrub/play both machines together, highlighting token consumption and step types.

Typical flow:

```bash
python3 -m pre3.tools.trace_viz --example balanced_parens --input "(())" --out trace.json
# then open pre3/tools/trace_viewer.html and load trace.json
```

If you want, I can next give you a **guided reading order in 15-20 minutes** (exact files + what to look for in each) so you can fully understand both the paper logic and the viewer mechanics quickly.

---

**User**

look at the paper and visualization tool and here is my vision for it
i have connected a local llm and it will throw a tokens and the visualization should show the incoming flow of tokens and maintain the pda and dpda and everything. it should basically maintain all the machines and show that in an easily understandable form. can you check the current tool and make a plan to meet my vision and make it way better than it already is

---

**Cursor**

I've now read through the full pipeline (paper-side and viewer-side). Below is a diagnosis of what the current tool actually does, where it falls short of your vision, and a concrete phased plan to push it well past where it is today.

## What the current tool actually is

It's a **two-stage, batch, post-hoc replay** — not a live system.

1. `pre3/tools/llm_pipeline.py` calls the LLM once (API or local HF), waits for the full output, parses JSON, drops non-grammar terminals → `tokens: list[str]`.
2. `pre3/tools/trace_viz.py` runs LR(1) and DPDA **to completion** on those tokens, dumps a JSON trace.
3. `pre3/tools/trace_viewer.html` loads the JSON and lets you scrub/play.

So the "trace" is finalized before the user sees a single thing.

## Where it falls short of your vision

| Vision | Today |
|---|---|
| Tokens **flow in** from a local LLM | LLM is awaited fully (`max_new_tokens=256`), then post-processed. No streaming. |
| Maintain **NPDA + DPDA + LR(1)** | Only LR(1) and DPDA are traced. NPDA simulator (`pda/simulator.py`) exists but is invisible in the viewer. |
| Visual machines that are easy to grasp | States are rendered as integer chips. No state graph, no stack-as-stack, no production highlight, no item-set view. |
| Live updates as input arrives | All steps are pre-computed; the viewer is a player, not a runner. |
| Show *why* the DPDA is the win of the paper | The `DPDASimulator.valid_symbols(...)` mask (the actual mechanism the paper uses to constrain LLM decoding) is computed but never displayed. The token-trie / mask-generator / logits-processor modules are invisible. |
| Show negative results | `wwR` (CFL ∖ DCFL) and `LRkAutomaton(strict=True)` exist, but the viewer can't demo "NPDA fans out, DPDA can't even be built". |

## Target architecture

```
                     ┌──────────────────────────────────┐
 Local HF model ────►│ TextIteratorStreamer  (one-token │
 (e.g. tiny-gpt2,    │   delta at a time, async thread) │
  Gemma, etc.)       └──────────────┬───────────────────┘
                                    ▼
                       ┌────────────────────────────┐
                       │ StreamingTerminalMatcher   │  greedy longest-match
                       │ (partial text → terminals) │  over grammar.terminals
                       └──────────────┬─────────────┘
                                      ▼
            ┌─────────────────────────┴─────────────────────────┐
            ▼                         ▼                         ▼
   IncrementalNPDA          IncrementalLR1           IncrementalDPDA
   (BFS frontier of        (shift/reduce          (deterministic step
    PDAConfigs, prunes      stack + ACTION/        + reductions chase
    dead branches)          GOTO)                   + valid_next mask)
            │                         │                         │
            └──────────────┬──────────┴────────────┬─────────────┘
                           ▼                       ▼
                    WebSocket frames        ──►  Rich browser viewer
                  (JSON, ~50–200/sec)            (Cytoscape graph +
                                                 stack viz + tape +
                                                 mask + grammar)
```

Two new ideas glue it together:

- A **streaming terminal matcher** that turns *raw model characters* into *grammar terminals* the moment they're recognized — so the user sees both the LLM's literal output and the validated stream.
- **Incremental** wrappers around the three simulators, each exposing a `feed(terminal) -> StepDelta` so the WebSocket can push one tiny event at a time.

## Phased plan

### Phase 1 — Streaming backend (the engine)

New files:
- `pre3/tools/live_server.py` — FastAPI app with:
  - `GET /` → serves `live_viewer.html`
  - `WS /run` → client sends `{grammar, prompt, model}`, server streams events.
- `pre3/tools/incremental.py` — `IncrementalDPDA`, `IncrementalLR1`, `IncrementalNPDA` (frontier-based, prunes), each producing structured step deltas.
- `pre3/tools/stream_match.py` — greedy longest-match over partial text using `grammar.terminals`, with a small lookahead buffer for ambiguous prefixes (`(` vs `(=` etc.).

Changes:
- `pre3/adapter/local_hf_llm.py` — add `stream_text()` using `transformers.TextIteratorStreamer`. Existing batch `emit()` stays for tests.
- `pre3/adapter/api_llm.py` — add `stream_text()` using OpenAI SSE (`stream: true`).

### Phase 2 — Three-pane live viewer (the UI)

New file: `pre3/tools/live_viewer.html` (extends, not replaces, the existing one).

Layout (top → bottom):

1. **LLM strip** — raw model text streaming (live, char by char), with badges showing each character that was matched into a terminal vs dropped, and a "filtered" tape directly underneath.
2. **Token tape** — terminal stream with three independent "read heads": NPDA frontier-deepest, LR(1) lookahead, DPDA consumed.
3. **Three machine panes** side-by-side:
   - **NPDA pane** — Cytoscape.js graph with the *frontier set of states* highlighted (multiple lit at once = nondeterminism). Stack(s) shown as vertical visual stacks (one per live branch, capped at e.g. 4).
   - **LR(1) pane** — same graph, single highlighted state, stack rendered visually with state numbers; a side popover shows the **item set** of the current state (closure items) so the viewer is genuinely pedagogical, not just numbers.
   - **DPDA pane** — graph with single state, real visual stack, edges colored by ACCEPTANCE vs REDUCTION as today, with the firing edge animated.
4. **Predicted-next-terminal panel** — uses `DPDASimulator.valid_symbols(config)` to render which terminals are allowed next. **This is the actual Pre³ paper mechanism**, currently invisible. Shown both as a simple chip list and as a "vocabulary mask" using `decoding/mask_generator.py`.
5. **Grammar pane** — productions list, with the production currently being reduced flashed.

### Phase 3 — LLM↔grammar inspector

A toggle that, every time the model emits a character/token, shows:

- Raw character(s) appended.
- Whether they completed a terminal (✓) / extended a partial match (…) / were dropped (✗).
- For dropped ones: the reason (not in `terminals`, or whitespace, or invalid prefix).
- For accepted ones: the DPDA's `valid_symbols` *before* the symbol — i.e. "would constrained decoding have allowed this?". This shows where an *unconstrained* model would have gone off-grammar and where Pre³ would have masked it.

This single panel is the most direct visual claim of the paper.

### Phase 4 — Determinism story (the theory payoff)

Add canned demos selectable from the UI:

- **`wwR` demo** — feeds `aabbaa` (or interactive). NPDA pane visibly *fans out* into multiple frontier branches; LR(1) pane shows a `GrammarConflictError` from `LRkAutomaton(strict=True)`; DPDA pane greys out with "no DPDA exists for this CFL". This is the cleanest possible visual of `CFL ⊋ DCFL`.
- **`a^n b^n c^n` demo** — show the pumping-lemma witness inline (already in `examples/anbncn.py`); no automaton built, just the textual proof rendered.
- **LR(0) vs LR(1) on balanced parens** — toggle `k` and watch LR(0) raise immediately while LR(1) is clean.

### Phase 5 — Polish & performance overlay

- Pause / step / rewind / "step until next reduction" / "step until next mismatch".
- Speed slider (already exists for replay).
- "Record this run → trace.json" button so live runs become saveable static traces compatible with the existing viewer.
- Performance overlay: preprocessing ms (LR build, DPDA build, mask compute), per-token decoding ms — drives home the perf claim of the paper.

## Tech choices (my recommendation, push back if you prefer otherwise)

- **Backend:** FastAPI + uvicorn. Native WebSockets, dead-simple, async friendly. (Stdlib `websockets` works but loses too much.)
- **Frontend:** Extend the existing single-file vanilla-HTML viewer. Add Cytoscape.js via CDN for the state graphs. No React/Vite — keeps the project zero-build, consistent with current style.
- **Streaming protocol:** Single WebSocket; messages are typed JSON events:
  - `model_text {char}`
  - `terminal {sym, raw_span}` / `terminal_dropped {raw, reason}`
  - `dpda_step {...}` / `lr_step {...}` / `npda_frontier {[states], [stacks]}`
  - `mask {valid_terminals}`
  - `verdict {npda, lr, dpda}`

## Risks / things I'll need to handle

- **Greedy terminal matching is grammar-sensitive** (e.g. `(` vs `((`, or `id` colliding with `if`). I'll do longest-match plus a small "needs-more-text" pending state so we never misclassify mid-token.
- **NPDA frontier explosion** on adversarial inputs. Already have `max_configs`; I'll surface "frontier capped at N" in the UI rather than freeze.
- **HF model latency on CPU**. Add a "demo / replay" mode that synthesizes a streaming source from a static token list, so the UI is testable without waiting for a real model.
- **Tokenizer mismatch** — local model's BPE tokens are not grammar terminals. The streaming matcher already handles this by matching characters, not BPE tokens.

## Files I'll add / touch

```
pre3/
├── tools/
│   ├── live_server.py          NEW  (FastAPI + WS)
│   ├── live_viewer.html        NEW  (rich UI)
│   ├── incremental.py          NEW  (Inc{NPDA,LR1,DPDA})
│   └── stream_match.py         NEW  (text → terminals stream)
├── adapter/
│   ├── local_hf_llm.py         + stream_text()
│   └── api_llm.py              + stream_text()
├── decoding/
│   └── mask_generator.py       (just imported by live_server, no edit)
├── tests/
│   ├── test_incremental.py     NEW
│   └── test_stream_match.py    NEW
└── requirements (or pyproject) +fastapi, +uvicorn, +websockets
```

Existing `tools/trace_viz.py` and `tools/trace_viewer.html` stay — the live viewer's "Record" button writes the same JSON schema, so all current tests and saved traces keep working.

Before I start building, two decisions from you would help. I'd default to the bolder option for both, but it's your call.

> **How aggressive should phase 1 be? (lower phases always come later)**
> - MVP: live FastAPI server + streaming local HF + 3-pane viewer with state graphs and visual stacks
> - MVP + the 'predicted-next-terminal' mask panel (the paper's actual contribution made visible)
> - Full Phase 1+2+3+4: streaming + UI + LLM inspector + wwR/anbncn negative-result demos

> **Where should the live UI live?**
> - Alongside: keep trace_viewer.html, add live_viewer.html (safer; both work)
> - Replace: live_viewer.html becomes the only UI, with a 'Replay saved trace' mode that subsumes the old viewer

> **Backend framework**
> - FastAPI + uvicorn (recommended, clean WebSockets)
> - Stdlib http.server + websockets package (zero new heavy deps)
> - Pick whichever is cleanest, you decide

---

**User**

live updates is not that important you can give it less prioity just focus on making the visualization better and bettter machines first. it should look very good in a demo. good work start implementing it

live updates is not that important you can give it less prioity just focus on making the visualization better and bettter machines first. it should look very good in a demo. good work start implementing it

---

**Cursor**

Got it. I’ll treat live streaming as later work and focus now on making the demo viewer substantially better: add the CFG→NPDA/PDA machine trace alongside LR(1) and DPDA, improve the visual story, and keep the generated trace format testable.

I’ve confirmed the current viewer only compares LR(1) and the derived DPDA. I’m going to add the CFG→NPDA/PDA side as a first-class machine in the trace output, then make the viewer present the pipeline as CFG/NPDA/LR/DPDA with a more demo-friendly visual layout.

The trace exporter is already partially upgraded in this working tree: it emits `format_version: 2`, includes NPDA frontier data, LR automaton metadata, DPDA graph metadata, and DPDA valid-next-token masks. I’ll build the viewer around that richer schema instead of redoing the exporter.

The viewer is more advanced than the earlier snapshot: it already has bundled samples, a launcher, Cytoscape graphs, NPDA frontier cards, LR item sets, DPDA masks, and a single demo scrubber. I’ll verify it now, then tighten anything broken or missing rather than rewriting stable pieces.

Focused tests are passing. I’m going to regenerate the bundled demo traces once so the sample JSONs definitely match the upgraded `format_version: 2` schema the viewer expects.

The generated viewer files are currently untracked in git, so I’m being careful not to disturb unrelated modified files. I’ll add focused assertions for the new visualization schema so the NPDA frontier, automaton graphs, and DPDA grammar-mask data are covered.

The focused tests now cover the upgraded schema and still pass. I’m going to do a quick local smoke run of the viewer launcher so we know the demo page and generated samples can be served cleanly.

A viewer server was already running on port `8765`, and both the HTML page and bundled sample JSON respond with HTTP 200. I’ll do one more sanity pass on the trace payload itself and then wrap up with what changed and how to demo it.

Implemented/verified the upgraded demo path.

The visualization now centers on the machines first:

- `NPDA`: CFG-derived nondeterministic machine with frontier trace and live configurations.
- `LR(1)`: parser machine with item sets, stack symbols, reductions, and highlighted productions.
- `DPDA`: Pre³ deterministic machine with graph, stack ops, reduction/acceptance edges, and visible next-token grammar mask.
- Viewer: `pre3/tools/trace_viewer.html` has the polished three-machine demo layout.
- Launcher: `python3 -m pre3.tools.viewer` serves the viewer and bundled examples.
- Samples regenerated under `pre3/tools/samples/`.

I also strengthened `pre3/tests/test_trace_viz.py` to assert the v2 schema includes NPDA frontier data, automaton graphs, LR stack-symbol payloads, and DPDA valid-next-terminal masks.

Verification passed:

```bash
python3 -m pytest tests/test_trace_viz.py tests/test_pda.py tests/test_dpda_equivalence.py -q
# 28 passed
```

Viewer smoke check passed too: `http://127.0.0.1:8765/trace_viewer.html` and `samples/parens.json` both return `200`.

