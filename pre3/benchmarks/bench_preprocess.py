"""
Benchmark: preprocessing time  (Grammar → LR(1) → DPDA → Optimize).

Run with:  python -m pre3.benchmarks.bench_preprocess
"""

from __future__ import annotations

import time
from typing import Callable

from pre3.grammar.cfg import ContextFreeGrammar
from pre3.grammar.grammar_loader import arithmetic, balanced_parens, from_rules, simple_json
from pre3.grammar.lr1 import LR1Automaton
from pre3.dpda.builder import build_dpda
from pre3.dpda.optimizer import optimize


def bench(name: str, grammar_fn: Callable[[], ContextFreeGrammar], runs: int = 5) -> None:
    times: list[float] = []
    for _ in range(runs):
        g = grammar_fn()
        t0 = time.perf_counter()
        lr1 = LR1Automaton(g)
        dpda = build_dpda(lr1)
        dpda_opt = optimize(dpda)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    best = min(times)
    print(
        f"  {name:30s}  avg={avg*1000:8.2f} ms  "
        f"best={best*1000:8.2f} ms  "
        f"states={dpda_opt.num_states}  edges={dpda_opt.edge_count}"
    )


def main() -> None:
    print("=== Pre³ Preprocessing Benchmark ===\n")

    bench("balanced_parens", balanced_parens)
    bench("arithmetic", arithmetic)
    bench("simple_json", simple_json)

    # Larger grammar: nested JSON with more terminals
    def large_json():
        return from_rules(
            {
                "Value": ["Object", "Array", "String", "Number", "true", "false", "null"],
                "Object": ["{ Members }", "{ }"],
                "Members": ["Pair , Members", "Pair"],
                "Pair": ["String : Value"],
                "Array": ["[ Elements ]", "[ ]"],
                "Elements": ["Value , Elements", "Value"],
                "String": ["QUOTE Chars QUOTE"],
                "Chars": ["Char Chars", ""],
                "Char": ["a", "b", "c", "d", "e", "0", "1", "2"],
                "Number": ["Digits"],
                "Digits": ["Digit Digits", "Digit"],
                "Digit": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            },
            start="Value",
        )

    bench("large_json", large_json)
    print("\nDone.")


if __name__ == "__main__":
    main()
