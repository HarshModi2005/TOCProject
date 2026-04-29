"""
Live pipeline:  OpenAI-compatible LLM  →  terminal tokens  →  DPDA accept?

Requires ``OPENAI_API_KEY`` in the environment (unless passed via flags).

  export OPENAI_API_KEY=sk-...
  python3 -m pre3.tools.llm_pipeline
  python3 -m pre3.tools.llm_pipeline --grammar anbn
  python3 -m pre3.tools.llm_pipeline --challenge   # many grammars, harder prompts
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from ..adapter.api_llm import OpenAILLMSource
from ..adapter.local_hf_llm import LocalHFLLMSource
from ..dpda.builder import build_dpda
from ..dpda.simulator import DPDASimulator
from ..dpda.verifier import determinism_certificate
from ..grammar.grammar_loader import arithmetic, balanced_parens, from_rules
from ..grammar.lr1 import LR1Automaton
from .trace_viz import build_trace_for_grammar

_GRAMMAR_CHOICE = {
    "parens",
    "anbn",
    "dyck2",
    "arith",
    "if_else",
    "wcwR",
}

_GRAMMAR_BUILDERS = {
    "parens": lambda: balanced_parens(),
    "anbn": lambda: from_rules({"S": ["a S b", ""]}, start="S"),
    "dyck2": lambda: from_rules(
        {"S": ["( S ) S", "[ S ] S", ""]},
        start="S",
    ),
    "arith": lambda: arithmetic(),
    "if_else": lambda: from_rules(
        {
            "S": [
                "if e then S else S",
                "if e then S",
                "x",
            ],
        },
        start="S",
    ),
    "wcwR": lambda: from_rules(
        {"S": ["a S a", "b S b", "c"]},
        start="S",
    ),
}


def _grammar(name: str):
    b = _GRAMMAR_BUILDERS.get(name)
    if b is None:
        raise SystemExit(f"Unknown grammar: {name}")
    return b()


def _trace_example_name(grammar_id: str) -> str:
    return {
        "parens": "balanced_parens",
        "arith": "arithmetic",
    }.get(grammar_id, grammar_id)


def _write_trace_bundle(path: str, traces: list[dict]) -> None:
    payload = {
        "format_version": 1,
        "kind": "trace_bundle",
        "traces": traces,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote viewer trace bundle to {path}")


def _challenge_sweep() -> list[tuple[str, str, list[str]]]:
    """
    (grammar_id, one-line label, [user message, ...]).
    Asks the model for nontrivial examples per language.
    """
    return [
        (
            "parens",
            "deep + invalid",
            [
                "Emit a JSON 'tokens' array of exactly 24 symbols: 12 '(' then 12 ')' (valid Dyck). "
                "Only characters '(' and ')' as single-character strings.",
                "Emit tokens for INVALID parens: 5 '(' and 3 ')' only (too few closes).",
            ],
        ),
        (
            "dyck2",
            "mixed + mismatch",
            [
                "Emit a VALID sequence using '(' ')' '[' ']' e.g. \"([()])\" as separate string tokens, "
                "no spaces inside tokens.",
                "Emit INVALID: \"([)]\" (wrong bracket at end) as separate tokens.",
            ],
        ),
        (
            "arith",
            "nested + bad",
            [
                "Emit a valid token list for: ( id + id * ( id + id ) ) * id  — one token per symbol.",
                "Emit INVALID: id + * id (consecutive bad operators) as token list.",
            ],
        ),
        (
            "if_else",
            "dangling-else + garbage",
            [
                "Emit a valid if/else program as tokens, nested: if e then if e then x else x",
                "Emit an INVALID extra 'else' at the end: if e then x else x else x",
            ],
        ),
        (
            "wcwR",
            "pal + wrong half",
            [
                "Emit a valid w c w^R: for w = a b, centre c, use tokens a, b, c, b, a",
                "Emit INVALID: a, b, c, a, b  (right half is not the reverse of left)",
            ],
        ),
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--grammar",
        default="parens",
        choices=sorted(_GRAMMAR_CHOICE),
        help="Which toy grammar the LLM should emit tokens for (default: parens).",
    )
    p.add_argument(
        "--backend",
        default="api",
        choices=["api", "local-hf"],
        help="Token source backend: remote API or local HuggingFace model.",
    )
    p.add_argument(
        "--challenge",
        action="store_true",
        help="Run a harder multi-grammar battery (several live API calls, varied prompts).",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="Defaults to $OPENAI_API_KEY.",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL (default: https://api.openai.com/v1).",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="Model name (default: gpt-4o-mini or $OPENAI_MODEL).",
    )
    p.add_argument(
        "--local-model",
        default="sshleifer/tiny-gpt2",
        help="HF model id/path for --backend local-hf (default: tiny open model).",
    )
    p.add_argument(
        "--local-download",
        action="store_true",
        help="Allow online download for local-hf if model is not cached.",
    )
    p.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token for gated models (defaults to $HF_TOKEN).",
    )
    p.add_argument(
        "--trace-out",
        default=None,
        help="Write trace JSON for the HTML viewer. Multiple LLM emissions are saved as a bundle.",
    )
    args = p.parse_args()

    if args.backend == "api" and not args.api_key:
        print(
            "Error: set OPENAI_API_KEY or pass --api-key <key>.\n"
            "  export OPENAI_API_KEY=sk-...",
            file=sys.stderr,
        )
        return 2

    if args.challenge:
        print("Challenge sweep — one grammar block per section\n")
        traces: list[dict] = []
        for gid, label, user_messages in _challenge_sweep():
            g = _grammar(gid)
            lr1 = LR1Automaton(g)
            dpda = build_dpda(lr1)
            sim = DPDASimulator(dpda)
            print()
            print(f"━━ {gid:8s}  ({label})  states={dpda.num_states}  edges={dpda.edge_count}")
            print(determinism_certificate(dpda))
            if args.backend == "api":
                src = OpenAILLMSource(
                    grammar=g,
                    user_messages=user_messages,
                    model=args.model,
                    api_key=args.api_key,
                    base_url=args.base_url,
                )
            else:
                src = LocalHFLLMSource(
                    grammar=g,
                    user_messages=user_messages,
                    model_id_or_path=args.local_model,
                    local_files_only=(not args.local_download),
                    hf_token=args.hf_token,
                )
            for j, tokens in enumerate(src.emit(), start=1):
                ok = sim.accepts(tokens)
                mark = "ACCEPT" if ok else "REJECT"
                print(f"  [{j}] {mark:6}  {tokens!s}")
                if args.trace_out:
                    traces.append(
                        build_trace_for_grammar(
                            _trace_example_name(gid),
                            g,
                            tokens,
                            source={
                                "kind": "llm",
                                "backend": args.backend,
                                "grammar": gid,
                                "model": args.model if args.backend == "api" else args.local_model,
                                "challenge": True,
                                "label": label,
                                "prompt_index": j,
                                "prompt": user_messages[j - 1],
                            },
                        )
                    )
        if args.trace_out:
            _write_trace_bundle(args.trace_out, traces)
        return 0

    g = _grammar(args.grammar)
    lr1 = LR1Automaton(g)
    dpda = build_dpda(lr1)
    sim = DPDASimulator(dpda)

    if args.grammar == "parens":
        prompts = [
            'Emit a *valid* balanced-parenthesis sequence of 4–8 tokens. '
            "Only use '(' and ')' in the JSON tokens array."
            ' Example shape: 3 nested pairs: "( ( ) )" as four tokens "(": "(" "(" then ")" ")".',
            "Emit a short *invalid* sequence (one extra closing paren) as the tokens list.",
        ]
    elif args.grammar == "anbn":
        prompts = [
            "Emit a valid a^n b^n string as tokens: n between 0 and 4 (only 'a' and 'b' in the array).",
            "Emit an invalid a/b sequence that is not in a^n b^n (describe why in a comment field? "
            "No — only output the JSON with tokens, nothing else).",
        ]
    elif args.grammar == "dyck2":
        prompts = [
            "Emit a valid interleaved () and [] sequence of 8–14 tokens, e.g. like ([()]) style.",
            "Emit an INVALID interleaved sequence with one bracket type mismatch (wrong close).",
        ]
    elif args.grammar == "arith":
        prompts = [
            "Emit a valid infix token list: at least id * ( id + id )  using only id + * ( )",
            "Emit INVALID: id + * id",
        ]
    elif args.grammar == "if_else":
        prompts = [
            "Emit valid tokens: if e then if e then x else x",
            "Emit INVALID: if e then x else x else x",
        ]
    else:  # wcwR
        prompts = [
            "Emit valid palindrome: a, b, a, c, a, b, a  (w=a b, centre c, w^R=b a? wait use w=aba) "
            "Actually emit: a, b, a, c, a, b, a for w=aba, c, w^R=aba",
            "Emit INVALID: a, b, c, a, b  (right side not w^R of left w)",
        ]

    print(determinism_certificate(dpda))
    print()
    traces: list[dict] = []

    if args.backend == "api":
        src = OpenAILLMSource(
            grammar=g,
            user_messages=prompts,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    else:
        src = LocalHFLLMSource(
            grammar=g,
            user_messages=prompts,
            model_id_or_path=args.local_model,
            local_files_only=(not args.local_download),
            hf_token=args.hf_token,
        )

    for i, tokens in enumerate(src.emit(), start=1):
        ok = sim.accepts(tokens)
        mark = "ACCEPT" if ok else "REJECT"
        print(f"  [{i}] {mark:6}  {tokens!s}")
        if args.trace_out:
            traces.append(
                build_trace_for_grammar(
                    _trace_example_name(args.grammar),
                    g,
                    tokens,
                    source={
                        "kind": "llm",
                        "backend": args.backend,
                        "grammar": args.grammar,
                        "model": args.model if args.backend == "api" else args.local_model,
                        "challenge": False,
                        "prompt_index": i,
                        "prompt": prompts[i - 1],
                    },
                )
            )

    if args.trace_out:
        _write_trace_bundle(args.trace_out, traces)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
