"""
Integration tests: LLM adapter + DPDA (``chat_completion`` mocked — no network).

For optional live API checks, set ``OPENAI_API_KEY`` and run with
``pytest -m integration`` (tests skip when unset).
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pytest

from pre3.adapter.api_llm import OpenAILLMSource, chat_completion
from pre3.grammar.grammar_loader import balanced_parens, from_rules
from pre3.grammar.lr1 import LR1Automaton
from pre3.dpda.builder import build_dpda
from pre3.dpda.simulator import DPDASimulator


# ------------------------------------------------------------------
# Full pipeline: MockLLM → OpenAILLMSource.emit → DPDASimulator
# ------------------------------------------------------------------


def test_llm_place_mocked_golden_parens_valid_then_invalid():
    """Two mocked completions: first string in L(G), second not."""
    g = balanced_parens()
    sim = DPDASimulator(build_dpda(LR1Automaton(g)))
    src = OpenAILLMSource(
        grammar=g,
        user_messages=["a", "b"],
        api_key="sk-test",
    )
    fakes = [
        json.dumps({"tokens": ["(", "(", ")", ")"]}),
        json.dumps({"tokens": [")", "("]}),
    ]
    with patch("pre3.adapter.api_llm.chat_completion", side_effect=fakes) as m:
        out = list(src.emit())
    assert m.call_count == 2
    assert sim.accepts(out[0]) is True
    assert sim.accepts(out[1]) is False


def test_llm_place_mocked_anbn():
    g = from_rules({"S": ["a S b", ""]}, start="S")
    sim = DPDASimulator(build_dpda(LR1Automaton(g)))
    src = OpenAILLMSource(
        grammar=g,
        user_messages=["x", "y"],
        api_key="sk-test",
    )
    fakes = [
        json.dumps({"tokens": ["a", "a", "b", "b"]}),
        json.dumps({"tokens": ["a", "b", "a"]}),
    ]
    with patch("pre3.adapter.api_llm.chat_completion", side_effect=fakes):
        t0, t1 = list(src.emit())
    assert sim.accepts(t0) is True
    assert sim.accepts(t1) is False


def test_llm_mocked_strips_non_terminals_before_dpda():
    """Model noise is dropped; only grammar terminals reach the automaton."""
    g = balanced_parens()
    sim = DPDASimulator(build_dpda(LR1Automaton(g)))
    src = OpenAILLMSource(
        grammar=g,
        user_messages=["x"],
        api_key="sk-test",
    )
    raw = json.dumps(
        {"tokens": ["<noise>", "(", "bogus", ")", "extra", ")"]}
    )
    with patch("pre3.adapter.api_llm.chat_completion", return_value=raw):
        toks = next(src.emit())
    assert toks == ["(", ")", ")"]
    assert sim.accepts(toks) is False


# ------------------------------------------------------------------
# tools.llm_pipeline main() — argv + mocked HTTP
# ------------------------------------------------------------------


def test_llm_pipeline_cli_parens_prints_accept_then_reject(capsys):
    fakes = [
        json.dumps({"tokens": ["(", "(", ")", ")"]}),
        json.dumps({"tokens": [")", "(", ")", ")", "(", ")", ")", ")", ")", ")"]}),
    ]
    with patch("pre3.adapter.api_llm.chat_completion", side_effect=fakes):
        with patch.object(
            sys,
            "argv",
            ["llm_pipeline", "--grammar", "parens", "--api-key", "k"],
        ):
            from pre3.tools import llm_pipeline

            assert llm_pipeline.main() == 0
    out = capsys.readouterr().out
    assert "DETERMINISM VERIFIED" in out
    assert "[1] ACCEPT" in out
    assert "[2] REJECT" in out


def test_llm_pipeline_cli_anbn(capsys):
    fakes = [
        json.dumps({"tokens": []}),
        json.dumps({"tokens": ["a", "b", "a"]}),
    ]
    with patch("pre3.adapter.api_llm.chat_completion", side_effect=fakes):
        with patch.object(
            sys,
            "argv",
            ["llm_pipeline", "--grammar", "anbn", "--api-key", "k"],
        ):
            from pre3.tools import llm_pipeline

            assert llm_pipeline.main() == 0
    out = capsys.readouterr().out
    assert "[1] ACCEPT" in out
    assert "[2] REJECT" in out


def test_llm_pipeline_exits_2_without_api_key(capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch.object(
        sys,
        "argv",
        ["llm_pipeline", "--grammar", "parens"],
    ):
        from pre3.tools import llm_pipeline

        assert llm_pipeline.main() == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err or "api-key" in err.lower() or "Error" in err


# ------------------------------------------------------------------
# HTTP client: User-Agent (Groq / Cloudflare)
# ------------------------------------------------------------------


def test_llm_pipeline_challenge_sweep_mocked(capsys):
    """``--challenge`` issues 2 prompts × 5 grammars; mock 10 API responses in order."""
    n = 12
    parens_valid = ["("] * n + [")"] * n
    parens_bad = ["("] * 5 + [")"] * 3

    fakes: list = [
        json.dumps({"tokens": parens_valid}),
        json.dumps({"tokens": parens_bad}),
        json.dumps({"tokens": list("([()])")}),
        json.dumps({"tokens": list("([)]")}),
        json.dumps(
            {
                "tokens": [
                    "(",
                    "id",
                    "+",
                    "id",
                    "*",
                    "(",
                    "id",
                    "+",
                    "id",
                    ")",
                    ")",
                    "*",
                    "id",
                ],
            }
        ),
        json.dumps({"tokens": ["id", "+", "*", "id"]}),
        json.dumps({"tokens": "if e then if e then x else x".split()}),
        json.dumps({"tokens": "if e then x else x else x".split()}),
        json.dumps(
            {
                "tokens": ["a", "b", "a", "c", "a", "b", "a"],
            }
        ),
        json.dumps(
            {
                "tokens": ["a", "b", "c", "a", "b"],
            }
        ),
    ]
    with patch("pre3.adapter.api_llm.chat_completion", side_effect=fakes):
        with patch.object(
            sys,
            "argv",
            [
                "llm_pipeline",
                "--challenge",
                "--api-key",
                "k",
            ],
        ):
            from pre3.tools import llm_pipeline

            assert llm_pipeline.main() == 0
    out = capsys.readouterr().out
    assert "parens" in out and "dyck2" in out and "Challenge sweep" in out
    assert out.count("ACCEPT") + out.count("REJECT") >= 10


def test_chat_completion_sends_user_agent():
    """Regression: missing User-Agent caused Groq 403 / 1010."""
    captured: dict = {}

    class _Resp:
        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {"message": {"content": json.dumps({"tokens": ["a"]})}}
                    ]
                }
            ).encode()

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(req, timeout: int = 60) -> _Resp:
        headers = {k: v for k, v in req.header_items()}
        captured["User-Agent"] = headers.get("User-agent") or headers.get("User-Agent")
        return _Resp()

    with patch("pre3.adapter.api_llm.urllib.request.urlopen", fake_urlopen):
        out = chat_completion(
            [{"role": "user", "content": "x"}],
            model="m",
            api_key="k",
        )
    assert "pre3-llm" in (captured.get("User-Agent") or "")
    assert '"tokens"' in out or "tokens" in out


# ------------------------------------------------------------------
# Optional live call (opt-in: key in env)
# ------------------------------------------------------------------


@pytest.mark.integration
def test_live_openai_compatible_skips_unless_env():
    """
    One real API round-trip. Skips unless RUN_LIVE_LLM=1 and OPENAI_API_KEY is set
    (avoids flaking default CI; use for manual confirmation).
    """
    if os.environ.get("RUN_LIVE_LLM", "").lower() not in ("1", "true", "yes"):
        pytest.skip("set RUN_LIVE_LLM=1 and OPENAI_API_KEY to run live test")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    g = from_rules({"S": ["a"]}, start="S")
    src = OpenAILLMSource(
        grammar=g,
        user_messages=['Reply with JSON only: {"tokens": ["a"]} — no other text.'],
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        use_json_object_mode=True,
    )
    toks = next(src.emit())
    sim = DPDASimulator(build_dpda(LR1Automaton(g)))
    assert sim.accepts(toks) is True
