"""Tests for local HuggingFace LLM adapter (mocked, no model load)."""

from __future__ import annotations

from unittest.mock import patch

from pre3.adapter.local_hf_llm import _extract_json


def test_extract_json_basic():
    s = 'prefix {"tokens":["a","b"]} suffix'
    assert _extract_json(s) == '{"tokens":["a","b"]}'


def test_llm_pipeline_local_backend_wiring(capsys):
    with patch("pre3.tools.llm_pipeline.LocalHFLLMSource") as src_cls:
        src_cls.return_value.emit.return_value = [["(", ")"], [")", "("]]
        with patch("sys.argv", ["llm_pipeline", "--backend", "local-hf", "--grammar", "parens"]):
            from pre3.tools import llm_pipeline

            rc = llm_pipeline.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "[1] ACCEPT" in out
    assert "[2] REJECT" in out
