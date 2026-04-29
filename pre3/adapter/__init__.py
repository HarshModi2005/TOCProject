"""
LLM-or-anything adapter:  the LEFT side of the pipeline.

Theory project:                    [ LLM stub ]  →  StringSource  →  DPDA → accept?
Production (later):                [ real LLM ]  →  StringSource  →  DPDA → accept?

This package defines the `StringSource` interface, a `MockLLMSource` for
tests, and :class:`OpenAILLMSource` in ``api_llm`` for OpenAI-compatible HTTP
(``OPENAI_API_KEY``).  Run ``python3 -m pre3.tools.llm_pipeline`` for a live
demo.
"""
