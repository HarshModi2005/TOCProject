"""
LLM-or-anything adapter:  the LEFT side of the pipeline.

Theory project:                    [ LLM stub ]  →  StringSource  →  DPDA → accept?
Production (later):                [ real LLM ]  →  StringSource  →  DPDA → accept?

This package defines the `StringSource` interface, plus a `MockLLMSource`
that emits canned strings for testing.  No real LLM is wired in yet.
"""
