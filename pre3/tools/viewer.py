"""
Local viewer launcher.

Generates the bundled showcase samples (if missing), then serves the
viewer's directory over HTTP and opens the browser pointed at the
viewer.

Usage::

    python -m pre3.tools.viewer
    python -m pre3.tools.viewer --port 9000
    python -m pre3.tools.viewer --no-open
    python -m pre3.tools.viewer --regen-samples       # rebuild all samples
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .trace_viz import build_trace_for_grammar


HERE = Path(__file__).resolve().parent
SAMPLES_DIR = HERE / "samples"
VIEWER_HTML = HERE / "trace_viewer.html"


# ----------------------------------------------------------------------
# Sample definitions
# ----------------------------------------------------------------------


def _g_parens():
    from ..grammar.grammar_loader import balanced_parens
    return balanced_parens()


def _g_arith():
    from ..grammar.grammar_loader import arithmetic
    return arithmetic()


def _g_anbn():
    from ..examples.anbn import grammar
    return grammar()


def _g_wcwR():
    from ..examples.wcwR import grammar
    return grammar()


def _g_wwR():
    from ..examples.wwR import grammar
    return grammar()


SAMPLES: List[Dict[str, Any]] = [
    {
        "name": "parens.json",
        "example": "balanced_parens",
        "grammar": _g_parens,
        "tokens": ["(", "(", ")", ")"],
        "raw_input": "(())",
        "label": "balanced parens — (())  · accept",
    },
    {
        "name": "parens_bad.json",
        "example": "balanced_parens",
        "grammar": _g_parens,
        "tokens": ["(", "(", ")"],
        "raw_input": "(()",
        "label": "balanced parens — (() · reject (missing close)",
    },
    {
        "name": "arith.json",
        "example": "arithmetic",
        "grammar": _g_arith,
        "tokens": ["id", "+", "id", "*", "id"],
        "raw_input": "id + id * id",
        "label": "arithmetic — id+id*id · accept (precedence)",
    },
    {
        "name": "anbn.json",
        "example": "anbn",
        "grammar": _g_anbn,
        "tokens": ["a", "a", "a", "b", "b", "b"],
        "raw_input": "aaabbb",
        "label": "aⁿbⁿ — aaabbb · accept (counter)",
    },
    {
        "name": "wcwR.json",
        "example": "wcwR",
        "grammar": _g_wcwR,
        "tokens": ["a", "b", "c", "b", "a"],
        "raw_input": "abcba",
        "label": "wcwᴿ — abcba · accept (DCFL palindrome)",
    },
    {
        "name": "wwR.json",
        "example": "wwR",
        "grammar": _g_wwR,
        "tokens": ["a", "b", "b", "a"],
        "raw_input": "abba",
        "label": "wwᴿ — abba · NPDA accepts; LR(1)/DPDA construction fails",
    },
]


def regenerate(force: bool = False) -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for spec in SAMPLES:
        path = SAMPLES_DIR / spec["name"]
        if path.exists() and not force:
            continue
        grammar = spec["grammar"]()
        try:
            trace = build_trace_for_grammar(
                spec["example"], grammar, spec["tokens"],
                raw_input=spec["raw_input"],
                source={
                    "kind": "showcase",
                    "label": spec["label"],
                    "input_mode": "raw",
                },
            )
        except Exception as e:
            sys.stderr.write(f"Could not build sample {spec['name']}: {e}\n")
            continue
        path.write_text(json.dumps(trace, indent=2))
        sys.stderr.write(f"  wrote {path.relative_to(HERE.parent.parent)}\n")


# ----------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        # Quiet by default; uncomment for debugging:
        # super().log_message(fmt, *args)
        pass

    def end_headers(self) -> None:  # type: ignore[override]
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()


def serve(port: int, open_browser: bool) -> None:
    os.chdir(HERE)
    handler: Callable[..., http.server.SimpleHTTPRequestHandler] = _Handler
    bind = ("127.0.0.1", port)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(bind, handler) as httpd:
        url = f"http://{bind[0]}:{port}/trace_viewer.html"
        sys.stderr.write(f"\nServing Pre³ viewer at:\n  {url}\n\nPress Ctrl-C to stop.\n\n")
        if open_browser:
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            sys.stderr.write("\nbye.\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    p.add_argument("--regen-samples", action="store_true", help="Force regenerate all bundled samples")
    p.add_argument("--just-samples", action="store_true", help="Only (re)generate samples and exit")
    args = p.parse_args()

    if not VIEWER_HTML.exists():
        sys.stderr.write(f"Viewer HTML not found at {VIEWER_HTML}\n")
        return 2

    sys.stderr.write("Generating bundled samples (if needed)...\n")
    regenerate(force=args.regen_samples)

    if args.just_samples:
        return 0

    serve(args.port, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
