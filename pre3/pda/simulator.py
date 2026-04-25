"""
NPDA simulator — accepts/rejects a string by BFS over configurations.

A configuration is the triple (state, remaining_input, stack).
We explore the reachable configurations breadth-first, bounded by a
configurable max_steps and max_stack_depth.

Two acceptance modes are supported:
  • "final_state"  — accept iff some BFS path ends with empty input AND state ∈ F.
  • "empty_stack"  — accept iff some BFS path ends with empty input AND empty stack.

The simulator also exposes a step-trace mode useful for visualization.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

from .pda import NPDA, PDATransition


@dataclass(frozen=True)
class PDAConfig:
    """A snapshot of an NPDA execution."""

    state: str
    remaining: Tuple[str, ...]
    stack: Tuple[str, ...]   # top = last element

    @property
    def stack_top(self) -> Optional[str]:
        return self.stack[-1] if self.stack else None

    def __repr__(self) -> str:
        rem = "".join(self.remaining) if self.remaining else "ε"
        stk = "".join(self.stack) if self.stack else "ε"
        return f"({self.state}, {rem!r}, [{stk}])"


@dataclass
class PDARunResult:
    accepted: bool
    mode: str
    configs_explored: int
    final_config: Optional[PDAConfig] = None
    reason: Optional[str] = None
    trace: List[PDAConfig] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "ACCEPTED" if self.accepted else "REJECTED"
        return (
            f"PDARunResult({status}, mode={self.mode}, "
            f"configs={self.configs_explored}, reason={self.reason})"
        )


class PDASimulator:
    """BFS simulator for NPDAs.  Sound and complete on bounded inputs."""

    def __init__(
        self,
        pda: NPDA,
        *,
        max_configs: int = 100_000,
        max_stack_depth: int = 256,
    ) -> None:
        self.pda = pda
        self.max_configs = max_configs
        self.max_stack_depth = max_stack_depth

    # ------------------------------------------------------------------
    # Single-step generator
    # ------------------------------------------------------------------

    def successors(self, cfg: PDAConfig) -> List[Tuple[PDAConfig, PDATransition]]:
        """All configurations reachable in one transition."""
        if not cfg.stack:
            return []
        top = cfg.stack[-1]
        next_input = cfg.remaining[0] if cfg.remaining else ""
        out: List[Tuple[PDAConfig, PDATransition]] = []

        # 1. ε-transitions
        for t in self.pda.epsilon_transitions(cfg.state, top):
            new_stack = list(cfg.stack[:-1])
            # leftmost in stack_push ends up on top → push reversed
            for s in reversed(t.stack_push):
                new_stack.append(s)
            if len(new_stack) > self.max_stack_depth:
                continue
            out.append((PDAConfig(t.next_state, cfg.remaining, tuple(new_stack)), t))

        # 2. Symbol-consuming transitions
        if next_input:
            for t in self.pda._index.get((cfg.state, next_input, top), []):
                new_stack = list(cfg.stack[:-1])
                for s in reversed(t.stack_push):
                    new_stack.append(s)
                if len(new_stack) > self.max_stack_depth:
                    continue
                out.append((PDAConfig(t.next_state, cfg.remaining[1:], tuple(new_stack)), t))

        return out

    # ------------------------------------------------------------------
    # Acceptance check
    # ------------------------------------------------------------------

    def accepts(
        self, input_string: Sequence[str], *, mode: str = "final_state",
    ) -> bool:
        return self.run(input_string, mode=mode).accepted

    def run(
        self, input_string: Sequence[str], *, mode: str = "final_state",
    ) -> PDARunResult:
        """BFS the configuration space, tracking visited to avoid loops."""
        if mode not in ("final_state", "empty_stack"):
            raise ValueError(f"mode must be final_state or empty_stack, got {mode!r}")

        start = PDAConfig(
            state=self.pda.start_state,
            remaining=tuple(input_string),
            stack=(self.pda.start_stack,),
        )

        def is_accepting(c: PDAConfig) -> bool:
            if c.remaining:
                return False
            if mode == "final_state":
                return c.state in self.pda.accept_states
            return len(c.stack) == 0

        visited: Set[PDAConfig] = {start}
        queue: deque[Tuple[PDAConfig, List[PDAConfig]]] = deque([(start, [start])])
        configs_explored = 0

        while queue:
            cfg, path = queue.popleft()
            configs_explored += 1
            if configs_explored > self.max_configs:
                return PDARunResult(
                    accepted=False, mode=mode,
                    configs_explored=configs_explored,
                    final_config=cfg,
                    reason=f"max_configs={self.max_configs} exceeded",
                    trace=path,
                )
            if is_accepting(cfg):
                return PDARunResult(
                    accepted=True, mode=mode,
                    configs_explored=configs_explored,
                    final_config=cfg, reason=None, trace=path,
                )
            for nxt, _t in self.successors(cfg):
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))

        return PDARunResult(
            accepted=False, mode=mode,
            configs_explored=configs_explored,
            final_config=None,
            reason="no accepting configuration reachable",
            trace=[],
        )
