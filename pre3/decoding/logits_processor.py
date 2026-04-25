"""
LogitsProcessor interface: plugs the DPDA mask into HuggingFace-style
generation loops.

Also provides a generic callback interface for any inference engine
that exposes a logits-modification hook.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol

from ..dpda.builder import DPDA
from ..dpda.simulator import DPDAConfig
from .mask_generator import CachedMaskGenerator, MaskGenerator
from .token_trie import TokenTrie


class LogitsProcessor:
    """
    Applies the DPDA vocabulary mask to raw logits at each decoding step.

    Usage with HuggingFace ``transformers``::

        processor = LogitsProcessor(dpda, trie, vocab)
        outputs = model.generate(
            input_ids,
            logits_processor=[processor],
        )

    For non-HF engines, call :meth:`__call__` manually on the logits
    tensor before sampling.
    """

    NEG_INF = -1e9

    def __init__(
        self,
        dpda: DPDA,
        trie: TokenTrie,
        vocab: dict[int, str],
        *,
        use_cache: bool = True,
    ) -> None:
        self.dpda = dpda
        self.trie = trie
        self.vocab = vocab
        self.mask_gen: MaskGenerator
        if use_cache:
            self.mask_gen = CachedMaskGenerator(dpda, trie)
        else:
            self.mask_gen = MaskGenerator(dpda, trie)

        # Per-request DPDA configs  (batch index → config)
        self.configs: dict[int, DPDAConfig] = {}

    def init_request(self, batch_idx: int) -> None:
        """Initialize a fresh DPDA config for a new request."""
        self.configs[batch_idx] = DPDAConfig(
            state=self.dpda.start_state,
            stack=[self.dpda.start_state],
        )

    def __call__(
        self,
        input_ids: Any,
        scores: Any,
    ) -> Any:
        """
        HuggingFace-compatible logits_processor callback.

        *input_ids* is (batch, seq_len), *scores* is (batch, vocab).
        Returns modified *scores*.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "LogitsProcessor.__call__ requires PyTorch.  "
                "Use process_logits_list() for a pure-Python path."
            )

        batch_size = scores.shape[0]
        for b in range(batch_size):
            if b not in self.configs:
                self.init_request(b)
            config = self.configs[b]
            mask = self.mask_gen.generate_mask(config)
            for token_id in range(len(mask)):
                if not mask[token_id]:
                    scores[b, token_id] = self.NEG_INF
        return scores

    def advance(self, batch_idx: int, token_id: int) -> None:
        """
        After sampling, advance the DPDA for the given request.

        Call this with the chosen token_id so the config is ready
        for the next step.
        """
        config = self.configs.get(batch_idx)
        if config is None:
            return
        text = self.vocab.get(token_id, "")
        for ch in text:
            edge = self.dpda.find_edge(config.state, ch, config.stack)
            if edge is None:
                break
            config.stack = edge.apply_stack_ops(config.stack)
            config.state = edge.target
            config.consumed += 1

    def process_logits_list(
        self, batch_idx: int, logits: list[float]
    ) -> list[float]:
        """
        Pure-Python logits masking (no PyTorch needed).
        Returns a new list with invalid tokens set to NEG_INF.
        """
        if batch_idx not in self.configs:
            self.init_request(batch_idx)
        config = self.configs[batch_idx]
        mask = self.mask_gen.generate_mask(config)
        return [
            s if m else self.NEG_INF
            for s, m in zip(logits, mask)
        ]
