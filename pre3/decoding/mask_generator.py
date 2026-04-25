"""
Vocabulary mask generator: the bridge between the DPDA and the LLM.

Given the current DPDA state + stack and a token trie, produces a
boolean mask over the vocabulary indicating which tokens are valid
next tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..dpda.builder import DPDA
from ..dpda.edge import PrefixConditionedEdge
from ..dpda.simulator import DPDAConfig
from .token_trie import TokenTrie, TrieNode


class MaskGenerator:
    """
    Produces per-step vocabulary masks for constrained decoding.

    For each token in the vocabulary, we simulate consuming its character
    sequence through the DPDA.  If all characters lead to valid transitions,
    the token is marked as allowed.
    """

    def __init__(self, dpda: DPDA, trie: TokenTrie) -> None:
        self.dpda = dpda
        self.trie = trie

    def generate_mask(self, config: DPDAConfig) -> list[bool]:
        """
        Return a boolean list of size vocab_size.  mask[token_id] = True
        means the token is valid given the current DPDA configuration.
        """
        mask = [False] * self.trie.vocab_size
        self._walk(config, self.trie.root, mask)
        return mask

    def generate_mask_batch(self, configs: list[DPDAConfig]) -> list[list[bool]]:
        """Batched version: one mask per request in the batch."""
        return [self.generate_mask(c) for c in configs]

    def allowed_token_ids(self, config: DPDAConfig) -> list[int]:
        """Return the list of valid token IDs (convenience wrapper)."""
        mask = self.generate_mask(config)
        return [i for i, v in enumerate(mask) if v]

    # ------------------------------------------------------------------
    # Internal: recursive trie + DPDA walk
    # ------------------------------------------------------------------

    def _walk(
        self,
        config: DPDAConfig,
        trie_node: TrieNode,
        mask: list[bool],
    ) -> None:
        """
        DFS over the trie.  At each trie node, try to advance the DPDA
        by the edge character.  If the transition is valid, recurse into
        the child trie node with the updated config.  If the trie node
        has token_ids, mark them in the mask.
        """
        # Any token that terminates here is valid
        for tid in trie_node.token_ids:
            mask[tid] = True

        for char, child_node in trie_node.children.items():
            edge = self.dpda.find_edge(config.state, char, config.stack)
            if edge is None:
                continue
            new_config = DPDAConfig(
                state=edge.target,
                stack=edge.apply_stack_ops(config.stack),
                consumed=config.consumed + 1,
            )
            self._walk(new_config, child_node, mask)


class CachedMaskGenerator(MaskGenerator):
    """
    Extends MaskGenerator with precomputed context-independent masks.

    For each DPDA state, we precompute the set of tokens that are valid
    regardless of stack content (= their edges have empty stack_match).
    At runtime, these are OR-ed with the dynamic context-dependent results.
    """

    def __init__(self, dpda: DPDA, trie: TokenTrie) -> None:
        super().__init__(dpda, trie)
        self._ci_masks: dict[int, list[bool]] = {}
        self._precompute()

    def _precompute(self) -> None:
        """
        For each state, find edges with empty stack_match (context-
        independent) and walk the trie to mark always-valid tokens.
        """
        for state_id in range(self.dpda.num_states):
            ci_config = DPDAConfig(state=state_id, stack=[])
            mask = [False] * self.trie.vocab_size
            self._walk_ci(ci_config, self.trie.root, mask)
            self._ci_masks[state_id] = mask

    def _walk_ci(
        self,
        config: DPDAConfig,
        trie_node: TrieNode,
        mask: list[bool],
    ) -> None:
        """Walk trie using only edges whose stack_match is empty."""
        for tid in trie_node.token_ids:
            mask[tid] = True

        for char, child_node in trie_node.children.items():
            # Only use context-independent edges
            for edge in self.dpda.lookup(config.state, char):
                if edge.stack_match == ():
                    new_config = DPDAConfig(
                        state=edge.target,
                        stack=edge.apply_stack_ops(config.stack),
                        consumed=config.consumed + 1,
                    )
                    self._walk_ci(new_config, child_node, mask)

    def generate_mask(self, config: DPDAConfig) -> list[bool]:
        ci = self._ci_masks.get(config.state)
        cd = super().generate_mask(config)
        if ci is None:
            return cd
        return [a or b for a, b in zip(ci, cd)]
