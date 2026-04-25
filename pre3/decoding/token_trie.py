"""
Token trie: maps LLM vocabulary tokens (multi-character strings) to
sequences of grammar terminal symbols.

At each decoding step, the trie is walked in tandem with the DPDA to
determine which vocabulary tokens lead to valid DPDA states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrieNode:
    children: dict[str, TrieNode] = field(default_factory=dict)
    token_ids: list[int] = field(default_factory=list)
    # token_ids lists vocabulary indices whose character sequence
    # terminates at this node.


class TokenTrie:
    """
    Prefix trie over vocabulary tokens, keyed by their character sequences.

    Example: if vocab = {0: "ab", 1: "a", 2: "abc"}, the trie has paths
        root -a-> n1 -b-> n2 -c-> n3
    with token_ids:  n1=[1], n2=[0], n3=[2].
    """

    def __init__(self) -> None:
        self.root = TrieNode()
        self.vocab_size = 0

    @classmethod
    def from_vocabulary(cls, vocab: dict[int, str]) -> TokenTrie:
        """
        Build a trie from a vocabulary mapping  token_id → string.

        Each character of the string becomes one trie edge.
        """
        trie = cls()
        trie.vocab_size = max(vocab.keys()) + 1 if vocab else 0
        for token_id, text in vocab.items():
            node = trie.root
            for ch in text:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.token_ids.append(token_id)
        return trie

    @classmethod
    def from_token_list(cls, tokens: list[str]) -> TokenTrie:
        """Build from a simple list where index = token_id."""
        return cls.from_vocabulary({i: t for i, t in enumerate(tokens)})

    def walk(self, node: Optional[TrieNode], char: str) -> Optional[TrieNode]:
        """Advance one character in the trie."""
        if node is None:
            return None
        return node.children.get(char)

    def all_tokens_from(self, node: TrieNode) -> list[int]:
        """Collect all token IDs reachable from a subtree."""
        result = list(node.token_ids)
        for child in node.children.values():
            result.extend(self.all_tokens_from(child))
        return result
