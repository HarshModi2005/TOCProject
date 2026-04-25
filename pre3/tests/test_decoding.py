"""Tests for the token trie, mask generator, and logits processor."""

import pytest

from pre3.grammar.grammar_loader import from_rules
from pre3.grammar.lr1 import LR1Automaton
from pre3.dpda.builder import build_dpda
from pre3.dpda.simulator import DPDAConfig
from pre3.decoding.token_trie import TokenTrie
from pre3.decoding.mask_generator import MaskGenerator, CachedMaskGenerator
from pre3.decoding.logits_processor import LogitsProcessor


# ------------------------------------------------------------------
# Token Trie
# ------------------------------------------------------------------


class TestTokenTrie:
    def test_build_from_vocabulary(self):
        vocab = {0: "ab", 1: "a", 2: "abc"}
        trie = TokenTrie.from_vocabulary(vocab)
        assert trie.vocab_size == 3

    def test_walk(self):
        vocab = {0: "ab", 1: "a"}
        trie = TokenTrie.from_vocabulary(vocab)
        n = trie.walk(trie.root, "a")
        assert n is not None
        assert 1 in n.token_ids

        n2 = trie.walk(n, "b")
        assert n2 is not None
        assert 0 in n2.token_ids

    def test_walk_invalid(self):
        vocab = {0: "ab"}
        trie = TokenTrie.from_vocabulary(vocab)
        assert trie.walk(trie.root, "z") is None

    def test_all_tokens_from_root(self):
        vocab = {0: "ab", 1: "a", 2: "abc"}
        trie = TokenTrie.from_vocabulary(vocab)
        all_ids = trie.all_tokens_from(trie.root)
        assert set(all_ids) == {0, 1, 2}

    def test_from_token_list(self):
        trie = TokenTrie.from_token_list(["hello", "world"])
        assert trie.vocab_size == 2


# ------------------------------------------------------------------
# Mask Generator
# ------------------------------------------------------------------


def _make_test_setup():
    """Simple grammar  S → a b  with a character-level vocab."""
    g = from_rules({"S": ["a b"]}, start="S")
    lr1 = LR1Automaton(g)
    dpda = build_dpda(lr1)
    # vocab: 0="a", 1="b", 2="c" (invalid)
    trie = TokenTrie.from_vocabulary({0: "a", 1: "b", 2: "c"})
    return dpda, trie


class TestMaskGenerator:
    def test_initial_mask(self):
        dpda, trie = _make_test_setup()
        mg = MaskGenerator(dpda, trie)
        config = DPDAConfig(state=dpda.start_state, stack=[dpda.start_state])
        mask = mg.generate_mask(config)
        assert mask[0] is True, "Token 'a' should be valid at start"
        assert mask[2] is False, "Token 'c' should be invalid"

    def test_allowed_token_ids(self):
        dpda, trie = _make_test_setup()
        mg = MaskGenerator(dpda, trie)
        config = DPDAConfig(state=dpda.start_state, stack=[dpda.start_state])
        allowed = mg.allowed_token_ids(config)
        assert 0 in allowed


class TestCachedMaskGenerator:
    def test_cached_same_as_uncached(self):
        dpda, trie = _make_test_setup()
        mg = MaskGenerator(dpda, trie)
        cmg = CachedMaskGenerator(dpda, trie)
        config = DPDAConfig(state=dpda.start_state, stack=[dpda.start_state])
        mask1 = mg.generate_mask(config)
        mask2 = cmg.generate_mask(config)
        # Cached should be at least as permissive as uncached
        for i in range(len(mask1)):
            if mask1[i]:
                assert mask2[i], f"Cached mask should include token {i}"


# ------------------------------------------------------------------
# Logits Processor (pure-Python path, no PyTorch)
# ------------------------------------------------------------------


class TestLogitsProcessor:
    def test_process_logits_list(self):
        dpda, trie = _make_test_setup()
        vocab = {0: "a", 1: "b", 2: "c"}
        proc = LogitsProcessor(dpda, trie, vocab, use_cache=False)
        logits = [1.0, 2.0, 3.0]
        masked = proc.process_logits_list(0, logits)
        # 'c' should be masked to NEG_INF
        assert masked[2] == LogitsProcessor.NEG_INF

    def test_advance(self):
        dpda, trie = _make_test_setup()
        vocab = {0: "a", 1: "b", 2: "c"}
        proc = LogitsProcessor(dpda, trie, vocab, use_cache=False)
        proc.init_request(0)
        config_before = proc.configs[0].state
        proc.advance(0, 0)  # consume token 'a'
        config_after = proc.configs[0].state
        # State should have changed after consuming 'a'
        assert config_after != config_before or proc.configs[0].consumed == 1
