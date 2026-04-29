"""Additional unit and integration tests for component interactions."""

from pre3.decoding.logits_processor import LogitsProcessor
from pre3.decoding.token_trie import TokenTrie
from pre3.dpda.builder import DPDA, build_dpda
from pre3.dpda.edge import EdgeKind, PrefixConditionedEdge, StackOp
from pre3.dpda.optimizer import aggregate_edges, optimize
from pre3.dpda.simulator import DPDASimulator
from pre3.grammar.grammar_loader import balanced_parens, from_rules
from pre3.grammar.lr1 import LR1Automaton


def test_aggregate_edges_unions_symbols_for_identical_structure():
    """Unit: aggregation keeps structure and unions accepted symbols."""
    e1 = PrefixConditionedEdge(
        source=0,
        target=1,
        accepted_symbols=frozenset({"a"}),
        stack_match=(0,),
        stack_ops=(StackOp.push(1),),
        kind=EdgeKind.ACCEPTANCE,
    )
    e2 = PrefixConditionedEdge(
        source=0,
        target=1,
        accepted_symbols=frozenset({"b"}),
        stack_match=(0,),
        stack_ops=(StackOp.push(1),),
        kind=EdgeKind.ACCEPTANCE,
    )

    aggregated = aggregate_edges([e1, e2])

    assert len(aggregated) == 1
    merged = aggregated[0]
    assert merged.accepted_symbols == frozenset({"a", "b"})
    assert merged.source == 0 and merged.target == 1
    assert merged.stack_match == (0,)
    assert merged.stack_ops == (StackOp.push(1),)


def test_find_edge_matches_stack_condition():
    """Unit: DPDA.find_edge returns only edges matching stack suffix."""
    dpda = DPDA(start_state=0, num_states=3)
    guarded = PrefixConditionedEdge(
        source=0,
        target=1,
        accepted_symbols=frozenset({"x"}),
        stack_match=(7,),
        stack_ops=(StackOp.push(1),),
        kind=EdgeKind.ACCEPTANCE,
    )
    dpda.add_edge(guarded)

    assert dpda.find_edge(0, "x", [0, 7]) is guarded
    assert dpda.find_edge(0, "x", [0, 6]) is None


def test_logits_processor_unknown_token_id_does_not_advance_state():
    """Unit: unknown token IDs are ignored by advance()."""
    grammar = from_rules({"S": ["a"]}, start="S")
    dpda = build_dpda(LR1Automaton(grammar))
    trie = TokenTrie.from_vocabulary({0: "a"})
    proc = LogitsProcessor(dpda, trie, {0: "a"}, use_cache=False)
    proc.init_request(0)

    before = proc.configs[0]
    before_state = before.state
    before_stack = list(before.stack)
    before_consumed = before.consumed

    proc.advance(0, 999)  # unknown token id -> empty text -> no transition

    after = proc.configs[0]
    assert after.state == before_state
    assert after.stack == before_stack
    assert after.consumed == before_consumed


def test_logits_processor_stepwise_masks_follow_grammar():
    """Integration: mask changes after consuming tokens according to grammar."""
    grammar = from_rules({"S": ["a b"]}, start="S")
    dpda = build_dpda(LR1Automaton(grammar))
    vocab = {0: "a", 1: "b", 2: "c"}
    trie = TokenTrie.from_vocabulary(vocab)
    proc = LogitsProcessor(dpda, trie, vocab, use_cache=False)

    # Step 0: only "a" should be valid.
    m0 = proc.process_logits_list(0, [0.1, 0.1, 0.1])
    assert m0[0] != LogitsProcessor.NEG_INF
    assert m0[1] == LogitsProcessor.NEG_INF
    assert m0[2] == LogitsProcessor.NEG_INF

    # Consume "a", then "b" must become available and "c" still invalid.
    proc.advance(0, 0)
    m1 = proc.process_logits_list(0, [0.2, 0.2, 0.2])
    assert m1[1] != LogitsProcessor.NEG_INF
    assert m1[2] == LogitsProcessor.NEG_INF


def test_optimized_dpda_preserves_acceptance_behavior():
    """Integration: optimizer should not change language acceptance."""
    lr1 = LR1Automaton(balanced_parens())
    base_dpda = build_dpda(lr1)
    opt_dpda = optimize(base_dpda)

    base_sim = DPDASimulator(base_dpda)
    opt_sim = DPDASimulator(opt_dpda)

    cases = [
        [],
        ["(", ")"],
        ["(", "(", ")", ")"],
        ["("],
        [")"],
        ["(", ")", ")"],
    ]
    for tokens in cases:
        assert base_sim.accepts(tokens) == opt_sim.accepts(tokens)
