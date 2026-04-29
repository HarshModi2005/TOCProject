from graphviz import Digraph

def build_pre3_pipeline():
    dot = Digraph("Pre3MainPipeline", format="png")
    dot.attr(rankdir="LR", splines="ortho", nodesep="0.6", ranksep="0.8")
    dot.attr("node", shape="box", style="rounded,filled", color="#1f2937",
             fillcolor="#e5e7eb", fontname="Helvetica", fontsize="11")
    dot.attr("edge", color="#374151", arrowsize="0.8", penwidth="1.2")

    # Main pipeline
    dot.node("cfg", "Input CFG\n(V, Σ, P, S)", fillcolor="#dbeafe")
    dot.node("npda", "CFG → NPDA Construction\n(pre3/pda/cfg_to_pda.py)", fillcolor="#ede9fe")
    dot.node("lr1", "LR(1) Automaton Build\n(pre3/grammar/lr1.py)", fillcolor="#ede9fe")
    dot.node("dpda", "LR(1) → DPDA Build\n(pre3/dpda/builder.py)", fillcolor="#ede9fe")
    dot.node("verify", "Determinism Verification\n(pre3/dpda/verifier.py)", fillcolor="#fef3c7")
    dot.node("sim", "DPDA Runtime State\n(state + stack)", fillcolor="#dcfce7")
    dot.node("mask", "Valid Next-Terminal Set\n(pre3/decoding/mask_generay)", fillcolor="#dcfce7")
    dot.node("trie", "Token Trie\n(pre3/decoding/token_trie.py)", fillcolor="#fce7f3")
    dot.node("logits", "Logits Masking\n(pre3/decoding/logits_processor.py)", fillcolor="#fce7f3")
    dot.node("llm", "LLM Sampling\n(only valid tokens survive)", fillcolor="#fee2e2")
    dot.node("advance", "Advance DPDA with chosen token\n(update state + stack)", fillcolor="#dcfce7")

    # Core edges
    dot.edge("cfg", "npda", label="theory path")
    dot.edge("cfg", "lr1", label="deterministic path")
    dot.edge("lr1", "dpda")
    dot.edge("dpda", "verify")
    dot.edge("verify", "sim")
    dot.edge("sim", "mask")
    dot.edge("trie", "mask", label="vocab structure")
    dot.edge("mask", "logits")
    dot.edge("logits", "llm")
    dot.edge("llm", "advance")
    dot.edge("advance", "sim", label="next decoding step")

    # Optional annotation node
    dot.node("note", "NPDA branch is for CFL equivalence/demo.\nDPDA branch drives constrained decoding.", 
             shape="note", fillcolor="#ffffff")
    dot.edge("npda", "note", style="dashed", arrowhead="none")
    dot.edge("dpda", "note", style="dashed", arrowhead="none")

    return dot

if __name__ == "__main__":
    chart = build_pre3_pipeline()
    out = chart.render("pre3_main_pipeline", cleanup=True)
    print(f"Flowchart generated: {out}")
