"""
Canonical example languages from the formal-languages textbooks.

Each module exposes (at least):
  - description       — one-paragraph English description with class
  - grammar()         — the CFG (when one exists)
  - npda()            — a hand-crafted NPDA (when illuminating)
  - positive_examples — list of strings in the language
  - negative_examples — list of strings NOT in the language

The MAIN value of these modules:
  • aⁿbⁿ, aⁿbⁿcⁿ      — pumping-lemma demonstrations
  • wcwᴿ, balanced    — DCFL examples (LR(1) → DPDA succeeds)
  • wwᴿ               — CFL but NOT DCFL (LR(1) construction MUST FAIL)
  • dyck2             — nested brackets, classic DCFL
"""
