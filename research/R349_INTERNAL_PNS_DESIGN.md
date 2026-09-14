# R349 — Internal Prospective Neural State intervention

R348 is a bridge from real frozen Qwen features into an external PNS descriptor. R349 moves the materialization boundary **inside the frozen Qwen residual stream**.

The falsification test splits Qwen2.5-0.5B at a late decoder layer. The prompt and all pre-barrier decoder blocks see no mutable world values. A cached pre-barrier hidden state is paired with canonical live references. At the materialization barrier, current world cells are converted into a deterministic World-Port residual vector and injected into the last-token residual stream; the remaining actual Qwen decoder blocks and LM head then execute normally.

The optimized PNS path must be full-vocabulary identical to a full current-world recomputation for every tested world revision while skipping the whole pre-barrier Transformer prefix. A matched early-materialization control injects the same world signal earlier and caches the resulting numeric hidden state; reusing that numeric cache after a world rewrite must demonstrably diverge from a fresh current-world recomputation.

This gate is intentionally about **pretrained internal activation semantics**, not task accuracy. R350 must add learned semantic World-Port behavior / language tasks once the exact internal split is established.
