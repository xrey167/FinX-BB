"""Opt-in GPT-2 last-token head, for CPU research, not an architecture claim.

The existing adapter exposes only the last non-padding token. Projecting all
sequence positions through GPT-2's vocabulary head is unnecessary for that API.
This variant preserves decoder hooks and gradients, and explicitly disables KV
caching (the adapter API does not return or consume a cache). Different GEMM
shapes may cause small floating-point differences; it is NOT bitwise-equivalent.
"""
from __future__ import annotations
import torch
from so.llm_adapter import KnowledgeAdapterLM


class LastTokenGPT2Adapter(KnowledgeAdapterLM):
    def forward(self, bank, input_ids, attention_mask, last_idx, cell_mask=None):
        if getattr(self.lm.config, "model_type", None) != "gpt2":
            raise TypeError("LastTokenGPT2Adapter is explicitly GPT-2-only")
        if bank is not None:
            enc = self.encode_bank(bank)
            allowed = enc["active"] if cell_mask is None else enc["active"] & cell_mask
            self._ctx = {"keys": enc["keys"], "values": enc["values"],
                         "allowed": allowed, "last_idx": last_idx, "routing": []}
        else:
            self._ctx = None
        try:
            out = self.lm.transformer(input_ids=input_ids, attention_mask=attention_mask,
                                      use_cache=False, return_dict=True)
            ar = torch.arange(input_ids.shape[0], device=input_ids.device)
            hidden = out.last_hidden_state[ar, last_idx]
            full = self.lm.lm_head(hidden)
            cand = full[:, self.candidate_ids]
            ctx = self._ctx
            routing = torch.stack(ctx["routing"], dim=1) if ctx is not None and ctx["routing"] else None
            self.last_query = torch.stack(ctx["query"], dim=1) if ctx is not None and ctx.get("query") else None
            return cand, full, routing, hidden
        finally:
            self._ctx = None
