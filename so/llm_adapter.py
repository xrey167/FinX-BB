"""Mutable knowledge layer attached to a frozen pretrained language model (GPT-2).

This is the CPU-feasible analogue of the ledger's outstanding C55–C57
"real-model" chain: the neural core is a *pretrained* transformer whose
weights are frozen; the SO knowledge layer is attached as an adapter
("symlink adapter", architecture document section 10) that reads cells at
one or more transformer blocks and writes the retrieved value into the
residual stream of the last token.

    text tokens ──► GPT-2 blocks 0..L ──┬──► blocks L+1..11 ──► LM head ──► object token
                                        │ read at block L (last token)
                                        ▼
                      routing attention over cell keys  (key = wte[subject] + R[relation])
                                        │
                                        ▼
                      gated cell value  (value = Wv · wte[object] ⊙ gate(marker))

Because the value is built from the model's own (tied) token embedding of the
object, adding it into the residual stream raises that token's logit through
the unchanged LM head.  The marker gate does not merely attenuate the value
(the RMS-matched injection would undo that): it selects between the payload
and the ' unknown' direction, so an unsigned payload reads as "unknown" — the knowledge participates in the model's own
computation instead of being pasted into the prompt.

Only the adapter is trained (relation embeddings, key/value/query/output
projections, marker gate, null cells).  The pretrained weights never change,
so no fact can be copied into them; the re-sampled worlds make sure no fact
can be copied into the adapter either.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class AdapterConfig:
    n_relations: int = 4
    read_layers: Tuple[int, ...] = (8, 10)
    d_key: int = 256
    marker_dim: int = 16
    use_marker_gate: bool = True
    hard_gate: bool = False        # verification mode: gate thresholded at 0.5

    def to_dict(self) -> Dict:
        return asdict(self)


class KnowledgeAdapterLM(nn.Module):
    def __init__(self, lm, cfg: AdapterConfig, entity_token_ids: Sequence[int], unknown_token_id: int):
        super().__init__()
        self.lm = lm
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.cfg = cfg
        d = lm.config.n_embd
        self.d = d
        self.register_buffer("entity_token_ids", torch.as_tensor(list(entity_token_ids), dtype=torch.long))
        self.register_buffer("candidate_ids", torch.as_tensor(list(entity_token_ids) + [unknown_token_id], dtype=torch.long))
        self.rel_emb = nn.Embedding(cfg.n_relations, d)
        nn.init.normal_(self.rel_emb.weight, std=0.02)
        self.ln_key = nn.LayerNorm(d)
        self.k_proj = nn.Linear(d, cfg.d_key, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.v_proj.weight)           # start as "write the object's own embedding direction"
        self.marker_gate = nn.Sequential(nn.Linear(cfg.marker_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.q_ln = nn.ModuleDict({str(l): nn.LayerNorm(d) for l in cfg.read_layers})
        self.q_proj = nn.ModuleDict({str(l): nn.Linear(d, cfg.d_key) for l in cfg.read_layers})
        # output projection: identity, no bias (a bias could implement a constant "unknown" shortcut)
        self.o_proj = nn.ModuleDict({str(l): nn.Linear(d, d, bias=False) for l in cfg.read_layers})
        for l in cfg.read_layers:
            nn.init.eye_(self.o_proj[str(l)].weight)
        # the injected read is RMS-matched to the residual stream (whose norm is ~30x a token embedding's)
        self.inject_gain = nn.Parameter(torch.full((len(cfg.read_layers),), 1.0))
        self.null_key = nn.Parameter(torch.randn(len(cfg.read_layers), cfg.d_key) * 0.02)
        with torch.no_grad():
            unk = lm.transformer.wte.weight[unknown_token_id].detach().clone()
        self.null_value = nn.Parameter(unk[None].repeat(len(cfg.read_layers), 1))   # "nothing found" -> ' unknown'
        self.scale = nn.Parameter(torch.tensor(1.0))
        self._ctx: Optional[Dict] = None
        self._hooks = [lm.transformer.h[l].register_forward_hook(self._make_hook(i, l)) for i, l in enumerate(cfg.read_layers)]

    @property
    def wte(self) -> torch.Tensor:
        return self.lm.transformer.wte.weight

    def adapter_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("lm.")]

    # ------------------------------------------------------------------ knowledge layer
    def gate_logits(self, marker: torch.Tensor) -> torch.Tensor:
        return self.marker_gate(marker)

    def gate(self, marker: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_marker_gate:
            return torch.ones(marker.shape[0], 1, device=marker.device)
        g = torch.sigmoid(self.gate_logits(marker))
        if self.cfg.hard_gate:
            g = (g > 0.5).to(g.dtype)
        return g

    def encode_bank(self, bank: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        subj = self.wte[self.entity_token_ids[bank["subject"]]]
        obj = self.wte[self.entity_token_ids[bank["obj"]]]
        keys = self.k_proj(self.ln_key(subj + self.rel_emb(bank["relation"])))
        g = self.gate(bank["marker"])
        payload = self.v_proj(obj)
        unk = self.v_proj(self.wte[self.candidate_ids[-1]][None])          # the ' unknown' direction
        # the gate selects between the payload and "unknown": an unsigned payload READS AS unknown.
        # (a mere attenuation would be undone by the RMS-matched injection downstream)
        values = payload * g + unk * (1 - g)
        return {"keys": keys, "values": values, "values_payload": payload, "gate": g.squeeze(-1), "active": bank["active"]}

    def _make_hook(self, read_index: int, layer: int):
        def hook(module, inputs, output):
            if self._ctx is None:
                return None
            h = output[0] if isinstance(output, tuple) else output
            ctx = self._ctx
            B = h.shape[0]
            ar = torch.arange(B, device=h.device)
            hl = h[ar, ctx["last_idx"]]                                    # (B, d)
            q = self.q_proj[str(layer)](self.q_ln[str(layer)](hl))
            keys = torch.cat([ctx["keys"], self.null_key[read_index][None]])
            values = torch.cat([ctx["values"], self.null_value[read_index][None]])
            allowed = torch.cat([ctx["allowed"], torch.ones(1, dtype=torch.bool, device=h.device)])
            scores = (q @ keys.t()) * (self.scale / self.cfg.d_key ** 0.5)
            scores = scores.masked_fill(~allowed[None], float("-inf"))
            p = torch.softmax(scores, dim=-1)
            read = self.o_proj[str(layer)](p @ values)                        # (B, d)
            rms_h = hl.detach().pow(2).mean(-1, keepdim=True).sqrt()
            rms_r = read.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6
            read = read * (rms_h / rms_r) * self.inject_gain[read_index]        # RMS-matched injection
            delta = torch.zeros_like(h)
            delta[ar, ctx["last_idx"]] = read
            ctx["routing"].append(p)
            h2 = h + delta
            return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2
        return hook

    # ------------------------------------------------------------------ forward
    def forward(self, bank: Optional[Dict[str, torch.Tensor]], input_ids: torch.Tensor, attention_mask: torch.Tensor,
                last_idx: torch.Tensor, cell_mask: Optional[torch.Tensor] = None):
        """Returns (candidate logits (B, n_entities+1), full-vocab logits at the last token, routing (B, R, C+1), hidden (B, d))."""
        if bank is not None:
            enc = self.encode_bank(bank)
            allowed = enc["active"] if cell_mask is None else enc["active"] & cell_mask
            self._ctx = {"keys": enc["keys"], "values": enc["values"], "allowed": allowed, "last_idx": last_idx, "routing": []}
        else:
            self._ctx = None
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        ar = torch.arange(input_ids.shape[0], device=input_ids.device)
        full = out.logits[ar, last_idx]                                    # (B, V)
        hidden = out.hidden_states[-1][ar, last_idx]
        cand = full[:, self.candidate_ids]
        routing = torch.stack(self._ctx["routing"], dim=1) if self._ctx is not None and self._ctx["routing"] else None
        self._ctx = None
        return cand, full, routing, hidden
