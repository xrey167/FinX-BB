"""Mini-Transformer neural core with a mutable knowledge interface.

Layout (architecture document section 15):

    query tokens ──► Neural Core (transformer encoder) ──► hop state h
                                                                │
                              ┌─────────────────────────────────┘
                              ▼   knowledge interface: one routed read per hop
                     Symlink / Routing Layer  (attention over cell keys, null cell)
                              │
                              ▼
                     Mutable Knowledge Layer (cells: key, value, marker, active flag)

The core never stores facts.  A cell's key is built from (subject, relation), its
value from the object; the value is multiplied by a *marker gate* that the model
has to learn — a payload whose marker is not signed must be unusable even though
it is physically present and routable (crypto-shredding analogy, section 12).
Multi-hop reasoning is composition: the state after hop *t* is the query of hop
*t + 1*, so the knowledge participates in the internal computation rather than
being pasted into the prompt.  The routing distribution of each hop is the
provenance trace (TRACE), and the attention mass / gated value norm per cell is
the observable used as a *biomarker* (ledger section 20).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_entities: int = 256
    n_relations: int = 4
    n_surface: int = 8
    max_hops: int = 3
    d_model: int = 128
    n_heads: int = 4
    n_core_layers: int = 2
    d_ff: int = 256
    marker_dim: int = 16
    use_marker_gate: bool = True   # ablation: without marker -> gate fixed to 1
    hard_gate: bool = False        # verification mode: gate thresholded at 0.5 (a payload is signed or it is not)
    use_routing: bool = True       # ablation: without routing -> no knowledge layer at all
    use_null_cell: bool = True
    # ---- E-000015 (explicit symlink cells).  Both default OFF, so every earlier configuration
    # builds exactly the same parameters and its checkpoints keep loading.
    use_links: bool = False        # the bank may contain alias rows whose payload is the TARGET'S KEY
    n_deref: int = 0               # dereference slots per hop (1 resolves an alias, 2 a chain of two)
    deref_query_from_state: bool = False   # ablation: let the deref query see the question, not only the pointer
    use_deref_passthrough: bool = True     # the deref slot may keep the value it was given ("this was no pointer")
    disable_deref: bool = False    # inference-time ablation: every deref slot passes through (no pointer is followed)

    def to_dict(self) -> Dict:
        return asdict(self)


class HopBlock(nn.Module):
    """One routed read from the knowledge layer followed by a feed-forward update."""

    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def read(self, h: torch.Tensor, rel: torch.Tensor, hop_emb: torch.Tensor, k_f: torch.Tensor,
             v_f: torch.Tensor, k_r: torch.Tensor, v_r: torch.Tensor, is_fwd: torch.Tensor,
             allowed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One routed read: returns ``(read value (B, d), routing (B, C))`` without touching the state."""
        q = self.q(self.ln_q(h + rel + hop_emb))
        scores = torch.where(is_fwd[:, None], q @ k_f.t(), q @ k_r.t())     # (B, C)
        scores = scores * (self.scale / k_f.shape[-1] ** 0.5)
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        p = torch.softmax(scores, dim=-1)
        read = torch.where(is_fwd[:, None], p @ v_f, p @ v_r)               # (B, d)
        return read, p

    def apply_read(self, h: torch.Tensor, read: torch.Tensor) -> torch.Tensor:
        h = h + self.o(read)
        return h + self.ff(self.ln_ff(h))

    def forward(self, h: torch.Tensor, rel: torch.Tensor, hop_emb: torch.Tensor, k_f: torch.Tensor,
                v_f: torch.Tensor, k_r: torch.Tensor, v_r: torch.Tensor, is_fwd: torch.Tensor,
                allowed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h, rel, hop_emb: (B, d); k_*/v_*: (C, d) shared by the batch; allowed: (C,) bool
        read, p = self.read(h, rel, hop_emb, k_f, v_f, k_r, v_r, is_fwd, allowed)
        return self.apply_read(h, read), p


class DerefBlock(nn.Module):
    """A second read whose query comes from the POINTER just read, not from the question (E-000015).

    The last column of the bank is the null column; in a dereference slot its value is the incoming
    read itself, so "what I just read was not a pointer" is expressible as keeping the value.
    Which column wins is learned; the passthrough itself is architectural.
    """

    def __init__(self, d: int, use_state: bool = False, use_passthrough: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.use_state = use_state
        self.use_passthrough = use_passthrough

    def forward(self, read: torch.Tensor, state: Optional[torch.Tensor], k_f: torch.Tensor, v_f: torch.Tensor,
                allowed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.q(self.ln(read if (state is None or not self.use_state) else read + state))
        scores = (q @ k_f.t()) * (self.scale / k_f.shape[-1] ** 0.5)
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        p = torch.softmax(scores, dim=-1)
        if self.use_passthrough:
            out = p[:, :-1] @ v_f[:-1] + p[:, -1:] * read      # last column: keep what came in
        else:
            out = p @ v_f
        return out, p


class MutableKnowledgeTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.ent_emb = nn.Embedding(cfg.n_entities, d)
        self.rel_emb = nn.Embedding(cfg.n_surface + 1, d)         # +1 = PAD
        self.cell_rel_emb = nn.Embedding(cfg.n_relations, d)      # canonical relation ids inside cells
        self.mode_emb = nn.Embedding(2, d)
        self.pos_emb = nn.Embedding(cfg.max_hops + 2, d)
        self.hop_emb = nn.Embedding(cfg.max_hops, d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.d_ff, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.core = nn.TransformerEncoder(layer, cfg.n_core_layers)
        self.core_ln = nn.LayerNorm(d)
        # knowledge layer projections
        self.ln_key = nn.LayerNorm(d)
        self.k_fwd = nn.Linear(d, d, bias=False)
        self.v_fwd = nn.Linear(d, d, bias=False)
        self.k_rev = nn.Linear(d, d, bias=False)
        self.v_rev = nn.Linear(d, d, bias=False)
        self.marker_gate = nn.Sequential(nn.Linear(cfg.marker_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.null_key = nn.Parameter(torch.randn(2, d) * 0.02)
        self.null_value = nn.Parameter(torch.randn(2, d) * 0.02)
        self.hop = HopBlock(d, cfg.d_ff)
        if cfg.use_links:
            self.v_link = nn.Linear(d, d, bias=False)              # an alias's value: the TARGET'S KEY, projected
            self.link_rev_key = nn.Parameter(torch.randn(d) * 0.02)  # aliases are not reverse-addressable
        if cfg.n_deref > 0:
            if not cfg.use_null_cell:
                raise ValueError("n_deref > 0 needs the null column: it carries the dereference passthrough")
            self.deref = nn.ModuleList([DerefBlock(d, cfg.deref_query_from_state, cfg.use_deref_passthrough)
                                        for _ in range(cfg.n_deref)])
        self.out_ln = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d)                            # tied read-out: logits = proj(h) · E^T
        self.unknown_head = nn.Linear(d, 1)                        # extra logit for UNKNOWN
        # fallback path used only by the "no routing" ablation: the core must memorise facts
        self.no_route_ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    # ------------------------------------------------------------------ knowledge layer
    def gate_logits(self, marker: torch.Tensor) -> torch.Tensor:
        return self.marker_gate(marker)

    def gate(self, marker: torch.Tensor) -> torch.Tensor:
        """Marker gate in (0, 1); shape (C, 1).  In ``hard_gate`` mode the gate is 0 or 1."""
        if not self.cfg.use_marker_gate:
            return torch.ones(marker.shape[0], 1, device=marker.device)
        g = torch.sigmoid(self.gate_logits(marker))
        if self.cfg.hard_gate:
            g = (g > 0.5).to(g.dtype)
        return g

    def encode_bank(self, bank: Dict[str, torch.Tensor], noise: float = 0.0,
                    generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        s, r, o = self.ent_emb(bank["subject"]), self.cell_rel_emb(bank["relation"]), self.ent_emb(bank["obj"])
        g = self.gate(bank["marker"])
        k_f = self.k_fwd(self.ln_key(s + r))
        v_f = self.v_fwd(o)
        k_r = self.k_rev(self.ln_key(o + r))
        v_r = self.v_rev(s)
        if self.cfg.use_links and "is_link" in bank:
            # Control-plane materialisation, exactly like the marker: the store decides WHICH payload a row
            # carries; the model is never told that a value it has read is a pointer — that it must learn.
            il = bank["is_link"][:, None]
            tgt_key = self.ln_key(self.ent_emb(bank["link_subject"]) + self.cell_rel_emb(bank["link_relation"]))
            v_f = torch.where(il, self.v_link(tgt_key), v_f)
            k_r = torch.where(il, self.link_rev_key[None].expand_as(k_r), k_r)
            v_r = torch.where(il, torch.zeros_like(v_r), v_r)
        v_f = v_f * g
        v_r = v_r * g
        if noise > 0:
            def jitter(x: torch.Tensor) -> torch.Tensor:
                rms = x.pow(2).mean().sqrt()
                return x + noise * rms * torch.randn(x.shape, generator=generator, device=x.device)
            k_f, v_f, k_r, v_r = jitter(k_f), jitter(v_f), jitter(k_r), jitter(v_r)
        return {"k_f": k_f, "v_f": v_f, "k_r": k_r, "v_r": v_r, "gate": g.squeeze(-1), "active": bank["active"],
                "gate_logits": self.gate_logits(bank["marker"]).squeeze(-1) if self.cfg.use_marker_gate else None}

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        z = self.out_ln(h)
        ent_logits = (self.out_proj(z) @ self.ent_emb.weight.t()) / (self.cfg.d_model ** 0.5)
        return torch.cat([ent_logits, self.unknown_head(z)], dim=-1)

    # ------------------------------------------------------------------ forward
    def forward(self, bank: Dict[str, torch.Tensor], mode: torch.Tensor, start: torch.Tensor, rels: torch.Tensor,
                hop_valid: torch.Tensor, noise: float = 0.0, generator: Optional[torch.Generator] = None,
                cell_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns ``(logits (B, n_entities+1), routing (B, H, C+1), extras)``.

        ``cell_mask`` (C,) bool optionally restricts routing further (causal interventions).
        """
        B, H = rels.shape
        d = self.cfg.d_model
        tokens = torch.stack([self.mode_emb(mode), self.ent_emb(start)] + [self.rel_emb(rels[:, t]) for t in range(H)], dim=1)
        tokens = tokens + self.pos_emb(torch.arange(H + 2, device=rels.device))[None]
        pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=rels.device), ~hop_valid], dim=1)
        x = self.core_ln(self.core(tokens, src_key_padding_mask=pad))
        h = x[:, 1] + self.mode_emb(mode)
        routing = torch.zeros(B, H, 1, device=rels.device)
        extras: Dict[str, torch.Tensor] = {}
        if not self.cfg.use_routing:
            for t in range(H):
                h_new = h + self.no_route_ff(x[:, 2 + t] + h)
                h = torch.where(hop_valid[:, t, None], h_new, h)
            extras["hidden"] = h
            return self.readout(h), routing, extras
        enc = self.encode_bank(bank, noise=noise, generator=generator)
        k_f, v_f, k_r, v_r = enc["k_f"], enc["v_f"], enc["k_r"], enc["v_r"]
        allowed = enc["active"]
        if cell_mask is not None:
            allowed = allowed & cell_mask
        if self.cfg.use_null_cell:
            k_f = torch.cat([k_f, self.null_key[0][None]]); v_f = torch.cat([v_f, self.null_value[0][None]])
            k_r = torch.cat([k_r, self.null_key[1][None]]); v_r = torch.cat([v_r, self.null_value[1][None]])
            allowed = torch.cat([allowed, torch.ones(1, dtype=torch.bool, device=rels.device)])
        is_fwd = (mode == 0)
        D = self.cfg.n_deref
        slots = H * (1 + D)
        routing = torch.zeros(B, slots, k_f.shape[0], device=rels.device)
        for t in range(H):
            rel_t = x[:, 2 + t]
            valid = hop_valid[:, t]
            read, p = self.hop.read(h, rel_t, self.hop_emb.weight[t][None], k_f, v_f, k_r, v_r, is_fwd, allowed)
            routing[:, t * (1 + D)] = torch.where(valid[:, None], p, torch.zeros_like(p))
            for dd in range(D):
                if self.cfg.disable_deref:
                    p = torch.zeros(B, k_f.shape[0], device=rels.device)
                    p[:, -1] = 1.0                                    # ablation: keep the value, follow nothing
                else:
                    read, p = self.deref[dd](read, h, k_f, v_f, allowed)
                routing[:, t * (1 + D) + 1 + dd] = torch.where(valid[:, None], p, torch.zeros_like(p))
            h_new = self.hop.apply_read(h, read)
            h = torch.where(valid[:, None], h_new, h)
        extras["gate"] = enc["gate"]
        extras["gate_logits"] = enc["gate_logits"]
        extras["hidden"] = h
        extras["value_norm"] = torch.cat([v_f.norm(dim=-1)[:-1], v_f.new_zeros(1)]) if self.cfg.use_null_cell else v_f.norm(dim=-1)
        return self.readout(h), routing, extras

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
