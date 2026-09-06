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
                      routing attention over cell keys  (key = W_in[subject] + R[relation])
                                        │
                                        ▼
                      gated cell value  (value = Wv · W_out[object] ⊙ gate(marker))

Because the value is built from the model's own OUTPUT embedding row for the
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

import numpy as np
import torch
import torch.nn as nn


def transformer_blocks(lm) -> nn.ModuleList:
    """The list of decoder blocks, for the several names the libraries use.

    The read layers are attached with forward hooks, so the layer needs a handle on the blocks
    themselves. ``lm.transformer.h`` is GPT-2's name for them; GPT-NeoX calls the same thing
    ``gpt_neox.layers`` and the Llama family ``model.layers``. Nothing else about the core is
    assumed: the hook takes the first element when a block returns a tuple and the tensor itself
    when it does not, which covers all three.
    """
    for path in ("transformer.h", "gpt_neox.layers", "model.layers", "transformer.blocks"):
        obj = lm
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise TypeError(f"cannot find the decoder blocks of {type(lm).__name__}; "
                    f"add its attribute path to transformer_blocks()")


def hidden_size(lm) -> int:
    cfg = lm.config
    for name in ("hidden_size", "n_embd", "d_model"):
        v = getattr(cfg, name, None)
        if isinstance(v, int):
            return v
    raise TypeError(f"cannot find the hidden size of {type(lm).__name__}")


@dataclass
class AdapterConfig:
    n_relations: int = 4
    read_layers: Tuple[int, ...] = (8, 10)
    d_key: int = 256
    marker_dim: int = 16
    use_marker_gate: bool = True
    hard_gate: bool = False        # verification mode: gate thresholded at 0.5
    status_gated: bool = False     # E-000012: revoked cells stay routable; the status flag is folded into the gate
    use_links: bool = False        # E-000020: the bank may contain alias rows whose payload is the TARGET'S KEY
    n_deref: int = 0               # E-000020: dereference reads after each read layer (1 resolves an alias)
    two_channel_null: bool = False # E-000022: split the null column into a payload-absence signal (inject the
                                   # unknown direction when a QUESTION finds no cell) and a query-relevance
                                   # signal (inject nothing when the text is not a question about a cell).
                                   # Without the split both live in one column and contradict each other.
    match_gate: bool = False       # E-000018: scale the injection by how well the query matches ANY real cell key.
                                   # The routing softmax always sums to one, so some cell always wins and the layer
                                   # injects into text it has no key for; an absolute match score can say "nothing".
    payload_from: str = "output"   # E-000027: which embedding matrix the injected value is built from.
                                   # 'output' is the only correct choice -- the payload has to raise the
                                   # object's logit at the LM head, and the head scores against the OUTPUT
                                   # matrix. GPT-2 ties the two, so 'input' is indistinguishable there;
                                   # on an untied model it is the control arm that should fail.
    fallback: str = "unknown"      # 'unknown': a null / unsigned / revoked read emits ' unknown';
                                   # 'prior' (E-000013): it injects nothing, so the pretrained distribution returns

    def to_dict(self) -> Dict:
        return asdict(self)


class KnowledgeAdapterLM(nn.Module):
    def __init__(self, lm, cfg: AdapterConfig, entity_token_ids: Sequence[int], unknown_token_id: int):
        super().__init__()
        self.lm = lm
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.cfg = cfg
        d = hidden_size(lm)
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
        if cfg.match_gate:
            self.match_tau = nn.Parameter(torch.full((len(cfg.read_layers),), 0.3))
            self.match_temp = nn.Parameter(torch.full((len(cfg.read_layers),), 10.0))
        if cfg.two_channel_null:
            # is this text a question about some subject and relation at all? Read from the state, not from
            # the cells, so it can fire for a question whose cell is missing and stay silent on prose.
            self.query_relevance = nn.ModuleDict(
                {str(l): nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1)) for l in cfg.read_layers})
        if cfg.use_links:
            self.v_link = nn.Linear(cfg.d_key, d, bias=False)      # an alias's value: its TARGET'S KEY, in value space
        if cfg.n_deref > 0:
            self.q_deref = nn.ModuleDict({str(l): nn.Linear(d, cfg.d_key) for l in cfg.read_layers})
            for q in self.q_deref.values():          # zero-init: every cell scores exactly 0 at the start, so
                nn.init.zeros_(q.weight)             # the passthrough bias below is the only thing that decides
                nn.init.zeros_(q.bias)               # and the slot is an identity until training moves it
            self.deref_ln = nn.ModuleDict({str(l): nn.LayerNorm(d) for l in cfg.read_layers})
            self.deref_scale = nn.Parameter(torch.ones(len(cfg.read_layers)))
            # Initialised so the dereference slot starts as a near-perfect PASSTHROUGH. Without it the slot
            # begins with a random query, spreads its mass over every cell and injects the average of all
            # values into the frozen model, which poisons training from the first step: the first attempt at
            # E-000020 collapsed to answering ' unknown' everywhere, direct reading included.
            # The bias competes against the SUM of the cell scores, not against one of them, so log(C) is
            # added at read time and this number is a log-odds against the whole bank, independent of its size.
            # A plain +5 leaves only 16% of the mass on the passthrough at 800 cells, which is why the first
            # fix did not work either.
            self.deref_pass_bias = nn.Parameter(torch.full((len(cfg.read_layers),), 7.0))
        self.null_key = nn.Parameter(torch.randn(len(cfg.read_layers), cfg.d_key) * 0.02)
        with torch.no_grad():
            out_emb = lm.get_output_embeddings()
            w_out = (lm.get_input_embeddings() if out_emb is None else out_emb).weight
            unk = w_out[unknown_token_id].detach().clone()
        self.null_value = nn.Parameter(unk[None].repeat(len(cfg.read_layers), 1))   # "nothing found" -> ' unknown'
        if cfg.fallback == "prior":
            # "nothing found" -> no injection at all (fixed, not learnable: no constant shortcut can be learned)
            self.null_value = nn.Parameter(torch.zeros(len(cfg.read_layers), d), requires_grad=False)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self._ctx: Optional[Dict] = None
        blocks = transformer_blocks(lm)
        self._hooks = [blocks[l].register_forward_hook(self._make_hook(i, l)) for i, l in enumerate(cfg.read_layers)]

    @property
    def w_in(self) -> torch.Tensor:
        """The input-embedding matrix: the space the residual stream is written in."""
        return self.lm.get_input_embeddings().weight

    @property
    def w_out(self) -> torch.Tensor:
        """The output-embedding matrix: the rows the LM head scores the residual against.

        On GPT-2 this is the SAME tensor as ``w_in`` -- the model ties them -- which is why the layer
        could be written against ``wte`` alone and still work.  Most models since do not tie them
        (Llama, Qwen, OLMo, Pythia all default to untied, and Pythia's two matrices have a row-wise
        cosine near zero), and there the distinction decides whether the mechanism works at all: a
        payload built from the INPUT embedding of the object raises nothing at the head.  Every
        logit-raising direction below therefore comes from ``w_out`` and every addressing direction
        from ``w_in``.
        """
        out = self.lm.get_output_embeddings()
        return self.w_in if out is None else out.weight

    @property
    def wte(self) -> torch.Tensor:
        """Retained name: the input embeddings. Kept so older call sites keep meaning what they meant."""
        return self.w_in

    @property
    def ties_embeddings(self) -> bool:
        return self.w_in.data_ptr() == self.w_out.data_ptr()

    def adapter_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("lm.") and p.requires_grad]

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
        subj = self.w_in[self.entity_token_ids[bank["subject"]]]     # a key, not a logit direction
        w_val = self.w_in if self.cfg.payload_from == "input" else self.w_out
        obj = w_val[self.entity_token_ids[bank["obj"]]]
        if "payload_delta" in bank:
            obj = obj + bank["payload_delta"]      # zero by default; see so.audit.certify_structural
        keys = self.k_proj(self.ln_key(subj + self.rel_emb(bank["relation"])))
        g = self.gate(bank["marker"])
        payload = self.v_proj(obj)
        if self.cfg.use_links and "is_link" in bank:
            # Control-plane materialisation, exactly like the marker: an alias row carries its TARGET'S KEY
            # instead of an object. What that does NOT buy is concealment: v_link is a separate projection
            # over a layer-normalised input, so a pointer's payload sits at a different scale from an
            # object's and E-000034 separates the two by norm alone at 1.0000. Recognising a pointer is
            # free; only following one is learned.
            tgt = self.k_proj(self.ln_key(self.w_in[self.entity_token_ids[bank["link_subject"]]]
                                          + self.rel_emb(bank["link_relation"])))
            payload = torch.where(bank["is_link"][:, None], self.v_link(tgt), payload)
        if self.cfg.status_gated:
            g = g * bank["active"].float()[:, None]          # an inactive (revoked) cell reads as unknown / as nothing
        if self.cfg.fallback == "prior":
            # unit-RMS payload: the injection downstream is scaled statically, so a closed gate or a null read
            # injects nothing and the frozen model's own distribution is what remains
            payload = payload / (payload.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
            values = payload * g
        else:
            unk = self.v_proj(w_val[self.candidate_ids[-1]][None])             # the ' unknown' direction
            # the gate selects between the payload and "unknown": an unsigned payload READS AS unknown.
            # (a mere attenuation would be undone by the RMS-matched injection downstream)
            values = payload * g + unk * (1 - g)
        allowed = bank["routable"] if (self.cfg.status_gated and "routable" in bank) else bank["active"]
        return {"keys": keys, "values": values, "values_payload": payload, "gate": g.squeeze(-1), "active": allowed}

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
            # E-000037: the ROUTING QUERY is recorded (with its graph, not detached). It is the only
            # tensor in the read path that depends on the phrasing: keys and values are built from the
            # bank alone (encode_bank), o_proj and inject_gain are shared, and everything after the
            # injection is the frozen core. Unlike the routing distribution p below, q does not depend
            # on which cells happen to be in this step's bank, so an invariance trained on it holds for
            # every bank.
            ctx.setdefault("query", []).append(q)
            keys = torch.cat([ctx["keys"], self.null_key[read_index][None]])
            values = torch.cat([ctx["values"], self.null_value[read_index][None]])
            allowed = torch.cat([ctx["allowed"], torch.ones(1, dtype=torch.bool, device=h.device)])
            scores = (q @ keys.t()) * (self.scale / self.cfg.d_key ** 0.5)
            scores = scores.masked_fill(~allowed[None], float("-inf"))
            p = torch.softmax(scores, dim=-1)
            val = p @ values                                                  # (B, d)
            w_null = p[:, -1:]                    # how much of val came from the null column
            if self.cfg.n_deref > 0:
                # the resolve slot is recorded HERE, before its dereferences, so the routing tensor reads
                # (resolve, deref...) per read layer and lines up with the supervision built by the
                # experiments. Appending it at the end instead silently swapped the two, which is what
                # actually collapsed the first attempts at E-000020: the resolve slot was trained on the
                # dereference target and the dereference slot on the resolve target.
                ctx["routing"].append(p)
                for _ in range(self.cfg.n_deref):
                    qd = self.q_deref[str(layer)](self.deref_ln[str(layer)](val))
                    sd = (qd @ keys.t()) * (self.deref_scale[read_index] / self.cfg.d_key ** 0.5)
                    sd = sd.masked_fill(~allowed[None], float("-inf"))
                    n_cells = max(int(ctx["allowed"].sum().item()), 1)
                    bias = self.deref_pass_bias[read_index] + float(np.log(n_cells))
                    sd = torch.cat([sd[:, :-1], sd[:, -1:] + bias], dim=-1)
                    pd = torch.softmax(sd, dim=-1)
                    # the null column carries the incoming value: "what I read was not a pointer"
                    val = pd[:, :-1] @ values[:-1] + pd[:, -1:] * val
                    w_null = w_null * pd[:, -1:]  # the null share survives only through the passthroughs
                    ctx["routing"].append(pd)
            null_c = w_null * values[-1][None]        # the null column's share of the read
            cell_c = val - null_c                     # everything the cells contributed
            if self.cfg.two_channel_null:
                # the null column stops carrying the ' unknown' direction on its own: its contribution is
                # scaled by how much this text looks like a question about a cell at all
                rel = torch.sigmoid(self.query_relevance[str(layer)](hl)).squeeze(-1)
                null_c = null_c * rel[:, None]
                ctx.setdefault("relevance", []).append(rel.detach())
            if self.cfg.match_gate:
                # absolute agreement with the best REAL cell key; the null key is excluded on purpose, because
                # the question is whether any cell matches at all, not which one wins the competition
                cells = ctx["keys"]
                qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
                kn = cells / (cells.norm(dim=-1, keepdim=True) + 1e-6)
                cos = (qn @ kn.t()).masked_fill(~ctx["allowed"][None], -1.0)
                cos_max = cos.max(dim=-1).values if cos.shape[-1] else torch.full((B,), -1.0, device=h.device)
                m = torch.sigmoid((cos_max - self.match_tau[read_index]) * self.match_temp[read_index].abs())
                ctx.setdefault("match", []).append(m.detach())
                cell_c = cell_c * m[:, None]          # a payload is injected only when a cell actually matches
            read = self.o_proj[str(layer)](cell_c + null_c)                   # (B, d)
            rms_h = hl.detach().pow(2).mean(-1, keepdim=True).sqrt()
            if self.cfg.fallback == "prior":
                read = read * rms_h * self.inject_gain[read_index]               # static: a zero read stays zero
            else:
                # The normaliser is computed from the UNGATED read. Taking it from the gated one divides the
                # gates straight back out — read * m, then divided by the RMS of read * m, is read again —
                # which is why the match gate arm of E-000018 measured no effect whatsoever. With the
                # reference ungated, a closed gate genuinely injects less, and the denominator can no longer
                # be driven to zero by a cancellation, which was the NaN in E-000022's second seed.
                ref = self.o_proj[str(layer)](val)
                rms_r = ref.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-3 * rms_h + 1e-6)
                read = read * (rms_h / rms_r) * self.inject_gain[read_index]
            delta = torch.zeros_like(h)
            delta[ar, ctx["last_idx"]] = read
            if self.cfg.n_deref == 0:
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
        # (B, len(read_layers), d_key), read-layer order. Kept as an attribute rather than a fifth
        # return value so that every existing call site keeps its four-tuple.
        self.last_query = (torch.stack(self._ctx["query"], dim=1)
                           if self._ctx is not None and self._ctx.get("query") else None)
        self._ctx = None
        return cand, full, routing, hidden
