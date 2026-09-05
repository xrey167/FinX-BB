"""E-000060 — scope before routing, with UNKNOWN only at the terminal read site.

The E-000052/053 seam is now structural.  The recorded two-channel implementation gates the NULL and
CELL contributions *after* the routing softmax, so generic prose can still receive an accidentally
matched cell and a broken question receives only the softmax-weighted fraction of UNKNOWN.  E-000057
makes three states explicit but, with two read sites, a legitimate one-hop query has an unused early
read whose 'no match' must mean BYPASS rather than UNKNOWN.

This experiment encodes the intended state machine directly:

  scope=0                              -> BYPASS (exact zero write at every read site)
  scope=1, match=1                    -> CELL
  scope=1, match=0, non-terminal read -> BYPASS
  scope=1, match=0, terminal read     -> full UNKNOWN

Training uses a differentiable analogue of the same state machine.  Then, with every other adapter
weight frozen, the scope heads receive E-000055's explicit binary calibration on answerable+broken
questions versus unrelated prose.  Evaluation uses the fixed 0.5 relevance decision and the model's
OWN learned match threshold -- there is no test-set threshold sweep.  Therefore a positive screen can
be promoted directly to independent seeds and attacks instead of being a grid-search artefact.
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.experiments.e000055_relevance_calibration import calibrate_relevance
from so.llm_adapter import AdapterConfig, KnowledgeAdapterLM

BARS = {
    "train/active_correct": (">=", .95),
    "heldout/active_correct": (">=", .95),
    "revoke_train_min": (">=", .95),
    "revoke_heldout_min": (">=", .95),
    "shred_heldout_min": (">=", .95),
    "heldout/revoked_deleted_object": ("<=", .02),
    "broken1_unknown": (">=", .90),
    "generic/kl_to_base": ("<=", .05),
}


class ScopeBeforeRoutingAdapter(KnowledgeAdapterLM):
    """Experiment-local two-read state machine; no dereference arm is used here."""
    def __init__(self, *args, **kwargs):
        self.scope_mode = "soft"
        super().__init__(*args, **kwargs)

    def _make_hook(self, read_index: int, layer: int):
        def hook(module, inputs, output):
            if self._ctx is None:
                return None
            if self.cfg.n_deref != 0:
                raise RuntimeError("E-000060 is pre-registered for n_deref=0")
            h = output[0] if isinstance(output, tuple) else output
            ctx = self._ctx
            B = h.shape[0]
            ar = torch.arange(B, device=h.device)
            hl = h[ar, ctx["last_idx"]]
            q = self.q_proj[str(layer)](self.q_ln[str(layer)](hl))
            ctx.setdefault("query", []).append(q)

            keys = torch.cat([ctx["keys"], self.null_key[read_index][None]])
            values = torch.cat([ctx["values"], self.null_value[read_index][None]])
            allowed = torch.cat([ctx["allowed"], torch.ones(1, dtype=torch.bool, device=h.device)])
            scores = (q @ keys.t()) * (self.scale / self.cfg.d_key ** 0.5)
            scores = scores.masked_fill(~allowed[None], float("-inf"))
            p = torch.softmax(scores, dim=-1)
            ctx["routing"].append(p)
            val = p @ values
            null_soft = p[:, -1:] * values[-1][None]
            cell_soft = val - null_soft

            # Scope is computed from the frozen-model state before any write at this read site.
            rel_logit = self.query_relevance[str(layer)](hl).squeeze(-1)
            rel = torch.sigmoid(rel_logit)
            ctx.setdefault("relevance", []).append(rel.detach())
            ctx.setdefault("relevance_logits", []).append(rel_logit)

            cells = ctx["keys"]
            qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            kn = cells / (cells.norm(dim=-1, keepdim=True) + 1e-6)
            cos = (qn @ kn.t()).masked_fill(~ctx["allowed"][None], -1.0)
            cos_max = cos.max(dim=-1).values if cos.shape[-1] else torch.full((B,), -1.0, device=h.device)
            tau = self.match_tau[read_index]
            m = torch.sigmoid((cos_max - tau) * self.match_temp[read_index].abs())
            ctx.setdefault("match", []).append(m.detach())

            terminal = read_index == len(self.cfg.read_layers) - 1
            if self.scope_mode == "soft":
                # Same semantics as the hard state machine, but gradients flow through both decisions.
                cell_c = cell_soft * (rel * m)[:, None]
                null_c = (values[-1][None] * (rel * (1.0 - m))[:, None]) if terminal else torch.zeros_like(cell_c)
            elif self.scope_mode == "hard":
                rh = (rel >= 0.5).to(val.dtype)
                mh = (cos_max >= tau).to(val.dtype)
                cell_c = cell_soft * (rh * mh)[:, None]
                null_c = (values[-1][None] * (rh * (1.0 - mh))[:, None]) if terminal else torch.zeros_like(cell_c)
            else:
                raise ValueError(self.scope_mode)

            read = self.o_proj[str(layer)](cell_c + null_c)
            rms_h = hl.detach().pow(2).mean(-1, keepdim=True).sqrt()
            if self.cfg.fallback == "prior":
                read = read * rms_h * self.inject_gain[read_index]
            else:
                # Keep E-000022's ungated reference normalisation: a zero state-machine write stays zero.
                ref = self.o_proj[str(layer)](val)
                rms_r = ref.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-3 * rms_h + 1e-6)
                read = read * (rms_h / rms_r) * self.inject_gain[read_index]
            delta = torch.zeros_like(h)
            delta[ar, ctx["last_idx"]] = read
            h2 = h + delta
            return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2
        return hook


def check(m: Dict[str, float]) -> Dict:
    out = {}
    for k, (op, bar) in BARS.items():
        v = float(m.get(k, float("nan")))
        ok = v >= bar if op == ">=" else v <= bar
        out[k] = {"value": v, "op": op, "bar": bar, "pass": bool(ok)}
    return out


def evaluate(gk, centre: np.ndarray, seed: int) -> Dict[str, float]:
    m = E17.evaluate_templates(gk, 13100 + seed, centre, E18.N_TRAIN_TEMPLATES)
    return {k: float(v) for k, v in m.items() if isinstance(v, (int, float, bool))}


def run(seed: int, steps: int, cal_steps: int, threads: int, outdir: str) -> Dict:
    if threads:
        torch.set_num_threads(threads)
    os.environ["SO_BOS"] = "1"
    old = E8.KnowledgeAdapterLM
    E8.KnowledgeAdapterLM = ScopeBeforeRoutingAdapter
    try:
        cfg = AdapterConfig(status_gated=True, match_gate=True, two_channel_null=True)
        gk = E8.GPT2Knowledge(cfg)
    finally:
        E8.KnowledgeAdapterLM = old

    t0 = time.time()
    gk.model.scope_mode = "soft"
    trained = E18.train_arm(gk, seed, steps, generic_share=0.25)
    centre = np.asarray(trained["centre"])
    before = evaluate(gk, centre, seed)

    # Directly teach only the scope decision; every other trained parameter is frozen by this helper.
    cal_hist = calibrate_relevance(gk, seed, cal_steps)
    soft = evaluate(gk, centre, seed)

    # Fixed hard policy: relevance 0.5, learned match_tau. No held-out/generic test metric selects either.
    gk.model.scope_mode = "hard"
    hard = evaluate(gk, centre, seed)
    c = check(hard)
    rec = {
        "experiment": "E-000060", "candidate_only": True, "seed": seed, "steps": steps,
        "cal_steps": cal_steps, "adapter": cfg.to_dict(),
        "learned_match_tau": [float(x) for x in gk.model.match_tau.detach()],
        "before_calibration": before, "calibrated_soft": soft,
        "hard_metrics": hard, "criteria": c,
        "screening_pass": all(x["pass"] for x in c.values()),
        "calibration_history": cal_hist, "seconds": time.time() - t0,
    }
    print(json.dumps({"seed": seed, "screening_pass": rec["screening_pass"],
                      "match_tau": rec["learned_match_tau"],
                      **{k: round(c[k]["value"], 4) for k in BARS}}, indent=2), flush=True)
    p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
    (p / f"e000060-seed{seed}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--cal-steps", type=int, default=250)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", default="ci-e60")
    a = ap.parse_args(); run(a.seed, a.steps, a.cal_steps, a.threads, a.results_dir)
