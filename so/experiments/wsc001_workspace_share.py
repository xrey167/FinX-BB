"""WSC-001 -- what a workspace basis accounts for in an external memory's write, and what it cannot.

THE QUESTION. Anthropic's Jacobian lens defines, at layer l, one direction per output token: v_u =
J_l^T W_U[u], the direction whose inner product with the residual gives the FIRST-ORDER contribution to
that token's final logit. J-Access (Song et al., arXiv:2608.11408) turns the lens into an accessibility
audit for unlearning: a fact is "still accessible" if its token enters the lens's top-k. That audit
reads exactly the span of those directions and nothing else.

An external memory does not write infinitesimally. This adapter's read is RMS-matched to the residual
stream before it is added (so/llm_adapter.py, ``read * (rms_h / rms_r) * inject_gain``), so the write
is the size of the state it is added to. A first-order basis is guaranteed to account for a
perturbation only in the limit of a small one. Whether it accounts for a write of THIS size is not a
theorem, it is a measurement, and this file makes it.

THE SUBSTRATE IS WHY THE MEASUREMENT IS POSSIBLE HERE. J-Access reports item-level AUROC at chance
because parametric unlearning gives no clean counterfactual: there is no model that never knew the
fact. A canonical-pod store gives exactly that -- ACTIVE, EVICTED and NEVER banks over the same frozen
weights and the same prompt -- and it gives a second thing no parametric setting has: the write is a
tensor the harness holds, so it can be decomposed and re-injected.

WHAT IS BY CONSTRUCTION, STATED BEFORE THE RUN.
  * The orthogonal arm has, by construction, no first-order effect on any entity logit. That is the
    definition of the lens and it is not a finding. What is NOT by construction is whether the model's
    answer survives on it, because the write is large and the frozen blocks after the read site are
    nonlinear.
  * The workspace arm's audit-completeness, in the confined follow-up, is by construction. Only its
    capability can fail.
  * keep + drop is the write exactly at the FIRST read site only; the two arms diverge afterwards
    because the earlier write is in the residual the later read builds its query from
    (so/tests/test_inject_projection.py). Arms are therefore reported per read site.

WHAT CAN FAIL. The matched-rank random subspace. Projecting onto any r-dimensional subspace keeps r/d
of a random vector's mass, so if the workspace arm behaves like the random arm of the same rank, this
experiment has measured rank and not the workspace, and says so.

Run:  SO_BOS=1 python -m so.experiments.wsc001_workspace_share [--seeds 0 1 2] [--threads 4]
      SO_BOS=1 python -m so.experiments.wsc001_workspace_share --smoke --seeds 0 --templates 3
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from so import ledger
from so.data import Bank, bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, _sha256
from so.jlens import random_basis, workspace_basis
from so.llm_adapter import AdapterConfig
from so.world import UNKNOWN

N_TEMPLATES = 12
ANCHOR_TEMPLATE = 3

# The corpus the lens is estimated on. Ordinary prose that contains none of the synthetic world's
# entities, so the basis is a property of the frozen model and not of the evaluation.
LENS_CORPUS = [
    "The capital of France is Paris and the capital of Japan is Tokyo.",
    "A researcher compared several methods before writing the report.",
    "The city centre was busy during the afternoon and quiet at night.",
    "Machine learning models transform sequences through many intermediate representations.",
    "Water freezes at zero degrees Celsius under ordinary conditions.",
    "A train crossed the bridge and continued toward the next station.",
    "The committee published a revised version of the document.",
    "Several countries maintain diplomatic relations with their neighbours.",
]

# Arms. Each is (basis kind, mode); "full" is the trained write and "none" the no-memory floor.
ARMS: Tuple[Tuple[str, str, str], ...] = (
    ("full", "none", ""),
    ("W", "workspace", "keep"),
    ("O", "workspace", "drop"),
    ("Wn", "workspace", "keep_renorm"),
    ("On", "workspace", "drop_renorm"),
    ("R", "random", "keep"),
    ("Rc", "random", "drop"),
    ("Rn", "random", "keep_renorm"),
    ("none", "none", "zero"),
)


def load_bos_adapter(seed: int, suffix: str = "_bos") -> Tuple[E8.GPT2Knowledge, np.ndarray, str]:
    """The BOS-trained symlink adapter: the strongest recorded reader in this repository."""
    path = CHECKPOINTS / f"e000020_gpt2{suffix}_seed{seed}.pt"
    if not path.exists():
        raise SystemExit(
            f"missing checkpoint {path}; run:\n"
            f"  SO_BOS=1 SO_CKPT_SUFFIX={suffix} python -m so.experiments.e000052_symlink_bos_train --seeds {seed}")
    ck = torch.load(path, weights_only=False)
    cfg = AdapterConfig(**ck["adapter_config"]) if "adapter_config" in ck else \
        AdapterConfig(status_gated=True, use_links=True, n_deref=1)
    gk = E8.GPT2Knowledge(cfg)
    gk.model.load_state_dict(ck["adapter"], strict=False)
    gk.model.eval()
    return gk, np.asarray(ck["centre"]), _sha256(path)


def bases_for(gk: E8.GPT2Knowledge, rank_energy: float = 0.99, null_seed: int = 0
              ) -> Dict[int, Dict[str, Any]]:
    """One workspace basis and one matched-rank random basis per read layer.

    The token family is the adapter's own candidate set: the 256 entity tokens plus ' unknown'. Those
    are exactly the tokens the reader's answer is scored over, so this is the subspace an audit of THIS
    memory would read, not a generic choice.
    """
    tok_ids = [int(t) for t in gk.model.candidate_ids]
    enc = gk.tok(LENS_CORPUS, return_tensors="pt", padding=True)
    w_out = gk.model.w_out
    out: Dict[int, Dict[str, Any]] = {}
    for layer in gk.model.cfg.read_layers:
        basis, sv, jl = workspace_basis(gk.model.lm, layer, tok_ids, enc["input_ids"], enc["attention_mask"],
                                        w_out, rank=None, energy=rank_energy, batch=4)
        r = basis.shape[1]
        out[layer] = {
            "workspace": basis,
            "random": random_basis(gk.model.d, r, seed=1000 * null_seed + layer),
            "rank": r,
            "n_tokens": len(tok_ids),
            "spectrum_top1_share": float((sv[0] ** 2 / (sv ** 2).sum())),
            "mean_abs_cos_between_atoms": float(
                ((jl.vectors @ jl.vectors.t()) - torch.eye(len(tok_ids))).abs().sum()
                / (len(tok_ids) * (len(tok_ids) - 1))),
            "cos_atom_to_unembedding": float(
                (jl.vectors * torch.nn.functional.normalize(w_out[torch.as_tensor(tok_ids)], dim=-1)).sum(-1).mean()),
        }
    return out


def _set_arm(gk: E8.GPT2Knowledge, bases: Dict[int, Dict[str, Any]], kind: str, mode: str,
             layers: Optional[Sequence[int]] = None) -> None:
    """Install one arm. The basis is a property of a layer, so it is installed per read site."""
    m = gk.model
    if kind == "none":
        m.set_inject_projection(None, "zero", layers=layers) if mode == "zero" else m.set_inject_projection(None)
        return
    m.set_inject_projection({l: bases[l][kind] for l in m.cfg.read_layers}, mode, layers=layers)


def answers(gk: E8.GPT2Knowledge, bank: Bank, keys: Sequence[Tuple[int, int]], template: int,
            batch: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Answers and the workspace share of the write, for one arm as currently set."""
    texts = [E17.TEMPLATES12[r][template].format(s=gk.names[s]) for s, r in keys]
    out, shares = [], []
    tensors = bank.tensors()
    for i in range(0, len(texts), batch):
        ids, am, last = E8.encode_texts(gk.tok, texts[i: i + batch])
        with torch.no_grad():
            cand, _, _, _ = gk.model(tensors, ids, am, last)
        a = cand.argmax(-1).numpy()
        out.append(np.where(a == gk.n_entities, UNKNOWN, a))
        inj = gk.model.last_injected
        shares.append(inj.norm(dim=-1).numpy() if inj is not None else np.zeros((a.shape[0], 1)))
    return np.concatenate(out), np.concatenate(shares)


def write_geometry(gk: E8.GPT2Knowledge, bases: Dict[int, Dict[str, Any]], bank: Bank,
                   keys: Sequence[Tuple[int, int]], template: int, batch: int = 64) -> Dict[str, float]:
    """The share of the trained write's squared norm that lies in each basis, per read site.

    Reported with the random basis beside it: an r-dimensional subspace holds r/d of a random vector,
    so the workspace share is only informative against that floor.
    """
    gk.model.set_inject_projection(None)
    texts = [E17.TEMPLATES12[r][template].format(s=gk.names[s]) for s, r in keys]
    tensors = bank.tensors()
    acc: Dict[str, List[float]] = {}
    for i in range(0, len(texts), batch):
        ids, am, last = E8.encode_texts(gk.tok, texts[i: i + batch])
        with torch.no_grad():
            gk.model(tensors, ids, am, last)
        inj = gk.model.last_injected            # (B, n_sites, d)
        for j, layer in enumerate(gk.model.cfg.read_layers):
            r = inj[:, j]
            n2 = r.pow(2).sum(-1).clamp_min(1e-12)
            for kind in ("workspace", "random"):
                share = ((r @ bases[layer][kind]) ** 2).sum(-1) / n2
                acc.setdefault(f"share/{kind}/layer{layer}", []).extend(share.tolist())
            acc.setdefault(f"write_rms/layer{layer}", []).extend(r.pow(2).mean(-1).sqrt().tolist())
    return {k: float(np.mean(v)) for k, v in acc.items()}


def evaluate_seed(seed: int, templates: Sequence[int], n_alias: Optional[int] = None,
                  n_direct: Optional[int] = None, verbose: bool = True) -> Dict[str, Any]:
    gk, centre, sha = load_bos_adapter(seed)
    bases = bases_for(gk, null_seed=seed)
    rng = np.random.default_rng(2000 + seed)
    world, spec = E15.sample_alias_world(rng, E20.EVAL["n_base"], E20.EVAL["n_groups"],
                                         E20.EVAL["n_alias_per_group"], gk.n_entities, 4,
                                         E20.N_TRAIN_TEMPLATES)
    store, kids = E15.load_arm(world, spec, centre, 2000 + seed, symlink=True)
    bank = bank_from_store(store)
    alias_keys = list(spec.alias_keys)
    base_keys = [f.key for f in world.facts if f.key not in spec.alias_of]
    pick = rng.choice(len(base_keys), size=min(n_direct or E20.EVAL["n_direct"], len(base_keys)), replace=False)
    direct_keys = [base_keys[int(i)] for i in pick]
    if n_alias is not None:
        alias_keys = alias_keys[:n_alias]
    truth_alias = np.array([world.index[spec.alias_of[k]] for k in alias_keys])
    truth_direct = np.array([world.index[k] for k in direct_keys])

    m: Dict[str, Any] = {"seed": seed, "checkpoint_sha256": sha}
    for layer, b in bases.items():
        m[f"basis/rank/layer{layer}"] = b["rank"]
        m[f"basis/spectrum_top1_share/layer{layer}"] = b["spectrum_top1_share"]
        m[f"basis/mean_abs_cos_between_atoms/layer{layer}"] = b["mean_abs_cos_between_atoms"]
        m[f"basis/cos_atom_to_unembedding/layer{layer}"] = b["cos_atom_to_unembedding"]
        m[f"basis/random_share_floor/layer{layer}"] = b["rank"] / gk.model.d

    t0 = time.time()
    for t in templates:
        m.update({f"t{t}/{k}": v for k, v in
                  write_geometry(gk, bases, bank, alias_keys, t).items()})
        for arm, kind, mode in ARMS:
            _set_arm(gk, bases, kind, mode)
            a_al, _ = answers(gk, bank, alias_keys, t)
            a_di, _ = answers(gk, bank, direct_keys, t)
            m[f"t{t}/{arm}/alias"] = float((a_al == truth_alias).mean())
            m[f"t{t}/{arm}/direct"] = float((a_di == truth_direct).mean())
            m[f"t{t}/{arm}/alias_unknown"] = float((a_al == UNKNOWN).mean())
        gk.model.set_inject_projection(None)
        if verbose:
            print(f"  seed {seed} t{t}: " + "  ".join(
                f"{arm} {m[f't{t}/{arm}/alias']:.3f}/{m[f't{t}/{arm}/direct']:.3f}" for arm, _, _ in ARMS)
                  + f"   [{time.time() - t0:.0f}s]", flush=True)
    m["eval_seconds"] = time.time() - t0
    return m


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--templates", type=int, nargs="*", default=list(range(N_TEMPLATES)))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--smoke", action="store_true", help="plumbing only: 40 alias reads, 40 direct, no record")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if os.environ.get("SO_BOS") != "1":
        raise SystemExit("run with SO_BOS=1: the substrate is the BOS-trained adapter and its prompts carry a BOS")

    per_seed = [evaluate_seed(s, args.templates,
                              n_alias=40 if args.smoke else None,
                              n_direct=40 if args.smoke else None)
                for s in args.seeds]
    if args.smoke:
        print("SMOKE ONLY -- nothing recorded")
        return {"per_seed": per_seed, "smoke": True}
    raise SystemExit("WSC-001 has no recorded criteria yet: the pre-registration must be committed first "
                     "(docs/novelty/wsc001-preregister.md), then this line is replaced by the scoring block.")


if __name__ == "__main__":
    main()
