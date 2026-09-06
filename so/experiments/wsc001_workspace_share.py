"""WSC-001 -- why a workspace accessibility audit cannot see an external memory, and what can.

WHERE THIS COMES FROM. E-000063 (the composed workspace-pod deletion certificate) ran for the first
time on 2026-09-06, after the J-lens was repaired to work on a frozen core (ledger 31.56), and its
seed-0 record reads:

    active_alias_correct        0.8795     the memory is read
    shred_alias_unknown         1.0000     one SHRED closes every alias at the output
    bystander_top1_agree        1.0000     locality is exact
    finalprobe(ACTIVE - NEVER)  0.9018     the memory is massively present in the final hidden state
    jprobe(ACTIVE - NEVER)     -0.0134     the J-space readout cannot tell live memory from none  <-- FAIL
    jprobe(SHRED - NEVER)      -0.0179     "no workspace trace after deletion"                    <-- passes

The second-to-last row is a pre-registered VALIDITY bar and it failed; the last row is the
certificate's headline and it passed. It passed because the instrument never saw the thing it was
certifying gone. This programme has caught that failure mode ten times under the rule *an instrument
that cannot fail is not evidence*; here it is an instrument that cannot SEE, and E-000063's own
validity bar caught it.

WHAT THIS FILE ASKS. Why is the workspace readout blind, and is anything else? Three explanations are
distinguishable and only one of them is about the workspace:

    DIMENSION  k coordinates are not enough, whatever they are
    DIRECTION  k coordinates are enough, but not those k
    DEPTH      the memory is not yet in linearly decodable form at the site the audit reads

PART A, the readout side. Probes trained on ACTIVE states ONLY and applied unchanged to SHRED and
NEVER (E-000063's transfer probe), over five matched feature families at the audit's own site and at
the final state: the J-lens atoms of the pod objects (the audit), an equally sized random projection
(the dimension-matched null), the top-k principal components of the ACTIVE states (the best k-dimensional
linear readout there is), the objects' unembedding rows (the vocabulary-basis analogue), and the raw
state (the capacity ceiling).

PART B, the causal side. A probe says what is readable; it does not say what the model uses. The write
is a tensor this harness holds, so it can be injected with the entire first-order channel removed:
projecting it out of the span of the J-lens atoms of ALL 257 scored tokens leaves a write whose
first-order effect on every scored logit is exactly zero (measured retention 1e-6 at full rank). If
the answer survives that, the audit's basis is not where the answer comes from.

BY CONSTRUCTION, DECLARED BEFORE THE RUN.
  * keep(W) ~= FULL and drop(W) ~= floor on the FIRST-ORDER term is algebra, not a finding: the atoms
    of the scored tokens are exactly the directions whose inner product with the write gives that term.
    What can fail is whether the ANSWER survives, which is second order.
  * A matched-rank ISOTROPIC random subspace cannot be a null for "workspaceness": at matched rank the
    atom span is the unique maximal first-order-retaining subspace, so the random arm must retain less
    (measured: keep-arm first-order retention 1.000 for the atom span against 0.583 for random). The
    null that can fail is a J-lens span built the same way over tokens that are NOT scored, which
    matches rank, construction and anisotropy and differs only in token identity.
  * At read layer 10 the lens target is one block away and the atoms ARE the unembedding rows (cos
    1.000, ledger 31.56). The second read site's "workspace" is vocabulary space, by construction.

WHAT CAN FAIL. Part A's PCA arm (if the best k-dimensional readout also sees nothing, the blindness is
not about direction), the token-matched null in both parts, the atom-write arm (a write that is by
construction the object's own output row), the content floor (another pod's write at the same
magnitude), and every validity row.

Run:  SO_BOS=1 python -m so.experiments.wsc001_workspace_share --seeds 0 1 2 --threads 4
      SO_BOS=1 python -m so.experiments.wsc001_workspace_share --smoke --seeds 0 --templates 3
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from so import ledger
from so.data import Bank, bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000063_workspace_pod_certificate as E63
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, _sha256
from so.jlens import first_order_retention, jlens_vectors, random_basis, token_matched_basis, workspace_basis
from so.llm_adapter import AdapterConfig, transformer_blocks
from so.world import UNKNOWN

# Entity-free prose: none of these tokens is in the adapter's 256-entity candidate set. (The corpus
# E-000063 estimates its lens on contains ' France' and ' Japan', which ARE candidates; a basis
# estimated on prompts containing the scored tokens is not independent of what it is used to score.)
LENS_CORPUS = [
    "The river moved slowly through the valley after the storm.",
    "She opened the window because the room had become warm.",
    "A compiler translates source code before the program is executed.",
    "Clouds formed over the mountains shortly before sunset.",
    "The old bridge was repaired during the summer months.",
    "Water freezes at zero degrees under ordinary conditions.",
    "A small garden grew behind the library near the courtyard.",
    "The committee published a revised version of the document.",
]

PART_B_ARMS = ("full", "dropW", "dropW_n", "dropT_n", "atom", "atom_dropW", "perm", "none")


# --------------------------------------------------------------------------------- part A: readouts
def probe_families(states: Dict[str, torch.Tensor], atoms: torch.Tensor, unembed: torch.Tensor,
                   k: int, seed: int) -> Dict[str, Dict[str, torch.Tensor]]:
    """Five feature families over the same states, four of them k-dimensional.

    ``states`` maps ACTIVE / SHRED / NEVER to (n, d). Cosine rather than inner product removes the
    residual's own norm, which is the confound E-000063 already controls for.
    """
    def cos(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), dim=-1) @ b.t()

    a = F.normalize(atoms.float(), dim=-1)
    u = F.normalize(unembed.float(), dim=-1)
    r = random_basis(states["active"].shape[1], k, seed=seed).t()          # (k, d), orthonormal rows
    mu = states["active"].float().mean(0, keepdim=True)
    _, _, v = torch.linalg.svd(states["active"].float() - mu, full_matrices=False)
    pca = v[:k]                                                           # (k, d) top-k of ACTIVE
    return {
        "jspace": {s: cos(x, a) for s, x in states.items()},
        "random": {s: cos(x, r) for s, x in states.items()},
        "pca": {s: cos(x, pca) for s, x in states.items()},
        "unembed": {s: cos(x, u) for s, x in states.items()},
        "raw": {s: x.float() for s, x in states.items()},
    }


def part_a(gk: E8.GPT2Knowledge, centre: np.ndarray, seed: int, n_groups: int,
           verbose: bool = True) -> Dict[str, float]:
    """E-000063's states and transfer probe, over five feature families at two sites."""
    rng = np.random.default_rng(63000 + seed)
    world, spec = E15.sample_alias_world(rng, 420, max(n_groups * 2, 48), 2, gk.n_entities, 4,
                                         E20.N_TRAIN_TEMPLATES)
    store, kids = E15.load_arm(world, spec, centre, 64000 + seed, symlink=True)
    selected, seen = [], set()
    for target, aliases in spec.groups:
        obj = int(world.index[target])
        if obj not in seen:
            selected.append((target, aliases, obj)); seen.add(obj)
        if len(selected) >= n_groups:
            break
    n = len(selected)
    obj_ids = [int(gk.model.entity_token_ids[o]) for _, _, o in selected]
    enc = gk.tok(LENS_CORPUS, return_tensors="pt", padding=True)
    w_out = gk.model.w_out
    jl = jlens_vectors(gk.model.lm, E63.JL_SOURCE, obj_ids, enc["input_ids"], enc["attention_mask"],
                       w_out, batch=4)
    cap = E63.Capture(gk)
    mid: Dict[str, List[torch.Tensor]] = {"active": [], "shred": [], "never": []}
    fin: Dict[str, List[torch.Tensor]] = {"active": [], "shred": [], "never": []}
    ys, groups = [], []
    for label, (target, aliases, obj) in enumerate(selected):
        texts, tg = E63.texts_for(aliases, gk.names, E63.TEMPLATES)
        sa, fa, _, _ = E63.eval_texts(gk, cap, store, texts)
        never = copy.deepcopy(store)
        for ak in aliases:
            if kids[ak] in never.cells:
                never.delete(kids[ak])
        if kids[target] in never.cells:
            never.delete(kids[target])
        sn, fn, _, _ = E63.eval_texts(gk, cap, never, texts)
        store.shred(kids[target])
        ss, fs, _, _ = E63.eval_texts(gk, cap, store, texts)
        store.resign(kids[target])
        for d, (a_, s_, n_) in ((mid, (sa, ss, sn)), (fin, (fa, fs, fn))):
            d["active"].append(a_); d["shred"].append(s_); d["never"].append(n_)
        ys.extend([label] * len(texts)); groups.extend(tg)
    cap.close()
    y = torch.tensor(ys, dtype=torch.long); g = torch.tensor(groups, dtype=torch.long)
    out: Dict[str, float] = {"partA/n_pods": float(n), "partA/chance": 1.0 / n}
    for site, d in (("mid", mid), ("final", fin)):
        states = {k_: torch.cat(v) for k_, v in d.items()}
        fams = probe_families(states, jl.vectors, w_out[torch.as_tensor(obj_ids)], n, seed)
        for fam, feats in fams.items():
            p = E63.transfer_probe(feats["active"], feats["shred"], feats["never"], y, g, n)
            for s_, v_ in p.items():
                out[f"partA/{site}/{fam}/{s_}"] = v_
            out[f"partA/{site}/{fam}/active_minus_never"] = p["active"] - p["never"]
            out[f"partA/{site}/{fam}/shred_minus_never"] = p["shred"] - p["never"]
            if verbose:
                print(f"  seed {seed} partA {site:5s} {fam:8s}: active {p['active']:.4f} "
                      f"shred {p['shred']:.4f} never {p['never']:.4f}  "
                      f"A-N {p['active'] - p['never']:+.4f}", flush=True)
    return out


# ---------------------------------------------------------------------------------- part B: causal
def bases_for(gk: E8.GPT2Knowledge, seed: int) -> Dict[int, Dict[str, Any]]:
    """Per read site: the full-span atom basis, a token-matched null, and their first-order retention.

    The basis at read layer l is estimated at ``hidden_states[l + 1]`` because the hook adds the write
    to the OUTPUT of block l. Getting this wrong measures a different layer's lens.
    """
    tok_ids = [int(t) for t in gk.model.candidate_ids]
    enc = gk.tok(LENS_CORPUS, return_tensors="pt", padding=True)
    w_out = gk.model.w_out
    out: Dict[int, Dict[str, Any]] = {}
    for layer in gk.model.cfg.read_layers:
        src = layer + 1
        B, sv, jl = workspace_basis(gk.model.lm, src, tok_ids, enc["input_ids"], enc["attention_mask"],
                                    w_out, rank=None, energy=1.0, batch=4)
        T, _, jlT, picked = token_matched_basis(gk.model.lm, src, tok_ids, len(tok_ids),
                                                enc["input_ids"], enc["attention_mask"], w_out,
                                                seed=7000 + 10 * seed + layer, batch=4)
        out[layer] = {
            "W": B, "T": T, "rank_W": B.shape[1], "rank_T": T.shape[1], "source_hidden_state": src,
            "drop_retention_max": float(first_order_retention(B, jl.vectors, drop=True).max()),
            "keep_retention_min": float(first_order_retention(B, jl.vectors, drop=False).min()),
            "tokenmatched_keep_retention_mean": float(first_order_retention(T, jl.vectors, drop=False).mean()),
            "random_keep_retention_mean": float(first_order_retention(
                random_basis(gk.model.d, B.shape[1], seed=layer), jl.vectors, drop=False).mean()),
            "cos_atom_unembedding_mean": float(
                (jl.vectors * F.normalize(w_out[torch.as_tensor(tok_ids)], dim=-1)).sum(-1).mean()),
        }
    return out


def _read(gk: E8.GPT2Knowledge, bank: Bank, keys: Sequence[Tuple[int, int]], truth: np.ndarray,
          template: int, arm: str, bases: Dict[int, Dict[str, Any]], batch: int = 64) -> Dict[str, float]:
    texts = [E17.TEMPLATES12[r][template].format(s=gk.names[s]) for s, r in keys]
    tensors = bank.tensors()
    m = gk.model
    got, share = [], []
    for i in range(0, len(texts), batch):
        chunk = slice(i, i + batch)
        ids, am, last = E8.encode_texts(gk.tok, texts[chunk])
        m.set_inject_projection(None); m.set_inject_override(None)
        if arm in ("atom", "atom_dropW", "perm"):
            tr = np.asarray(truth[chunk])
            if arm == "perm":
                # A DERANGEMENT OVER OBJECTS, not a roll. The smoke caught the defect a roll has here:
                # alias keys are emitted group by group and the two aliases of a pod share its object,
                # so rolling by one hands half the batch its OWN object and the floor reads 0.55
                # instead of chance. Each row is given the nearest object in the batch that differs
                # from its own, so "something was retrieved, of the wrong content" is what is measured.
                order = np.arange(len(tr))
                for j in range(len(tr)):
                    cand = np.flatnonzero(tr != tr[j])
                    order[j] = cand[(j + 1) % len(cand)] if len(cand) else j
                tr = tr[order]
            rows = m.w_out[m.entity_token_ids[torch.as_tensor(tr, dtype=torch.long)]]
            m.set_inject_override(rows.detach())
        if arm in ("dropW", "atom_dropW"):
            m.set_inject_projection({l: bases[l]["W"] for l in m.cfg.read_layers}, "drop")
        elif arm == "dropW_n":
            m.set_inject_projection({l: bases[l]["W"] for l in m.cfg.read_layers}, "drop_renorm")
        elif arm == "dropT_n":
            m.set_inject_projection({l: bases[l]["T"] for l in m.cfg.read_layers}, "drop_renorm")
        elif arm == "none":
            m.set_inject_projection(None, "zero")
        with torch.no_grad():
            cand, _, _, _ = m(tensors, ids, am, last)
        a = cand.argmax(-1).numpy()
        got.append(np.where(a == gk.n_entities, UNKNOWN, a))
        inj = m.last_injected
        if inj is not None:
            r0 = inj[:, 0]
            share.append((((r0 @ bases[m.cfg.read_layers[0]]["W"]) ** 2).sum(-1)
                          / r0.pow(2).sum(-1).clamp_min(1e-12)).numpy())
    m.set_inject_projection(None); m.set_inject_override(None)
    got = np.concatenate(got)
    return {"acc": float((got == truth).mean()), "unknown": float((got == UNKNOWN).mean()),
            "share_W": float(np.mean(np.concatenate(share))) if share else float("nan")}


def part_b(gk: E8.GPT2Knowledge, centre: np.ndarray, seed: int, templates: Sequence[int],
           n_alias: Optional[int], n_direct: int, verbose: bool = True) -> Dict[str, float]:
    bases = bases_for(gk, seed)
    rng = np.random.default_rng(2000 + seed)
    world, spec = E15.sample_alias_world(rng, E20.EVAL["n_base"], E20.EVAL["n_groups"],
                                         E20.EVAL["n_alias_per_group"], gk.n_entities, 4,
                                         E20.N_TRAIN_TEMPLATES)
    store, _ = E15.load_arm(world, spec, centre, 2000 + seed, symlink=True)
    bank = bank_from_store(store)
    alias_keys = list(spec.alias_keys)[: n_alias] if n_alias else list(spec.alias_keys)
    base_keys = [f.key for f in world.facts if f.key not in spec.alias_of]
    pick = rng.choice(len(base_keys), size=min(n_direct, len(base_keys)), replace=False)
    direct_keys = [base_keys[int(i)] for i in pick]
    truth_alias = np.array([world.index[spec.alias_of[k]] for k in alias_keys])
    truth_direct = np.array([world.index[k] for k in direct_keys])

    out: Dict[str, float] = {}
    for layer, b in bases.items():
        for key in ("rank_W", "rank_T", "source_hidden_state", "drop_retention_max", "keep_retention_min",
                    "tokenmatched_keep_retention_mean", "random_keep_retention_mean",
                    "cos_atom_unembedding_mean"):
            out[f"partB/basis/layer{layer}/{key}"] = float(b[key])
    t0 = time.time()
    for t in templates:
        for arm in PART_B_ARMS:
            al = _read(gk, bank, alias_keys, truth_alias, t, arm, bases)
            di = _read(gk, bank, direct_keys, truth_direct, t, arm, bases)
            out[f"partB/t{t}/{arm}/alias"] = al["acc"]
            out[f"partB/t{t}/{arm}/alias_unknown"] = al["unknown"]
            out[f"partB/t{t}/{arm}/direct"] = di["acc"]
            if arm == "full":
                out[f"partB/t{t}/share_W_site0"] = al["share_W"]
        if verbose:
            print(f"  seed {seed} partB t{t}: " + " ".join(
                f"{a}={out[f'partB/t{t}/{a}/alias']:.3f}" for a in PART_B_ARMS)
                  + f"  shareW={out[f'partB/t{t}/share_W_site0']:.3f}  [{time.time() - t0:.0f}s]", flush=True)
    out["partB/eval_seconds"] = time.time() - t0
    return out


def load_bos_adapter(seed: int, suffix: str = "_bos") -> Tuple[E8.GPT2Knowledge, np.ndarray, str]:
    path = CHECKPOINTS / f"e000020_gpt2{suffix}_seed{seed}.pt"
    if not path.exists():
        raise SystemExit(f"missing checkpoint {path}; run:\n  SO_BOS=1 SO_CKPT_SUFFIX={suffix} "
                         f"python -m so.experiments.e000052_symlink_bos_train --seeds {seed}")
    ck = torch.load(path, weights_only=False)
    cfg = AdapterConfig(**ck["adapter_config"]) if "adapter_config" in ck else \
        AdapterConfig(status_gated=True, use_links=True, n_deref=1)
    gk = E8.GPT2Knowledge(cfg)
    gk.model.load_state_dict(ck["adapter"], strict=False)
    gk.model.eval()
    return gk, np.asarray(ck["centre"]), _sha256(path)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--templates", type=int, nargs="*", default=list(range(12)))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--n-groups", type=int, default=16)
    ap.add_argument("--n-direct", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if os.environ.get("SO_BOS") != "1":
        raise SystemExit("run with SO_BOS=1: the substrate is the BOS-trained adapter")

    per_seed: List[Dict[str, Any]] = []
    for s in args.seeds:
        gk, centre, sha = load_bos_adapter(s)
        m: Dict[str, Any] = {"seed": s, "checkpoint_sha256": sha}
        t0 = time.time()
        if not args.skip_a:
            m.update(part_a(gk, centre, s, 6 if args.smoke else args.n_groups))
        if not args.skip_b:
            m.update(part_b(gk, centre, s, args.templates,
                            20 if args.smoke else None, 20 if args.smoke else args.n_direct))
        m["seconds"] = time.time() - t0
        per_seed.append(m)
    if args.smoke:
        print("SMOKE ONLY -- nothing recorded")
        return {"per_seed": per_seed, "smoke": True}
    raise SystemExit("WSC-001 is not scored until docs/novelty/wsc001-preregister.md is committed; "
                     "this line is then replaced by the scoring block.")


if __name__ == "__main__":
    main()
