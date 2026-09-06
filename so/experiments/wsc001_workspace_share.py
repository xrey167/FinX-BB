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

WHAT THIS FILE ASKS, AND THE ANSWER IT FOUND. Why is the workspace readout blind, and is anything
else? FOUR explanations are distinguishable, and the first was added after a measurement, not from the
armchair -- the first draft of this file listed only the last three:

    SITING     the audit reads a state the memory has not reached yet
    DEPTH      the write reaches that state but is not linearly decodable there
    DIRECTION  k coordinates suffice there, but not those k
    DIMENSION  k coordinates are too few, whatever they are

Separating SITING from the rest needs a MEDIATOR, and that is what the first draft lacked: at every
candidate site, how far does the state move when the pod is shredded, and when it was never written?
A readout that sees nothing where the state does not move is not blind, it is correctly sited and
pointed at the wrong block. Measured on three seeds: the injected write at the FIRST read site moves
2.81 when a pod is shredded and 89.70 at the second, a ratio of 0.032, and E-000063 captures the
first. The answer is SITING.

PART A, the readout side. Probes trained on ACTIVE states ONLY and applied unchanged to SHRED and
NEVER (E-000063's transfer probe), over five matched feature families -- the J-lens atoms of the pod
objects (the audit), an equally sized random projection (dimension-matched), the top-k principal
components of the ACTIVE states (the best k-dimensional linear readout there is), the objects'
unembedding rows (the vocabulary-basis analogue), and the raw state (the capacity ceiling) -- at every
candidate site (blocks 8, 9, 10 and the final state), under two ADDRESS MODES: the pod's own canonical
key, and a LINK alias the model must dereference. Each site carries its own mediator.

PART B, the causal side -- AND ITS RESULT IS VOID, WHICH IS RECORDED HERE SO THE FILE IS NOT READ AS
IF IT STOOD. A probe says what is readable; it does not say what the model uses. The write is a tensor
this harness holds, so it can be injected with the entire first-order channel removed: projecting it
out of the span of the J-lens atoms of ALL 257 scored tokens leaves a write whose first-order effect
on every scored logit is exactly zero (measured retention 1e-6 at full rank). On the recorded run the
TOKEN-MATCHED NULL fired -- a span built identically over 257 tokens that are NOT scored costs the
answer up to 0.26 against a 0.20 bar -- so the registered sentence is not licensed and part B is void.
Its numbers are kept as a table with no sentence attached. See docs/novelty/audit-siting-claim.md.

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


# Block outputs, plus "final" (the last hidden state, after ln_f). Block 11 is not a site: its output
# IS hidden_states[12], and the lens's target hidden_states[-2] is upstream of it, so the Jacobian is
# not defined there (the vector-Jacobian product raises "not used in the graph"). At the final state
# the lens degenerates to the unembedding row by construction -- ledger 31.56 measures cos = 1.000 at
# layer 11 -- so the family used there is W_U itself, which is the same object and is labelled as such.
SITES: Tuple[int, ...] = (8, 9, 10)


class MultiCapture:
    """The output of every named block at the last token, plus the final state.

    E-000063 captures ONE block, chosen as "the first adapter read site" (`CAPTURE_BLOCK = 8`). That
    choice is the reason its audit could not see the pod: measured on the BOS-trained adapter, the
    write at the first read site is BIT-IDENTICAL between a live pod and a shredded one (0.0000 over
    eight pods), while the write at the second read site differs by 68.84 and the block-10 and block-11
    states differ by 68.84 and 288.87. The first read site injects nothing pod-specific for these
    prompts; the content enters at the second. An audit sited at block 8 is therefore upstream of the
    write it audits, and this class exists so the siting question is measured rather than assumed.
    """

    def __init__(self, gk: E8.GPT2Knowledge, sites: Sequence[int] = SITES):
        self.sites = tuple(sites)
        self.buf: Dict[int, torch.Tensor] = {}
        blocks = transformer_blocks(gk.model.lm)
        self.handles = [blocks[l].register_forward_hook(self._mk(l)) for l in self.sites]

    def _mk(self, l: int):
        def hook(module, inputs, output):
            self.buf[l] = (output[0] if isinstance(output, tuple) else output).detach()
            return None
        return hook

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def eval_states(gk: E8.GPT2Knowledge, cap: MultiCapture, store, texts: Sequence[str], batch: int = 64):
    """Per-site states at the last token, the final state, the answers, and the injected writes."""
    tensors = bank_from_store(store).tensors()
    per: Dict[str, List[torch.Tensor]] = {}
    answers, injected = [], []
    for i in range(0, len(texts), batch):
        ids, am, last = E8.encode_texts(gk.tok, texts[i:i + batch])
        with torch.no_grad():
            cand, _, _, final = gk.model(tensors, ids, am, last)
        ar = torch.arange(ids.shape[0])
        for l in cap.sites:
            per.setdefault(f"site{l}", []).append(cap.buf[l][ar, last].cpu())
        per.setdefault("final", []).append(final.detach().cpu())
        answers.append(cand.argmax(-1).cpu())
        if gk.model.last_injected is not None:
            injected.append(gk.model.last_injected.cpu())
    return ({k: torch.cat(v) for k, v in per.items()}, torch.cat(answers),
            torch.cat(injected) if injected else None)


def part_a(gk: E8.GPT2Knowledge, centre: np.ndarray, seed: int, n_groups: int,
           verbose: bool = True) -> Dict[str, float]:
    """Where can an accessibility audit of this memory be read, and where does the memory enter?

    Two address modes (the pod's own canonical key; a LINK alias the model must dereference), five
    feature families, and every candidate site from the first read layer to the final state. Each site
    carries its own MEDIATOR -- how far the state moves when the pod is shredded, and when it was never
    written -- so "the audit sees nothing here" can be separated from "nothing is here".
    """
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
    # One lens family per site, each at the index whose state it is used to read.
    jl = {f"site{l}": jlens_vectors(gk.model.lm, l + 1, obj_ids, enc["input_ids"], enc["attention_mask"],
                                    w_out, batch=4) for l in SITES}
    class _Identity:                      # the lens at the final state: J = I, so v_u = W_U[u]
        vectors = F.normalize(w_out[torch.as_tensor(obj_ids)].float(), dim=-1)
    jl["final"] = _Identity()
    cap = MultiCapture(gk)
    acc: Dict[Tuple[str, str, str], List[torch.Tensor]] = {}
    ys: Dict[str, List[int]] = {"alias": [], "direct": []}
    gs: Dict[str, List[int]] = {"alias": [], "direct": []}
    ans: Dict[str, List[torch.Tensor]] = {"alias": [], "direct": []}
    truths: Dict[str, List[int]] = {"alias": [], "direct": []}
    med: Dict[Tuple[str, str, str], List[float]] = {}
    for label, (target, aliases, obj) in enumerate(selected):
        never = copy.deepcopy(store)
        for ak in aliases:
            if kids[ak] in never.cells:
                never.delete(kids[ak])
        if kids[target] in never.cells:
            never.delete(kids[target])
        for mode, keys in (("alias", aliases), ("direct", [target])):
            texts, tg = E63.texts_for(keys, gk.names, E63.TEMPLATES)
            sa, aa, inj_a = eval_states(gk, cap, store, texts)
            sn, _, _ = eval_states(gk, cap, never, texts)
            store.shred(kids[target])
            ss, _, inj_s = eval_states(gk, cap, store, texts)
            store.resign(kids[target])
            for site in sa:
                acc.setdefault((mode, site, "active"), []).append(sa[site])
                acc.setdefault((mode, site, "shred"), []).append(ss[site])
                acc.setdefault((mode, site, "never"), []).append(sn[site])
                med.setdefault((mode, site, "shred_moves"), []).append(
                    float((sa[site] - ss[site]).abs().max()))
                med.setdefault((mode, site, "never_moves"), []).append(
                    float((sa[site] - sn[site]).abs().max()))
            if inj_a is not None and inj_s is not None:
                for j, l in enumerate(gk.model.cfg.read_layers):
                    med.setdefault((mode, f"write{l}", "shred_moves"), []).append(
                        float((inj_a[:, j] - inj_s[:, j]).abs().max()))
            ys[mode].extend([label] * len(texts)); gs[mode].extend(tg)
            ans[mode].append(aa); truths[mode].extend([obj] * len(texts))
    cap.close()
    out: Dict[str, float] = {"partA/n_pods": float(n), "partA/chance": 1.0 / n}
    for (mode, site, what), v in med.items():
        out[f"partA/{mode}/{site}/{what}"] = float(np.mean(v))
    for mode in ("alias", "direct"):
        y = torch.tensor(ys[mode], dtype=torch.long); g = torch.tensor(gs[mode], dtype=torch.long)
        truth = torch.tensor(truths[mode], dtype=torch.long)
        out[f"partA/{mode}/answer_correct"] = float((torch.cat(ans[mode]) == truth).float().mean())
        for site in list(f"site{l}" for l in SITES) + ["final"]:
            states = {st: torch.cat(acc[(mode, site, st)]) for st in ("active", "shred", "never")}
            fams = probe_families(states, jl[site].vectors, w_out[torch.as_tensor(obj_ids)], n, seed)
            for fam, feats in fams.items():
                p = E63.transfer_probe(feats["active"], feats["shred"], feats["never"], y, g, n)
                for s_, v_ in p.items():
                    out[f"partA/{mode}/{site}/{fam}/{s_}"] = v_
                out[f"partA/{mode}/{site}/{fam}/active_minus_never"] = p["active"] - p["never"]
                out[f"partA/{mode}/{site}/{fam}/shred_minus_never"] = p["shred"] - p["never"]
            if verbose:
                print(f"  seed {seed} partA {mode:6s} {site:6s} "
                      f"moves(shred) {out.get(f'partA/{mode}/{site}/shred_moves', float('nan')):8.3f}  "
                      + "  ".join(f"{f}={out[f'partA/{mode}/{site}/{f}/active_minus_never']:+.3f}"
                                  for f in ("jspace", "pca", "raw")), flush=True)
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


# ------------------------------------------------------------------------------- scoring
SUMMARY_KEYS = ("B/full_alias_min", "B/full_direct_min", "B/dropW_alias_max", "B/dropW_n_alias_max",
                "B/dropT_n_deficit_min", "B/perm_alias_max", "B/none_alias_max",
                "B/n_templates_dropWn_ge_50", "B/share_W_mean", "B/atom_dropW_alias_max",
                "A/write8_shred_moves", "A/write10_shred_moves", "A/write_site_ratio",
                "A/state_site_ratio",
                "A/site8_max_amn", "A/site10_max_amn", "A/final_raw_amn",
                "A/alias/site8/shred_moves", "A/alias/site10/shred_moves", "A/alias/final/shred_moves",
                "A/alias/final/jspace", "A/direct/final/jspace", "A/indirection/jspace",
                "A/alias/final/raw", "A/direct/final/raw", "A/indirection/raw",
                "A/shred_residual_direct", "A/shred_residual_alias", "A/shred_residual_asymmetry")


def summarise(m: Dict[str, Any], templates: Sequence[int]) -> Dict[str, float]:
    """Per-seed reductions the pre-registered bars are read on. Worst direction per row."""
    g = lambda k, d=float("nan"): float(m.get(k, d))
    out: Dict[str, float] = {}
    ts = [t for t in templates if f"partB/t{t}/full/alias" in m]
    if ts:
        out["B/full_alias_min"] = min(g(f"partB/t{t}/full/alias") for t in ts)
        out["B/full_direct_min"] = min(g(f"partB/t{t}/full/direct") for t in ts)
        out["B/dropW_alias_max"] = max(g(f"partB/t{t}/dropW/alias") for t in ts)
        out["B/dropW_n_alias_max"] = max(g(f"partB/t{t}/dropW_n/alias") for t in ts)
        out["B/atom_dropW_alias_max"] = max(g(f"partB/t{t}/atom_dropW/alias") for t in ts)
        out["B/dropT_n_deficit_min"] = min(g(f"partB/t{t}/dropT_n/alias") - g(f"partB/t{t}/full/alias")
                                           for t in ts)
        out["B/perm_alias_max"] = max(g(f"partB/t{t}/perm/alias") for t in ts)
        out["B/none_alias_max"] = max(g(f"partB/t{t}/none/alias") for t in ts)
        out["B/n_templates_dropWn_ge_50"] = float(sum(g(f"partB/t{t}/dropW_n/alias") >= 0.50 for t in ts))
        out["B/share_W_mean"] = float(np.mean([g(f"partB/t{t}/share_W_site0") for t in ts]))
    fams = ("jspace", "random", "pca", "unembed", "raw")
    sites = [f"site{l}" for l in SITES] + ["final"]
    if any(f"partA/alias/{sites[0]}/{f}/active_minus_never" in m for f in fams):
        for mode in ("alias", "direct"):
            for site in sites:
                for f in fams:
                    out[f"A/{mode}/{site}/{f}"] = g(f"partA/{mode}/{site}/{f}/active_minus_never")
                out[f"A/{mode}/{site}/max_amn"] = max(out[f"A/{mode}/{site}/{f}"] for f in fams)
                out[f"A/{mode}/{site}/shred_moves"] = g(f"partA/{mode}/{site}/shred_moves")
                out[f"A/{mode}/{site}/raw_shred_minus_never"] = g(
                    f"partA/{mode}/{site}/raw/shred_minus_never")
            out[f"A/{mode}/answer_correct"] = g(f"partA/{mode}/answer_correct")
        # The write mediator: how far the INJECTED vector moves when the pod is shredded, per read
        # site, and the RATIO between the two sites. The ratio is what the siting branch reads: the
        # first site's write is not exactly zero on every prompt, it is small, and "small relative to
        # the site that carries the content" is the honest form of the statement.
        for l in (8, 10):
            out[f"A/write{l}_shred_moves"] = max(g(f"partA/alias/write{l}/shred_moves", 0.0),
                                                 g(f"partA/direct/write{l}/shred_moves", 0.0))
        denom = max(out.get("A/write10_shred_moves", 0.0), 1e-9)
        out["A/write_site_ratio"] = out.get("A/write8_shred_moves", 0.0) / denom
        out["A/state_site_ratio"] = max(
            g("partA/alias/site8/shred_moves", 0.0), g("partA/direct/site8/shred_moves", 0.0)) / max(
            max(g("partA/alias/site10/shred_moves", 0.0), g("partA/direct/site10/shred_moves", 0.0)), 1e-9)
        # A6 indirection, at the site where the audit is valid (the final state).
        for f in fams:
            out[f"A/indirection/{f}"] = out[f"A/direct/final/{f}"] - out[f"A/alias/final/{f}"]
        # A7, the disclosed post-hoc row.
        out["A/shred_residual_direct"] = out["A/direct/final/raw_shred_minus_never"]
        out["A/shred_residual_alias"] = out["A/alias/final/raw_shred_minus_never"]
        out["A/shred_residual_asymmetry"] = (out["A/shred_residual_direct"]
                                             - out["A/shred_residual_alias"])
        out["A/final_raw_amn"] = min(out["A/alias/final/raw"], out["A/direct/final/raw"])
        out["A/site8_max_amn"] = max(out["A/alias/site8/max_amn"], out["A/direct/site8/max_amn"])
        out["A/site10_max_amn"] = max(out["A/alias/site10/max_amn"], out["A/direct/site10/max_amn"])
    return out


def criteria() -> Dict[str, Tuple[str, float]]:
    """Exactly the bars of docs/novelty/wsc001-preregister.md, read on the worst seed."""
    return {
        "B/full_alias_min": (">=", 0.80), "B/full_direct_min": (">=", 0.90),      # V1
        "B/perm_alias_max": ("<=", 0.05),                                          # V2
        "B/none_alias_max": ("<=", 0.05),                                          # V3
        "A/final_raw_amn": (">=", 0.30),                                           # V4
        "A/write8_shred_moves": (">=", 0.0),          # reported: the first read site's write mediator
        "A/site10_max_amn": (">=", 0.0),              # reported: the second read site's readability
        "B/dropW_n_alias_max": ("<=", 0.20),                                       # B1a
        "B/dropT_n_deficit_min": (">=", -0.20),                                    # B1b
    }


def decide(agg: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """Every branch of the pre-registered decision rule, evaluated on the worst seed."""
    w = lambda k, lower_is_better: ledger.worst(agg[k], lower_is_better) if k in agg else float("nan")
    v = {"V1": w("B/full_alias_min", False) >= 0.80 and w("B/full_direct_min", False) >= 0.90,
         "V2": w("B/perm_alias_max", True) <= 0.05,
         "V3": w("B/none_alias_max", True) <= 0.05,
         "V4": w("A/final_raw_amn", False) >= 0.30,
         "V5": all(agg[k]["max"] <= 1e-4 for k in agg if k.endswith("drop_retention_max"))}
    out = {"validity": "PASS" if all(v.values()) else "FAIL: " + ",".join(k for k, ok in v.items() if not ok)}
    if v["V1"] and v["V2"] and v["V3"] and v["V5"]:
        null_fired = w("B/dropT_n_deficit_min", False) < -0.20
        b1 = (w("B/dropW_n_alias_max", True) <= 0.20) and not null_fired
        b2 = w("B/n_templates_dropWn_ge_50", True) >= 3
        out["partB"] = ("VOID-NULL (B3): the token-matched span costs the answer too, so the result is "
                        "about removing a rank-257 lens span of this construction" if null_fired else
                        "B1 CONFIRMED: the scored-token atom span is the causal channel of the write "
                        "and the token-matched null does not reproduce it" if b1 else
                        "B2: the answer survives with the first-order channel removed" if b2 else
                        "B4 mixed: neither branch holds across templates")
    else:
        out["partB"] = "VOID (validity)"
    if v["V4"]:
        site8 = w("A/site8_max_amn", True)
        ratio = w("A/write_site_ratio", True)
        out["partA"] = ("A0 SITING: E-000063's capture block is not where the pod's content enters -- "
                        "the write at the first read site moves by a small fraction of the second's "
                        "when the pod is shredded -- so no readout there can attribute the memory, and "
                        "the certificate audits a state the pod has effectively not reached"
                        if (site8 <= 0.05 and ratio <= 0.05) else
                        "A1 DEPTH: the write reaches the first read site but no linear readout there "
                        "attributes it, at any dimension up to the full residual" if site8 <= 0.05 else
                        "A4 mixed: no siting sentence is licensed")
    else:
        out["partA"] = "VOID (V4: no readout anywhere sees the memory)"
    if v["V4"]:
        ind = w("A/indirection/jspace", True)
        out["A6_indirection"] = ("closed: an audit calibrated on the canonical key transfers to the "
                                 "pointer path" if ind <= 0.10 else
                                 "open: the audit loses sensitivity through the pointer" if ind >= 0.20
                                 else "inconclusive between 0.10 and 0.20")
        d, a = w("A/shred_residual_direct", False), w("A/shred_residual_alias", True)
        out["A7_shred_residual"] = ("asymmetric (DISCLOSED POST-HOC ROW): a shredded pod stays "
                                    "decodable from the final state when addressed by its own key and "
                                    "not through a pointer" if (d >= 0.20 and d - a >= 0.20)
                                    else "no asymmetry recorded; table only")
    return out


def record(per_seed: List[Dict[str, Any]], args) -> Dict[str, Any]:
    keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], (int, float))
            and k not in ("seed",)]
    agg = ledger.aggregate(per_seed, keys)
    check = ledger.check_criteria(agg, criteria())
    verdict = decide(agg)
    rows = [(k, f"{agg[k]['mean']:.4f}", f"{ledger.worst(agg[k], k in LOWER):.4f}")
            for k in SUMMARY_KEYS if k in agg]
    ts = [t for t in args.templates if f"partB/t{t}/full/alias" in per_seed[0]]
    # 'full' and the null are capability rows (worst = min); every other arm is a leak row (worst = max)
    arm_rows = [[f"t{t}"] + [f"{ledger.worst(agg[f'partB/t{t}/{a}/alias'], a not in ('full', 'dropT_n')):.4f}"
                             for a in PART_B_ARMS] for t in ts]
    site_names = [f"site{l}" for l in SITES] + ["final"]
    fam_rows = [[f] + [f"{ledger.worst(agg[f'partA/{mode}/{site}/{f}/active_minus_never'], False):.4f}"
                       for mode in ("direct", "alias") for site in site_names]
                for f in ("jspace", "random", "pca", "unembed", "raw")
                if f"partA/alias/{site_names[0]}/{f}/active_minus_never" in agg]
    med_rows = [[site] + [f"{ledger.worst(agg[f'partA/{mode}/{site}/{w_}'], False):.4f}"
                          for mode in ("direct", "alias") for w_ in ("shred_moves", "never_moves")]
                for site in site_names if f"partA/alias/{site}/shred_moves" in agg]
    rec = {
        "experiment": "WSC-001",
        "title": "Where an accessibility audit of an external memory must read",
        "evidence_level": "E5",
        "preregistration": "docs/novelty/wsc001-preregister.md",
        "claim_supported": check["claim_supported"],
        "verdict": verdict,
        "not_claimed": ("the Jacobian lens, J-lens auditing, probing, projection, external memory, pods or "
                        "pointer aliases; no deletion guarantee; the atom arm is an ORACLE and is never a "
                        "capability comparison"),
        "by_construction": ["keep/drop of the first-order term (ledger 31.39)",
                            "atom/alias = 1.0 (the arm is handed the ground-truth object)",
                            "at read layer 10 the atoms are the unembedding rows (31.56)"],
        "config": {"seeds": args.seeds, "templates": args.templates, "n_groups": args.n_groups,
                   "n_direct": args.n_direct, "lens_corpus": LENS_CORPUS, "arms": list(PART_B_ARMS)},
        "criteria": check["criteria"], "per_seed": per_seed, "aggregate": agg,
    }
    md = "\n".join([
        "# WSC-001 — where an accessibility audit of an external memory must read", "",
        f"Pre-registered: `docs/novelty/wsc001-preregister.md`. Seeds {args.seeds}; worst seed reported.",
        f"Validity: **{verdict['validity']}**.  Part A: **{verdict.get('partA','-')}**.  "
        f"Part B: **{verdict.get('partB','-')}**.", "",
        "## Part B — the causal side (alias reads, worst seed)", "",
        ledger.table(["template"] + list(PART_B_ARMS), arm_rows), "",
        "## Part A — the readout side (probe trained on ACTIVE only, worst seed)", "",
        ledger.table(["family"] + [f"{m} {s_}" for m in ("direct", "alias") for s_ in site_names],
                     fam_rows), "",
        "### The mediator: how far the state moves at each site when the pod is shredded, and when it "
        "was never written", "",
        ledger.table(["site", "direct SHRED", "direct NEVER", "alias SHRED", "alias NEVER"], med_rows), "",
        "## Summary rows", "", ledger.table(["measure", "mean over seeds", "worst seed"], rows), "",
        "## Pre-registered criteria (worst seed)", "", ledger.criteria_table(check), "",
        "By construction: " + "; ".join(rec["by_construction"]) + ".", "",
        "Not claimed: " + rec["not_claimed"],
    ])
    path = ledger.save("wsc001_workspace_share", rec, md)
    print(md); print(f"\nsaved {path}")
    return rec


LOWER = {"B/dropW_alias_max", "B/dropW_n_alias_max", "B/perm_alias_max", "B/none_alias_max",
         "B/atom_dropW_alias_max", "A/mid_max_active_minus_never", "A/mid_jspace_amn",
         "A/mid_jspace_shred_minus_never", "A/final_raw_shred_minus_never"}


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
    for m in per_seed:
        m.update(summarise(m, args.templates))
    return record(per_seed, args)


if __name__ == "__main__":
    main()
