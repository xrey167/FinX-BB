"""Experiment E-000039 — tie the ADDRESS, not the carrier.

THE CLAIM.  A fact's erasure cost in a store-backed model has two independent factors: how many
RECORDS hold it, and how many ACCESS PATHS address it.  ``so/closure.py`` measures the first and
proves that canonicalisation collapses it to one.  This experiment says the second factor lives in
one tensor and names it: the ROUTING QUERY

    q_l(x) = q_proj[l]( q_ln[l]( h_l[last] ) )          so/llm_adapter.py :: KnowledgeAdapterLM._make_hook

and that making ``q_l`` invariant across phrasings of the same question is what makes one record
deletion propagate to phrasings nobody trained on.

WHY THAT TENSOR AND NOT ANOTHER.  In this adapter the injected value is ``values[cell]``, built in
``encode_bank`` from the bank alone -- the SAME vector whatever the phrasing.  ``o_proj`` and
``inject_gain`` are shared across phrasings, and everything downstream of the injection is the frozen
core.  With ``match_gate`` and ``two_channel_null`` off (they are off in E-000017's configuration,
``AdapterConfig(status_gated=True)``), ``q_l`` is the ONLY tensor in the read path that depends on
the wording.  A loss that tied the injected carrier would therefore be tying something that is
already constant, and would prove nothing.

WHY q AND NOT THE ROUTING DISTRIBUTION p.  ``p = softmax(q K^T / sqrt(d_key))`` depends on the bank
drawn at that step: two phrasings can share a p on an easy bank and diverge on a hard one, so an
invariance trained on p is a property of (phrasing, bank) and does not constrain the 1000-cell
evaluation bank.  ``q`` is bank-independent -- a deterministic function of the text through the
frozen core and ``q_proj`` -- so tying q makes p identical for EVERY bank at once.  E-000017's
existing ``--consistency`` arm ties p (``routing[:k, -1]``); it has never been run (the only recorded
E-000017-B has ``consistency: 0.0``), and it draws its alternative template uniformly from the eight
trained ones, so one pair in eight is the template paired with itself and contributes exactly zero.

THE STRATIFICATION, AND WHERE IT COMES FROM.  ``subject_initial_templates()`` below splits the twelve
surface forms by the GPT-2 token index of the subject name -- a property of the tokenizer, computed
here, never of the recorded accuracies.  It returns {0, 2, 6, 8, 11} for all four relations.  Pairs
cross that split.  The prediction is specific: the held-out subject-initial forms recover, because the
invariance trained spans the axis they differ on.

WHAT THE DECIDE PHASE ALREADY FOUND, AND WHY IT SETS THE BAR RATHER THAN CLEARING IT.  Run on
E-000017-B's three recorded checkpoints (seed 0 at 64 targets, seeds 1 and 2 at 32):

  * reading never exceeds ADDRESSING at any of the twelve templates, and both split on the same axis:
    seed 0, t8 read 0.578 / route hit 0.625, t11 0.438 / 0.563, t9 0.969 / 0.969, t10 1.000 / 1.000.
  * over the five subject-initial forms and three seeds, natural reading is 0.6562 (worst cell 0.3125)
    and natural addressing 0.7927 (worst 0.5312).
  * prepending ``PREFIX`` -- no weight changed -- gives addressing 640/640 (95% CI 0.9943-1.0000) and
    reading 639/640.  On the two HELD-OUT subject-initial forms it moves reading 0.4661 -> 1.0000 and
    addressing 0.6302 -> 1.0000, worst cell in both cases 1.0000.
  * the tie is not vacuous: the mean cosine of the routing query between two phrasings of the same
    fact is 0.7868 at read layer 10 (0.7745 at layer 8) against 0.2723 between different facts.

So the gap is addressing, its cause is positional, and a prompt-side patch already closes it.  The
training run is therefore not "can this be fixed" but "can the invariance be made intrinsic": the arm
has to reach the prefixed numbers WITHOUT the prefix, on phrasings it never saw.  If it cannot, the
honest recommendation is to normalise the prompt in the read path, note the scope on the deletion
certificate, and not train at all.

Run:  python -m so.experiments.e000039_address_tying --phase decide  [--seeds 0 1 2]
      python -m so.experiments.e000039_address_tying --phase train --arm address [--seeds 0 1 2]
      python -m so.experiments.e000039_address_tying --phase train --arm read    [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from so import ledger
from so.data import bank_from_store, bank_from_world, sample_training_queries
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, guard_recorded_checkpoint, _sha256
from so.experiments.e000012_status_gated_revoke import route_targets_status_gated
from so.experiments.e000017_paraphrase_gap import EVAL, N_HELDOUT, N_TRAIN_DEFAULT, TEMPLATES12, query_text_pc
from so.llm_adapter import AdapterConfig
from so.mvcc import MVCCStore
from so.reference import load_world
from so.train import TrainConfig, lr_at, make_centre, routing_loss
from so.world import Query, UNKNOWN, World, fill_random

N_TRAIN = N_TRAIN_DEFAULT
N_T = N_TRAIN + N_HELDOUT
TAU = 0.1
# The positional control. Prepending ANY context moves the subject off token 0 without changing a
# weight; if reading recovers, the failure was in the address and not in the carrier. The string is
# borrowed from template 10 so it is not a phrasing invented for this test.
PREFIX = "It is known that "


# ------------------------------------------------------------------ the axis the pairs must cross
def subject_initial_templates(tok, relation: int = 0, name: str = "Anna") -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Split the templates by the token index of the subject name: (initial, medial).

    This is read off the tokenizer, not off any recorded accuracy, so using it to build training
    pairs is not tuning on the measurement it is supposed to move.
    """
    initial, medial = [], []
    for t in range(N_T):
        text = TEMPLATES12[relation][t].format(s=name)
        toks = tok.convert_ids_to_tokens(tok(text)["input_ids"])
        idx = next(i for i, s in enumerate(toks) if name in s)
        (initial if idx == 0 else medial).append(t)
    return tuple(initial), tuple(medial)


# ---------------------------------------------------------------------------------- the objective
def infonce(a: torch.Tensor, b: torch.Tensor, same: Optional[torch.Tensor] = None,
            tau: float = TAU) -> torch.Tensor:
    """Symmetric InfoNCE between two aligned renderings of the same facts.

    ``a`` and ``b`` are (B, d) or (B, R, d); row i of both is the same (subject, relation).  ``same``
    is a (B, B) bool matrix marking pairs that are the SAME fact off the diagonal -- they are masked
    out rather than used as negatives.

    Collapse is the WORST point of this loss, not its minimum: if every query were the same vector
    the similarity matrix would be constant and the cross-entropy would sit at its maximum, log B.
    A plain ||a - b||^2 or a cosine pull has its global minimum exactly there, which is why neither
    is used.
    """
    if a.dim() == 2:
        a, b = a[:, None], b[:, None]
    ua, ub = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    B, R = ua.shape[0], ua.shape[1]
    tgt = torch.arange(B, device=ua.device)
    total = ua.sum() * 0
    for r in range(R):
        s = (ua[:, r] @ ub[:, r].t()) / tau
        if same is not None:
            off = same & ~torch.eye(B, dtype=torch.bool, device=s.device)
            s = s.masked_fill(off, float("-inf"))
        total = total + 0.5 * (F.cross_entropy(s, tgt) + F.cross_entropy(s.t(), tgt))
    return total / R


def tie_tensor(model, arm: str) -> torch.Tensor:
    """The tensor the arm ties, taken from the forward that has just run.

    ``address``: ``model.last_query`` -- (B, len(read_layers), d_key), the routing query at each read
    site, recorded by ``_make_hook`` with its graph intact.
    ``read``: the post-injection readout state at the answer position, which is the ``hidden`` the
    forward returns; the caller passes it in.  Both arms use the SAME objective, the same pairs, the
    same schedule and the same weight, so the only difference between them is which tensor is tied.
    """
    if arm == "address":
        q = model.last_query
        if q is None:
            raise RuntimeError("model.last_query is None: so/llm_adapter.py must record the routing query")
        return q
    raise ValueError(arm)


# ------------------------------------------------------------------------------------ the trainer
def train_arm(gk: E8.GPT2Knowledge, seed: int, steps: int, arm: str, tie_weight: float = 0.5,
              n_pair: int = 8, batch_size: int = 32, route_weight: float = 1.0, gate_weight: float = 5.0,
              lr: float = 2e-3, route_only_steps: int = 300, p_revoked: float = 0.20, p_shred: float = 0.10,
              extra_unanswerable: float = 0.2, verbose: bool = True) -> Dict[str, Any]:
    """E-000017-B's trainer, unchanged, plus one tying term on ``n_pair`` stratified pairs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    centre = make_centre(seed, gk.model.cfg.marker_dim)
    model = gk.model
    params = model.adapter_parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    tcfg = TrainConfig(seed=seed, n_steps=steps, lr=lr, warmup=50)
    mix = {"fwd1": 0.7, "fwd2": 0.3}
    initial, medial = subject_initial_templates(gk.tok)
    tr_initial = [t for t in initial if t < N_TRAIN]
    tr_medial = [t for t in medial if t < N_TRAIN]
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    n_extra = int(round(batch_size * extra_unanswerable))
    n_reads = len(model.cfg.read_layers)
    for step in range(steps):
        model.train()
        route_only = step < route_only_steps
        n_cells = int(rng.integers(150, 301)) if route_only else int(rng.integers(700, 1001))
        world = World.sample(rng, gk.n_entities, 4, n_cells, N_TRAIN)
        bank = bank_from_world(rng, world, centre, p_revoked, p_shred, 0.05)
        queries = sample_training_queries(rng, world, bank, batch_size - n_extra, mix)
        queries += world.sample_queries(rng, n_extra, 1, "fwd", require_answer=False, index=bank.index_view)
        ids, am, last = E8.encode_texts(gk.tok, [query_text_pc(q, gk.names, N_TRAIN) for q in queries])
        target = E8.targets_of(queries, bank, world)
        route = route_targets_status_gated(queries, bank, world, n_reads)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg)
        tensors = bank.tensors()
        cand, _, routing, _ = model(tensors, ids, am, last)
        loss_ans = F.cross_entropy(cand, target)
        loss_route = routing_loss(routing, route)
        valid = tensors["marker_valid"].float()
        per_cell = F.binary_cross_entropy_with_logits(model.gate_logits(tensors["marker"]).squeeze(-1), valid,
                                                      reduction="none")
        n_pos, n_neg = valid.sum().clamp_min(1), (1 - valid).sum().clamp_min(1)
        loss_gate = 0.5 * (per_cell * valid).sum() / n_pos + 0.5 * (per_cell * (1 - valid)).sum() / n_neg

        loss_tie = cand.sum() * 0
        if tie_weight > 0 and not route_only and n_pair > 0:
            # facts whose cell is present and usable; the pair is one subject-INITIAL rendering and
            # one subject-MEDIAL rendering of the same question, both from the eight trained forms
            pool = [q for q in queries if q.hops == 1 and (q.start, q.path[0]) in bank.kid_of_key]
            if len(pool) >= 2:
                sub = pool[:n_pair]
                ta = [tr_initial[int(rng.integers(0, len(tr_initial)))] for _ in sub]
                tb = [tr_medial[int(rng.integers(0, len(tr_medial)))] for _ in sub]
                texts = ([query_text_pc(q, gk.names, N_TRAIN, t) for q, t in zip(sub, ta)]
                         + [query_text_pc(q, gk.names, N_TRAIN, t) for q, t in zip(sub, tb)])
                ids2, am2, last2 = E8.encode_texts(gk.tok, texts)
                cand2, _, _, hidden2 = model(tensors, ids2, am2, last2)   # one forward, both renderings
                k = len(sub)
                keys = [(q.start, q.path[0]) for q in sub]
                same = torch.as_tensor([[keys[i] == keys[j] for j in range(k)] for i in range(k)])
                if arm == "address":
                    t_all = tie_tensor(model, "address")                  # (2k, R, d_key)
                    loss_tie = infonce(t_all[:k], t_all[k:], same)
                elif arm == "read":
                    loss_tie = infonce(hidden2[:k], hidden2[k:], same)    # the readout state, same objective
                else:
                    raise ValueError(arm)

        loss = (loss_route + gate_weight * loss_gate) if route_only else \
            (loss_ans + route_weight * loss_route + gate_weight * loss_gate + tie_weight * loss_tie)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            acc = (cand.argmax(-1) == target).float().mean().item()
            rec = {"step": step + 1, "loss": float(loss.item()), "answer_loss": float(loss_ans.item()),
                   "route_loss": float(loss_route.item()), "gate_loss": float(loss_gate.item()),
                   "tie_loss": float(loss_tie.item()), "batch_acc": acc, "elapsed_s": time.time() - t0}
            history.append(rec)
            if verbose:
                print(f"  step {rec['step']:5d}  loss {rec['loss']:.4f}  ans {rec['answer_loss']:.4f}  "
                      f"route {rec['route_loss']:.4f}  tie {rec['tie_loss']:.4f}  acc {acc:.3f}  "
                      f"{rec['elapsed_s']:.0f}s", flush=True)
    model.eval()
    return {"centre": centre, "history": history, "train_seconds": time.time() - t0}


# ---------------------------------------------------------------- the measurement that decides
@torch.no_grad()
def decompose(gk: E8.GPT2Knowledge, seed: int, centre: np.ndarray, n_targets: int = 100,
               oracle: bool = True) -> Dict[str, Any]:
    """Split reading, per template, into ADDRESSING and TRANSPORT.

      read(t)   top-1 = object, everything addressable      (what E-000017-B reports)
      hit(t)    argmax of the LAST read slot's routing = the cell that holds the fact
      orc(t)    top-1 = object when the ONLY addressable cell is that one (``cell_mask``)

    ``orc`` is an intervention, not a conditional: conditioning reading on ``hit`` would condition on
    a post-treatment variable and could not separate the two factors.
    """
    rng = np.random.default_rng(seed)
    world = fill_random(rng, World(gk.n_entities, 4, N_TRAIN, []), EVAL["n_cells"])
    store = MVCCStore(marker_dim=centre.shape[0], seed=seed, marker_centre=centre)
    load_world(store, world)
    bank = bank_from_store(store)
    tensors = bank.tensors()
    idx = rng.choice(len(world.facts), size=min(n_targets, len(world.facts)), replace=False)
    targets = [world.facts[int(i)] for i in idx]
    truth = np.array([f.obj for f in targets])
    cstar = np.array([bank.kid_of_key[f.key] for f in targets])
    gk.model.eval()
    m: Dict[str, Any] = {"seed": seed, "n_targets": len(targets)}

    read = np.zeros((N_T, len(targets)), dtype=np.int64)
    hit = np.zeros((N_T, len(targets)), dtype=bool)
    cell = np.zeros((N_T, len(targets)), dtype=np.int64)
    null0 = np.zeros((N_T, len(targets)), dtype=bool)
    qs = []
    for t in range(N_T):
        texts = [TEMPLATES12[f.relation][t].format(s=gk.names[f.subject]) for f in targets]
        qt = []
        for i in range(0, len(texts), 32):
            ids, am, last = E8.encode_texts(gk.tok, texts[i:i + 32])
            cand, _, routing, _ = gk.model(tensors, ids, am, last)
            a = cand.argmax(-1).numpy()
            sl = slice(i, i + ids.shape[0])
            read[t, sl] = np.where(a == gk.n_entities, UNKNOWN, a)
            cell[t, sl] = routing[:, -1].numpy().argmax(-1)
            hit[t, sl] = cell[t, sl] == cstar[sl]
            # The EARLIER read site is supervised to the null column for a 1-hop question
            # (``route_targets_status_gated`` sets route[i, 0] = -1). If it mis-routes it injects a
            # wrong entity that competes with the correct one injected at the later site, which is how
            # a query can hit the right cell at read layer 10 and still answer wrongly.
            null0[t, sl] = routing[:, 0].numpy().argmax(-1) == routing.shape[-1] - 1
            qt.append(gk.model.last_query.numpy())
        qs.append(np.concatenate(qt))
    # Is the address already invariant?  Mean cosine of the routing query between two phrasings of the
    # SAME fact, against between different facts.  If the first is already ~1 the tie is vacuous.
    Q = np.stack(qs)                                                   # (T, N, R, d_key)
    U = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-9)
    for r in range(U.shape[2]):
        M = U[:, :, r]
        within = float(np.mean([(M[i] * M[j]).sum(-1).mean() for i in range(N_T) for j in range(i + 1, N_T)]))
        G = M.reshape(-1, M.shape[-1]) @ M.reshape(-1, M.shape[-1]).T
        lab = np.repeat(np.arange(len(targets))[None], N_T, 0).reshape(-1)
        m[f"query_cos_within_fact/read{r}"] = within
        m[f"query_cos_between_fact/read{r}"] = float(G[lab[:, None] != lab[None, :]].mean())
    # Do different facts still get different addresses?  Distinct cells addressed, over the held-out forms.
    m["address_collision"] = float(np.mean([1.0 - len(set(cell[t].tolist())) / len(targets)
                                            for t in range(N_TRAIN, N_T)]))

    # The mask leaves the NULL column addressable (``_make_hook`` always appends it), so a phrasing
    # that still routes to null answers ' unknown' rather than the object: ``oracle_read`` is a lower
    # bound on transport and ``oracle_unknown`` says how much of the shortfall is that choice.
    orc = np.zeros((N_T, len(targets)), dtype=np.int64)
    for j, f in enumerate(targets if oracle else []):    # one mask per fact; all twelve forms in one batch
        mask = torch.zeros(bank.size, dtype=torch.bool)
        mask[cstar[j]] = True
        ids, am, last = E8.encode_texts(gk.tok, [TEMPLATES12[f.relation][t].format(s=gk.names[f.subject])
                                                 for t in range(N_T)])
        cand, _, _, _ = gk.model(tensors, ids, am, last, cell_mask=mask)
        a = cand.argmax(-1).numpy()
        orc[:, j] = np.where(a == gk.n_entities, UNKNOWN, a)

    r = {t: float((read[t] == truth).mean()) for t in range(N_T)}
    o = {t: (float((orc[t] == truth).mean()) if oracle else float("nan")) for t in range(N_T)}
    for t in range(N_T):
        kind = "train" if t < N_TRAIN else "heldout"
        m[f"t{t}/{kind}/read"] = r[t]
        m[f"t{t}/{kind}/route_hit"] = float(hit[t].mean())
        m[f"t{t}/{kind}/first_read_null"] = float(null0[t].mean())
        m[f"t{t}/{kind}/oracle_read"] = o[t]
        m[f"t{t}/{kind}/oracle_unknown"] = float((orc[t] == UNKNOWN).mean())
    # the positional intervention: same weights, same fact, subject no longer token 0
    initial, _ = subject_initial_templates(gk.tok)
    for t in initial:
        texts = [PREFIX + TEMPLATES12[f.relation][t].format(s=gk.names[f.subject]) for f in targets]
        pa, ph = [], []
        for i in range(0, len(texts), 32):
            ids, am, last = E8.encode_texts(gk.tok, texts[i:i + 32])
            cand, _, routing, _ = gk.model(tensors, ids, am, last)
            a = cand.argmax(-1).numpy()
            pa.append(np.where(a == gk.n_entities, UNKNOWN, a))
            ph.append(routing[:, -1].numpy().argmax(-1) == cstar[i:i + ids.shape[0]])
        m[f"t{t}/prefixed_read"] = float((np.concatenate(pa) == truth).mean())
        m[f"t{t}/prefixed_route_hit"] = float(np.concatenate(ph).mean())
    m["prefixed/read_min"] = min(m[f"t{t}/prefixed_read"] for t in initial)
    m["prefixed/route_hit_min"] = min(m[f"t{t}/prefixed_route_hit"] for t in initial)

    best = max(range(N_T), key=lambda t: r[t])
    gaps, resid = [], []
    for t in range(N_TRAIN, N_T):
        gaps.append(max(r[best] - r[t], 0.0))
        resid.append(max(o[best] - o[t], 0.0))
    g, s = float(sum(gaps)), float(sum(resid))
    m["best_template"] = best
    m["heldout/gap"] = g
    m["heldout/residual_gap"] = s
    # the deciding number: the share of the held-out reading gap that forcing the address closes
    m["heldout/routing_share"] = (float(np.clip(1.0 - s / g, 0.0, 1.0)) if (oracle and g > 1e-9)
                                  else float("nan"))
    return m


# -------------------------------------------------------------------------- pre-registered bars
def criteria_groups() -> Dict[str, Dict[str, Tuple[str, float]]]:
    """Fixed before either arm is trained.

    ``reading_generalises`` is E-000017-B's own bar (observed there: 0.7288 worst seed).
    ``deletion_propagates`` is the claim: one SHRED of one record refuses on the WORST held-out
    phrasing (observed there: 0.8650).
    The two collapse groups are what stops a closure of one being bought by destroying the model.
    """
    return {
        # the ceiling is not 1.0 in the abstract: E-000039-A measures what the SAME weights do when a
        # neutral prefix moves the subject off token 0, and the arm must reach that number WITHOUT the
        # prefix. Anything less and the honest recommendation is to normalise the prompt instead.
        "reading_generalises": {"heldout/active_correct": (">=", 0.95), "train/active_correct": (">=", 0.95)},
        "addressing_generalises": {"heldout/route_hit_min": (">=", 0.95)},
        "deletion_propagates": {"shred_heldout_min": (">=", 0.95), "revoke_heldout_min": (">=", 0.95),
                                "heldout/revoked_deleted_object": ("<=", 0.02)},
        "addresses_do_not_collapse": {"query_cos_between_fact/read1": ("<=", 0.33),
                                      "address_collision": ("<=", 0.02)},
        "no_new_collateral": {"broken1_unknown": (">=", 0.63), "generic/kl_to_base": ("<=", 3.65)},
    }


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["decide", "train"], required=True)
    ap.add_argument("--arm", choices=["address", "read"], default="address")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-targets", type=int, default=100)
    ap.add_argument("--no-oracle", action="store_true",
                    help="skip the cell_mask arm: the prefix intervention is the cheap decisive one "
                         "(~5 min/seed against ~13)")
    ap.add_argument("--tie-weight", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.phase == "decide":
        per_seed = []
        for seed in args.seeds:
            gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))
            path = CHECKPOINTS / f"e000017_t8_c0{CKPT_SUFFIX}_seed{seed}.pt"
            ck = torch.load(path, weights_only=False)
            gk.model.load_state_dict(ck["adapter"], strict=False)
            gk.model.eval()
            print(f"=== seed {seed}: decomposing E-000017-B's checkpoint ===", flush=True)
            m = decompose(gk, 1700 + seed, np.asarray(ck["centre"]), args.n_targets, not args.no_oracle)
            m["checkpoint_sha256"] = _sha256(path)
            per_seed.append(m)
            print({k: v for k, v in m.items() if "heldout/" in k or k == "best_template"}, flush=True)
        keys = [k for k in per_seed[0] if k not in ("seed", "checkpoint_sha256", "best_template")]
        agg = ledger.aggregate(per_seed, keys)
        record = {
            "experiment": "E-000039-A", "title": "Addressing versus transport in the paraphrase gap",
            "evidence_level": "E5", "deletion_level": None,
            "no_training": "E-000017-B's three checkpoints are evaluated as recorded; nothing is trained.",
            "decision_rule": "routing_share >= 0.7 -> train the address arm alone; <= 0.3 -> train the read "
                             "arm alone; in between -> train both. Fixed before either arm was trained.",
            "per_seed": per_seed, "aggregate": agg,
        }
        rows = [(k, f"{agg[k]['mean']:.4f}", f"{agg[k]['min']:.4f}") for k in
                ["heldout/gap", "heldout/residual_gap", "heldout/routing_share"] if k in agg]
        md = "\n".join([f"# E-000039-A — {record['title']}", "", record["no_training"], "",
                        record["decision_rule"], "",
                        ledger.table(["measure", "mean over seeds", "worst seed"], rows)])
        print(md); print(f"\nsaved {ledger.save('e000039a_address_decision', record, md)}")
        return record

    per_seed = []
    for seed in args.seeds:
        gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))
        path = CHECKPOINTS / f"e000039_{args.arm}{CKPT_SUFFIX}_seed{seed}.pt"
        print(f"=== seed {seed}: arm {args.arm}, tie weight {args.tie_weight:g} ===", flush=True)
        if path.exists() and not args.force:
            ck = torch.load(path, weights_only=False)
            gk.model.load_state_dict(ck["adapter"], strict=False)
            gk.model.eval()
            out = {"centre": ck["centre"], "history": ck["history"], "train_seconds": ck["train_seconds"]}
        else:
            out = train_arm(gk, seed, args.steps, args.arm, args.tie_weight)
            CHECKPOINTS.mkdir(parents=True, exist_ok=True)
            guard_recorded_checkpoint(path)
            torch.save({"adapter": E8.adapter_state(gk.model), "centre": out["centre"],
                        "history": out["history"], "train_seconds": out["train_seconds"],
                        "adapter_config": gk.model.cfg.to_dict(), "arm": args.arm,
                        "tie_weight": args.tie_weight}, path)
        out["checkpoint_sha256"] = _sha256(path)
        from so.experiments.e000017_paraphrase_gap import evaluate_templates
        m = evaluate_templates(gk, 1700 + seed, np.asarray(out["centre"]), N_TRAIN)
        d = decompose(gk, 1700 + seed, np.asarray(out["centre"]), args.n_targets, not args.no_oracle)
        m["heldout/route_hit_min"] = min(d[f"t{t}/heldout/route_hit"] for t in range(N_TRAIN, N_T))
        m["heldout/routing_share"] = d["heldout/routing_share"]
        # the two collapse criteria are computed by decompose() and were never copied into the
        # record, so the first run reported them as '-' (FAIL by absence). Copied now; the run that
        # recorded them is an evaluation-only re-run from the saved checkpoints (ledger §31.37).
        for k, v in d.items():
            if (k.startswith("query_cos_between_fact/") or k == "address_collision"
                    or "/heldout/" in k or k.startswith("prefixed/") or k.endswith("_route_hit")
                    or k.endswith("_read")) and k not in m and isinstance(v, (int, float, bool)):
                # the per-template held-out addressing and the prefix ceiling: the prediction was
                # about the subject-initial held-out forms (t8, t11), so their numbers must be on record
                m[k] = float(v)
        m["train_seconds"] = out["train_seconds"]; m["checkpoint_sha256"] = out["checkpoint_sha256"]
        per_seed.append(m)
    keys = [k for k in per_seed[0] if k not in ("seed", "checkpoint_sha256")]
    agg = ledger.aggregate(per_seed, keys)
    groups = criteria_groups()
    check = ledger.check_criteria(agg, {k: v for g in groups.values() for k, v in g.items()})
    met = {g: all(check["criteria"][k]["pass"] for k in ks if k in check["criteria"]) for g, ks in groups.items()}
    record = {"experiment": f"E-000039-B-{args.arm}", "evidence_level": "E5",
              "title": f"Tying the {'routing query' if args.arm == 'address' else 'readout state'} "
                       f"across phrasings of the same fact",
              "claim_groups_met": met, "criteria": check["criteria"],
              "claim_supported": check["claim_supported"], "per_seed": per_seed, "aggregate": agg,
              "control": "E-000017-B (e000017b_templates8) — same trainer, same budget, tie weight 0."}
    md = "\n".join([f"# {record['experiment']} — {record['title']}", "",
                    ledger.table(["claim group", "supported"], [(g, "yes" if v else "**no**") for g, v in met.items()]),
                    "", ledger.criteria_table(check)])
    print(md); print(f"\nsaved {ledger.save(f'e000039b_{args.arm}', record, md)}")
    return record


if __name__ == "__main__":
    main()
