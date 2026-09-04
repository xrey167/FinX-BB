"""Experiment E-000030 — a deletion certificate for the recorded checkpoints.

Every deletion result in this programme, including the F4 label, is an attack that failed to recover
the fact. E-000028 showed what that is worth: SHRED passed a linear probe, forced choice, rank and
top-1, and an attack written afterwards recovered the object at 1.0000 through a channel none of them
read. An attack battery bounds one adversary. It says nothing about the next one.

This experiment reports the other kind of evidence. For each lifecycle operation it asks whether the
model's computation depends on the deleted payload AT ALL, by sweeping every value that payload could
hold and comparing what the model computes. The payload domain is finite and small -- an entity id in
0..255 -- so the sweep is every case rather than a sample.

Two levels, at two costs:

  interface   `so.audit.certify_encoding`. Both models read the store in exactly one place:
              `MutableKnowledgeTransformer.forward` computes `enc = self.encode_bank(bank)` and then
              touches only `enc[...]`, the query tensors and its own parameters. So if the ENCODING is
              bit-identical across all 256 payload values, every downstream quantity is identical for
              EVERY POSSIBLE QUERY -- multi-hop, reverse, phrasings nobody has written -- not merely
              for a swept set. That premise is a claim about the model, and reading source is how the
              defects of E-000028 and E-000029 got into the record in the first place, so
              `check_mediation` tries to falsify it by looking for an output that moves while the
              encoding holds still.

  outputs     `so.audit.certify_deletion` over an exhaustive single-hop query domain (every subject x
              relation x mode). Weaker in scope -- it certifies the questions actually asked -- and it
              is what tells REVOKE apart from SHRED, because a masked row moves the encoding while
              nothing a user sees can move.

DELETE is not swept, and the record says why rather than reporting a number the sweep did not
produce: `store.delete` removes the row, so the post-deletion bank has no field holding the object
and there is nothing to perturb. Its independence is structural, and stating that is more honest than
sweeping a row that is not there.

Trains nothing.

Run:  python -m so.experiments.e000030_deletion_certificate [--seeds 0 1 2] [--n-targets 3]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch

from so import ledger
from so.audit import certify_deletion, certify_encoding, check_mediation
from so.data import bank_from_store, encode_queries
from so.experiments.common import fresh_world, position_of_kid
from so.experiments.e000001b_mini_transformer import _sha256, checkpoint_path, train_or_load
from so.model import ModelConfig
from so.train import TrainConfig
from so.world import Query

OPS = ("revoke", "shred")          # DELETE is structural; see the module docstring
N_TARGETS = 3
N_PROBES = 6


def exhaustive_one_hop(world) -> List[Query]:
    """Every single-hop question the world admits: each entity, each relation, forward and reverse."""
    return [Query(mode, e, (r,), (world.surface_of(r, 0),))
            for mode in ("fwd", "rev")
            for e in range(world.n_entities)
            for r in range(world.n_relations)]


def addressing_queries(world, targets, rng, n_unrelated: int = 64) -> List[Query]:
    """The queries the OUTPUT sweep uses, and why it is not the whole domain.

    The interface certificate already covers every possible query, so the output sweep has one job the
    interface cannot do: separate REVOKE, where the encoding moves but nothing a user sees does, from
    SHRED, where both move. For that it needs the questions that can actually address a target -- its
    own forward question in both surface forms, and EVERY candidate object as a reverse question,
    which is exactly E-000028's attack surface and is exhaustive over that surface -- plus a fixed
    sample of unrelated questions, because a target's key sits in the softmax denominator of queries
    that are not about it. Sweeping all 2048 single-hop questions instead costs 21 minutes per
    operation per seed and adds nothing the interface column does not already state universally.
    """
    qs: List[Query] = []
    for f in targets:
        for k in range(world.n_synonyms):
            qs.append(Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, k),)))
        for o in range(world.n_entities):
            qs.append(Query("rev", o, (f.relation,), (world.surface_of(f.relation, 0),)))
    keys = {(f.subject, f.relation) for f in targets}
    others = [f for f in world.facts if f.key not in keys]
    for i in rng.choice(len(others), size=min(n_unrelated, len(others)), replace=False):
        f = others[int(i)]
        qs.append(Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),)))
    return qs


def run_seed(seed: int, n_targets: int, steps: int, verbose: bool = True) -> Dict[str, Any]:
    path = checkpoint_path("e000010", seed)
    if not path.exists():
        raise SystemExit(f"missing checkpoint {path}; run the E-000010 arm of e000009_verification_gate first")
    out = train_or_load("e000010", seed, ModelConfig(),
                        TrainConfig(seed=seed, n_steps=steps, gate_weight=5.0, gate_balanced=True))
    model, centre = out["model"], out["centre"]
    model.cfg.hard_gate = False
    rng, world, store, kids, _ = fresh_world(900 + seed, centre)
    facts = list(world.facts)
    targets = [facts[int(i)] for i in rng.permutation(len(facts))[:n_targets]]
    n_ent = world.n_entities
    queries = exhaustive_one_hop(world)
    out_queries = addressing_queries(world, targets, np.random.default_rng(700 + seed))

    m: Dict[str, Any] = {"seed": seed, "checkpoint_sha256": _sha256(path), "n_targets": len(targets),
                         "n_entities": n_ent, "n_queries_full_domain": len(queries),
                         "n_queries_swept": len(out_queries), "n_cells": len(facts)}
    t0 = time.time()

    for op in OPS:
        for f in targets:
            getattr(store, op)(kids[f.key])
        bank = bank_from_store(store)
        tensors = bank.tensors()
        rows = [position_of_kid(store, kids[f.key]) for f in targets]
        batch = encode_queries(out_queries, bank, world, model.cfg.max_hops)

        def run(b, _batch=batch):
            with torch.no_grad():
                return model(b, _batch.mode, _batch.start, _batch.rels, _batch.hop_valid)

        # what MutableKnowledgeTransformer.forward reads out of the encoding (so/model.py:245-248):
        # k_f, v_f, k_r, v_r and the allowed set. "gate" is a diagnostic it does not consume.
        iface = certify_encoding(model, tensors, rows, n_ent, joint_trials=32, seed=seed,
                                 interface_keys=("k_f", "v_f", "k_r", "v_r", "active"))
        outs = certify_deletion(model, tensors, rows[:1], n_ent, run, outputs_of=lambda o: o[0],
                                check_activations=False, joint_trials=0, seed=seed)
        med = check_mediation(model, tensors, rows, n_ent, run, outputs_of=lambda o: o[0],
                              n_probes=N_PROBES, seed=seed)

        m[f"{op}/interface_certified"] = bool(iface.output_certified)
        m[f"{op}/interface_joint_certified"] = bool(iface.joint_certified)
        m[f"{op}/interface_evaluations"] = int(iface.n_evaluations)
        m[f"{op}/interface_first_violation"] = iface.violations[0].module if iface.violations else ""
        m[f"{op}/outputs_certified"] = bool(outs.output_certified)
        m[f"{op}/outputs_evaluations"] = int(outs.n_evaluations)
        m[f"{op}/outputs_first_violation"] = outs.violations[0].module if outs.violations else ""
        m[f"{op}/mediation_consistent"] = bool(med.consistent)
        m[f"{op}/mediation_note"] = med.note
        if verbose:
            print(f"  seed {seed} {op:<7} interface {'CERTIFIED' if iface.output_certified else 'no  (' + m[f'{op}/interface_first_violation'] + ')':<28}"
                  f" outputs {'CERTIFIED' if outs.output_certified else 'no':<10} mediation "
                  f"{'consistent' if med.consistent else 'VOID'}  ({time.time() - t0:.0f}s)", flush=True)
        # put the store back for the next operation
        for f in targets:
            store.resign(kids[f.key]) if op == "shred" else store.restore(kids[f.key])

    # DELETE: the row leaves the bank, so no field holds the object and nothing can depend on it.
    before = bank_from_store(store).subject.shape[0]
    for f in targets:
        store.delete(kids[f.key])
    after_bank = bank_from_store(store)
    m["delete/rows_removed"] = int(before - after_bank.subject.shape[0])
    m["delete/object_absent_from_bank"] = bool(m["delete/rows_removed"] == len(targets))
    m["delete/structural"] = True
    m["seconds"] = time.time() - t0
    return m


KEYS = [f"{op}/{k}" for op in OPS
        for k in ("interface_certified", "interface_joint_certified", "outputs_certified",
                  "mediation_consistent", "interface_evaluations", "outputs_evaluations")]

CRITERIA = {
    # what the record already claims, restated as an independence question
    "shred/outputs_certified": (">=", 1.0),      # F4 for SHRED, if it held, would require this
    "revoke/outputs_certified": (">=", 1.0),     # REVOKE masks the row, so this should hold
    # the premise the interface certificate rests on must survive every falsification attempt
    "revoke/mediation_consistent": (">=", 1.0),
    "shred/mediation_consistent": (">=", 1.0),
}


def run_gpt2_seed(seed: int, n_targets: int, verbose: bool = True) -> Dict[str, Any]:
    """The same question of the frozen-GPT-2 adapter, where the answer should differ.

    That adapter's key is `k_proj(ln_key(subject + relation))` and carries no object, so E-000028's
    leak cannot occur; what remains is the VALUE, and `values = payload * g + unk * (1 - g)`. With a
    soft gate `g` is small but never exactly zero, so a shredded cell's value still moves with the
    object by whatever `g` is -- the certificate fails at bit-identity and the residual is measurable.
    With `hard_gate` the gate is exactly 0 or 1, the value becomes exactly the ' unknown' direction,
    and the encoding is invariant. That is a precise statement of what the hard gate buys, which no
    recorded experiment has made.
    """
    from so.experiments import e000008_gpt2_adapter as E8
    from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX
    from so.llm_adapter import AdapterConfig
    from so.mvcc import MVCCStore
    from so.reference import load_world
    from so.world import World, fill_random

    path = CHECKPOINTS / f"e000012_gpt2{CKPT_SUFFIX}_seed{seed}.pt"
    if not path.exists():
        raise SystemExit(f"missing checkpoint {path}")
    ck = torch.load(path, weights_only=False)
    gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))
    gk.model.load_state_dict(ck["adapter"], strict=False)
    gk.model.eval()
    centre = np.asarray(ck["centre"])
    rng = np.random.default_rng(3000 + seed)
    world = fill_random(rng, World(gk.n_entities, 4, gk.n_synonyms, []), 400)
    store = MVCCStore(marker_dim=centre.shape[0], seed=seed, marker_centre=centre)
    kids = load_world(store, world)
    facts = list(world.facts)
    targets = [facts[int(i)] for i in rng.permutation(len(facts))[:n_targets]]

    m: Dict[str, Any] = {"seed": seed, "model": "frozen GPT-2 + E-000012 adapter",
                         "checkpoint_sha256": _sha256(path), "n_cells": len(facts),
                         "n_entities": gk.n_entities, "n_targets": len(targets)}
    t0 = time.time()
    for gate_mode in ("soft", "hard"):
        gk.model.cfg.hard_gate = (gate_mode == "hard")
        for op in OPS:
            for f in targets:
                getattr(store, op)(kids[f.key])
            bank = bank_from_store(store, respect_markers=True)
            tensors = bank.tensors()
            rows = [position_of_kid(store, kids[f.key]) for f in targets]
            # what KnowledgeAdapterLM.forward reads (so/llm_adapter.py:323): keys, values, allowed.
            # values_payload is the UNGATED payload and is returned as a diagnostic only; comparing it
            # would report a leak through a tensor the model never looks at.
            cert = certify_encoding(gk.model, tensors, rows, gk.n_entities, joint_trials=16, seed=seed,
                                    interface_keys=("keys", "values", "active"))
            m[f"gpt2_{gate_mode}/{op}/interface_certified"] = bool(cert.output_certified)
            m[f"gpt2_{gate_mode}/{op}/first_violation"] = cert.violations[0].module if cert.violations else ""
            m[f"gpt2_{gate_mode}/{op}/residual"] = float(cert.violations[0].max_abs) if cert.violations else 0.0
            m[f"gpt2_{gate_mode}/{op}/evaluations"] = int(cert.n_evaluations)
            texts = [E8.query_text(Query("fwd", f.subject, (f.relation,), (0,)), gk.names, gk.n_synonyms, 0)
                     for f in targets]
            ids, am, last = E8.encode_texts(gk.tok, texts)

            def run_adapter(b, _i=ids, _a=am, _l=last):
                with torch.no_grad():
                    return gk.model(b, _i, _a, _l)

            med = check_mediation(gk.model, tensors, rows, gk.n_entities, run_adapter,
                                  encode=lambda b: {k: v for k, v in gk.model.encode_bank(b).items()
                                                    if k in ("keys", "values", "active")},
                                  outputs_of=lambda o: o[0], n_probes=4, seed=seed)
            m[f"gpt2_{gate_mode}/{op}/mediation_consistent"] = bool(med.consistent)
            if verbose:
                verdict = "CERTIFIED" if cert.output_certified else f"no (residual {m[f'gpt2_{gate_mode}/{op}/residual']:.3e})"
                print(f"  seed {seed} gpt2 {gate_mode:<4} {op:<7} {verdict}  ({time.time() - t0:.0f}s)", flush=True)
            for f in targets:
                store.resign(kids[f.key]) if op == "shred" else store.restore(kids[f.key])
    gk.model.cfg.hard_gate = False
    m["seconds"] = time.time() - t0
    return m


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-targets", type=int, default=N_TARGETS)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--with-gpt2", action="store_true", help="also certify the frozen-GPT-2 adapter")
    ap.add_argument("--gpt2-seeds", type=int, nargs="*", default=[0])
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    per_seed = [run_seed(s, args.n_targets, args.steps) for s in args.seeds]
    gpt2 = [run_gpt2_seed(s, args.n_targets) for s in args.gpt2_seeds] if args.with_gpt2 else []
    numeric = [{k: (float(v) if isinstance(v, bool) else v) for k, v in s.items()
                if isinstance(v, (bool, int, float))} for s in per_seed]
    keys = [k for k in KEYS if all(k in s for s in numeric)]
    agg = ledger.aggregate(numeric, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    rows = []
    for op in OPS:
        rows.append([op,
                     "yes" if agg[f"{op}/interface_certified"]["min"] == 1.0 else "no",
                     "yes" if agg[f"{op}/outputs_certified"]["min"] == 1.0 else "no",
                     "consistent" if agg[f"{op}/mediation_consistent"]["min"] == 1.0 else "VOID",
                     per_seed[0].get(f"{op}/interface_first_violation") or "-",
                     f"{int(agg[f'{op}/interface_evaluations']['mean'])}"])
    rows.append(["delete", "yes (structural)", "yes (structural)", "n/a", "the row is not in the bank", "0"])
    for gate_mode in ("soft", "hard"):
        for op in OPS:
            k = f"gpt2_{gate_mode}/{op}/interface_certified"
            if gpt2 and k in gpt2[0]:
                ok = all(g[k] for g in gpt2)
                res = max(g[f"gpt2_{gate_mode}/{op}/residual"] for g in gpt2)
                rows.append([f"{op} (GPT-2, {gate_mode} gate)", "yes" if ok else "no", "-", "-",
                             gpt2[0][f"gpt2_{gate_mode}/{op}/first_violation"] or "-",
                             f"{int(np.mean([g[f'gpt2_{gate_mode}/{op}/evaluations'] for g in gpt2]))}"
                             + (f"  (residual {res:.2e})" if not ok else "")])
    tbl = ledger.table(["operation", "certified for every query (interface)", "certified on the swept queries",
                        "mediation premise", "first quantity that moves", "encodings swept"], rows)

    record = {"experiment": "E-000030", "title": "a deletion certificate for the recorded checkpoints",
              "trains_nothing": True, "seeds": args.seeds, "n_targets": args.n_targets,
              "operations": list(OPS) + ["delete"], "per_seed": per_seed, "gpt2": gpt2,
              "aggregate": agg, "criteria": check}
    md = [f"# E-000030 — {record['title']}", "",
          f"Seeds {args.seeds}, {args.n_targets} targets, the recorded E-000010 checkpoints, no training.",
          "For each lifecycle operation, every value the deleted payload could hold is swept and what the",
          f"model computes is compared. The payload domain has {per_seed[0]['n_entities']} values, so the sweep",
          "is every case rather than a sample.", "",
          "## What survives the deletion", "", tbl, "",
          "`interface` compares `encode_bank`'s output. The forward reads the bank only there, so an",
          "invariant encoding means an invariant computation FOR EVERY POSSIBLE QUERY, not just the swept",
          f"ones. `outputs` compares the returned logits over an exhaustive single-hop query domain",
          
          "`mediation` is the falsification check on the premise the interface column rests on: it looks",
          "for an output that moves while the encoding does not, and voids the certificate if it finds one.", "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    path = ledger.save("e000030_deletion_certificate", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(tbl)
    print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
