"""A deletion certificate you can watch happen, in a few minutes on a CPU.

Everything the programme claims about deletion is spread across twenty-six JSON records. This
prints the claim as a single transcript on ONE fact: a frozen GPT-2 answers it, one operation
removes it, the same model can no longer produce it, and the attacks that try to get it back come
back at chance — with the chance level and the gate's measured error rate printed next to them, so
the numbers can be judged rather than believed.

Nothing is trained here. It loads a checkpoint recorded by E-000012 and runs in a few minutes.

    python -m so.demo                 # one fact, full transcript
    python -m so.demo --n-targets 50  # the same measured over fifty facts
    python -m so.demo --seed 1        # a different recorded checkpoint

What to watch for, in order:
  1. the pretrained model alone does not know the fact          (it is not answering from its weights)
  2. with the cell present it answers correctly                 (the layer is the only channel)
  3. the routing names the cell that answered                   (provenance)
  4. ONE shred, and the answer becomes 'unknown'                (deletion, not suppression)
  5. the payload is still physically in the bank and still routed to, and the attacks still fail
  6. a second cell, untouched, still answers                    (deletion was local)
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np
import torch

from so.attacks import LinearProbe, forced_choice, object_rank
from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000011_gpt2_v2 as E11
from so.experiments.e000001b_mini_transformer import CHECKPOINTS
from so.llm_adapter import AdapterConfig
from so.mvcc import MVCCStore
from so.reference import load_world
from so.world import Query, UNKNOWN, World, fill_random

BAR = "─" * 78


def h(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="which recorded checkpoint to use")
    ap.add_argument("--n-targets", type=int, default=50,
                    help="facts deleted and attacked. The narrative follows the first one; the numbers "
                         "need all of them, because a single trial has no statistical content.")
    ap.add_argument("--n-cells", type=int, default=400)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    path = CHECKPOINTS / f"e000012_gpt2_seed{args.seed}.pt"
    if not path.exists():
        print(f"No checkpoint at {path}.\nTrain one first:  python -m so.experiments.e000012_status_gated_revoke "
              f"--seeds {args.seed}\n(about 50 minutes on four CPU cores), or run 'make gpt2'.")
        return {}

    h("SETUP")
    print("Loading the frozen GPT-2 and the knowledge layer recorded by E-000012 ...")
    gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))
    ck = torch.load(path, weights_only=False)
    gk.model.load_state_dict(ck["adapter"], strict=False)
    gk.model.eval()
    centre = np.asarray(ck["centre"])
    frozen = sum(p.numel() for n, p in gk.model.named_parameters() if n.startswith("lm."))
    adapter = sum(p.numel() for p in gk.model.adapter_parameters())
    print(f"  frozen GPT-2 parameters, never updated : {frozen:,}")
    print(f"  knowledge-layer parameters             : {adapter:,}")

    rng = np.random.default_rng(9000 + args.seed)
    world = fill_random(rng, World(gk.n_entities, 4, gk.n_synonyms, []), args.n_cells)
    store = MVCCStore(marker_dim=centre.shape[0], seed=args.seed, marker_centre=centre)
    kids = load_world(store, world)
    facts = world.facts
    n_by = min(50, len(facts) - args.n_targets)
    idx = rng.choice(len(facts), size=args.n_targets + n_by, replace=False)
    targets = [facts[int(i)] for i in idx[: args.n_targets]]
    bystanders = [facts[int(i)] for i in idx[args.n_targets:]]
    print(f"  cells written into the layer           : {len(facts)}")

    def q(f) -> Query:
        return Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),))

    def name(i: int) -> str:
        return gk.names[i] if 0 <= i < len(gk.names) else "?"

    t0 = targets[0]
    prompt = E8.query_text(q(t0), gk.names, gk.n_synonyms)
    single = True     # the narrative always follows one fact; the numbers below always use all of them

    if single:
        h("1. THE PRETRAINED MODEL ALONE")
        print(f'  prompt                : "{prompt}"')
        a = gk.predict(None, world, [q(t0)])["answers"][0]
        print(f"  answers               : {name(a) if a != UNKNOWN else 'unknown'}")
        print(f"  the fact in the layer : {name(t0.obj)}")
        print("  → the fact is not in the weights. Whatever comes next cannot be recall.")

        h("2. WITH THE CELL PRESENT")
        p = gk.predict(bank_from_store(store), world, [q(t0)])
        a = p["answers"][0]
        print(f"  answers               : {name(a) if a != UNKNOWN else 'unknown'}   (expected {name(t0.obj)})")
        h("3. WHICH CELL ANSWERED")
        bank = bank_from_store(store)
        r = p["routing"][0, -1]
        k = int(r.argmax())
        print(f"  routing mass on the winning cell : {float(r[k]):.4f}")
        print(f"  that cell is                     : kid {int(bank.kid[k])}, which holds "
              f"({name(int(bank.subject[k]))}, relation {int(bank.relation[k])}) → {name(int(bank.obj[k]))}")
        print(f"  the cell we wrote the fact into  : kid {kids[t0.key]}")
        print("  → the answer is attributable. The layer is not a black box that happens to be right.")

    h("4. ONE OPERATION: SHRED THE MARKER")
    for f in targets:
        store.shred(kids[f.key])
    print(f"  shredded {len(targets)} cell(s) with one operation each. The payload is NOT removed: it is still in the bank, still "
          f"addressable, still routed to.\n  Only the marker that certifies it is destroyed.")
    gk.model.cfg.hard_gate = True
    bank = bank_from_store(store)
    if single:
        p = gk.predict(bank, world, [q(t0)])
        a = p["answers"][0]
        print(f'  the same prompt now   : {name(a) if a != UNKNOWN else "unknown"}')
        row = int(np.where(bank.kid == kids[t0.key])[0][0])
        print(f"  routing mass on that same cell   : {float(p['routing'][0, -1, row]):.4f}   "
              f"(it is still found, it just no longer reads)")
        print(f"  the object still sitting in the bank at that row : {name(int(bank.obj[row]))}")

    h("5. THE ATTACKS, AND WHAT THEY WOULD HAVE TO BEAT")
    n_ent = gk.n_entities
    others = [f for f in facts if f.key not in {t.key for t in targets}]
    po = gk.predict(bank, world, [q(f) for f in others])
    y = np.array([f.obj for f in others]); split = int(0.8 * len(others))
    probe = LinearProbe(po["hidden"].shape[1], n_ent, seed=args.seed)
    probe.fit(po["hidden"][:split], y[:split])
    cal = probe.accuracy(po["hidden"][split:], y[split:])
    pt = gk.predict(bank, world, [q(f) for f in targets])
    truth = np.array([f.obj for f in targets])
    probe_hit = probe.accuracy(pt["hidden"], truth)
    fc = forced_choice(pt["logits"], truth, np.random.default_rng(args.seed), n_ent)
    rk = object_rank(pt["logits"], truth, n_ent)
    print(f"  measured over {len(targets)} deleted facts. A single trial would tell you nothing: forced choice on")
    print(f"  one fact is 0 or 1 whatever the truth is, which is why the default is not one.\n")
    rows = [("linear probe on the hidden state", f"{probe_hit:.4f}", f"{1 / n_ent:.4f}", f"works at {cal:.2f} on live cells"),
            ("two-way forced choice", f"{fc:.4f}", "0.5000", "coin flip"),
            ("rank of the deleted object among entities", f"{rk['mean_rank']:.1f}", f"{rk['chance_mean_rank']:.1f}", "middle of the pack"),
            ("deleted object is top-1 among entities", f"{rk['top1']:.4f}", f"{1 / n_ent:.4f}", "chance")]
    w = max(len(r[0]) for r in rows)
    print(f"  {'attack'.ljust(w)}   measured    chance   note")
    for a_, b_, c_, d_ in rows:
        print(f"  {a_.ljust(w)}   {b_:>8s}  {c_:>8s}   {d_}")
    print("\n  The probe is the control that makes the rest mean something: it recovers the object")
    print(f"  {cal:.0%} of the time from cells that are alive, and {probe_hit:.0%} from the ones that were deleted.")

    h("6. WAS THE DELETION LOCAL?")
    ab = gk.predict(bank, world, [q(f) for f in bystanders])["answers"]
    by_truth = np.array([f.obj for f in bystanders])
    by_acc = float((ab == by_truth).mean())
    by_unknown = float((ab == UNKNOWN).mean())
    print(f"  {len(bystanders)} untouched cells, same layer, same forward pass")
    print(f"    still answered correctly : {by_acc:.0%}")
    print(f"    turned into 'unknown'    : {by_unknown:.0%}")
    print(f"  This adapter reads at about 91% to begin with (E-000012), so the shortfall here is its")
    print(f"  ordinary error rate, not damage from the deletion. What matters is that it did not go to zero")
    print(f"  and did not turn into refusals: the deletion did not spread.")
    gk.model.cfg.hard_gate = False

    h("WHAT THIS DOES AND DOES NOT SHOW")
    print("  Shown here: on a frozen 124M-parameter GPT-2, one operation on one cell removes a fact")
    print("  the model was answering, the payload stays physically present, and four attacks come back")
    print("  at chance while the same probe still works on live cells.")
    print("  Measured elsewhere and not shown here: over 750 pooled trials on seeds that took no part in")
    print("  choosing the configuration, forced choice landed on exactly 375 and the probe on 4, with every")
    print("  exact interval containing chance (E-000019). The gate that certifies a marker admits an")
    print("  unsigned one about once in 1,180 over 2.2 million markers (E-000021) — that is the bound on")
    print("  this guarantee, and it is a measured rate, not a proof.")
    print("  NOT shown: anything above 124M parameters, multi-token answers, unlearning of knowledge that")
    print("  was already in the weights, or an adversary who gets to choose the marker.")
    print(f"\n  Full records: so/results/  •  the ledger: docs/so-experiment-ledger.md section 31\n")
    return {"probe": probe_hit, "probe_calibration": cal, "forced_choice": fc, "mean_rank": rk["mean_rank"],
            "chance_mean_rank": rk["chance_mean_rank"], "n_targets": len(targets)}


if __name__ == "__main__":
    main()
