"""SIT-001 -- where an accessibility audit of this memory MAY be read, and what it says when it is.

THE QUESTION THIS ANSWERS. E-000063 returns the verdict *no workspace trace after deletion* from a
block the pod's content never reaches (ledger 31.57, WSC-001). That establishes the certificate is
vacuous; it does not say whether any siting would make it mean something. A site is admissible only if
BOTH halves of ``so.siting`` hold there -- ARRIVAL, the state moves when the pod is removed, and
DISTINCTNESS, the lens is not already the token's unembedding row -- and on the recorded adapter those
two are anti-aligned: the pod arrives at block 10, one block from the output, where the lens has cosine
1.000 to ``W_U``. The window is empty, so no siting of E-000063 on that adapter certifies anything.

BOTH HALVES OF THAT ARE ALREADY IN THE LEDGER (arrival in 31.57, the cosine curve in 31.56). This file
assembles them into one rule and MEASURES THEM TOGETHER ON THE SAME CHECKPOINTS, which had not been
done, but the recorded arm's empty window is reported as a consolidation and never as a finding.

WHAT IS ACTUALLY UNDER TEST is the second arm: the identical adapter with ``read_layers=(4, 6)``, so
the write lands four blocks earlier and there is depth left between it and the output. Bars,
predictions, the site-selection rule and the six decision branches are fixed in
``docs/novelty/sit001-preregister.md``, committed before any number here existed:

    VE1-VE3  capability: the early adapter must read a pod through its aliases (>= 0.80), delete it at
             the output (>= 0.90 unknown) and leave bystanders alone (>= 0.98). VE1 is how this arm
             most plausibly dies, and the window is worthless if it is bought by breaking the memory.
    P0       the siting rule replicates: the FIRST read site again fails ARRIVAL.
    P1       the window opens: some site passes both halves on all three seeds.
    P2       at the registered site -- the lowest-numbered block in the window on all three seeds --
             ``jprobe(ACTIVE - NEVER) >= 0.30``. This is E-000063's FAILED validity row.
    P3       given P2, ``jprobe(SHRED - NEVER) <= 0.05``: the deletion verdict, now non-vacuous.

P2 failing while P1 passes is a registered outcome, not a surprise: the window would then be necessary
and not sufficient, and the audit blind at an admissible site for one of WSC-001's other reasons.

EVERYTHING ELSE IS HELD FIXED so the two arms differ in one number. Same trainer, same 3000 steps, same
BOS, same seeds, same world sampler and pod selection as E-000063, same templates, same alias address
mode, same transfer probe (trained on ACTIVE only, applied unchanged to SHRED and NEVER), same
entity-free lens corpus as WSC-001.

Run:  SO_BOS=1 python -m so.experiments.sit001_audit_window --arms recorded early --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from so import ledger
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000063_workspace_pod_certificate as E63
from so.experiments import wsc001_workspace_share as WSC
from so.jlens import jlens_vectors
from so.llm_adapter import AdapterConfig
from so.siting import ARRIVAL_FLOOR, DEGENERACY_CEILING, audit_window, certify, lens_alignment, window_sites

CHECKPOINTS = Path("so/results/checkpoints")

# The two placements. "recorded" is the adapter every number in the claim document was measured on;
# "early" is the one design change under test. Nothing else differs.
ARMS: Dict[str, Dict[str, Any]] = {
    "recorded": {"read_layers": (8, 10), "suffix": "_bos"},
    "early":    {"read_layers": (4, 6),  "suffix": "_bos_early"},
}
# The last candidate block is 10. Block 11's output IS hidden_states[12], and the lens's target
# hidden_states[-2] is upstream of it, so the Jacobian is not defined there (WSC-001 hit exactly this).
LAST_BLOCK = 10


def candidate_sites(read_layers: Sequence[int]) -> List[str]:
    """Every block from the first read site to block 10, plus the final state.

    The sweep starts AT the first read site rather than before it: a site upstream of every write
    cannot carry the content by construction, so including it would pad the window rule with sites it
    cannot fail on.
    """
    return [f"site{l}" for l in range(min(read_layers), LAST_BLOCK + 1)] + ["final"]


def site_block(site: str) -> int:
    return LAST_BLOCK + 1 if site == "final" else int(site[4:])


def lens_families(gk: E8.GPT2Knowledge, sites: Sequence[str], obj_ids: Sequence[int]):
    """One lens family per site, plus its cosine to the unembedding rows of the same tokens.

    The family at block l is estimated at ``hidden_states[l + 1]``, because the adapter's hook adds the
    write to the OUTPUT of block l; estimating at ``l`` would measure a state the write has not entered.
    At the final state the lens is the identity Jacobian, so the family IS ``W_U`` and the cosine is
    1.000 by construction -- it is computed rather than hard-coded so that a change to the lens would
    show up here instead of being assumed away.
    """
    enc = gk.tok(WSC.LENS_CORPUS, return_tensors="pt", padding=True)
    w_out = gk.model.w_out
    rows = w_out[torch.as_tensor(list(obj_ids))].float()
    fams: Dict[str, torch.Tensor] = {}
    cos: Dict[str, float] = {}
    for s in sites:
        if s == "final":
            fams[s] = F.normalize(rows, dim=-1)
        else:
            jl = jlens_vectors(gk.model.lm, site_block(s) + 1, list(obj_ids), enc["input_ids"],
                               enc["attention_mask"], w_out, batch=4)
            fams[s] = jl.vectors.cpu()
        cos[s] = lens_alignment(fams[s], rows)
    return fams, cos


def run_arm(arm: str, seed: int, n_groups: int, verbose: bool = True,
            families: bool = False) -> Dict[str, Any]:
    """One arm, one seed: capability, mediators, lens cosines, the window, and the audit at every site."""
    spec_arm = ARMS[arm]
    ckpt = CHECKPOINTS / f"e000020_gpt2{spec_arm['suffix']}_seed{seed}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} does not exist; train it first "
                                f"(recorded: e000052_symlink_bos_train, early: sit001_early_read_train)")
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF,
                        read_layers=tuple(spec_arm["read_layers"]))
    gk = E8.GPT2Knowledge(cfg)
    ck = torch.load(ckpt, weights_only=False)
    saved = tuple(ck.get("adapter_config", {}).get("read_layers", spec_arm["read_layers"]))
    if saved != tuple(spec_arm["read_layers"]):
        raise RuntimeError(f"{ckpt} was trained with read_layers={saved}, not {spec_arm['read_layers']}: "
                           "loading it under this arm would attribute one placement's numbers to another")
    gk.model.load_state_dict(ck["adapter"], strict=False)
    gk.model.eval()
    centre = np.asarray(ck["centre"])

    # Identical world, pod selection and templates to E-000063, so the rows are comparable to its record.
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

    sites = candidate_sites(cfg.read_layers)
    fams, cos = lens_families(gk, sites, obj_ids)
    cap = WSC.MultiCapture(gk, sites=[site_block(s) for s in sites if s != "final"])

    acc: Dict[Tuple[str, str], List[torch.Tensor]] = {}
    med: Dict[Tuple[str, str], List[float]] = {}
    ys: List[int] = []; gs: List[int] = []; truths: List[int] = []
    ans_a: List[torch.Tensor] = []; ans_s: List[torch.Tensor] = []; ans_n: List[torch.Tensor] = []
    by_agree: List[float] = []

    for label, (target, aliases, obj) in enumerate(selected):
        texts, tg = E63.texts_for(aliases, gk.names, E63.TEMPLATES)
        never = copy.deepcopy(store)
        for ak in aliases:
            if kids[ak] in never.cells:
                never.delete(kids[ak])
        if kids[target] in never.cells:
            never.delete(kids[target])
        sa, aa, inj_a = WSC.eval_states(gk, cap, store, texts)
        sn, an, _ = WSC.eval_states(gk, cap, never, texts)
        by_keys = [a[1][0] for j, a in enumerate(selected) if j != label][:8]
        by_text, _ = E63.texts_for(by_keys, gk.names, [9])
        _, ba0, _ = WSC.eval_states(gk, cap, store, by_text)
        store.shred(kids[target])
        ss, as_, inj_s = WSC.eval_states(gk, cap, store, texts)
        _, ba1, _ = WSC.eval_states(gk, cap, store, by_text)
        store.resign(kids[target])

        for s in sites:
            acc.setdefault((s, "active"), []).append(sa[s])
            acc.setdefault((s, "shred"), []).append(ss[s])
            acc.setdefault((s, "never"), []).append(sn[s])
            med.setdefault((s, "shred_moves"), []).append(float((sa[s] - ss[s]).abs().max()))
            med.setdefault((s, "never_moves"), []).append(float((sa[s] - sn[s]).abs().max()))
        if inj_a is not None and inj_s is not None:
            for j, l in enumerate(cfg.read_layers):
                med.setdefault((f"write{l}", "shred_moves"), []).append(
                    float((inj_a[:, j] - inj_s[:, j]).abs().max()))
        ys.extend([label] * len(texts)); gs.extend(tg); truths.extend([obj] * len(texts))
        ans_a.append(aa); ans_s.append(as_); ans_n.append(an)
        if len(by_text):
            by_agree.extend((ba0 == ba1).float().tolist())
    cap.close()

    y = torch.tensor(ys, dtype=torch.long); g = torch.tensor(gs, dtype=torch.long)
    truth = torch.tensor(truths, dtype=torch.long)
    unknown = gk.n_entities
    a_ans, s_ans, n_ans = torch.cat(ans_a), torch.cat(ans_s), torch.cat(ans_n)
    capability = {
        "active_alias_correct": float((a_ans == truth).float().mean()),
        "shred_alias_unknown": float((s_ans == unknown).float().mean()),
        "shred_alias_true_object": float((s_ans == truth).float().mean()),
        "never_alias_true_object": float((n_ans == truth).float().mean()),   # kill condition 1: the floor
        "bystander_top1_agree": float(np.mean(by_agree)) if by_agree else float("nan"),
    }

    mediators = {s: float(np.mean(med[(s, "shred_moves")])) for s in sites}
    never_med = {s: float(np.mean(med[(s, "never_moves")])) for s in sites}
    writes = {f"write{l}": float(np.mean(med[(f"write{l}", "shred_moves")]))
              for l in cfg.read_layers if (f"write{l}", "shred_moves") in med}
    readings = audit_window(mediators, cos, order=sites)

    # The audit itself, at EVERY site. The registered site is chosen across seeds afterwards, by the
    # rule in the pre-registration, so no site is picked here by what it answered.
    per_site: Dict[str, Dict[str, float]] = {}
    per_family: Dict[str, Dict[str, Dict[str, float]]] = {}
    rows = gk.model.w_out[torch.as_tensor(obj_ids)]
    for s in sites:
        states = {st: torch.cat(acc[(s, st)]) for st in ("active", "shred", "never")}
        feats = {st: F.normalize(v.float(), dim=-1) @ fams[s].t() for st, v in states.items()}
        p = E63.transfer_probe(feats["active"], feats["shred"], feats["never"], y, g, n)
        per_site[s] = {"jprobe_active": p["active"], "jprobe_shred": p["shred"], "jprobe_never": p["never"],
                       "jprobe_active_minus_never": p["active"] - p["never"],
                       "jprobe_shred_minus_never": p["shred"] - p["never"]}
        if families:
            # EXPLORATORY, and off by default: not part of the pre-registered verdict, which reads the
            # audit's own family alone. It answers the question a reader asks next -- at an admissible
            # site, is the AUDIT'S BASIS doing the work, or would any k-dimensional readout do? WSC-001
            # found a dimension-matched random projection at 0.741-0.871 downstream on the recorded
            # adapter, so the question is live and the answer belongs in the record either way.
            fam = WSC.probe_families(states, fams[s], rows, n, seed)
            per_family[s] = {}
            for name, fe in fam.items():
                q = E63.transfer_probe(fe["active"], fe["shred"], fe["never"], y, g, n)
                per_family[s][name] = {"active_minus_never": q["active"] - q["never"],
                                       "shred_minus_never": q["shred"] - q["never"], "never": q["never"]}
            if verbose:
                print("    families " + s + ": " + "  ".join(
                    f"{k}={v['active_minus_never']:+.3f}" for k, v in per_family[s].items()), flush=True)
        if verbose:
            r = [x for x in readings if x.site == s][0]
            print(f"  {arm:8s} seed {seed} {s:7s} moves {mediators[s]:8.2f} (ratio {r.arrival_ratio:.3f})  "
                  f"cos {cos[s]:.3f}  window {'YES' if r.in_window else 'no ':3s}  "
                  f"jprobe A-N {per_site[s]['jprobe_active_minus_never']:+.4f}  "
                  f"S-N {per_site[s]['jprobe_shred_minus_never']:+.4f}", flush=True)

    first_read = f"site{min(cfg.read_layers)}"
    return {
        "arm": arm, "seed": seed, "read_layers": list(cfg.read_layers), "n_pods": n,
        "chance": 1.0 / n, "checkpoint": str(ckpt), "sites": sites,
        "capability": capability, "mediators": mediators, "never_mediators": never_med,
        "write_mediators": writes, "lens_cos": cos,
        "siting": [r.to_dict() for r in readings], "window": window_sites(readings),
        "audit": per_site,
        "audit_families_exploratory": per_family or None,
        "e000063_site": first_read,
        "e000063_certificate": certify(readings, first_read, per_site[first_read]),
    }


def decide(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The pre-registered bars, applied to the worst seed of each arm. No bar is invented here."""
    out: Dict[str, Any] = {"arrival_floor": ARRIVAL_FLOOR, "degeneracy_ceiling": DEGENERACY_CEILING}
    for arm in sorted({r["arm"] for r in records}):
        rs = [r for r in records if r["arm"] == arm]
        cap_keys = ("active_alias_correct", "shred_alias_unknown", "shred_alias_true_object",
                    "never_alias_true_object", "bystander_top1_agree")
        worst = {k: (min if k in ("active_alias_correct", "shred_alias_unknown", "bystander_top1_agree")
                     else max)(float(r["capability"][k]) for r in rs) for k in cap_keys}
        common = set(rs[0]["window"])
        for r in rs[1:]:
            common &= set(r["window"])
        window = sorted(common, key=site_block)
        first_read = rs[0]["e000063_site"]
        a = {
            "seeds": sorted(r["seed"] for r in rs),
            "capability_worst_seed": worst,
            "VE1_active_alias_correct_ge_080": worst["active_alias_correct"] >= 0.80,
            "VE2_shred_alias_unknown_ge_090": worst["shred_alias_unknown"] >= 0.90,
            "VE2_shred_alias_true_object_le_005": worst["shred_alias_true_object"] <= 0.05,
            "VE3_bystander_top1_agree_ge_098": worst["bystander_top1_agree"] >= 0.98,
            "kill1_never_floor_le_005": worst["never_alias_true_object"] <= 0.05,
            "window_all_seeds": window,
            "P0_first_read_site_fails_arrival": all(first_read not in r["window"] for r in rs) and all(
                [x for x in r["siting"] if x["site"] == first_read][0]["arrival_ratio"] < ARRIVAL_FLOOR
                for r in rs),
            "P1_window_non_empty": bool(window),
        }
        if window:
            site = window[0]                       # the registered site: lowest block in the window
            a["registered_site"] = site
            a["P2_jprobe_active_minus_never_ge_030"] = all(
                r["audit"][site]["jprobe_active_minus_never"] >= 0.30 for r in rs)
            a["P3_jprobe_shred_minus_never_le_005"] = all(
                r["audit"][site]["jprobe_shred_minus_never"] <= 0.05 for r in rs)
            a["registered_site_rows"] = {
                str(r["seed"]): {k: r["audit"][site][k] for k in
                                 ("jprobe_active_minus_never", "jprobe_shred_minus_never")} for r in rs}
        else:
            a["registered_site"] = None
            a["P2_jprobe_active_minus_never_ge_030"] = None
            a["P3_jprobe_shred_minus_never_le_005"] = None
        out[arm] = a
    return out


def markdown(records: List[Dict[str, Any]], verdict: Dict[str, Any]) -> str:
    lines = ["# SIT-001 — the audit window", "",
             f"Pre-registered at `docs/novelty/sit001-preregister.md`. Bars: arrival ratio ≥ "
             f"{ARRIVAL_FLOOR:.2f}, lens |cos| to the unembedding ≤ {DEGENERACY_CEILING:.2f}.", ""]
    for arm in sorted({r["arm"] for r in records}):
        rs = sorted((r for r in records if r["arm"] == arm), key=lambda r: r["seed"])
        lines += [f"## arm `{arm}` — read_layers {rs[0]['read_layers']}", "",
                  "| site | moves(shred) | arrival | lens cos | in window | jprobe(A−N) | jprobe(S−N) |",
                  "|---|---|---|---|---|---|---|"]
        for s in rs[0]["sites"]:
            mv = float(np.mean([r["mediators"][s] for r in rs]))
            ar = float(np.mean([[x for x in r["siting"] if x["site"] == s][0]["arrival_ratio"] for r in rs]))
            co = float(np.mean([r["lens_cos"][s] for r in rs]))
            inw = all(s in r["window"] for r in rs)
            an = min(r["audit"][s]["jprobe_active_minus_never"] for r in rs)
            sn = max(r["audit"][s]["jprobe_shred_minus_never"] for r in rs)
            lines.append(f"| `{s}` | {mv:.2f} | {ar:.3f} | {co:.3f} | {'**yes**' if inw else 'no'} | "
                         f"{an:+.4f} | {sn:+.4f} |")
        v = verdict[arm]
        lines += ["", "Worst seed: " + ", ".join(f"`{k}` {val:.4f}" for k, val in
                                                 v["capability_worst_seed"].items()), "",
                  "| check | result |", "|---|---|"]
        for k in sorted(k for k in v if k.startswith(("VE", "P0", "P1", "P2", "P3", "kill"))):
            lines.append(f"| `{k}` | {v[k]} |")
        lines += ["", f"Window on all seeds: `{v['window_all_seeds'] or 'EMPTY'}`; "
                      f"registered site: `{v['registered_site']}`.", ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["recorded", "early"], choices=sorted(ARMS))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-groups", type=int, default=16)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "4")))
    ap.add_argument("--families", action="store_true",
                    help="also score WSC-001's five feature families at every site (EXPLORATORY, not "
                         "part of the pre-registered verdict; use SO_RESULT_SUFFIX to keep it apart)")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    records: List[Dict[str, Any]] = []
    for arm in a.arms:
        for seed in a.seeds:
            records.append(run_arm(arm, seed, a.n_groups, families=a.families))
    verdict = decide(records)
    rec = {"experiment": "SIT-001", "candidate_only": True,
           "preregistration": "docs/novelty/sit001-preregister.md",
           "arms": {k: {"read_layers": list(v["read_layers"]), "suffix": v["suffix"]} for k, v in ARMS.items()},
           "bos": os.environ.get("SO_BOS", "") == "1", "seconds": time.time() - t0,
           "families_exploratory": bool(a.families),
           "records": records, "verdict": verdict}
    ledger.save("sit001_audit_window", rec, markdown(records, verdict))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
