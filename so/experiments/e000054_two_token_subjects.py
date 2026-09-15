"""Experiment E-000054 -- two-token subjects: is the subject-initial failure the single-token collapse
condition, or subject-at-position-0 in general?

WHERE THIS COMES FROM. E-000050 (ledger 31.38, corrected in 31.44) measured that the recorded GPT-2
adapter fails the held-out phrasings whose subject is the position-0 token (0.37 / 0.54 read / route,
worst seed) and reads them at 0.97 / 0.98 with a token prepended at inference. The field census (31.43)
then found, through its completeness critic, that every subject in this repository is ONE BPE token
(``select_entities``: the first 256 tokens matching ``G[A-Z][a-z]{3,}``), so the failing case is exactly
Yang et al.'s collapse condition for ROME on GPT-2-XL -- subject-initial AND single-token, 77 / 77 of
their collapse cases, 80 / 21,919 = 0.36% of CounterFact -- and NOT the 80% condition of the published
efficacy prompts, which is a multi-token subject whose first token sits at position 0. On that condition
ROME (reading the subject's LAST token) shows no penalty, and this repository has no measurement. The
critic named the cheapest decisive measurement: change the string a subject is rendered as, and nothing
else.

SURFACES. The object side, the keys (``encode_bank`` keys a cell on ``w_in[entity_token_ids[subject]]``,
a single row per entity) and the trainer are untouched; only ``gk.names`` -- the string every prompt
renders a subject as -- changes:
  product   256 subjects = 16 first-tokens x 16 second-tokens, both needed for identity (a 16 x 16
            product code: if the sink eats the first token it eats four of the eight bits)
  second    256 subjects = a REDUNDANT first token (one of the same 16, a fixed function of the entity)
            followed by the entity's own single token (identity in the second token alone: if the sink
            eats the first token it eats nothing)
  single    the record -- E-000050 arms A and B -- not re-run
The 32 part-tokens are entity-shaped tokens 256..287 of the same selection, outside the object pool, so
no subject part collides with an object logit. Both surfaces put the subject's FIRST token at position
0 on the subject-initial templates (t0, t2, t6 trained; t8, t11 held out) and at position >= 1 on the
medial ones; the tokenizer verifies this (``surface_positions``).

ARMS. Per surface and seed, ONE adapter trained without a BOS (E-000017-B's trainer through E-000039's
``train_arm`` at tie weight 0, the same budget as E-000050-B), read three ways with no weight changed:
  N  bare, the record's protocol
  B  ``<|endoftext|>`` at inference (E-000050 arm B's reading)
  S  a lone space at inference (E-000050-A's cheapest prefix; on its seed 0 the only one with no medial price)
Nothing is trained with a BOS: 31.44 established that a BOS at training time costs capability and that
the finding is the inference-time token's.

WHAT COULD FAIL, EACH REGISTERED BELOW. V: a surface the adapter cannot learn on the trained MEDIAL
templates (where position 0 is not the subject) voids the run -- nothing else is read. H1: under the
sink reading, ``product`` read bare fails the held-out subject-initial forms (<= 0.50) and recovers with
a space or a BOS at position 0 (>= 0.90); if it reads them bare, the single-token case is special and
E-000050's failure does not extend even to a product code. H2: ``second`` read bare reads the held-out
subject-initial forms (>= 0.90); if it fails, ANY subject-initial multi-token surface fails for this
learned router, and the 80% field condition is live for adapters of this kind. M: both surfaces read the
held-out MEDIAL form bare at >= 0.90, as the record does (0.95) -- the medial control that says the
surface change did not break generalisation itself. The trained subject-initial rows and the space
reading's medial price are reported and never scored (31.44's at-risk clause).

Prior art, so nothing is claimed as a mechanism: the position-0 anomaly is Xiao et al. (2023), Sun et
al. (2024), Gu et al. (ICLR 2025) and Ran-Milo et al. (2026); the single-token subject-initial collapse
of a weight edit and its prefix remedy are Yang et al. ("The Fall of ROME", Findings of EMNLP 2024, and
"The Butterfly Effect of Model Editing", Findings of ACL 2024); that a later subject token can carry the
identity ROME reads is their ``subject_last`` design. What is measured is whether a LEARNED routing query
read at the last prompt token recovers a subject whose identity is spread over two tokens when the first
sits on the sink -- the condition the census puts at 80% of the published efficacy prompts.

Run:  python -m so.experiments.e000054_two_token_subjects [--seeds 0 1 2] [--steps 3000] [--surfaces product second]
      python -m so.experiments.e000054_two_token_subjects --quick --steps 30 --n-targets 8 --seeds 0 \
          --results-dir /path/to/scratch                                   (a smoke run; records nothing)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from so import ledger
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000039_address_tying as E39
from so.experiments import e000050_bos_artefact as E50A
from so.experiments import e000050_bos_trained as E50
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256, guard_recorded_checkpoint
from so.llm_adapter import AdapterConfig

SURFACES = ("product", "second")
READINGS = ("N", "B", "S")
N_PARTS = 16
BOS = "<|endoftext|>"


def surfaces_for(tok, entity_ids: List[int]) -> Dict[str, List[str]]:
    """The two two-token renderings of the 256 entities; the object pool is untouched."""
    n = len(entity_ids)
    if n != N_PARTS * N_PARTS:
        raise ValueError(f"the product code needs n_entities = {N_PARTS * N_PARTS}, got {n}")
    ids = E8.select_entities(tok, n + 2 * N_PARTS)
    if ids[:n] != list(entity_ids):
        raise ValueError("the entity pool is not the first 256 of the selection; the part tokens would collide")
    parts = [tok.decode([i]) for i in ids[n:]]
    first, second = parts[:N_PARTS], parts[N_PARTS:]
    single = [tok.decode([i]) for i in entity_ids]                 # the record's names, whatever gk.names is now
    product = [first[i // N_PARTS] + second[i % N_PARTS] for i in range(n)]
    rng = np.random.default_rng(54)
    redundant = rng.integers(0, N_PARTS, size=n)
    second_surface = [first[int(redundant[i])] + single[i] for i in range(n)]
    return {"product": product, "second": second_surface}


def surface_positions(tok, names: List[str]) -> Dict[str, Any]:
    """Tokenizer-verified: every name is two tokens; the subject's first token sits at index 0 on the
    subject-initial templates and at >= 1 on the medial ones."""
    n_tok = [len(tok(nm)["input_ids"]) for nm in names]
    initial, medial = E39.subject_initial_templates(tok)
    idx = {}
    for t in range(E39.N_T):
        text = E17.TEMPLATES12[0][t].format(s=names[17])
        toks = tok.convert_ids_to_tokens(tok(text)["input_ids"])
        key = names[17].split()[0]
        idx[t] = next(i for i, s in enumerate(toks) if key in s)
    return {"tokens_per_name_min": min(n_tok), "tokens_per_name_max": max(n_tok),
            "initial_at_0": all(idx[t] == 0 for t in initial),
            "medial_at_ge1": all(idx[t] >= 1 for t in medial), "index_by_template": idx}


QUICK = False     # set by --quick: checkpoints go under a _smoke suffix and nothing is recorded


def _train(gk: E8.GPT2Knowledge, surface: str, seed: int, steps: int, force: bool) -> Tuple[np.ndarray, str, float]:
    path = CHECKPOINTS / f"e000054_{surface}{'_smoke' if QUICK else CKPT_SUFFIX}_seed{seed}.pt"
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        if ck.get("names") != gk.names:
            raise RuntimeError(f"{path} was trained on a different surface")
        gk.model.load_state_dict(ck["adapter"], strict=False)
        gk.model.eval()
        return np.asarray(ck["centre"]), _sha256(path), float(ck["train_seconds"])
    os.environ["SO_BOS"] = "0"
    out = E39.train_arm(gk, seed, steps, "address", tie_weight=0.0)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    guard_recorded_checkpoint(path)
    torch.save({"adapter": E8.adapter_state(gk.model), "centre": out["centre"], "history": out["history"],
                "train_seconds": out["train_seconds"], "adapter_config": gk.model.cfg.to_dict(),
                "bos": False, "surface": surface, "names": list(gk.names)}, path)
    return np.asarray(out["centre"]), _sha256(path), float(out["train_seconds"])


def _read(gk: E8.GPT2Knowledge, seed: int, centre: np.ndarray, reading: str, n_targets: int) -> Dict[str, float]:
    """E-000050-B's ``_measure`` under one prompt condition; adds the trained-template split."""
    os.environ["SO_BOS"] = "0"
    if reading == "N":
        m = E50._measure(gk, seed, centre, False, oracle=False, n_targets=n_targets)
    elif reading == "B":
        m = E50._measure(gk, seed, centre, True, oracle=False, n_targets=n_targets)
    elif reading == "S":
        with E50A.prefixed(" "):
            m = E50._measure(gk, seed, centre, False, oracle=False, n_targets=n_targets)
    else:
        raise ValueError(reading)
    initial, medial = E39.subject_initial_templates(gk.tok)
    tr_i = [t for t in initial if t < E39.N_TRAIN]
    tr_m = [t for t in medial if t < E39.N_TRAIN]
    m["train_initial/read_min"] = min(m[f"t{t}/train/read"] for t in tr_i)
    m["train_initial/route_hit_min"] = min(m[f"t{t}/train/route_hit"] for t in tr_i)
    m["train_medial/read_min"] = min(m[f"t{t}/train/read"] for t in tr_m)
    m["train_medial/route_hit_min"] = min(m[f"t{t}/train/route_hit"] for t in tr_m)
    return m


def run_seed(seed: int, surfaces: List[str], steps: int, n_targets: int, force: bool,
             results_dir: Optional[str], verbose: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    out: Dict[str, Any] = {"seed": seed}
    for surface in surfaces:
        gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))
        names = surfaces_for(gk.tok, list(gk.entity_ids))[surface]
        gk.names = names
        pos = surface_positions(gk.tok, names)
        if not (pos["tokens_per_name_min"] == pos["tokens_per_name_max"] == 2 and pos["initial_at_0"] and pos["medial_at_ge1"]):
            raise RuntimeError(f"surface {surface} is not two tokens with the first at position 0: {pos}")
        out[f"{surface}/tokens_per_name"] = 2.0
        out[f"{surface}/example"] = names[17]
        centre, sha, train_s = _train(gk, surface, seed, steps, force)
        out[f"{surface}/sha256"] = sha
        out[f"{surface}/train_seconds"] = train_s
        for reading in READINGS:
            m = _read(gk, seed, centre, reading, n_targets)
            out.update({f"{surface}/{reading}/{k}": v for k, v in m.items()})
            if verbose:
                print(f"  seed {seed} {surface:8s} {reading}: held-out initial read/route "
                      f"{m['heldout_initial/read_min']:.2f}/{m['heldout_initial/route_hit_min']:.2f}  "
                      f"medial {m['heldout_medial/read_min']:.2f}/{m['heldout_medial/route_hit_min']:.2f}  "
                      f"trained initial/medial read_min {m['train_initial/read_min']:.2f}/{m['train_medial/read_min']:.2f}  "
                      f"heldout_read {m['heldout/active_correct']:.4f} train_read {m['train/active_correct']:.4f} "
                      f"generic_kl {m['generic/kl_to_base']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, f"e000054_{surface}_seed{seed}.json"), "w") as f:
                json.dump({k: v for k, v in out.items() if k == "seed" or k.startswith(surface + "/")}, f, indent=1)
    out["seconds"] = time.time() - t0
    return out


# Worst seed. Fixed before the run.
CRITERIA: Dict[str, Tuple[str, float]] = {
    # V: the surface is learnable where position 0 is not the subject (trained medial templates, bare)
    "product/N/train_medial/read_min": (">=", 0.85),
    "second/N/train_medial/read_min": (">=", 0.85),
    # H1: a product code with its first token on the sink fails bare and recovers with a token at position 0
    "product/N/heldout_initial/read_min": ("<=", 0.50),
    "product/S/heldout_initial/read_min": (">=", 0.90),
    "product/B/heldout_initial/read_min": (">=", 0.90),
    # H2: a redundant first token on the sink costs nothing
    "second/N/heldout_initial/read_min": (">=", 0.90),
    # M: the medial held-out form is read bare by both surfaces, as the record's is (0.95)
    "product/N/heldout_medial/read_min": (">=", 0.90),
    "second/N/heldout_medial/read_min": (">=", 0.90),
}

DECISION_RULE = (
    "Worst seed. VOID if either V row fails: the surface is not learnable at this budget and nothing else "
    "is read. With V and M holding: H1 and H2 both PASS -> the failure is IDENTITY at position 0 -- a "
    "multi-token subject whose first token is redundant is read bare, one whose first token is needed is "
    "not, and the census's 80% condition splits by whether the first token carries identity, which no "
    "benchmark records. H1 PASS, H2 FAIL -> any subject-initial multi-token surface fails for this "
    "learned router, so the 80% condition is live for adapters of this kind (and says nothing against "
    "ROME, which reads the last subject token). H1's bare row FAILS (product reads bare) with H2 PASS -> "
    "the single-token case is special: E-000050's failure is Yang et al.'s 0.36% condition and does not "
    "extend even to a product code. H1's recovery rows FAIL (product stays low under a space and a BOS) "
    "-> the product surface fails for a reason other than position 0 and no sink sentence is read. M "
    "failing on a surface -> that surface's rows are reported and not interpreted. The trained "
    "subject-initial rows and the space reading's medial price are reported and never scored.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--surfaces", nargs="*", default=list(SURFACES))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-targets", type=int, default=100)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true", help="shrink E-000017's evaluation; records nothing")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    if os.environ.get("SO_BOS") == "1":
        raise SystemExit("E-000054 trains and reads its bare arm without a BOS; unset SO_BOS")
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.quick:
        global QUICK
        QUICK = True
        if not args.results_dir:
            raise SystemExit("--quick records nothing: give --results-dir")
        E17.EVAL.update(n_cells=200, n_targets=40, n_broken=40, n_generic=40)
        os.environ["SO_RESULT_SUFFIX"] = "-smoke"
    per_seed = [run_seed(s, args.surfaces, args.steps, args.n_targets, args.force, args.results_dir) for s in args.seeds]
    keys = sorted(k for k in per_seed[0] if isinstance(per_seed[0][k], (int, float)) and k != "seed")
    agg = ledger.aggregate(per_seed, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})
    rows = []
    for surface in args.surfaces:
        for reading in READINGS:
            p = f"{surface}/{reading}"
            rows.append([surface, {"N": "bare", "B": BOS + " at inference", "S": "a lone space at inference"}[reading],
                         f"{agg[f'{p}/heldout_initial/read_min']['min']:.2f} / {agg[f'{p}/heldout_initial/route_hit_min']['min']:.2f}",
                         f"{agg[f'{p}/heldout_medial/read_min']['min']:.2f} / {agg[f'{p}/heldout_medial/route_hit_min']['min']:.2f}",
                         f"{agg[f'{p}/train_initial/read_min']['min']:.2f} / {agg[f'{p}/train_medial/read_min']['min']:.2f}",
                         f"{agg[f'{p}/heldout/active_correct']['min']:.4f}",
                         f"{agg[f'{p}/train/active_correct']['min']:.4f}",
                         f"{agg[f'{p}/generic/kl_to_base']['max']:.3f}"])
    rows.append(["single (the record)", "bare / BOS (E-000050 A / B)", "0.37 / 0.54 ; 0.97 / 0.98", "0.95 / 0.94 ; 0.70 / 0.64",
                 "0.75 / — ; — / —", "0.7288 ; 0.9175", "0.9119 ; 0.9719", "3.647 ; 3.920"])
    tbl = ledger.table(["surface", "reading", "held-out subject-initial read / route", "held-out subject-medial read / route",
                        "trained subject-initial / medial read_min", "held-out reading", "trained reading",
                        "mean generic KL (worst seed)"], rows)
    record = {"experiment": "E-000054", "title": "two-token subjects: identity at position 0, or the single-token collapse",
              "evidence_level": "E5", "seeds": args.seeds, "surfaces": args.surfaces, "steps": args.steps,
              "n_targets": args.n_targets, "quick": args.quick, "decision_rule": DECISION_RULE,
              "per_seed": per_seed, "aggregate": agg, "criteria": check,
              "control": "E-000050 arms A and B on the recorded single-token surface (not re-run)"}
    md = [f"# E-000054 — {record['title']}", "",
          f"Seeds {args.seeds}, {args.steps} steps per surface and seed, trained without a BOS; {args.n_targets} targets per",
          "seed for the decomposition. Only the string a subject is rendered as changes; keys, objects and the trainer",
          "are E-000050-B's. Worst seed everywhere.", "", tbl, "",
          "## The rule, fixed before the run", "", DECISION_RULE, "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    text = "\n".join(md)
    path = None
    if not args.quick:                      # a real run is always recorded under so/results
        path = ledger.save("e000054_two_token_subjects", record, text)
    if args.results_dir:                    # ... and copied beside the per-seed partials
        os.makedirs(args.results_dir, exist_ok=True)
        name = "e000054_two_token_subjects" + os.environ.get("SO_RESULT_SUFFIX", "")
        record.setdefault("environment", ledger.environment())
        with open(os.path.join(args.results_dir, name + ".json"), "w") as f:
            json.dump(ledger._to_jsonable(record), f, indent=1, sort_keys=True)
        with open(os.path.join(args.results_dir, name + ".md"), "w") as f:
            f.write(text.rstrip("\n") + "\n")
        path = path or os.path.join(args.results_dir, name + ".md")
    print("\n".join(md[1:])); print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
