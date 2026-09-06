"""Experiment E-000029 — what the marker gate actually certifies.

E-000021 measured the gate's false-accept rate at 8.49e-04 and called that the bound on the deletion
guarantee. It drew its UNSIGNED class from ``so.data.invalid_markers``, which rejects every draw
within 0.7 of the marker centre — and the store calls everything beyond 0.35 deleted. So the class
that number was measured on excludes the annulus 0.35 < ‖m − κ‖ < 0.7 entirely: the region nearest
the accepting cap, where a gate is most likely to accept, and which no training and no evaluation
distribution in the programme has ever populated. `MVCCStore.new_invalid_marker` rejects the same
band (``2 * valid_radius``), so the gap is in the store as well as in the experiment.

This measures the gate over the geometry instead of over a convenient sample, without training
anything:

  band profile      accept rate and mean score in 20 shells of ‖m − κ‖ from 0 to 2, sampled by
                    construction (``m = cos θ·κ + sin θ·u``) rather than by rejection, which cannot
                    reach the near shells at all: a uniform unit vector in 16 dimensions concentrates
                    at ‖m − κ‖ ≈ √2
  operational       the smallest shell at which the accept rate is exactly zero, against the store's
  radius            declared 0.35
  uniform rate      accept rate over unit vectors with NO rejection, the honest "a random marker" rate
  annulus rate      accept rate inside 0.35 < ‖m − κ‖ < 0.7, the band the store deletes and nothing
                    ever tested

and asks a second question, about the centre rather than the gate. `new_valid_marker` returns
``normalise(κ + N(0, 0.05²·I))``, and every signed marker in a bank is therefore a noisy copy of κ.
The mean of the signed markers estimates κ; the experiment reports that error, whether the estimate
itself passes `marker_valid`, and the yield of markers minted from it. `make_centre` derives κ from
``10_000 + seed``, and every checkpoint serialises κ verbatim, so this is not a hypothetical about a
leaked secret.

Run:  python -m so.experiments.e000029_marker_geometry
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import torch

from so import ledger
from so.data import invalid_markers, valid_markers
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256, checkpoint_path
from so.experiments.e000021_gate_error_rates import FAMILIES, gate_scores
from so.model import ModelConfig, MutableKnowledgeTransformer
from so.mvcc import MVCCStore

N_BANDS = 20
BAND_MAX = 2.0
N_PER_BAND = 20_000
N_UNIFORM = 500_000
N_ANNULUS = 200_000
N_SIGNED_FOR_ESTIMATE = 950        # about what EVAL_CONFIG's bank holds
VALID_RADIUS = 0.35


def shell(rng: np.random.Generator, centre: np.ndarray, dist: float, n: int) -> np.ndarray:
    """Unit vectors at exactly ``dist`` from ``centre``, by construction rather than by rejection.

    Two unit vectors at chord distance d subtend cos θ = 1 − d²/2, so m = cos θ·κ + sin θ·u with u a
    unit vector orthogonal to κ. Rejection sampling cannot produce these: in 16 dimensions a uniform
    unit vector sits at ‖m − κ‖ ≈ √2 and essentially never lands in the near shells.
    """
    d = centre.shape[0]
    cos_t = 1.0 - dist ** 2 / 2.0
    if not -1.0 <= cos_t <= 1.0:
        return np.zeros((0, d))
    sin_t = float(np.sqrt(max(0.0, 1.0 - cos_t ** 2)))
    u = rng.normal(size=(n, d))
    u -= (u @ centre)[:, None] * centre[None, :]                 # project onto the orthogonal complement
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return cos_t * centre[None, :] + sin_t * u


def uniform_sphere(rng: np.random.Generator, d: int, n: int) -> np.ndarray:
    m = rng.normal(size=(n, d))
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def annulus(rng: np.random.Generator, centre: np.ndarray, n: int, lo: float, hi: float) -> np.ndarray:
    """Uniform in the chord-distance band: draw each distance, then place the vector on that shell."""
    d = centre.shape[0]
    dists = rng.uniform(lo, hi, size=n)
    cos_t = 1.0 - dists ** 2 / 2.0
    sin_t = np.sqrt(np.clip(1.0 - cos_t ** 2, 0.0, None))
    u = rng.normal(size=(n, d))
    u -= (u @ centre)[:, None] * centre[None, :]
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return cos_t[:, None] * centre[None, :] + sin_t[:, None] * u


def measure(name: str, seed: int, rng: np.random.Generator) -> Dict[str, Any]:
    path = checkpoint_path(name, seed)
    if not path.exists():
        raise SystemExit(f"missing checkpoint {path}")
    ck = torch.load(path, weights_only=False)
    model = MutableKnowledgeTransformer(ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    centre = np.asarray(ck["centre"], dtype=float)
    m: Dict[str, Any] = {"checkpoint": f"{name}_seed{seed}", "seed": seed, "family": name,
                         "checkpoint_sha256": _sha256(path), "centre_in_checkpoint": True}

    # ---- the gate over the geometry
    accept, mean_score, bands = [], [], []
    for b in range(1, N_BANDS + 1):
        dist = BAND_MAX * b / N_BANDS
        v = shell(rng, centre, dist, N_PER_BAND)
        if not v.shape[0]:
            continue
        s = gate_scores(model, v)
        bands.append(dist); accept.append(float((s > 0.5).mean())); mean_score.append(float(s.mean()))
    m["bands"] = bands; m["band_accept"] = accept; m["band_mean_score"] = mean_score
    zero = [d for d, a in zip(bands, accept) if a == 0.0]
    m["operational_radius"] = float(min(zero)) if zero else float("nan")
    m["declared_radius"] = VALID_RADIUS

    s_uniform = gate_scores(model, uniform_sphere(rng, centre.shape[0], N_UNIFORM))
    m["uniform_accept_rate"] = float((s_uniform > 0.5).mean())
    m["uniform_n"] = int(N_UNIFORM)
    s_ann = gate_scores(model, annulus(rng, centre, N_ANNULUS, VALID_RADIUS, 0.7))
    m["annulus_accept_rate"] = float((s_ann > 0.5).mean())
    m["annulus_n"] = int(N_ANNULUS)
    # the class E-000021 actually measured, for a like-for-like column
    s_rej = gate_scores(model, invalid_markers(rng, centre, N_ANNULUS))
    m["rejection_sampled_accept_rate"] = float((s_rej > 0.5).mean())
    s_signed = gate_scores(model, valid_markers(rng, centre, N_ANNULUS))
    m["false_reject_rate"] = float((s_signed <= 0.5).mean())

    # ---- the centre, from the bank alone
    store = MVCCStore(marker_dim=centre.shape[0], seed=seed, marker_centre=centre)
    signed = np.stack([store.new_valid_marker() for _ in range(N_SIGNED_FOR_ESTIMATE)])
    est = signed.mean(0)
    est = est / np.linalg.norm(est)
    m["centre_estimate_error"] = float(np.linalg.norm(est - centre))
    m["centre_estimate_passes_marker_valid"] = bool(store.marker_valid(est))
    m["one_donor_marker_passes"] = bool(store.marker_valid(signed[0]))
    minted = np.stack([(est + rng.normal(scale=0.05, size=centre.shape[0])) for _ in range(10_000)])
    minted /= np.linalg.norm(minted, axis=1, keepdims=True)
    m["minted_marker_valid_rate"] = float(np.mean([store.marker_valid(x) for x in minted]))
    m["minted_gate_accept_rate"] = float((gate_scores(model, minted) > 0.5).mean())
    return m


KEYS = ["operational_radius", "uniform_accept_rate", "annulus_accept_rate", "rejection_sampled_accept_rate",
        "false_reject_rate", "centre_estimate_error", "minted_marker_valid_rate", "minted_gate_accept_rate"]

CRITERIA = {
    # E-000021's own pre-registered bar, applied to the distribution it did not measure
    "annulus_accept_rate": ("<=", 1e-3),
    "uniform_accept_rate": ("<=", 1e-3),
}


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    rng = np.random.default_rng(29)

    per_ckpt: List[Dict[str, Any]] = []
    for name in args.families:
        for seed in FAMILIES[name]:
            if not checkpoint_path(name, seed).exists():
                print(f"  (skipping missing {name}_seed{seed})", flush=True)
                continue
            m = measure(name, seed, rng)
            per_ckpt.append(m)
            print(f"  {m['checkpoint']:<18} operational radius {m['operational_radius']:.2f} "
                  f"(declared {VALID_RADIUS}) | uniform {m['uniform_accept_rate']:.2e} | "
                  f"annulus {m['annulus_accept_rate']:.2e} | rejection-sampled "
                  f"{m['rejection_sampled_accept_rate']:.2e} | centre error {m['centre_estimate_error']:.4f}",
                  flush=True)
    if not per_ckpt:
        raise SystemExit("no checkpoints found")

    agg = ledger.aggregate(per_ckpt, KEYS)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})
    n_ck = len(per_ckpt)
    pooled = {
        "uniform": (int(round(sum(m["uniform_accept_rate"] * m["uniform_n"] for m in per_ckpt))),
                    sum(m["uniform_n"] for m in per_ckpt)),
        "annulus": (int(round(sum(m["annulus_accept_rate"] * m["annulus_n"] for m in per_ckpt))),
                    sum(m["annulus_n"] for m in per_ckpt)),
        "rejection_sampled": (int(round(sum(m["rejection_sampled_accept_rate"] * m["annulus_n"] for m in per_ckpt))),
                              sum(m["annulus_n"] for m in per_ckpt)),
    }
    ci = {k: ledger.clopper_pearson(s, n) for k, (s, n) in pooled.items()}
    tbl = ledger.table(["marker distribution", "accepted", "of", "rate", "95% CI lower", "95% CI upper"],
                       [[k, pooled[k][0], pooled[k][1], f"{pooled[k][0] / pooled[k][1]:.3e}",
                         f"{ci[k]['lower']:.3e}", f"{ci[k]['upper']:.3e}"] for k in pooled])

    bands = per_ckpt[0]["bands"]
    band_tbl = ledger.table(["distance from the centre", "accept rate (mean over checkpoints)", "mean gate score"],
                            [[f"{d:.2f}", f"{np.mean([m['band_accept'][i] for m in per_ckpt]):.4f}",
                              f"{np.mean([m['band_mean_score'][i] for m in per_ckpt]):.4f}"]
                             for i, d in enumerate(bands)])

    record = {"experiment": "E-000029", "title": "what the marker gate actually certifies",
              "trains_nothing": True, "n_checkpoints": n_ck, "declared_radius": VALID_RADIUS,
              "n_per_band": N_PER_BAND, "n_uniform": N_UNIFORM, "n_annulus": N_ANNULUS,
              "pooled": pooled, "pooled_intervals": ci,
              "per_checkpoint": per_ckpt, "aggregate": agg, "criteria": check}
    md = [f"# E-000029 — {record['title']}", "",
          f"{n_ck} recorded checkpoints, no training.", "",
          "E-000021 reported the gate's false-accept rate as 8.49e-04 and called it the bound on the deletion",
          "guarantee. Its unsigned class comes from `invalid_markers`, which rejects every draw within 0.7 of",
          f"the centre, while the store calls everything beyond {VALID_RADIUS} deleted. The band in between was",
          "measured by nothing. These are the three distributions side by side.", "",
          "## The gate's accept rate, by where the marker is", "", tbl, "",
          "## Accept rate by distance from the centre", "", band_tbl, "",
          f"Operational radius (smallest shell with an accept rate of exactly zero), worst checkpoint: "
          f"{agg['operational_radius']['max']:.2f} against a declared {VALID_RADIUS}.", "",
          "## The centre is not a secret", "",
          f"A signed marker is `normalise(centre + N(0, 0.05^2))`, so the mean of {N_SIGNED_FOR_ESTIMATE} of them",
          f"estimates the centre to {agg['centre_estimate_error']['max']:.4f} on the worst checkpoint, well inside",
          f"the {VALID_RADIUS} radius. Markers minted from that estimate pass `marker_valid` at "
          f"{agg['minted_marker_valid_rate']['min']:.4f} and the learned gate at "
          f"{agg['minted_gate_accept_rate']['min']:.4f}. `make_centre` derives the centre from `10_000 + seed`,",
          "and every checkpoint stores it verbatim, so shipping a model ships the ability to mint signatures.", "",
          "## Pre-registered criteria", "", ledger.criteria_table(check, basis="worst checkpoint"), ""]
    path = ledger.save("e000029_marker_geometry", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(tbl)
    print(ledger.criteria_table(check, basis="worst checkpoint"))
    return record


if __name__ == "__main__":
    main()
