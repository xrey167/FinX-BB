from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STAGE = "R355-WORLD-NATURAL-NEURAL-LAYER-ALGEBRA"
REPORT_PATH = Path(os.environ.get("SO_R355_REPORT", "ci-r355/report.json"))
SEED = 3550914
PROGRAMS = int(os.environ.get("SO_R355_PROGRAMS", "1200"))
WORLDS_PER_PROGRAM = int(os.environ.get("SO_R355_WORLDS", "128"))
WORLD_CELLS = 128
H = 16
R = 6
LAYERS = 8
TOL = 2e-10


@dataclass
class PNS:
    h: np.ndarray                 # world-independent numeric continuation
    refs: np.ndarray              # canonical addresses, shape [R]
    coeff: np.ndarray             # reference-valued linear form, shape [R,R]

    def materialize(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = world[self.refs]
        z = self.coeff @ values
        return self.h.copy(), z

    def digest(self) -> str:
        x = hashlib.sha256()
        x.update(self.h.tobytes()); x.update(self.refs.tobytes()); x.update(self.coeff.tobytes())
        return x.hexdigest()


class Layer:
    kind: str
    def lift(self, p: PNS) -> PNS: raise NotImplementedError
    def numeric(self, h: np.ndarray, z: np.ndarray) -> tuple[np.ndarray,np.ndarray]: raise NotImplementedError


@dataclass
class StaticMLP(Layer):
    w: np.ndarray
    b: np.ndarray
    kind: str = "static_mlp"
    def lift(self, p: PNS) -> PNS:
        return PNS(np.tanh(self.w @ p.h + self.b), p.refs.copy(), p.coeff.copy())
    def numeric(self, h, z):
        return np.tanh(self.w @ h + self.b), z.copy()


@dataclass
class RefLinear(Layer):
    a: np.ndarray
    kind: str = "ref_linear"
    def lift(self, p: PNS) -> PNS:
        return PNS(p.h.copy(), p.refs.copy(), self.a @ p.coeff)
    def numeric(self, h, z):
        return h.copy(), self.a @ z


@dataclass
class RefResidual(Layer):
    a: np.ndarray
    kind: str = "ref_residual"
    def lift(self, p: PNS) -> PNS:
        return PNS(p.h.copy(), p.refs.copy(), p.coeff + self.a @ p.coeff)
    def numeric(self, h, z):
        return h.copy(), z + self.a @ z


@dataclass
class HControlledScale(Layer):
    u: np.ndarray                 # [R,H]
    bias: np.ndarray              # [R]
    kind: str = "h_controlled_ref_scale"
    def scales(self, h):
        x = self.u @ h + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
    def lift(self, p: PNS) -> PNS:
        s = self.scales(p.h)
        return PNS(p.h.copy(), p.refs.copy(), s[:,None] * p.coeff)
    def numeric(self, h, z):
        return h.copy(), self.scales(h) * z


@dataclass
class RefPermute(Layer):
    perm: np.ndarray
    kind: str = "ref_permute_output"
    def lift(self, p: PNS) -> PNS:
        return PNS(p.h.copy(), p.refs.copy(), p.coeff[self.perm])
    def numeric(self, h, z):
        return h.copy(), z[self.perm]


def make_layer(rng: np.random.Generator, py: random.Random) -> Layer:
    kind = py.randrange(5)
    if kind == 0:
        return StaticMLP(rng.normal(0, 0.22, (H,H)), rng.normal(0, 0.08, H))
    if kind == 1:
        return RefLinear(rng.normal(0, 0.22, (R,R)))
    if kind == 2:
        return RefResidual(rng.normal(0, 0.08, (R,R)))
    if kind == 3:
        return HControlledScale(rng.normal(0, 0.16, (R,H)), rng.normal(0,0.1,R))
    return RefPermute(rng.permutation(R))


def make_pns(rng: np.random.Generator) -> PNS:
    h = rng.normal(0, 1, H)
    refs = rng.choice(WORLD_CELLS, size=R, replace=False).astype(np.int64)
    coeff = rng.normal(0, 0.5, (R,R))
    return PNS(h, refs, coeff)


def verify_program(p0: PNS, layers: list[Layer], worlds: list[np.ndarray]):
    p = p0
    for layer in layers:
        p = layer.lift(p)
    max_err = 0.0
    mismatch = 0
    for world in worlds:
        h0,z0 = p0.materialize(world)
        h,z = h0,z0
        for layer in layers:
            h,z = layer.numeric(h,z)
        hp,zp = p.materialize(world)
        max_err = max(max_err, float(np.max(np.abs(h-hp))), float(np.max(np.abs(z-zp))))
        # Treat combined state argmax as a simple downstream observable.
        mismatch += int(int(np.argmax(np.concatenate([h,z]))) != int(np.argmax(np.concatenate([hp,zp]))))
    return p, max_err, mismatch


def retrospective_control(p0: PNS, layers: list[Layer], old_world: np.ndarray, future_worlds: list[np.ndarray], rng: np.random.Generator):
    # Execute world-natural prefix, then prematurely fuse world values into h.
    cut = len(layers) // 2
    p = p0
    for layer in layers[:cut]:
        p = layer.lift(p)
    h_old,z_old = p.materialize(old_world)
    fuse = rng.normal(0, 0.35, (H,R))
    cached_h = np.tanh(h_old + fuse @ z_old)
    readout = rng.normal(0, 0.35, (7,H))
    stale_mismatches = 0
    max_stale_delta = 0.0
    for w in future_worlds:
        h_cur,z_cur = p.materialize(w)
        fresh_h = np.tanh(h_cur + fuse @ z_cur)
        fresh = readout @ fresh_h
        stale = readout @ cached_h
        max_stale_delta = max(max_stale_delta, float(np.max(np.abs(fresh-stale))))
        stale_mismatches += int(int(np.argmax(fresh)) != int(np.argmax(stale)))
    return stale_mismatches, max_stale_delta


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    py = random.Random(SEED)
    all_max_err = 0.0
    all_mismatch = 0
    retrospective_mismatch = 0
    retrospective_max_delta = 0.0
    digests_changed = 0
    layer_hist = {k:0 for k in ["static_mlp","ref_linear","ref_residual","h_controlled_ref_scale","ref_permute_output"]}
    timings = []

    for _ in range(PROGRAMS):
        p0 = make_pns(rng)
        layers = [make_layer(rng,py) for _ in range(LAYERS)]
        for l in layers: layer_hist[l.kind] += 1
        worlds = [rng.normal(0,1,WORLD_CELLS) for _ in range(WORLDS_PER_PROGRAM)]
        before = p0.digest()
        ts = time.perf_counter_ns()
        p, err, mis = verify_program(p0,layers,worlds)
        timings.append(time.perf_counter_ns()-ts)
        all_max_err = max(all_max_err,err); all_mismatch += mis
        # Original PNS bytes are not touched by future worlds.
        digests_changed += int(before != p0.digest())
        sm,sd = retrospective_control(p0,layers,worlds[0],worlds[1:],rng)
        retrospective_mismatch += sm; retrospective_max_delta=max(retrospective_max_delta,sd)

    # Composition law is the main gate: every randomly composed lifted prefix must
    # commute with materialization for every tested future world.
    report = {
        "stage": STAGE,
        "architecture_candidate": "World-Natural Neural Layer Algebra",
        "programs": PROGRAMS,
        "layers_per_program": LAYERS,
        "future_worlds_per_program": WORLDS_PER_PROGRAM,
        "layer_histogram": layer_hist,
        "max_commuting_square_abs_error": all_max_err,
        "world_natural_downstream_observable_mismatches": all_mismatch,
        "retained_pns_digest_changes_from_world_evaluation": digests_changed,
        "retrospective_early_fusion_downstream_mismatches": retrospective_mismatch,
        "retrospective_early_fusion_max_logit_delta": retrospective_max_delta,
        "median_verify_program_ns_python": statistics.median(timings),
        "materialization_boundary_rule": (
            "A block may remain before the mutable-world barrier iff a lifted implementation satisfies M_W(L^(K)) = L(M_W(K)) "
            "for every legal W. Composition of such blocks preserves the law; a decoder-visible early fusion of historical value payload does not."
        ),
        "mechanism": (
            "R355 treats PNS safety as a layer interface law rather than a hand-chosen split point. World-independent nonlinear continuation, "
            "reference transforms, residual reference mixing and h-controlled reference scaling are lifted without dereference and mechanically "
            "checked against their numeric materialized counterparts over arbitrary future worlds."
        ),
        "claim_boundary": (
            "Commuting diagrams/naturality, lifted operators, partial evaluation and staged computation are established. This is a PNS compiler "
            "formalism and falsification framework, not a standalone novelty claim."
        ),
        "dod_status": "NOT_DOD; compositional PNS layer-law gate",
    }
    report["contract_pass"] = (
        all_max_err <= TOL and all_mismatch == 0 and digests_changed == 0 and retrospective_mismatch > 0
    )
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode(); report["report_sha256"]=hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if report["contract_pass"] else 2

if __name__ == "__main__": raise SystemExit(main())
