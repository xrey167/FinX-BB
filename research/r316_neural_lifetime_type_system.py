from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path


STAGE = "R316-NEURAL-LIFETIME-TYPE-SYSTEM"
PROGRAMS = int(os.environ.get("SO_R316_PROGRAMS", "250000"))
REPORT_PATH = Path(os.environ.get("SO_R316_REPORT", "r316_report.json"))
SEED = 3160914


class TypeErrorCKVM(Exception):
    pass


@dataclass(frozen=True)
class Ty:
    domain: str  # static | plan | borrowed | sealed
    plan_atoms: frozenset[str] = frozenset()
    gen_atoms: frozenset[str] = frozenset()
    sealed_atoms: frozenset[str] = frozenset()


def join(a: Ty, b: Ty) -> Ty:
    plan = a.plan_atoms | b.plan_atoms
    gens = a.gen_atoms | b.gen_atoms
    sealed = a.sealed_atoms | b.sealed_atoms
    if gens:
        return Ty("borrowed", plan, gens, sealed)
    if sealed:
        return Ty("sealed", plan, frozenset(), sealed)
    if plan:
        return Ty("plan", plan)
    return Ty("static")


def check(program: list[tuple]) -> None:
    regs: dict[str, Ty] = {}
    committed: frozenset[str] | None = None
    for ins in program:
        op = ins[0]
        if op == "CONST":
            _, dst = ins
            regs[dst] = Ty("static")
        elif op == "PLAN":
            _, dst, atom = ins
            regs[dst] = Ty("plan", frozenset([atom]))
        elif op == "READ":
            _, dst, plan_reg, gen_atom = ins
            p = regs[plan_reg]
            if p.gen_atoms:
                raise TypeErrorCKVM("plan register already value-tainted")
            regs[dst] = Ty("borrowed", p.plan_atoms, frozenset([gen_atom]))
        elif op == "COMBINE":
            _, dst, x, y = ins
            regs[dst] = join(regs[x], regs[y])
        elif op == "SEAL":
            _, dst, src, claimed = ins
            t = regs[src]
            claimed = frozenset(claimed)
            if t.domain != "borrowed" or claimed != t.gen_atoms:
                raise TypeErrorCKVM("lifetime seal must cover exact generation read-set")
            regs[dst] = Ty("sealed", t.plan_atoms, frozenset(), claimed)
        elif op == "COMMIT":
            _, claimed = ins
            claimed = frozenset(claimed)
            # A static checker cannot prove currentness, but it can prove that a
            # commit barrier is supplied the exact dynamic generation read-set.
            live_borrowed = frozenset().union(*(t.gen_atoms for t in regs.values()))
            if claimed != live_borrowed:
                raise TypeErrorCKVM("commit barrier read-set mismatch")
            committed = claimed
        elif op == "CACHE_B":
            _, src = ins
            t = regs[src]
            if t.gen_atoms or t.sealed_atoms:
                raise TypeErrorCKVM("mutable knowledge may not enter reusable B cache")
        elif op == "RETAIN_J":
            _, src = ins
            t = regs[src]
            if t.domain != "sealed" or not t.sealed_atoms:
                raise TypeErrorCKVM("retained J state requires sealed post-commit lifetime")
        elif op == "PUBLISH":
            _, src = ins
            t = regs[src]
            needed = t.gen_atoms if t.domain == "borrowed" else t.sealed_atoms
            if needed and (committed is None or not needed.issubset(committed)):
                raise TypeErrorCKVM("publication requires matching Neural Commit Barrier")
        elif op == "ERASE_LIFETIME":
            raise TypeErrorCKVM("lifetime erasure/declassification is forbidden")
        else:
            raise TypeErrorCKVM(f"unknown opcode {op}")


def make_valid(rng: random.Random, idx: int) -> list[tuple]:
    p = [("PLAN", "p", f"alias:{idx % 97}:schema:{idx % 11}")]
    reads = rng.randint(1, 4)
    read_regs = []
    gens = []
    for i in range(reads):
        atom = f"pod:{rng.randrange(4096)}:g{rng.randrange(1, 100000)}"
        reg = f"r{i}"
        p.append(("READ", reg, "p", atom))
        read_regs.append(reg)
        gens.append(atom)
    cur = read_regs[0]
    for i, reg in enumerate(read_regs[1:], 1):
        dst = f"j{i}"
        p.append(("COMBINE", dst, cur, reg))
        cur = dst
    # Commit the exact dynamic read-set before any external publication.
    p.append(("COMMIT", tuple(gens)))
    sealed = "sealed"
    p.append(("SEAL", sealed, cur, tuple(gens)))
    p.append(("RETAIN_J", sealed))
    # Plans, but never value-bearing derivatives, are reusable in B.
    p.append(("CACHE_B", "p"))
    p.append(("PUBLISH", sealed))
    return p


def mutate(program: list[tuple], rng: random.Random) -> tuple[list[tuple], str]:
    q = list(program)
    mode = rng.randrange(6)
    if mode == 0:
        # Direct mutable leak into reusable B cache.
        src = next(ins[1] for ins in q if ins[0] == "READ")
        q.insert(-1, ("CACHE_B", src))
        return q, "borrowed_to_b_cache"
    if mode == 1:
        # Retain a borrowed neural derivative without attaching a lifetime factor.
        src = next(ins[1] for ins in q if ins[0] == "READ")
        q.insert(-1, ("RETAIN_J", src))
        return q, "retain_unsealed_j"
    if mode == 2:
        # Seal only a strict subset of the dynamic generations.
        si = next(i for i, ins in enumerate(q) if ins[0] == "SEAL")
        ins = q[si]
        claimed = list(ins[3])
        claimed = claimed[:-1] if len(claimed) > 1 else []
        q[si] = ("SEAL", ins[1], ins[2], tuple(claimed))
        return q, "incomplete_lifetime_factor"
    if mode == 3:
        # Commit a strict subset / empty read set.
        ci = next(i for i, ins in enumerate(q) if ins[0] == "COMMIT")
        claimed = list(q[ci][1])
        claimed = claimed[:-1] if len(claimed) > 1 else []
        q[ci] = ("COMMIT", tuple(claimed))
        return q, "incomplete_commit_readset"
    if mode == 4:
        # Remove commit entirely while publishing generation-derived state.
        q = [ins for ins in q if ins[0] != "COMMIT"]
        return q, "publish_without_commit"
    # Attempt explicit declassification.
    q.insert(-1, ("ERASE_LIFETIME", "sealed"))
    return q, "lifetime_erasure"


def main() -> None:
    rng = random.Random(SEED)
    valid_rejects = 0
    mutant_accepts = 0
    mutant_types: dict[str, int] = {}
    t0 = time.perf_counter_ns()
    for i in range(PROGRAMS):
        p = make_valid(rng, i)
        try:
            check(p)
        except TypeErrorCKVM:
            valid_rejects += 1
        m, kind = mutate(p, rng)
        mutant_types[kind] = mutant_types.get(kind, 0) + 1
        try:
            check(m)
            mutant_accepts += 1
        except TypeErrorCKVM:
            pass
    elapsed_ns = time.perf_counter_ns() - t0
    total_checks = PROGRAMS * 2
    report = {
        "stage": STAGE,
        "architecture_candidate": "Neural Lifetime Type System (NLTS) / CKVM temporal borrow checker",
        "mechanism": (
            "Every CKVM register is statically classified as static/plan/borrowed/sealed. Generation-bearing values "
            "cannot enter reusable B-plane caches, retained J artifacts cannot escape a transaction without an exact "
            "lifetime seal, and publication requires a commit barrier covering the dynamic generation read-set."
        ),
        "programs_valid": PROGRAMS,
        "programs_mutated": PROGRAMS,
        "mutant_type_counts": mutant_types,
        "valid_program_false_rejects": valid_rejects,
        "lifecycle_leak_mutants_false_accepts": mutant_accepts,
        "lifecycle_leak_detection_rate": 1.0 - mutant_accepts / PROGRAMS,
        "mean_typecheck_ns_per_program": elapsed_ns / total_checks,
        "dod_status": "NOT_DOD; static lifecycle-safety construction gate",
        "claim_boundary": (
            "Type systems, taint tracking, affine/borrow checking, and information-flow control are established. "
            "R316 tests a CKVM-specific rule set for neural artifact lifetimes; it is not a standalone novelty claim."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
