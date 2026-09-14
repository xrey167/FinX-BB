from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

STAGE = "R327-CAUSAL-PREDICATE-GENERATIONS"
COMPANIES = int(os.environ.get("SO_R327_COMPANIES", "2500"))
SUPPLIERS = int(os.environ.get("SO_R327_SUPPLIERS", "12000"))
EDGES_PER_COMPANY = int(os.environ.get("SO_R327_EDGES", "12"))
CACHED_QUERIES = int(os.environ.get("SO_R327_CACHES", "80000"))
MUTATIONS = int(os.environ.get("SO_R327_MUTATIONS", "30000"))
SERVE_PROBES = int(os.environ.get("SO_R327_SERVES", "240000"))
REPORT_PATH = Path(os.environ.get("SO_R327_REPORT", "r327_report.json"))
SEED = 3270914


@dataclass
class Supplier:
    risk: int
    generation: int = 1


@dataclass
class RelationSet:
    members: set[int]
    generation: int = 1


@dataclass
class CachedQuery:
    company: int
    relation_generation: int
    member_generations: tuple[tuple[int, int], ...]
    result: int


class World:
    def __init__(self, rng: random.Random) -> None:
        self.suppliers = [Supplier(rng.randrange(1000)) for _ in range(SUPPLIERS)]
        self.relations = []
        for _ in range(COMPANIES):
            self.relations.append(RelationSet(set(rng.sample(range(SUPPLIERS), EDGES_PER_COMPANY))))

    def evaluate(self, company: int) -> tuple[int, tuple[tuple[int, int], ...], int]:
        rel = self.relations[company]
        deps = tuple(sorted((sid, self.suppliers[sid].generation) for sid in rel.members))
        result = max((self.suppliers[sid].risk for sid in rel.members), default=-1)
        return rel.generation, deps, result

    def cache(self, company: int) -> CachedQuery:
        rg, deps, result = self.evaluate(company)
        return CachedQuery(company, rg, deps, result)

    def naive_member_valid(self, q: CachedQuery) -> bool:
        return all(self.suppliers[sid].generation == g for sid, g in q.member_generations)

    def predicate_valid(self, q: CachedQuery) -> bool:
        rel = self.relations[q.company]
        return rel.generation == q.relation_generation and self.naive_member_valid(q)

    def mutate_risk(self, rng: random.Random, same_value: bool) -> tuple[int, int]:
        sid = rng.randrange(SUPPLIERS)
        s = self.suppliers[sid]
        old = s.generation
        old_risk = s.risk
        s.generation += 1
        s.risk = old_risk if same_value else rng.randrange(1000)
        return sid, old

    def insert_edge(self, rng: random.Random) -> tuple[int, int | None]:
        c = rng.randrange(COMPANIES)
        rel = self.relations[c]
        if len(rel.members) >= SUPPLIERS:
            return c, None
        sid = rng.randrange(SUPPLIERS)
        for _ in range(30):
            if sid not in rel.members:
                break
            sid = rng.randrange(SUPPLIERS)
        if sid in rel.members:
            return c, None
        rel.members.add(sid)
        rel.generation += 1
        return c, sid

    def delete_edge(self, rng: random.Random) -> tuple[int, int | None]:
        c = rng.randrange(COMPANIES)
        rel = self.relations[c]
        if not rel.members:
            return c, None
        sid = rng.choice(tuple(rel.members))
        rel.members.remove(sid)
        rel.generation += 1
        return c, sid


def main() -> None:
    rng = random.Random(SEED)
    world = World(rng)
    caches = [world.cache(rng.randrange(COMPANIES)) for _ in range(CACHED_QUERIES)]

    naive_semantic_false_serves = 0
    full_semantic_false_serves = 0
    naive_lifetime_false_accepts = 0
    full_lifetime_false_accepts = 0
    correct_live_serves = 0
    relation_mutations = 0
    risk_mutations = 0
    same_value_risk_rewrites = 0
    probes = 0
    mutation_ns = []
    validity_ns = []

    t0 = time.perf_counter()
    for m in range(MUTATIONS):
        r = rng.random()
        ts = time.perf_counter_ns()
        if r < 0.35:
            world.insert_edge(rng)
            relation_mutations += 1
        elif r < 0.65:
            world.delete_edge(rng)
            relation_mutations += 1
        else:
            same = rng.random() < 0.45
            world.mutate_risk(rng, same)
            risk_mutations += 1
            same_value_risk_rewrites += int(same)
        mutation_ns.append(time.perf_counter_ns() - ts)

        # Sample several retained neural/semantic graph query artifacts after each
        # mutation. This approximates admission from a long-lived cache population.
        local = max(1, SERVE_PROBES // MUTATIONS)
        for _ in range(local):
            q = caches[rng.randrange(CACHED_QUERIES)]
            ts = time.perf_counter_ns()
            nv = world.naive_member_valid(q)
            fv = world.predicate_valid(q)
            validity_ns.append(time.perf_counter_ns() - ts)
            cur_rg, cur_deps, cur_result = world.evaluate(q.company)
            semantic_stale = q.result != cur_result
            lifetime_stale = q.relation_generation != cur_rg or q.member_generations != cur_deps

            if nv and semantic_stale:
                naive_semantic_false_serves += 1
            if fv and semantic_stale:
                full_semantic_false_serves += 1
            if nv and lifetime_stale:
                naive_lifetime_false_accepts += 1
            if fv and lifetime_stale:
                full_lifetime_false_accepts += 1
            if fv and not lifetime_stale:
                correct_live_serves += 1
            probes += 1

        # Replace a small number of random cache entries with current snapshots so
        # the population contains both old and fresh artifacts throughout the run.
        for _ in range(3):
            idx = rng.randrange(CACHED_QUERIES)
            caches[idx] = world.cache(caches[idx].company)

    elapsed = time.perf_counter() - t0
    report = {
        "stage": STAGE,
        "architecture_candidate": "Causal Predicate Generations (CPG) / phantom-safe CKVM graph reads",
        "companies": COMPANIES,
        "suppliers": SUPPLIERS,
        "initial_edges_per_company": EDGES_PER_COMPANY,
        "cached_queries": CACHED_QUERIES,
        "mutations": MUTATIONS,
        "relation_insert_delete_mutations": relation_mutations,
        "supplier_risk_mutations": risk_mutations,
        "same_value_risk_generation_rewrites": same_value_risk_rewrites,
        "serve_probes": probes,
        "naive_member_only_semantic_false_serves": naive_semantic_false_serves,
        "predicate_generation_semantic_false_serves": full_semantic_false_serves,
        "naive_member_only_lifetime_false_accepts": naive_lifetime_false_accepts,
        "predicate_generation_lifetime_false_accepts": full_lifetime_false_accepts,
        "correct_live_serves": correct_live_serves,
        "median_mutation_ns_python": statistics.median(mutation_ns),
        "median_naive_plus_predicate_validity_check_ns_python": statistics.median(validity_ns),
        "elapsed_seconds": elapsed,
        "contract_pass": full_semantic_false_serves == 0 and full_lifetime_false_accepts == 0 and naive_lifetime_false_accepts > 0,
        "mechanism": (
            "A graph traversal reads not only the generations of members it enumerated but also a generation for the relation/predicate set "
            "it traversed. Edge insert/delete increments that predicate generation. A cached neural result therefore expires when the set "
            "of possible members changes, even if none of the previously enumerated member value generations changed."
        ),
        "architectural_hypothesis": (
            "For mutable relational knowledge, exact neural lifetimes require predicate/set generations in addition to value generations. "
            "Otherwise edge insertion/deletion creates a phantom resurrection path analogous to database phantom reads."
        ),
        "dod_status": "NOT_DOD; relational-lifetime mechanism gate",
        "claim_boundary": (
            "Predicate/range versioning and phantom-read protection are established database ideas. R327 tests the corresponding generation "
            "lifetime requirement for CKVM graph-derived neural artifacts; standalone novelty is not claimed."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("predicate-generation graph contract failed")


if __name__ == "__main__":
    main()
