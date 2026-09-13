from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

OUT = Path(os.environ.get("SO_R310_REPORT", "ci-r310/report.json"))


@dataclass
class Page:
    generation: int
    value: int
    flag: bool


@dataclass(frozen=True)
class Program:
    op: int
    pods: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class State:
    output: int
    deps: tuple[tuple[int, int], ...]


class World:
    def __init__(self, n: int, rng: random.Random):
        self.pages = [Page(1, rng.randrange(100_000), bool(rng.getrandbits(1))) for _ in range(n)]

    def update(self, pid: int, rng: random.Random):
        p = self.pages[pid]
        p.generation += 1
        # Same-value/flag rewrites are intentionally common: generation identity still changes.
        if rng.random() >= 0.25:
            p.value = rng.randrange(100_000)
        if rng.random() >= 0.25:
            p.flag = bool(rng.getrandbits(1))


def dep(world: World, pid: int) -> tuple[int, int]:
    return (pid, world.pages[pid].generation)


def execute_lazy(world: World, p: Program) -> State:
    a,b,c,d,e=p.pods
    if p.op == 0:  # direct A
        return State(world.pages[a].value, (dep(world,a),))
    if p.op == 1:  # flag(A) ? B : C
        if world.pages[a].flag:
            return State(world.pages[b].value, tuple(sorted((dep(world,a),dep(world,b)))))
        return State(world.pages[c].value, tuple(sorted((dep(world,a),dep(world,c)))))
    if p.op == 2:  # A==B ? C : A
        da,db=dep(world,a),dep(world,b)
        if world.pages[a].value == world.pages[b].value:
            return State(world.pages[c].value, tuple(sorted((da,db,dep(world,c)))))
        return State(world.pages[a].value, tuple(sorted((da,db))))
    if p.op == 3:  # flag(A) ? (flag(B)?C:D) : E
        deps=[dep(world,a)]
        if world.pages[a].flag:
            deps.append(dep(world,b))
            if world.pages[b].flag:
                deps.append(dep(world,c)); out=world.pages[c].value
            else:
                deps.append(dep(world,d)); out=world.pages[d].value
        else:
            deps.append(dep(world,e)); out=world.pages[e].value
        return State(out, tuple(sorted(set(deps))))
    if p.op == 4:  # if A==B then (flag(C)?D:E) else B
        deps=[dep(world,a),dep(world,b)]
        if world.pages[a].value == world.pages[b].value:
            deps.append(dep(world,c))
            if world.pages[c].flag:
                deps.append(dep(world,d));out=world.pages[d].value
            else:
                deps.append(dep(world,e));out=world.pages[e].value
        else:
            out=world.pages[b].value
        return State(out, tuple(sorted(set(deps))))
    raise ValueError(p.op)


def execute_eager(world: World, p: Program) -> State:
    # Same semantic execution, but materialize every potential operand first.
    a,b,c,d,e=p.pods
    deps=tuple(sorted({dep(world,x) for x in p.pods}))
    va,vb,vc,vd,ve=(world.pages[x].value for x in p.pods)
    fa,fb,fc=(world.pages[x].flag for x in (a,b,c))
    if p.op==0:out=va
    elif p.op==1:out=vb if fa else vc
    elif p.op==2:out=vc if va==vb else va
    elif p.op==3:out=(vc if fb else vd) if fa else ve
    elif p.op==4:out=(vd if fc else ve) if va==vb else vb
    else:raise ValueError(p.op)
    return State(out,deps)


def current_valid(world: World, s: State) -> bool:
    return all(world.pages[pid].generation==gen for pid,gen in s.deps)


def build_states(world,programs,executor):
    states=[];rev=defaultdict(set);dep_counts=[]
    for sid,p in enumerate(programs):
        s=executor(world,p);states.append(s);dep_counts.append(len(s.deps))
        for d in s.deps:rev[d].add(sid)
    return states,rev,dep_counts


def main()->int:
    OUT.parent.mkdir(parents=True,exist_ok=True)
    rng=random.Random(310)
    pod_count=50_000
    program_count=200_000
    world=World(pod_count,rng)
    programs=[]
    for _ in range(program_count):
        programs.append(Program(rng.randrange(5),tuple(rng.sample(range(pod_count),5))))

    started=time.perf_counter()
    epochs=20;updates_per_epoch=100
    lazy_invalid_fracs=[];eager_invalid_fracs=[];lazy_dep_means=[];eager_dep_means=[]
    output_mismatches=0;lazy_safety_mismatches=0;eager_safety_mismatches=0
    same_value_rewrite_witnesses=0
    lazy_build_seconds=[];eager_build_seconds=[]

    for epoch in range(epochs):
        t=time.perf_counter();lazy_states,lazy_rev,lazy_deps=build_states(world,programs,execute_lazy);lazy_build_seconds.append(time.perf_counter()-t)
        t=time.perf_counter();eager_states,eager_rev,eager_deps=build_states(world,programs,execute_eager);eager_build_seconds.append(time.perf_counter()-t)
        lazy_dep_means.append(statistics.mean(lazy_deps));eager_dep_means.append(statistics.mean(eager_deps))
        output_mismatches += sum(int(a.output!=b.output) for a,b in zip(lazy_states,eager_states))

        lazy_invalid=set();eager_invalid=set()
        for _ in range(updates_per_epoch):
            pid=rng.randrange(pod_count);old_gen=world.pages[pid].generation;old_value=world.pages[pid].value;old_flag=world.pages[pid].flag
            old_key=(pid,old_gen)
            lazy_invalid.update(lazy_rev.get(old_key,()))
            eager_invalid.update(eager_rev.get(old_key,()))
            world.update(pid,rng)
            if world.pages[pid].value==old_value and world.pages[pid].flag==old_flag:
                # Every old state depending on this generation must still become invalid despite same semantics.
                if lazy_rev.get(old_key):
                    same_value_rewrite_witnesses += int(all(not current_valid(world,lazy_states[sid]) for sid in list(lazy_rev[old_key])[:50]))

        lazy_invalid_fracs.append(len(lazy_invalid)/program_count);eager_invalid_fracs.append(len(eager_invalid)/program_count)

        # Audit marked-vs-explicit generation validity and prove unmarked cached results remain semantically current.
        audit_ids=rng.sample(range(program_count),10_000)
        for sid in audit_ids:
            lv=current_valid(world,lazy_states[sid]);ev=current_valid(world,eager_states[sid])
            lazy_safety_mismatches += int((sid not in lazy_invalid) != lv)
            eager_safety_mismatches += int((sid not in eager_invalid) != ev)
            if lv:
                fresh=execute_lazy(world,programs[sid])
                lazy_safety_mismatches += int(fresh.output!=lazy_states[sid].output)
            if ev:
                fresh=execute_eager(world,programs[sid])
                eager_safety_mismatches += int(fresh.output!=eager_states[sid].output)

    mean_lazy_invalid=statistics.mean(lazy_invalid_fracs);mean_eager_invalid=statistics.mean(eager_invalid_fracs)
    report={
        "stage":"R310-CKVM-LAZY-READSETS",
        "architecture_candidate":"Causal Knowledge Virtual Machine (CKVM) / Late-Bound K-SSA",
        "pod_count":pod_count,"program_count":program_count,"epochs":epochs,"updates_per_epoch":updates_per_epoch,"total_world_updates":epochs*updates_per_epoch,
        "eager_vs_lazy_output_mismatches":output_mismatches,
        "lazy_validity_or_output_audit_mismatches":lazy_safety_mismatches,"eager_validity_or_output_audit_mismatches":eager_safety_mismatches,
        "mean_lazy_dependency_count":statistics.mean(lazy_dep_means),"mean_eager_dependency_count":statistics.mean(eager_dep_means),
        "dependency_edge_reduction_fraction":1.0-statistics.mean(lazy_dep_means)/statistics.mean(eager_dep_means),
        "mean_lazy_invalidated_state_fraction_per_100_updates":mean_lazy_invalid,
        "mean_eager_invalidated_state_fraction_per_100_updates":mean_eager_invalid,
        "invalidation_fanout_reduction_fraction":1.0-mean_lazy_invalid/max(mean_eager_invalid,1e-12),
        "same_value_generation_rewrite_witnesses":same_value_rewrite_witnesses,
        "median_lazy_build_seconds":statistics.median(lazy_build_seconds),"median_eager_build_seconds":statistics.median(eager_build_seconds),
        "elapsed_seconds":time.perf_counter()-started,
        "mechanism":(
            "CKVM executes typed K-SSA lazily. Only generations on the actually executed control-flow path become source "
            "lifetimes of a J state. Eager execution materializes every potential branch operand and therefore creates a "
            "conservative dependency superset. Both are safe; late binding aims to reduce invalidation fanout and stale-neural surface."
        ),
        "claim_boundary":"SSA, lazy evaluation and dynamic read sets are established. This gate tests their lifecycle cost inside CKCA, not standalone novelty.",
        "dod_status":"NOT_DOD; CKVM runtime scalability gate",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode();report["report_sha256"]=hashlib.sha256(canonical).hexdigest();OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
    if output_mismatches!=0:return 2
    if lazy_safety_mismatches!=0 or eager_safety_mismatches!=0:return 3
    if report["dependency_edge_reduction_fraction"]<=0:return 4
    if report["invalidation_fanout_reduction_fraction"]<=0:return 5
    if same_value_rewrite_witnesses==0:return 6
    return 0

if __name__=="__main__":raise SystemExit(main())
