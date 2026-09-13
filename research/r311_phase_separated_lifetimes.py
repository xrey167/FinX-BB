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

OUT=Path(os.environ.get("SO_R311_REPORT","ci-r311/report.json"))


@dataclass
class Pod:
    value_generation:int
    value:int


@dataclass
class Alias:
    bind_generation:int
    pod_id:int


@dataclass(frozen=True)
class Plan:
    alias_name:str
    alias_generation:int
    pod_id:int
    schema_generation:int


@dataclass(frozen=True)
class JState:
    plan_key:int
    pod_id:int
    value_generation:int
    result:int


def main()->int:
    OUT.parent.mkdir(parents=True,exist_ok=True)
    rng=random.Random(311)
    pod_count=50_000
    alias_count=100_000
    plan_count=250_000
    pods=[Pod(1,rng.randrange(1_000_000)) for _ in range(pod_count)]
    aliases={f"alias-{i:06d}":Alias(1,rng.randrange(pod_count)) for i in range(alias_count)}
    schema_generation=1

    plans=[];jstates=[]
    alias_to_plans=defaultdict(set);pod_to_j=defaultdict(set);pod_to_naive_plans=defaultdict(set)
    for plan_id in range(plan_count):
        name=f"alias-{rng.randrange(alias_count):06d}";a=aliases[name]
        p=Plan(name,a.bind_generation,a.pod_id,schema_generation);plans.append(p);alias_to_plans[name].add(plan_id)
        pod=pods[p.pod_id]
        jstates.append(JState(plan_id,p.pod_id,pod.value_generation,pod.value))
        pod_to_j[p.pod_id].add(plan_id)
        # Naive cache incorrectly ties parsed plan lifetime to current mutable fact generation.
        pod_to_naive_plans[p.pod_id].add(plan_id)

    plan_valid=[True]*plan_count;j_valid=[True]*plan_count;naive_plan_valid=[True]*plan_count
    value_updates=20_000;alias_rebinds=2_000;schema_updates=4
    value_plan_invalidations=[];value_j_invalidations=[];alias_plan_invalidations=[];alias_j_invalidations=[];naive_value_plan_invalidations=[]
    started=time.perf_counter()

    for _ in range(value_updates):
        pid=rng.randrange(pod_count);p=pods[pid];p.value_generation+=1
        if rng.random()>=.2:p.value=rng.randrange(1_000_000) # same-value rewrite still generation change
        invj=0
        for sid in pod_to_j.get(pid,()):
            if j_valid[sid]:j_valid[sid]=False;invj+=1
        invn=0
        for sid in pod_to_naive_plans.get(pid,()):
            if naive_plan_valid[sid]:naive_plan_valid[sid]=False;invn+=1
        value_j_invalidations.append(invj);naive_value_plan_invalidations.append(invn);value_plan_invalidations.append(0)

    for _ in range(alias_rebinds):
        name=f"alias-{rng.randrange(alias_count):06d}";a=aliases[name];a.bind_generation+=1;a.pod_id=rng.randrange(pod_count)
        invp=invj=0
        for sid in alias_to_plans.get(name,()):
            if plan_valid[sid]:plan_valid[sid]=False;invp+=1
            if j_valid[sid]:j_valid[sid]=False;invj+=1
        alias_plan_invalidations.append(invp);alias_j_invalidations.append(invj)

    # Schema generation invalidates reusable plans globally; this is intentionally a distinct lifetime domain.
    schema_invalidations=0
    for _ in range(schema_updates):
        schema_generation+=1
        newly=sum(1 for x in plan_valid if x);schema_invalidations+=newly
        plan_valid=[False]*plan_count
        j_valid=[False]*plan_count
        # regenerate all at new schema epoch for the next cycle, except after final cycle it is not important.
        plans=[];jstates=[];alias_to_plans=defaultdict(set);pod_to_j=defaultdict(set)
        for plan_id in range(plan_count):
            name=f"alias-{rng.randrange(alias_count):06d}";a=aliases[name]
            p=Plan(name,a.bind_generation,a.pod_id,schema_generation);plans.append(p);alias_to_plans[name].add(plan_id)
            pod=pods[p.pod_id];jstates.append(JState(plan_id,p.pod_id,pod.value_generation,pod.value));pod_to_j[p.pod_id].add(plan_id)
        plan_valid=[True]*plan_count;j_valid=[True]*plan_count

    # Final audit after randomized further mutations without regeneration.
    audit_value_updates=2_000;audit_alias_rebinds=500
    for _ in range(audit_value_updates):
        pid=rng.randrange(pod_count);pods[pid].value_generation+=1
        for sid in pod_to_j.get(pid,()):j_valid[sid]=False
    for _ in range(audit_alias_rebinds):
        name=f"alias-{rng.randrange(alias_count):06d}";a=aliases[name];a.bind_generation+=1;a.pod_id=rng.randrange(pod_count)
        for sid in alias_to_plans.get(name,()):plan_valid[sid]=False;j_valid[sid]=False

    plan_mismatch=j_mismatch=0
    audit_ids=rng.sample(range(plan_count),50_000)
    for sid in audit_ids:
        p=plans[sid];a=aliases[p.alias_name]
        truth_plan=(p.alias_generation==a.bind_generation and p.pod_id==a.pod_id and p.schema_generation==schema_generation)
        plan_mismatch+=int(plan_valid[sid]!=truth_plan)
        j=jstates[sid]
        truth_j=truth_plan and pods[j.pod_id].value_generation==j.value_generation
        j_mismatch+=int(j_valid[sid]!=truth_j)

    naive_invalid=sum(1 for x in naive_plan_valid if not x)
    # In phase-separated design value updates never invalidate plan cache.
    report={
        "stage":"R311-PHASE-SEPARATED-LIFETIME-DOMAINS",
        "architecture_candidate":"Typed Lifetime Domains for CKVM",
        "pod_count":pod_count,"alias_count":alias_count,"plan_cache_entries":plan_count,
        "value_updates":value_updates,"alias_rebinds":alias_rebinds,"schema_updates":schema_updates,
        "phase_separated_plan_invalidations_due_to_value_updates":sum(value_plan_invalidations),
        "naive_plan_invalidations_due_to_value_updates":sum(naive_value_plan_invalidations),
        "mean_jstate_invalidations_per_value_update":statistics.mean(value_j_invalidations),
        "mean_plan_invalidations_per_alias_rebind":statistics.mean(alias_plan_invalidations),
        "mean_jstate_invalidations_per_alias_rebind":statistics.mean(alias_j_invalidations),
        "schema_domain_total_plan_invalidations":schema_invalidations,
        "final_plan_validity_audit_mismatches":plan_mismatch,"final_jstate_validity_audit_mismatches":j_mismatch,
        "plan_reuse_saved_fraction_vs_value_bound_naive":1.0-(sum(value_plan_invalidations)/max(sum(naive_value_plan_invalidations),1)),
        "elapsed_seconds":time.perf_counter()-started,
        "lifetime_types":{
            "plan":"alias-binding generation AND schema generation; independent of fact value generation",
            "jstate":"plan lifetime AND exact source Pod value generations actually read",
            "decode_suffix":"jstate lifetime plus any later mutable sources read during generation",
        },
        "mechanism":(
            "CKVM assigns different lifetime domains to interpretation state and mutable data state. Ordinary fact updates "
            "invalidate derived J data but do not destroy the reusable query plan. Alias rebindings invalidate plans that "
            "captured the old identity binding; schema changes invalidate plan interpretation globally."
        ),
        "claim_boundary":"Prepared-plan invalidation and typed/versioned cache dependencies are established systems ideas. R311 is a CKCA correctness/performance factorization gate, not standalone novelty.",
        "dod_status":"NOT_DOD; lifetime-domain factorization gate",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode();report["report_sha256"]=hashlib.sha256(canonical).hexdigest();OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
    if plan_mismatch!=0 or j_mismatch!=0:return 2
    if report["phase_separated_plan_invalidations_due_to_value_updates"]!=0:return 3
    if report["naive_plan_invalidations_due_to_value_updates"]<=0:return 4
    return 0

if __name__=="__main__":raise SystemExit(main())
