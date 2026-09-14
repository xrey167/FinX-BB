from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

STAGE = "R356-TRAINABLE-REFERENCE-GENERATING-PNS"
REPORT_PATH = Path(os.environ.get("SO_R356_REPORT", "ci-r356/report.json"))
SEED = 3560914
PODS = int(os.environ.get("SO_R356_PODS", "256"))
SEM = 32
QDIM = 48
MAX_HOPS = 6
VALUE_COUNT = 64
TRAIN_STEPS = int(os.environ.get("SO_R356_TRAIN_STEPS", "1600"))
BATCH = int(os.environ.get("SO_R356_BATCH", "512"))
DESCRIPTORS = int(os.environ.get("SO_R356_DESCRIPTORS", "6000"))
FUTURE_WORLDS = int(os.environ.get("SO_R356_FUTURE_WORLDS", "64"))
UPDATES = int(os.environ.get("SO_R356_UPDATES", "30000"))


@dataclass
class World:
    next_pod: torch.Tensor
    payload: torch.Tensor
    generation: torch.Tensor


def make_world(g: torch.Generator) -> World:
    return World(
        torch.randint(0, PODS, (PODS,), generator=g),
        torch.randint(0, VALUE_COUNT, (PODS,), generator=g),
        torch.ones(PODS, dtype=torch.long),
    )


def follow(world: World, start: torch.Tensor, hops: torch.Tensor) -> tuple[torch.Tensor, list[list[tuple[int,int]]]]:
    out = torch.empty_like(start)
    deps: list[list[tuple[int,int]]] = []
    for i in range(len(start)):
        pid = int(start[i]); path=[]
        for _ in range(int(hops[i])):
            path.append((pid, int(world.generation[pid])))
            pid = int(world.next_pod[pid])
        path.append((pid, int(world.generation[pid])))
        out[i] = world.payload[pid]
        deps.append(path)
    return out, deps


def descriptor_digest(start, hops):
    return hashlib.sha256(start.to(torch.int16).numpy().tobytes()+hops.to(torch.int8).numpy().tobytes()).hexdigest()


class ProgramCompiler(nn.Module):
    """Learned query -> (start reference, hop count) residual address program."""
    def __init__(self, keys: torch.Tensor):
        super().__init__(); self.register_buffer("keys", keys)
        self.start_head = nn.Sequential(nn.Linear(QDIM,96),nn.GELU(),nn.Linear(96,SEM))
        self.key_proj = nn.Linear(SEM,SEM,bias=False)
        self.hop_head = nn.Sequential(nn.Linear(QDIM,64),nn.GELU(),nn.Linear(64,MAX_HOPS))
    def forward(self,q):
        k=F.normalize(self.key_proj(self.keys),dim=-1)
        qq=F.normalize(self.start_head(q),dim=-1)
        return qq@k.T*12.0, self.hop_head(q)
    @torch.inference_mode()
    def compile(self,q):
        s,h=self(q); return s.argmax(-1), h.argmax(-1)+1


class SnapshotFinalRef(nn.Module):
    """Retrospective control trained to emit the final ref for one home topology."""
    def __init__(self, keys: torch.Tensor):
        super().__init__(); self.register_buffer("keys",keys)
        self.head=nn.Sequential(nn.Linear(QDIM,128),nn.GELU(),nn.Linear(128,SEM))
        self.key_proj=nn.Linear(SEM,SEM,bias=False)
    def forward(self,q):
        k=F.normalize(self.key_proj(self.keys),dim=-1); x=F.normalize(self.head(q),dim=-1)
        return x@k.T*12.0


def final_ref(world: World, start: torch.Tensor, hops: torch.Tensor):
    out=torch.empty_like(start)
    for i in range(len(start)):
        pid=int(start[i])
        for _ in range(int(hops[i])): pid=int(world.next_pod[pid])
        out[i]=pid
    return out


def main()->int:
    REPORT_PATH.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","2")))
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED); rng=random.Random(SEED); g=torch.Generator().manual_seed(SEED)

    keys=F.normalize(torch.randn(PODS,SEM,generator=g),dim=-1)
    hop_seed=torch.randn(MAX_HOPS,12,generator=g)
    mix=torch.randn(SEM+12,QDIM,generator=g)/(SEM**0.5)
    def qvec(start,hops,noise=True):
        raw=torch.cat([keys[start],hop_seed[hops-1]],-1); q=torch.tanh(raw@mix)
        if noise:q=q+0.015*torch.randn(q.shape,generator=g)
        return q

    home=make_world(g)
    compiler=ProgramCompiler(keys); snap=SnapshotFinalRef(keys)
    opt=torch.optim.AdamW(list(compiler.parameters())+list(snap.parameters()),lr=2.2e-3,weight_decay=1e-4)
    for _ in range(TRAIN_STEPS):
        start=torch.randint(0,PODS,(BATCH,),generator=g); hops=torch.randint(1,MAX_HOPS+1,(BATCH,),generator=g)
        q=qvec(start,hops)
        ls,lh=compiler(q); target_final=final_ref(home,start,hops); lf=snap(q)
        loss=F.cross_entropy(ls,start)+F.cross_entropy(lh,hops-1)+F.cross_entropy(lf,target_final)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(list(compiler.parameters())+list(snap.parameters()),1.0);opt.step()
    compiler.eval();snap.eval()

    start=torch.randint(0,PODS,(DESCRIPTORS,),generator=g);hops=torch.randint(1,MAX_HOPS+1,(DESCRIPTORS,),generator=g);q=qvec(start,hops,False)
    with torch.inference_mode():
        ps,ph=compiler.compile(q); snapshot_ref=snap(q).argmax(-1)
    program_acc=float((ps.eq(start)&ph.eq(hops)).float().mean())
    snapshot_home_acc=float(snapshot_ref.eq(final_ref(home,start,hops)).float().mean())
    digest_before=descriptor_digest(ps,ph)

    future_program=[];future_snapshot=[];exact_payload=[]
    for _ in range(FUTURE_WORLDS):
        w=make_world(g)
        truth_ref=final_ref(w,start,hops); truth_payload=w.payload[truth_ref]
        prog_payload,_=follow(w,ps,ph); snap_payload=w.payload[snapshot_ref]
        future_program.append(float(prog_payload.eq(truth_payload).float().mean()))
        future_snapshot.append(float(snap_payload.eq(truth_payload).float().mean()))
        exact_payload.append(int(prog_payload.eq(truth_payload).sum()))

    # Update a single live world for race and write-maintenance analysis.
    world=make_world(g);same_target=0;same_payload=0
    snapshot_patch_counterfactual=0
    # Snapshot final references would have to be reconsidered after every topology write in worst case.
    for _ in range(UPDATES):
        pid=rng.randrange(PODS); world.generation[pid]+=1
        if rng.random()<0.55:
            if rng.random()<0.30:same_target+=1
            else:world.next_pod[pid]=rng.randrange(PODS)
            snapshot_patch_counterfactual+=DESCRIPTORS
        else:
            if rng.random()<0.35:same_payload+=1
            else:world.payload[pid]=rng.randrange(VALUE_COUNT)
    digest_after=descriptor_digest(ps,ph)

    expected=detected=escaped=retries=mismatch=0
    for _ in range(12000):
        i=rng.randrange(DESCRIPTORS); pstart=ps[i:i+1];phops=ph[i:i+1]
        cand,deps=follow(world,pstart,phops);seen=deps[0]
        raced=rng.random()<0.09
        if raced:
            pid,_=seen[rng.randrange(len(seen))];world.generation[pid]+=1;expected+=1
            if rng.random()<0.5:world.next_pod[pid]=rng.randrange(PODS)
            elif rng.random()<0.5:world.payload[pid]=rng.randrange(VALUE_COUNT)
        valid=all(world.generation[pid]==gen for pid,gen in seen)
        if not valid:
            detected+=1;retries+=1;cand,_=follow(world,pstart,phops)
        elif raced:escaped+=1
        truth_payload,_=follow(world,start[i:i+1],hops[i:i+1]);mismatch+=int(int(cand[0])!=int(truth_payload[0]))

    report={
        "stage":STAGE,
        "architecture_candidate":"Trainable Reference-Generating Prospective Program (TRG-PNS)",
        "pods":PODS,"max_hops":MAX_HOPS,"descriptors":DESCRIPTORS,
        "program_descriptor_accuracy":program_acc,"snapshot_home_final_ref_accuracy":snapshot_home_acc,
        "mean_future_program_payload_accuracy":statistics.mean(future_program),
        "mean_future_snapshot_payload_accuracy":statistics.mean(future_snapshot),
        "future_gain_program_minus_snapshot":statistics.mean(future_program)-statistics.mean(future_snapshot),
        "descriptor_digest_unchanged_after_updates":digest_before==digest_after,
        "write_time_program_patches_or_invalidations":0,
        "snapshot_topology_patch_counterfactual":snapshot_patch_counterfactual,
        "world_updates":UPDATES,"same_target_generation_rewrites":same_target,"same_payload_generation_rewrites":same_payload,
        "race_conflicts_expected":expected,"race_conflicts_detected":detected,"race_conflicts_escaped":escaped,"race_retries":retries,
        "race_semantic_mismatches":mismatch,
        "mechanism":(
            "A trainable attention compiler emits an unresolved address-generating program (start PodRef + hop count), not the final address observed "
            "in the training topology. Serving follows the current authoritative pointer graph and returns the current payload. The matched neural control "
            "instead caches the final reference learned under one home topology."
        ),
        "claim_boundary":(
            "Neural program synthesis, pointer networks and graph traversal are established. R356 only shows that reference-generating PNS can be a learned "
            "neural descriptor rather than a hand-written program; it is not a standalone novelty claim."
        ),
        "dod_status":"NOT_DOD; learned dynamic-reference PNS gate",
    }
    report["contract_pass"]=(program_acc>=0.99 and statistics.mean(future_program)>=0.98 and digest_before==digest_after and detected==expected and escaped==0 and mismatch==0 and statistics.mean(future_snapshot)<0.90)
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode();report["report_sha256"]=hashlib.sha256(canonical).hexdigest();REPORT_PATH.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True));return 0 if report["contract_pass"] else 2

if __name__=="__main__":raise SystemExit(main())
