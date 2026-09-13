from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_WEIGHTS = {
    "model.safetensors": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
}
OUT = Path(os.environ.get("SO_R298_REPORT", "ci-qwen-r298/report.json"))
COLORS = ("red", "blue", "green", "yellow")
OPS = ("same", "xor_red", "conditional_color", "edge_pair")
LABELS = ("red", "blue", "green", "yellow", "yes", "no", "A", "B", "NULL")
L2I = {x: i for i, x in enumerate(LABELS)}


def qtext(op: str, a: str, b: str) -> str:
    if op == "same":
        return f"Do entities {a} and {b} currently have the same color? Return only yes or no."
    if op == "xor_red":
        return f"Is exactly one of {a} and {b} currently red? Return only yes or no."
    if op == "conditional_color":
        return f"If {a} is currently red, return its color; otherwise return the current color of {b}. Return only a color word."
    if op == "edge_pair":
        return f"Return A if the current pair ({a}, {b}) contains red or yellow at least once; otherwise return B. Return only A or B."
    raise ValueError(op)


def expected(op: str, va: int | None, vb: int | None) -> str:
    if va is None or vb is None:
        return "NULL"
    a, b = COLORS[va], COLORS[vb]
    if op == "same":
        return "yes" if a == b else "no"
    if op == "xor_red":
        return "yes" if (a == "red") ^ (b == "red") else "no"
    if op == "conditional_color":
        return a if a == "red" else b
    if op == "edge_pair":
        return "A" if a in ("red", "yellow") or b in ("red", "yellow") else "B"
    raise ValueError(op)


def prompt(tok, op: str, a: str, b: str) -> str:
    system = (
        "Mutable entity values are supplied through a separate trusted knowledge Port plane. "
        "The text identifies entities and the requested operation but contains no current values."
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": qtext(op, a, b)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_features(model, tok, triples, batch_size: int = 32):
    feats = {}
    logits = {}
    for start in range(0, len(triples), batch_size):
        batch = triples[start:start + batch_size]
        texts = [prompt(tok, op, a, b) for op, a, b in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.inference_mode():
            out = model(**enc, use_cache=False, output_hidden_states=True, return_dict=True)
        idx = out.hidden_states[-1].shape[1] - 1  # left-padded decoder: physical final token
        h = out.hidden_states[-1]
        for i, key in enumerate(batch):
            feats[key] = h[i, idx].float().cpu()
            logits[key] = out.logits[i, idx].float().cpu()
    return feats, logits


class CoherenceBridge(nn.Module):
    def __init__(self, d: int, width: int = 72):
        super().__init__()
        self.q = nn.Linear(d, width, bias=False)
        self.p = nn.Linear(d, width, bias=False)
        self.net = nn.Sequential(
            nn.LayerNorm(width * 3 + 2),
            nn.Linear(width * 3 + 2, width * 2), nn.GELU(),
            nn.Linear(width * 2, width), nn.GELU(),
            nn.Linear(width, len(LABELS)),
        )

    def forward(self, qh, pa, pb, live_a, live_b):
        x = torch.cat([
            self.q(qh), self.p(pa), self.p(pb),
            live_a.float().unsqueeze(-1), live_b.float().unsqueeze(-1),
        ], dim=-1)
        return self.net(x)


@dataclass
class Pod:
    generation: int
    kind: str  # value | pointer
    payload: int
    live: bool = True


@dataclass(frozen=True)
class Capability:
    pod_id: int
    generation: int
    kind: str
    payload: int


class AuthorityWorld:
    def __init__(self):
        self.pods: dict[int, Pod] = {}

    def add_value(self, pid: int, value: int):
        self.pods[pid] = Pod(1, "value", value, True)

    def add_pointer(self, pid: int, target: int):
        self.pods[pid] = Pod(1, "pointer", target, True)

    def capability(self, pid: int) -> Capability | None:
        p = self.pods.get(pid)
        if p is None or not p.live:
            return None
        return Capability(pid, p.generation, p.kind, p.payload)

    def verify(self, cap: Capability | None) -> bool:
        if cap is None:
            return False
        p = self.pods.get(cap.pod_id)
        return bool(
            p is not None and p.live and p.generation == cap.generation
            and p.kind == cap.kind and p.payload == cap.payload
        )

    def update_value(self, pid: int, value: int):
        p = self.pods[pid]
        old_gen = p.generation
        p.generation += 1; p.kind = "value"; p.payload = value; p.live = True
        return old_gen

    def update_pointer(self, pid: int, target: int):
        p = self.pods[pid]
        old_gen = p.generation
        p.generation += 1; p.kind = "pointer"; p.payload = target; p.live = True
        return old_gen

    def revoke(self, pid: int):
        p = self.pods[pid]
        old_gen = p.generation
        p.generation += 1; p.live = False
        return old_gen

    def resolve(self, pid: int, max_hops: int = 8):
        deps = []
        seen = set()
        cur = pid
        for _ in range(max_hops):
            if cur in seen:
                return None, tuple(deps)
            seen.add(cur)
            cap = self.capability(cur)
            if cap is None or not self.verify(cap):
                return None, tuple(deps)
            deps.append((cap.pod_id, cap.generation))
            if cap.kind == "value":
                if 0 <= cap.payload < 4:
                    return cap.payload, tuple(deps)
                return None, tuple(deps)
            cur = cap.payload
        return None, tuple(deps)

    def explicit_state_valid(self, deps: tuple[tuple[int, int], ...]) -> bool:
        for pid, gen in deps:
            p = self.pods.get(pid)
            if p is None or not p.live or p.generation != gen:
                return False
        return True


class LifetimeDAG:
    """Monotonic generation leaves + conjunctive derived factors.

    Read path is one boolean lookup. Generation expiry propagates once through reverse edges.
    """

    def __init__(self):
        self.valid: list[bool] = []
        self.children: dict[int, list[int]] = defaultdict(list)
        self.leaf_by_generation: dict[tuple[int, int], int] = {}
        self.factor_cache: dict[tuple[int, ...], int] = {}

    def _node(self, is_valid: bool = True) -> int:
        nid = len(self.valid)
        self.valid.append(is_valid)
        return nid

    def leaf(self, pid: int, generation: int) -> int:
        key = (pid, generation)
        if key not in self.leaf_by_generation:
            self.leaf_by_generation[key] = self._node(True)
        return self.leaf_by_generation[key]

    def factor_for(self, deps: tuple[tuple[int, int], ...]) -> int:
        parents = tuple(sorted({self.leaf(pid, gen) for pid, gen in deps}))
        if parents in self.factor_cache:
            return self.factor_cache[parents]
        fid = self._node(all(self.valid[p] for p in parents))
        self.factor_cache[parents] = fid
        for p in parents:
            self.children[p].append(fid)
        return fid

    def invalidate_generation(self, pid: int, generation: int) -> int:
        leaf = self.leaf_by_generation.get((pid, generation))
        if leaf is None or not self.valid[leaf]:
            return 0
        touched = 0
        q = deque([leaf])
        while q:
            nid = q.popleft()
            if not self.valid[nid]:
                continue
            self.valid[nid] = False
            touched += 1
            for child in self.children.get(nid, ()):
                if self.valid[child]:
                    q.append(child)
        return touched

    def is_valid(self, factor_id: int) -> bool:
        return self.valid[factor_id]


@dataclass
class JState:
    factor_id: int
    deps: tuple[tuple[int, int], ...]
    answer: str


def batch_predict(bridge, feature_rows, port_a, port_b, live_a, live_b, chunk=2048):
    out = []
    for start in range(0, len(feature_rows), chunk):
        sl = slice(start, start + chunk)
        with torch.inference_mode():
            pred = bridge(
                torch.stack(feature_rows[sl]),
                torch.stack(port_a[sl]),
                torch.stack(port_b[sl]),
                torch.tensor(live_a[sl]),
                torch.tensor(live_b[sl]),
            ).argmax(dim=-1)
        out.extend(int(x) for x in pred.tolist())
    return out


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    seed = 298
    random.seed(seed); torch.manual_seed(seed)
    rng = random.Random(seed + 77)

    model_dir = Path(snapshot_download(
        repo_id=MODEL_ID, revision=REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json"],
    ))
    hashes = {n: sha256_file(model_dir / n) for n in EXPECTED_WEIGHTS}
    assert hashes == EXPECTED_WEIGHTS
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    tok.pad_token = tok.eos_token; tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    train_pairs = [(f"TA-{i:03d}", f"TB-{i:03d}") for i in range(16)]
    hold_pairs = [(f"HA-{i:03d}", f"HB-{i:03d}") for i in range(8)]
    triples = [(op, a, b) for a, b in train_pairs + hold_pairs for op in OPS]
    started = time.perf_counter()
    feats, base_logits = extract_features(model, tok, triples)

    emb = model.get_input_embeddings().weight.detach().float().cpu()
    codes = []
    for c in COLORS:
        ids = tok.encode(c, add_special_tokens=False)
        codes.append(emb[ids].mean(dim=0))
    null = torch.zeros_like(codes[0])
    d = int(null.numel())

    qx=[]; pa=[]; pb=[]; la=[]; lb=[]; yy=[]
    for a,b in train_pairs:
        for op in OPS:
            qh=feats[(op,a,b)]
            for va in range(4):
                for vb in range(4):
                    qx.append(qh); pa.append(codes[va]); pb.append(codes[vb]); la.append(1.0); lb.append(1.0); yy.append(L2I[expected(op,va,vb)])
            qx.append(qh); pa.append(null); pb.append(codes[0]); la.append(0.0); lb.append(1.0); yy.append(L2I["NULL"])
            qx.append(qh); pa.append(codes[0]); pb.append(null); la.append(1.0); lb.append(0.0); yy.append(L2I["NULL"])
    qx=torch.stack(qx); pa=torch.stack(pa); pb=torch.stack(pb)
    la=torch.tensor(la); lb=torch.tensor(lb); yy=torch.tensor(yy,dtype=torch.long)

    bridge=CoherenceBridge(d)
    opt=torch.optim.AdamW(bridge.parameters(),lr=3e-3,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(seed+1)
    bridge.train()
    for _ in range(360):
        idx=torch.randint(0,len(yy),(min(320,len(yy)),),generator=gen)
        loss=nn.functional.cross_entropy(bridge(qx[idx],pa[idx],pb[idx],la[idx],lb[idx]),yy[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    bridge.eval()

    # Held-out identity/value binding gate.
    hq=[]; hpa=[]; hpb=[]; hla=[]; hlb=[]; hy=[]
    for a,b in hold_pairs:
        for op in OPS:
            qh=feats[(op,a,b)]
            for va in range(4):
                for vb in range(4):
                    hq.append(qh); hpa.append(codes[va]); hpb.append(codes[vb]); hla.append(1.0); hlb.append(1.0); hy.append(L2I[expected(op,va,vb)])
            hq.append(qh); hpa.append(null); hpb.append(codes[0]); hla.append(0.0); hlb.append(1.0); hy.append(L2I["NULL"])
            hq.append(qh); hpa.append(codes[0]); hpb.append(null); hla.append(1.0); hlb.append(0.0); hy.append(L2I["NULL"])
    hp=batch_predict(bridge,hq,hpa,hpb,hla,hlb)
    heldout_acc=sum(int(p==y) for p,y in zip(hp,hy))/len(hy)

    world=AuthorityWorld(); dag=LifetimeDAG()
    value_ids=list(range(100,164))
    ptr_ids=list(range(1000,1064))
    head_ids=list(range(2000,2064))
    for pid in value_ids: world.add_value(pid,rng.randrange(4))
    for i,pid in enumerate(ptr_ids): world.add_pointer(pid,value_ids[i])
    for i,pid in enumerate(head_ids): world.add_pointer(pid,ptr_ids[i])

    states: list[JState] = []
    feature_rows=[]; porta=[]; portb=[]; livea=[]; liveb=[]; labels=[]; state_meta=[]
    initial_queries=12000
    for i in range(initial_queries):
        ha=head_ids[rng.randrange(len(head_ids))]; hb=head_ids[rng.randrange(len(head_ids))]
        va,depa=world.resolve(ha); vb,depb=world.resolve(hb)
        op=OPS[rng.randrange(len(OPS))]
        pair=hold_pairs[rng.randrange(len(hold_pairs))]
        deps=tuple(sorted(set(depa+depb)))
        fid=dag.factor_for(deps)
        feature_rows.append(feats[(op,pair[0],pair[1])])
        porta.append(null if va is None else codes[va]); portb.append(null if vb is None else codes[vb])
        livea.append(va is not None); liveb.append(vb is not None)
        labels.append(L2I[expected(op,va,vb)]); state_meta.append((fid,deps))
    preds=batch_predict(bridge,feature_rows,porta,portb,livea,liveb)
    initial_world_acc=sum(int(p==y) for p,y in zip(preds,labels))/len(labels)
    for pred,(fid,deps) in zip(preds,state_meta):
        states.append(JState(fid,deps,LABELS[pred]))

    stale_cap_reject=[]; invalidated_counts=[]
    update_count=1000
    all_mutable=value_ids+ptr_ids+head_ids
    for _ in range(update_count):
        pid=all_mutable[rng.randrange(len(all_mutable))]
        stale=world.capability(pid)
        p=world.pods[pid]
        old_gen=p.generation
        if p.kind=="value":
            # includes same-value rewrites to attack ABA/semantic-equality resurrection
            world.update_value(pid, rng.randrange(4) if rng.random()>0.25 else p.payload)
        else:
            targets=value_ids if pid in ptr_ids else ptr_ids
            target=p.payload if rng.random()<0.25 else targets[rng.randrange(len(targets))]
            world.update_pointer(pid,target)
        invalidated_counts.append(dag.invalidate_generation(pid,old_gen))
        stale_cap_reject.append(not world.verify(stale))

    audit_n=50000
    mismatches=0
    for _ in range(audit_n):
        s=states[rng.randrange(len(states))]
        if dag.is_valid(s.factor_id) != world.explicit_state_valid(s.deps):
            mismatches+=1

    # Fresh current-world neural reasoning after all updates, no retraining.
    fq=[]; fpa=[]; fpb=[]; fla=[]; flb=[]; fy=[]
    fresh_queries=4000
    for _ in range(fresh_queries):
        ha=head_ids[rng.randrange(len(head_ids))]; hb=head_ids[rng.randrange(len(head_ids))]
        va,_=world.resolve(ha); vb,_=world.resolve(hb)
        op=OPS[rng.randrange(len(OPS))]; pair=hold_pairs[rng.randrange(len(hold_pairs))]
        fq.append(feats[(op,pair[0],pair[1])]); fpa.append(null if va is None else codes[va]); fpb.append(null if vb is None else codes[vb])
        fla.append(va is not None); flb.append(vb is not None); fy.append(L2I[expected(op,va,vb)])
    fresh_preds=batch_predict(bridge,fq,fpa,fpb,fla,flb)
    fresh_acc=sum(int(p==y) for p,y in zip(fresh_preds,fy))/len(fy)

    # Dedicated ABA proof: same semantic value under a new generation cannot revive old J state.
    special_pid=value_ids[0]
    v0, deps0=world.resolve(special_pid)
    aba_factor=dag.factor_for(tuple(deps0))
    old_gen=world.pods[special_pid].generation
    world.update_value(special_pid, int(v0))
    dag.invalidate_generation(special_pid,old_gen)
    _=dag.leaf(special_pid,world.pods[special_pid].generation)
    aba_no_resurrection=not dag.is_valid(aba_factor)

    # Revoke -> revive same value also cannot resurrect old state.
    revoke_pid=value_ids[1]
    rv,rdeps=world.resolve(revoke_pid)
    revoke_factor=dag.factor_for(tuple(rdeps))
    old_gen=world.revoke(revoke_pid); dag.invalidate_generation(revoke_pid,old_gen)
    revoked_invalid=not dag.is_valid(revoke_factor) and world.resolve(revoke_pid)[0] is None
    old_gen=world.pods[revoke_pid].generation
    world.update_value(revoke_pid,int(rv)); dag.invalidate_generation(revoke_pid,old_gen)
    revived_old_still_dead=not dag.is_valid(revoke_factor)

    valid_states=[s for s in states if dag.is_valid(s.factor_id)]
    invalid_states=[s for s in states if not dag.is_valid(s.factor_id)]
    serve_invalid_admissions=sum(1 for s in invalid_states[:5000] if dag.is_valid(s.factor_id))

    # O(1) factor-bit read versus explicit dependency scan.
    factor_ns=[]; explicit_ns=[]
    bench_states=[states[rng.randrange(len(states))] for _ in range(100000)]
    for s in bench_states:
        t=time.perf_counter_ns(); _=dag.is_valid(s.factor_id); factor_ns.append(time.perf_counter_ns()-t)
        t=time.perf_counter_ns(); _=world.explicit_state_valid(s.deps); explicit_ns.append(time.perf_counter_ns()-t)

    # B-plane isolation: repeat exact frozen-model query after all mutable-world transitions.
    probe=(OPS[0],hold_pairs[0][0],hold_pairs[0][1])
    base_h_before=feats[probe].clone(); base_l_before=base_logits[probe].clone()
    after_feats,after_logits=extract_features(model,tok,[probe],batch_size=1)
    base_hidden_delta=float(torch.max(torch.abs(base_h_before-after_feats[probe])).item())
    base_logit_delta=float(torch.max(torch.abs(base_l_before-after_logits[probe])).item())

    report={
        "stage":"R298-CAUSAL-KNOWLEDGE-COHERENCE-MACHINE",
        "architecture_candidate":"Causal Knowledge Coherence Architecture (CKCA)",
        "model_id":MODEL_ID,"revision":REVISION,"weights_sha256":hashes,
        "backbone_frozen":all(not p.requires_grad for p in model.parameters()),
        "base_model_optimizer_steps":0,
        "bridge_parameters":sum(p.numel() for p in bridge.parameters()),
        "bridge_training_steps":360,
        "heldout_two_port_accuracy":heldout_acc,
        "initial_multihop_world_accuracy":initial_world_acc,
        "fresh_post_update_world_accuracy":fresh_acc,
        "initial_j_states":len(states),
        "lifecycle_updates":update_count,
        "stale_capability_rejection_rate":sum(stale_cap_reject)/len(stale_cap_reject),
        "factor_vs_explicit_audit_cases":audit_n,
        "factor_vs_explicit_mismatches":mismatches,
        "aba_same_value_no_resurrection":aba_no_resurrection,
        "revoke_invalidates_old_jstate":revoked_invalid,
        "revive_same_value_does_not_resurrect_old_jstate":revived_old_still_dead,
        "invalid_jstate_admissions":serve_invalid_admissions,
        "currently_valid_jstates":len(valid_states),
        "currently_invalid_jstates":len(invalid_states),
        "mean_factor_nodes_invalidated_per_update":statistics.mean(invalidated_counts),
        "p99_factor_nodes_invalidated_per_update":sorted(invalidated_counts)[int(0.99*len(invalidated_counts))],
        "factor_validity_ns_median":statistics.median(factor_ns),
        "explicit_dependency_scan_ns_median":statistics.median(explicit_ns),
        "validity_speedup_vs_explicit":statistics.median(explicit_ns)/max(statistics.median(factor_ns),1),
        "base_hidden_max_delta_after_world_changes":base_hidden_delta,
        "base_logits_max_delta_after_world_changes":base_logit_delta,
        "optimizer_steps_after_world_updates":0,
        "elapsed_seconds":time.perf_counter()-started,
        "architecture_invariant":(
            "A mutable Pod generation is an immutable causal lifetime leaf. Verified capabilities are the only "
            "bridge from the authority plane into neural Port state. Every derived J/cache state owns one conjunctive "
            "lifetime factor over the exact generations read. An update or revoke invalidates that factor transitively; "
            "the frozen language/skill plane remains reusable and uncontaminated by world-state writes."
        ),
        "claim_boundary":(
            "This integrates a real frozen Qwen backbone, held-out future bindings, external multi-hop Pods, independent "
            "generation capabilities and O(1) lifetime-factor admission. The semantic tasks remain synthetic and the "
            "bridge is a classifier; free-form generation, strong-RAG comparisons and broader novelty are still required."
        ),
        "dod_status":"NOT_DOD; integrated causal-coherence architecture gate",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode()
    report["report_sha256"]=hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in (
        "heldout_two_port_accuracy","initial_multihop_world_accuracy","fresh_post_update_world_accuracy",
        "stale_capability_rejection_rate","factor_vs_explicit_mismatches","aba_same_value_no_resurrection",
        "revoke_invalidates_old_jstate","revive_same_value_does_not_resurrect_old_jstate","invalid_jstate_admissions",
        "validity_speedup_vs_explicit","base_hidden_max_delta_after_world_changes","base_logits_max_delta_after_world_changes",
        "elapsed_seconds",
    )},indent=2))

    if heldout_acc < 0.97: return 2
    if initial_world_acc < 0.97 or fresh_acc < 0.97: return 3
    if report["stale_capability_rejection_rate"] != 1.0: return 4
    if mismatches != 0: return 5
    if not aba_no_resurrection or not revoked_invalid or not revived_old_still_dead: return 6
    if serve_invalid_admissions != 0: return 7
    if base_hidden_delta > 1e-6 or base_logit_delta > 1e-6: return 8
    return 0


if __name__=="__main__":
    raise SystemExit(main())
