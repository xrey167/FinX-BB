from __future__ import annotations

import hashlib
import json
import os
import random
import time
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
OUT = Path(os.environ.get("SO_R296_REPORT", "ci-qwen-r296/report.json"))
COLORS = ("red", "blue", "green", "yellow")
OPS = ("same", "both_warm", "xor_red", "conditional_color", "one_blue", "edge_pair")
LABELS = ("red", "blue", "green", "yellow", "yes", "no", "A", "B", "NULL")
L2I = {x: i for i, x in enumerate(LABELS)}


def qtext(op: str, a: str, b: str) -> str:
    if op == "same":
        return f"Do entities {a} and {b} currently have the same color? Return only yes or no."
    if op == "both_warm":
        return f"Are both {a} and {b} currently warm colors (red or yellow)? Return only yes or no."
    if op == "xor_red":
        return f"Is exactly one of {a} and {b} currently red? Return only yes or no."
    if op == "conditional_color":
        return f"If {a} is currently red, return its color; otherwise return the current color of {b}. Return only a color word."
    if op == "one_blue":
        return f"Is at least one of {a} and {b} currently blue? Return only yes or no."
    if op == "edge_pair":
        return f"Return A if the current pair ({a}, {b}) contains red or yellow at least once; otherwise return B. Return only A or B."
    raise ValueError(op)


def expected(op: str, va: int | None, vb: int | None) -> str:
    if va is None or vb is None:
        return "NULL"
    a, b = COLORS[va], COLORS[vb]
    if op == "same":
        return "yes" if a == b else "no"
    if op == "both_warm":
        return "yes" if a in ("red", "yellow") and b in ("red", "yellow") else "no"
    if op == "xor_red":
        return "yes" if (a == "red") ^ (b == "red") else "no"
    if op == "conditional_color":
        return a if a == "red" else b
    if op == "one_blue":
        return "yes" if a == "blue" or b == "blue" else "no"
    if op == "edge_pair":
        return "A" if a in ("red", "yellow") or b in ("red", "yellow") else "B"
    raise ValueError(op)


def prompt(tok, op: str, a: str, b: str) -> str:
    system = (
        "Mutable entity values are supplied through a separate trusted Port plane. "
        "The text identifies entities and the requested operation but contains no current values."
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": qtext(op, a, b)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_features(model, tok, triples, batch_size: int = 16):
    feats = {}
    for start in range(0, len(triples), batch_size):
        batch = triples[start:start + batch_size]
        texts = [prompt(tok, op, a, b) for op, a, b in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.inference_mode():
            out = model(**enc, use_cache=False, output_hidden_states=True, return_dict=True)
        last = enc.attention_mask.sum(dim=1) - 1
        h = out.hidden_states[-1]
        for i, key in enumerate(batch):
            feats[key] = h[i, int(last[i])].float().cpu()
    return feats


class MultiPortBridge(nn.Module):
    def __init__(self, d: int, width: int = 224):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.LayerNorm(width))
        self.p = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.LayerNorm(width))
        self.net = nn.Sequential(
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


@dataclass
class JState:
    answer: str
    deps: frozenset[tuple[int, int]]


class World:
    def __init__(self):
        self.pods: dict[int, Pod] = {}

    def add_value(self, pid: int, value: int):
        self.pods[pid] = Pod(1, "value", value, True)

    def add_pointer(self, pid: int, target: int):
        self.pods[pid] = Pod(1, "pointer", target, True)

    def update_value(self, pid: int, value: int):
        p = self.pods[pid]
        p.generation += 1; p.kind = "value"; p.payload = value; p.live = True

    def update_pointer(self, pid: int, target: int):
        p = self.pods[pid]
        p.generation += 1; p.kind = "pointer"; p.payload = target; p.live = True

    def revoke(self, pid: int):
        self.pods[pid].generation += 1
        self.pods[pid].live = False

    def resolve(self, pid: int, max_hops: int = 8):
        deps = []
        seen = set()
        cur = pid
        for _ in range(max_hops):
            if cur in seen or cur not in self.pods:
                return None, frozenset(deps)
            seen.add(cur)
            p = self.pods[cur]
            deps.append((cur, p.generation))
            if not p.live:
                return None, frozenset(deps)
            if p.kind == "value":
                return p.payload, frozenset(deps)
            cur = p.payload
        return None, frozenset(deps)

    def valid(self, state: JState) -> bool:
        for pid, gen in state.deps:
            p = self.pods.get(pid)
            if p is None or not p.live or p.generation != gen:
                return False
        return True


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    seed = 296
    random.seed(seed); torch.manual_seed(seed)

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

    train_pairs = [(f"TA-{i:03d}", f"TB-{i:03d}") for i in range(40)]
    hold_pairs = [(f"HA-{i:03d}", f"HB-{i:03d}") for i in range(20)]
    triples = [(op, a, b) for a, b in train_pairs + hold_pairs for op in OPS]
    started = time.perf_counter()
    feats = extract_features(model, tok, triples)

    emb = model.get_input_embeddings().weight.detach().float().cpu()
    codes = []
    for c in COLORS:
        ids = tok.encode(c, add_special_tokens=False)
        codes.append(emb[ids].mean(dim=0))
    null = torch.zeros_like(codes[0])
    d = int(null.numel())

    qx=[]; pa=[]; pb=[]; la=[]; lb=[]; y=[]
    for a,b in train_pairs:
        for op in OPS:
            qh=feats[(op,a,b)]
            for va in range(4):
                for vb in range(4):
                    qx.append(qh); pa.append(codes[va]); pb.append(codes[vb]); la.append(1.0); lb.append(1.0); y.append(L2I[expected(op,va,vb)])
            # one-side and both-side revoke semantics
            qx.append(qh); pa.append(null); pb.append(codes[0]); la.append(0.0); lb.append(1.0); y.append(L2I["NULL"])
            qx.append(qh); pa.append(codes[0]); pb.append(null); la.append(1.0); lb.append(0.0); y.append(L2I["NULL"])
    qx=torch.stack(qx); pa=torch.stack(pa); pb=torch.stack(pb)
    la=torch.tensor(la); lb=torch.tensor(lb); y=torch.tensor(y,dtype=torch.long)

    bridge=MultiPortBridge(d)
    opt=torch.optim.AdamW(bridge.parameters(),lr=2.2e-3,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(seed+1)
    bridge.train()
    for _ in range(1050):
        idx=torch.randint(0,len(y),(320,),generator=gen)
        logits=bridge(qx[idx],pa[idx],pb[idx],la[idx],lb[idx])
        loss=nn.functional.cross_entropy(logits,y[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    bridge.eval()

    def pred(op,a,b,va,vb):
        q=feats[(op,a,b)].unsqueeze(0)
        ca=(null if va is None else codes[va]).unsqueeze(0)
        cb=(null if vb is None else codes[vb]).unsqueeze(0)
        with torch.inference_mode():
            i=int(bridge(q,ca,cb,torch.tensor([va is not None]),torch.tensor([vb is not None])).argmax(-1).item())
        return LABELS[i]

    held=[]
    for a,b in hold_pairs:
        for op in OPS:
            for va in range(4):
                for vb in range(4):
                    held.append(pred(op,a,b,va,vb)==expected(op,va,vb))
            held.append(pred(op,a,b,None,0)=="NULL")
            held.append(pred(op,a,b,0,None)=="NULL")

    # Build canonical multi-hop Pod graph. Pointer generations are included in read-sets.
    rng=random.Random(seed+9)
    world=World()
    value_ids=list(range(100,180))
    for pid in value_ids: world.add_value(pid,rng.randrange(4))
    pointer_ids=list(range(1000,1080))
    for i,pid in enumerate(pointer_ids): world.add_pointer(pid,value_ids[i])
    head_ids=list(range(2000,2080))
    for i,pid in enumerate(head_ids): world.add_pointer(pid,pointer_ids[i])

    jstates=[]; world_checks=[]; dependency_checks=[]; unaffected_checks=[]
    pair=a,b=hold_pairs[0]
    a_name,b_name=pair
    for step in range(5000):
        ia=rng.randrange(len(head_ids)); ib=rng.randrange(len(head_ids))
        pida,pidb=head_ids[ia],head_ids[ib]
        va,depa=world.resolve(pida); vb,depb=world.resolve(pidb)
        op=OPS[rng.randrange(len(OPS))]
        ans=pred(op,a_name,b_name,va,vb)
        exp=expected(op,va,vb)
        state=JState(ans,frozenset(set(depa)|set(depb)))
        jstates.append(state); world_checks.append(ans==exp and world.valid(state))

        # Every 25th step mutate one dependency and prove this state dies while a disjoint
        # state remains valid. This is the J-Space lifetime-union primitive.
        if step % 25 == 0:
            candidate_unaffected=None
            for old in reversed(jstates[:-1]):
                if old.deps.isdisjoint(state.deps) and world.valid(old):
                    candidate_unaffected=old; break
            dep_pid=next(iter(state.deps))[0]
            p=world.pods[dep_pid]
            if p.kind=="value": world.update_value(dep_pid,rng.randrange(4))
            else: world.update_pointer(dep_pid,p.payload)
            dependency_checks.append(not world.valid(state))
            if candidate_unaffected is not None:
                unaffected_checks.append(world.valid(candidate_unaffected))

    # Revocation closure on a pointer in the chain.
    revoke_pid=pointer_ids[0]
    vh,dh=world.resolve(head_ids[0])
    control_state=JState("probe",dh)
    world.revoke(revoke_pid)
    revoke_invalidates=not world.valid(control_state)
    resolved_after_revoke,_=world.resolve(head_ids[0])

    report={
        "stage":"R296-MULTIPOD-JSPACE",
        "architecture_candidate":"Multi-Pod Ephemeral J-Space with Exact Lifetime Union",
        "model_id":MODEL_ID,"revision":REVISION,"weights_sha256":hashes,
        "backbone_frozen":all(not p.requires_grad for p in model.parameters()),
        "bridge_parameters":sum(p.numel() for p in bridge.parameters()),
        "heldout_two_port_operation_accuracy":sum(held)/len(held),
        "online_multihop_world_accuracy":sum(world_checks)/len(world_checks),
        "dependent_jstate_invalidation_rate":sum(dependency_checks)/len(dependency_checks),
        "disjoint_jstate_survival_rate":sum(unaffected_checks)/len(unaffected_checks) if unaffected_checks else None,
        "revoked_pointer_invalidates_prior_jstate":revoke_invalidates,
        "revoked_pointer_resolves_to_null":resolved_after_revoke is None,
        "world_steps":5000,
        "optimizer_steps_after_world_updates":0,
        "max_pointer_depth_tested":2,
        "dependency_rule":"deps(output)=union(deps(all Port reads)); valid iff every (pod,generation) remains live and authoritative",
        "mechanism":(
            "Frozen Qwen supplies operation/language features; mutable values arrive only through two Port codes. "
            "J-Space outputs carry the union of every pointer/value generation read. Updating any member kills only "
            "dependent J states; disjoint J states remain valid. Pointer generations are first-class dependencies, "
            "so retargeting an intermediate Symlink invalidates derived neural state even when the leaf value is unchanged."
        ),
        "scientific_scope":"Real frozen Qwen2.5-0.5B plus trained sidecar; synthetic color relations and external pointer graph; not yet free-form generation or real QA.",
        "elapsed_seconds":time.perf_counter()-started,
        "dod_status":"NOT_DOD; multi-Pod lifetime composition gate",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode(); report["report_sha256"]=hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    if report["heldout_two_port_operation_accuracy"]<0.98:return 2
    if report["online_multihop_world_accuracy"]<0.98:return 3
    if report["dependent_jstate_invalidation_rate"]!=1.0:return 4
    if unaffected_checks and report["disjoint_jstate_survival_rate"]<0.99:return 5
    if not revoke_invalidates or resolved_after_revoke is not None:return 6
    return 0

if __name__=="__main__": raise SystemExit(main())
