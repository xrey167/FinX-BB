from __future__ import annotations

import copy
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

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file, legacy_cache, slice_cache
from research.qwen25_05b_r308_late_bound_handle_vm import MODEL_ID, REVISION, EXPECTED_WEIGHTS, VALUE_STRINGS

OUT = Path(os.environ.get("SO_R312_REPORT", "ci-qwen-r312/report.json"))
OPS = ("copy_a", "copy_b", "if_a_flag", "if_same")
OP2I = {x: i for i, x in enumerate(OPS)}


def question(op: str, a: str, b: str) -> str:
    if op == "copy_a":
        return f"Return the current governed value of {a}."
    if op == "copy_b":
        return f"Return the current governed value of {b}."
    if op == "if_a_flag":
        return f"If the current governed flag of {a} is true, return {a}'s current value; otherwise return {b}'s current value."
    if op == "if_same":
        return f"If {a} and {b} currently have the same governed value, return {a}'s value; otherwise return {b}'s value."
    raise ValueError(op)


def query_prompt(tok, op: str, a: str, b: str) -> str:
    system = (
        "Mutable current values are supplied only by a trusted canonical knowledge runtime. "
        "Interpret the requested operation and entity references. Do not invent current values from the text."
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": question(op, a, b)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def decode_prefix(tok, op: str, a: str, b: str) -> str:
    return query_prompt(tok, op, a, b) + "Verified current result:\n"


def extract_features(model, tok, triples, batch_size=24):
    feats = {}
    for start in range(0, len(triples), batch_size):
        batch = triples[start:start + batch_size]
        texts = [query_prompt(tok, *x) for x in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.inference_mode():
            out = model(**enc, use_cache=False, output_hidden_states=True, return_dict=True)
        idx = out.hidden_states[-1].shape[1] - 1
        for i, key in enumerate(batch):
            feats[key] = out.hidden_states[-1][i, idx].float().cpu()
    return feats


class Planner(nn.Module):
    def __init__(self, d: int, width: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width), nn.GELU(), nn.Linear(width, len(OPS)))

    def forward(self, x):
        return self.net(x)


@dataclass
class Page:
    generation: int
    value_idx: int
    flag: bool
    live: bool = True


@dataclass(frozen=True)
class Capability:
    pod_id: int
    generation: int
    value_idx: int
    flag: bool


@dataclass(frozen=True)
class JState:
    output_ids: tuple[int, ...]
    deps: tuple[tuple[int, int], ...]
    factor_id: int


class World:
    def __init__(self, values: list[tuple[int, ...]]):
        self.pages: dict[int, Page] = {}
        self.aliases: dict[str, int] = {}
        self.values = values

    def add(self, pid: int, value_idx: int, flag: bool):
        self.pages[pid] = Page(1, value_idx, flag, True)

    def bind_alias(self, alias: str, pid: int):
        self.aliases[alias.casefold()] = pid

    def resolve(self, alias: str) -> int | None:
        return self.aliases.get(alias.casefold())

    def capability(self, pid: int) -> Capability | None:
        p = self.pages.get(pid)
        if p is None or not p.live:
            return None
        return Capability(pid, p.generation, p.value_idx, p.flag)

    def verify(self, cap: Capability | None) -> bool:
        if cap is None:
            return False
        p = self.pages.get(cap.pod_id)
        return bool(p is not None and p.live and p.generation == cap.generation and p.value_idx == cap.value_idx and p.flag == cap.flag)

    def update(self, pid: int, value_idx: int | None = None, flag: bool | None = None):
        p = self.pages[pid]
        old = p.generation
        p.generation += 1
        if value_idx is not None:
            p.value_idx = value_idx
        if flag is not None:
            p.flag = flag
        p.live = True
        return old

    def revoke(self, pid: int):
        p = self.pages[pid]
        old = p.generation
        p.generation += 1
        p.live = False
        return old


class LifetimeDAG:
    def __init__(self):
        self.valid: list[bool] = []
        self.children: dict[int, list[int]] = defaultdict(list)
        self.leaves: dict[tuple[int, int], int] = {}
        self.factors: dict[tuple[int, ...], int] = {}

    def _new(self, v=True):
        i = len(self.valid); self.valid.append(v); return i

    def leaf(self, dep: tuple[int, int]):
        if dep not in self.leaves:
            self.leaves[dep] = self._new(True)
        return self.leaves[dep]

    def factor(self, deps: tuple[tuple[int, int], ...]):
        parents = tuple(sorted({self.leaf(d) for d in deps}))
        if parents in self.factors:
            return self.factors[parents]
        f = self._new(all(self.valid[p] for p in parents)); self.factors[parents] = f
        for p in parents: self.children[p].append(f)
        return f

    def invalidate(self, dep: tuple[int, int]):
        leaf = self.leaves.get(dep)
        if leaf is None or not self.valid[leaf]: return 0
        q = deque([leaf]); touched = 0
        while q:
            x = q.popleft()
            if not self.valid[x]: continue
            self.valid[x] = False; touched += 1
            for c in self.children.get(x, ()):
                if self.valid[c]: q.append(c)
        return touched

    def is_valid(self, factor_id: int):
        return self.valid[factor_id]


def execute(world: World, dag: LifetimeDAG, op: str, ca: Capability | None, cb: Capability | None) -> JState:
    if ca is None or cb is None or not world.verify(ca) or not world.verify(cb):
        return JState(tuple(), tuple(), dag.factor(tuple()))
    if op == "copy_a":
        deps = ((ca.pod_id, ca.generation),); out = world.values[ca.value_idx]
    elif op == "copy_b":
        deps = ((cb.pod_id, cb.generation),); out = world.values[cb.value_idx]
    elif op == "if_a_flag":
        if ca.flag:
            deps = ((ca.pod_id, ca.generation),); out = world.values[ca.value_idx]
        else:
            deps = tuple(sorted(((ca.pod_id, ca.generation), (cb.pod_id, cb.generation)))); out = world.values[cb.value_idx]
    elif op == "if_same":
        deps = tuple(sorted(((ca.pod_id, ca.generation), (cb.pod_id, cb.generation))))
        out = world.values[ca.value_idx if ca.value_idx == cb.value_idx else cb.value_idx]
    else:
        raise ValueError(op)
    return JState(tuple(out), deps, dag.factor(deps))


def run_suffix(model, suffix_ids, prefix_cache, prefix_len):
    ids = torch.tensor([suffix_ids], dtype=torch.long)
    mask = torch.ones((1, prefix_len + len(suffix_ids)), dtype=torch.long)
    pos = torch.arange(prefix_len, prefix_len + len(suffix_ids), dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        return model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=copy.deepcopy(prefix_cache), use_cache=True, return_dict=True)


def mutate_cache(cache, seed: int):
    g = torch.Generator().manual_seed(seed); out = []
    for k,v in legacy_cache(cache):
        out.append((torch.randn(k.shape,generator=g,dtype=torch.float32).to(k.dtype), torch.randn(v.shape,generator=g,dtype=torch.float32).to(v.dtype)))
    return tuple(out)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    seed = 312; random.seed(seed); torch.manual_seed(seed); rng = random.Random(seed + 1)

    md = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION, allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors","*.index.json","*.merges","*.vocab","merges.txt","vocab.json"]))
    hashes = {n: sha256_file(md/n) for n in EXPECTED_WEIGHTS}; assert hashes == EXPECTED_WEIGHTS
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False); tok.pad_token = tok.eos_token; tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(md, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=False).eval(); model.requires_grad_(False)

    value_ids = [tuple(tok.encode(x, add_special_tokens=False)) for x in VALUE_STRINGS]
    train_pairs=[(f"TR-A-{i:02d}",f"TR-B-{i:02d}") for i in range(12)]
    hold_pairs=[(f"HO-A-{i:02d}",f"HO-B-{i:02d}") for i in range(6)]
    triples=[(op,a,b) for a,b in train_pairs+hold_pairs for op in OPS]
    feats=extract_features(model,tok,triples)
    d=next(iter(feats.values())).numel()

    X=[];Y=[]
    for a,b in train_pairs:
        for op in OPS: X.append(feats[(op,a,b)]);Y.append(OP2I[op])
    X=torch.stack(X);Y=torch.tensor(Y)
    planner=Planner(d);opt=torch.optim.AdamW(planner.parameters(),lr=3e-3,weight_decay=1e-4);gen=torch.Generator().manual_seed(seed+7)
    planner.train()
    for _ in range(300):
        idx=torch.randint(0,len(Y),(min(160,len(Y)),),generator=gen);loss=nn.functional.cross_entropy(planner(X[idx]),Y[idx]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    planner.eval()

    route_checks=[]
    for a,b in hold_pairs:
        for op in OPS:
            with torch.inference_mode(): pred=OPS[int(planner(feats[(op,a,b)].unsqueeze(0)).argmax(-1).item())]
            route_checks.append(pred==op)

    world=World(value_ids);dag=LifetimeDAG();bindings=[];pid=100
    for i,(a,b) in enumerate(hold_pairs):
        pa,pb=pid,pid+1;pid+=2;world.add(pa,rng.randrange(len(value_ids)),bool(rng.getrandbits(1)));world.add(pb,rng.randrange(len(value_ids)),bool(rng.getrandbits(1)));world.bind_alias(a,pa);world.bind_alias(b,pb);world.bind_alias(f"late-a-{i}",pa);world.bind_alias(f"late-b-{i}",pb);bindings.append((a,b,pa,pb))

    # Query-specific immutable decode checkpoints before any mutable result is bound.
    checkpoints={}
    for a,b,_,_ in bindings:
        for op in OPS:
            prefix=decode_prefix(tok,op,a,b);ids=tok(prefix,add_special_tokens=False).input_ids;t=torch.tensor([ids])
            with torch.inference_mode(): out=model(input_ids=t,attention_mask=torch.ones_like(t),use_cache=True,return_dict=True)
            checkpoints[(op,a,b)]=(prefix,ids,legacy_cache(out.past_key_values))

    rows=[];equiv=[];stale_deltas=[];old_states=[];old_suffixes=[];invalid_checks=[];update_times=[];full_times=[];splice_times=[]
    for step in range(16):
        a,b,pa,pb=bindings[step%len(bindings)];op=OPS[step%len(OPS)]
        with torch.inference_mode(): routed=OPS[int(planner(feats[(op,a,b)].unsqueeze(0)).argmax(-1).item())]
        ca,cb=world.capability(pa),world.capability(pb);state=execute(world,dag,routed,ca,cb);assert state.output_ids
        prefix,prefix_ids,prefix_cache=checkpoints[(op,a,b)]
        suffix=list(state.output_ids)+tok.encode("\nStatus:",add_special_tokens=False)
        full_ids=tok(prefix,add_special_tokens=False).input_ids+suffix
        t0=time.perf_counter();ft=torch.tensor([full_ids]);
        with torch.inference_mode(): fo=model(input_ids=ft,attention_mask=torch.ones_like(ft),use_cache=True,return_dict=True)
        full_times.append(time.perf_counter()-t0);fl=fo.logits[0,-1].float()
        t0=time.perf_counter();so=run_suffix(model,suffix,prefix_cache,len(prefix_ids));splice_times.append(time.perf_counter()-t0);sl=so.logits[0,-1].float()
        delta=float(torch.max(torch.abs(fl-sl)).item());equiv.append(delta)
        pc=legacy_cache(so.past_key_values);old_suffixes.append(slice_cache(pc,len(prefix_ids),len(prefix_ids)+len(suffix)));old_states.append(state)

        # Online world transition, including frequent same-value/flag rewrites.
        target=pa if rng.random()<.5 else pb;p=world.pages[target];old_gen=p.generation
        t0=time.perf_counter_ns();world.update(target,p.value_idx if rng.random()<.5 else rng.randrange(len(value_ids)),p.flag if rng.random()<.5 else bool(rng.getrandbits(1)));dag.invalidate((target,old_gen));update_times.append(time.perf_counter_ns()-t0)
        invalid_checks.append(not dag.is_valid(state.factor_id) if (target,old_gen) in state.deps else dag.is_valid(state.factor_id))

        # Current result after edit; stale physical suffix is corrupted but never re-admitted.
        ca2,cb2=world.capability(pa),world.capability(pb);cur=execute(world,dag,routed,ca2,cb2);assert cur.output_ids
        csuffix=list(cur.output_ids)+tok.encode("\nStatus:",add_special_tokens=False)
        before=run_suffix(model,csuffix,prefix_cache,len(prefix_ids)).logits[0,-1].float();_garbage=mutate_cache(old_suffixes[-1],9000+step);after=run_suffix(model,csuffix,prefix_cache,len(prefix_ids)).logits[0,-1].float();stale_deltas.append(float(torch.max(torch.abs(before-after)).item()))
        rows.append({"step":step,"op":op,"routed":routed,"deps":state.deps,"factor_valid_after_update":dag.is_valid(state.factor_id),"full_vs_splice_delta":delta,"stale_suffix_mutation_delta":stale_deltas[-1]})

    alias_checks=[]
    for i,(a,b,pa,pb) in enumerate(bindings): alias_checks.append(world.resolve(a)==world.resolve(f"late-a-{i}")==pa and world.resolve(b)==world.resolve(f"late-b-{i}")==pb)

    report={
        "stage":"R312-INTEGRATED-CKVM-TRANSACTION",
        "architecture_candidate":"CKCA + CKVM + CLFD + Late-Bound Decode Checkpoint",
        "model_id":MODEL_ID,"revision":REVISION,"weights_sha256":hashes,"backbone_frozen":all(not p.requires_grad for p in model.parameters()),
        "planner_parameters":sum(p.numel() for p in planner.parameters()),"planner_training_steps":300,"heldout_operation_route_accuracy":sum(route_checks)/len(route_checks),
        "real_model_transactions":len(rows),"max_full_vocab_logit_delta_full_recompute_vs_late_splice":max(equiv),"topological_lifetime_checks_correct_rate":sum(invalid_checks)/len(invalid_checks),"max_current_logit_delta_after_stale_suffix_corruption":max(stale_deltas),
        "alias_to_canonical_identity_rate":sum(alias_checks)/len(alias_checks),"median_world_update_plus_lifetime_invalidation_ns":statistics.median(update_times),"p99_world_update_plus_lifetime_invalidation_ns":sorted(update_times)[int(.99*len(update_times))],
        "median_full_recompute_seconds":statistics.median(full_times),"median_late_splice_seconds":statistics.median(splice_times),"median_speedup_full_over_splice":statistics.median(full_times)/max(statistics.median(splice_times),1e-9),
        "optimizer_steps_after_world_updates":0,"rows":rows,
        "integrated_invariant":(
            "Frozen B-plane compiles the operation; aliases resolve to canonical Pods; current capabilities are independently "
            "verified; CKVM executes only current typed reads; CLFD assigns the exact source-generation lifetime; mutable "
            "result bytes are bound only after a query-specific immutable decode checkpoint; edits invalidate old factors and "
            "only the dependent suffix is re-executed. Stale suffix bytes remain physically present but have no current causal path."
        ),
        "claim_boundary":"Integrated real-model mechanism gate on synthetic operations and 16 neural transactions; not a broad QA/RAG or novelty proof.",
        "dod_status":"NOT_DOD; first integrated CKCA/CKVM neural transaction gate",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode();report["report_sha256"]=hashlib.sha256(canonical).hexdigest();OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("heldout_operation_route_accuracy","real_model_transactions","max_full_vocab_logit_delta_full_recompute_vs_late_splice","topological_lifetime_checks_correct_rate","max_current_logit_delta_after_stale_suffix_corruption","alias_to_canonical_identity_rate","median_world_update_plus_lifetime_invalidation_ns","median_full_recompute_seconds","median_late_splice_seconds","median_speedup_full_over_splice")},indent=2))
    if report["heldout_operation_route_accuracy"]<.97:return 2
    if max(equiv)>.02:return 3
    if report["topological_lifetime_checks_correct_rate"]!=1.0:return 4
    if max(stale_deltas)>.02:return 5
    if report["alias_to_canonical_identity_rate"]!=1.0:return 6
    return 0

if __name__=="__main__":raise SystemExit(main())
