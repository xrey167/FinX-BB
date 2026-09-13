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
from torch import nn
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_WEIGHTS = {"model.safetensors": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"}
OUT = Path(os.environ.get("SO_R308_REPORT", "ci-qwen-r308/report.json"))

OPS = ("copy_a", "copy_b", "same", "if_a_flag", "if_same")
OP2I = {x: i for i, x in enumerate(OPS)}
VALUE_STRINGS = (
    "New Zealand", "Buenos Aires", "San Francisco", "South Korea", "Alpha Centauri",
    "Blue Mountain", "Red Valley", "North Harbor", "Silver Lake", "Green Meadow",
    "vexa 173", "orion node 42", "delta sector 9", "quantum bay 17", "lumen field 88",
    "Nova Station", "Amber Coast", "Crystal Ridge", "Echo Point", "River Gate",
    "Solar District", "Cloud Harbor", "Moon Valley", "Atlas Zone", "Cedar Plains",
    "Vector City", "Lambda Port", "Sigma Ridge", "Kappa Field", "Omega Basin",
)


def question(op: str, a: str, b: str) -> str:
    if op == "copy_a":
        return f"Return the current value of {a} exactly."
    if op == "copy_b":
        return f"Return the current value of {b} exactly."
    if op == "same":
        return f"Do {a} and {b} currently have exactly the same value? Return yes or no."
    if op == "if_a_flag":
        return f"If the current flag of {a} is true, return {a}'s current value; otherwise return {b}'s current value."
    if op == "if_same":
        return f"If {a} and {b} currently have the same value, return {a}'s value; otherwise return {b}'s value."
    raise ValueError(op)


def prompt(tok, op: str, a: str, b: str) -> str:
    system = (
        "Mutable values and flags are held in a separate trusted Port plane. "
        "Interpret only the requested operation and entity references; no current value is present in the text."
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": question(op, a, b)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_features(model, tok, triples, batch_size=24):
    feats = {}
    token_counts = {}
    for start in range(0, len(triples), batch_size):
        batch = triples[start:start + batch_size]
        texts = [prompt(tok, *x) for x in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.inference_mode():
            out = model(**enc, use_cache=False, output_hidden_states=True, return_dict=True)
        idx = out.hidden_states[-1].shape[1] - 1  # left padding: physical last token
        for i, key in enumerate(batch):
            feats[key] = out.hidden_states[-1][i, idx].float().cpu()
            token_counts[key] = int(enc.attention_mask[i].sum().item())
    return feats, token_counts


class OpPlanner(nn.Module):
    def __init__(self, d: int, width: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, width), nn.GELU(),
            nn.Linear(width, len(OPS)),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class Page:
    generation: int
    value_idx: int
    flag: bool
    live: bool = True


@dataclass(frozen=True)
class JState:
    output_ids: tuple[int, ...]
    deps: tuple[tuple[int, int], ...]


class World:
    def __init__(self, value_token_ids: list[tuple[int, ...]]):
        self.pages: dict[int, Page] = {}
        self.value_token_ids = value_token_ids
        self.aliases: dict[str, int] = {}

    def add(self, pid: int, value_idx: int, flag: bool):
        self.pages[pid] = Page(1, value_idx, flag, True)

    def alias(self, text: str, pid: int):
        self.aliases[text.casefold()] = pid

    def resolve_alias(self, text: str) -> int | None:
        return self.aliases.get(text.casefold())

    def update(self, pid: int, value_idx: int | None = None, flag: bool | None = None):
        p = self.pages[pid]
        p.generation += 1
        if value_idx is not None:
            p.value_idx = value_idx
        if flag is not None:
            p.flag = flag
        p.live = True

    def revoke(self, pid: int):
        p = self.pages[pid]
        p.generation += 1
        p.live = False

    def page(self, pid: int) -> Page | None:
        p = self.pages.get(pid)
        return p if p is not None and p.live else None

    def valid(self, state: JState) -> bool:
        for pid, generation in state.deps:
            p = self.pages.get(pid)
            if p is None or not p.live or p.generation != generation:
                return False
        return True


def execute(world: World, op: str, pa: int, pb: int, yes_ids: tuple[int, ...], no_ids: tuple[int, ...]) -> JState:
    a = world.page(pa)
    b = world.page(pb)
    if op == "copy_a":
        if a is None:
            return JState(tuple(), tuple())
        return JState(world.value_token_ids[a.value_idx], ((pa, a.generation),))
    if op == "copy_b":
        if b is None:
            return JState(tuple(), tuple())
        return JState(world.value_token_ids[b.value_idx], ((pb, b.generation),))
    if op == "same":
        if a is None or b is None:
            return JState(tuple(), tuple())
        out = yes_ids if a.value_idx == b.value_idx else no_ids
        return JState(out, tuple(sorted(((pa, a.generation), (pb, b.generation)))))
    if op == "if_a_flag":
        if a is None:
            return JState(tuple(), tuple())
        if a.flag:
            return JState(world.value_token_ids[a.value_idx], ((pa, a.generation),))
        if b is None:
            return JState(tuple(), tuple())
        return JState(world.value_token_ids[b.value_idx], tuple(sorted(((pa, a.generation), (pb, b.generation)))))
    if op == "if_same":
        if a is None or b is None:
            return JState(tuple(), tuple())
        selected = a if a.value_idx == b.value_idx else b
        selected_pid = pa if a.value_idx == b.value_idx else pb
        # both values were causally read to decide the branch, so both generations remain dependencies.
        return JState(world.value_token_ids[selected.value_idx], tuple(sorted(((pa, a.generation), (pb, b.generation)))))
    raise ValueError(op)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    seed = 308
    random.seed(seed); torch.manual_seed(seed)
    rng = random.Random(seed + 1)

    md = Path(snapshot_download(
        repo_id=MODEL_ID, revision=REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json"],
    ))
    hashes = {n: sha256_file(md / n) for n in EXPECTED_WEIGHTS}; assert hashes == EXPECTED_WEIGHTS
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    tok.pad_token = tok.eos_token; tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        md, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=False,
    ).eval(); model.requires_grad_(False)

    value_token_ids = [tuple(tok.encode(x, add_special_tokens=False)) for x in VALUE_STRINGS]
    if any(len(x) == 0 for x in value_token_ids):
        raise RuntimeError("empty value tokenization")
    yes_ids = tuple(tok.encode("yes", add_special_tokens=False))
    no_ids = tuple(tok.encode("no", add_special_tokens=False))

    train_pairs = [(f"train-A-{i:03d}", f"train-B-{i:03d}") for i in range(24)]
    held_pairs = [(f"future-A-{i:03d}", f"future-B-{i:03d}") for i in range(12)]
    alias_pairs = [
        (f"supplier nickname {i} / A", f"alias::{i}::B") for i in range(12)
    ]
    triples = [(op, a, b) for a, b in train_pairs + held_pairs + alias_pairs for op in OPS]
    features, token_counts = extract_features(model, tok, triples)
    d = next(iter(features.values())).numel()

    X=[]; Y=[]
    for a,b in train_pairs:
        for op in OPS:
            X.append(features[(op,a,b)]); Y.append(OP2I[op])
    X=torch.stack(X); Y=torch.tensor(Y,dtype=torch.long)
    planner=OpPlanner(d)
    opt=torch.optim.AdamW(planner.parameters(),lr=3e-3,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(seed+9)
    planner.train()
    for _ in range(300):
        idx=torch.randint(0,len(Y),(min(192,len(Y)),),generator=gen)
        loss=nn.functional.cross_entropy(planner(X[idx]),Y[idx])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    planner.eval()

    def route_accuracy(pairs):
        rows=[]; labels=[]
        for a,b in pairs:
            for op in OPS:
                rows.append(features[(op,a,b)]);labels.append(OP2I[op])
        with torch.inference_mode(): pred=planner(torch.stack(rows)).argmax(-1).tolist()
        return sum(int(p==y) for p,y in zip(pred,labels))/len(labels), pred, labels

    held_route,_,_=route_accuracy(held_pairs)
    alias_route,_,_=route_accuracy(alias_pairs)

    # Runtime world. Each logical pair maps through aliases to canonical Pods.
    world=World(value_token_ids)
    pair_bindings=[]
    next_pid=100
    for i,(a,b) in enumerate(held_pairs+alias_pairs):
        pa,pb=next_pid,next_pid+1;next_pid+=2
        world.add(pa,rng.randrange(len(VALUE_STRINGS)),bool(rng.getrandbits(1)))
        world.add(pb,rng.randrange(len(VALUE_STRINGS)),bool(rng.getrandbits(1)))
        world.alias(a,pa);world.alias(b,pb)
        # add several late aliases per canonical identity
        world.alias(f"late synonym A {i}",pa);world.alias(f"late synonym B {i}",pb)
        pair_bindings.append((a,b,pa,pb))

    # Exact future-value output gate: values never appeared in planner training because planner never saw values.
    runtime_checks=[]; states=[]; factual_context_tokens=[]
    started=time.perf_counter()
    for a,b,pa,pb in pair_bindings:
        for op in OPS:
            with torch.inference_mode(): routed=OPS[int(planner(features[(op,a,b)].unsqueeze(0)).argmax(-1).item())]
            state=execute(world,routed,pa,pb,yes_ids,no_ids)
            oracle=execute(world,op,pa,pb,yes_ids,no_ids)
            runtime_checks.append(routed==op and state.output_ids==oracle.output_ids and state.deps==oracle.deps)
            states.append(state)
            pga=world.page(pa);pgb=world.page(pb)
            if pga is not None and pgb is not None:
                # optimistic text-RAG would need to serialize both current values for generic two-entity operations.
                factual_context_tokens.append(len(value_token_ids[pga.value_idx])+len(value_token_ids[pgb.value_idx])+8)

    # 50k future world updates after all neural training. Planner features remain reusable.
    update_checks=[]; update_ns=[]; old_state_invalidations=[]
    bindings_only=[(pa,pb) for _a,_b,pa,pb in pair_bindings]
    for step in range(50_000):
        pa,pb=bindings_only[rng.randrange(len(bindings_only))]
        pid=pa if rng.random()<0.5 else pb
        # Capture a state before mutation, then prove its generation lifetime expires.
        op=OPS[rng.randrange(len(OPS))]
        old=execute(world,op,pa,pb,yes_ids,no_ids)
        t0=time.perf_counter_ns()
        if rng.random()<0.03:
            world.revoke(pid)
        else:
            # Includes same-value/same-flag rewrites: generation still changes.
            p=world.pages[pid]
            new_idx=p.value_idx if rng.random()<0.2 else rng.randrange(len(VALUE_STRINGS))
            new_flag=p.flag if rng.random()<0.2 else bool(rng.getrandbits(1))
            world.update(pid,new_idx,new_flag)
        update_ns.append(time.perf_counter_ns()-t0)
        old_state_invalidations.append(not world.valid(old) if old.deps else True)

        # revive a revoked page immediately in half the revoke cases; old state must stay dead.
        if not world.pages[pid].live and rng.random()<0.5:
            world.update(pid,rng.randrange(len(VALUE_STRINGS)),bool(rng.getrandbits(1)))
            old_state_invalidations.append(not world.valid(old) if old.deps else True)

        # Periodic correctness check on current world.
        if step % 10 == 0:
            a,b,paa,pbb=pair_bindings[rng.randrange(len(pair_bindings))]
            op=OPS[rng.randrange(len(OPS))]
            with torch.inference_mode(): routed=OPS[int(planner(features[(op,a,b)].unsqueeze(0)).argmax(-1).item())]
            got=execute(world,routed,paa,pbb,yes_ids,no_ids); oracle=execute(world,op,paa,pbb,yes_ids,no_ids)
            update_checks.append(routed==op and got.output_ids==oracle.output_ids and got.deps==oracle.deps)

    # Alias resolution equivalence: different late aliases reach the same exact canonical execution state.
    alias_identity=[]
    for i,(a,b,pa,pb) in enumerate(pair_bindings[:12]):
        la=f"late synonym A {i}";lb=f"late synonym B {i}"
        ra=world.resolve_alias(a);rb=world.resolve_alias(b);rla=world.resolve_alias(la);rlb=world.resolve_alias(lb)
        alias_identity.append(ra==rla==pa and rb==rlb==pb)

    # Storage: late-bound value page stores token IDs + generation/flag metadata rather than KV or hidden vector.
    token_storage_bytes=[4*len(ids) for ids in value_token_ids]
    mean_value_token_bytes=statistics.mean(token_storage_bytes)
    bf16_hidden_bytes=d*2

    report={
        "stage":"R308-LATE-BOUND-HANDLE-VM",
        "architecture_candidate":"Late-Bound Knowledge Execution (LBKE) over CKCA",
        "model_id":MODEL_ID,"revision":REVISION,"weights_sha256":hashes,"backbone_frozen":all(not p.requires_grad for p in model.parameters()),
        "planner_parameters":sum(p.numel() for p in planner.parameters()),"planner_training_steps":300,
        "heldout_operation_routing_accuracy":held_route,"late_alias_operation_routing_accuracy":alias_route,
        "future_multitoken_value_count":len(VALUE_STRINGS),"mean_future_value_token_length":statistics.mean(len(x) for x in value_token_ids),"max_future_value_token_length":max(len(x) for x in value_token_ids),
        "initial_runtime_exactness":sum(runtime_checks)/len(runtime_checks),
        "online_post_training_update_exactness":sum(update_checks)/len(update_checks),"online_world_updates":50000,"optimizer_steps_after_world_updates":0,
        "old_jstate_generation_invalidation_rate":sum(old_state_invalidations)/len(old_state_invalidations),
        "alias_to_canonical_pod_identity_rate":sum(alias_identity)/len(alias_identity),
        "median_runtime_update_ns":statistics.median(update_ns),"p99_runtime_update_ns":sorted(update_ns)[int(.99*len(update_ns))],
        "mean_late_bound_value_payload_bytes_token_ids_only":mean_value_token_bytes,"bf16_hidden_vector_bytes":bf16_hidden_bytes,"payload_ratio_vs_one_bf16_hidden_vector":mean_value_token_bytes/bf16_hidden_bytes,
        "mean_text_rag_factual_context_tokens_avoided_per_generic_two_entity_query":statistics.mean(factual_context_tokens),
        "execution_seconds_after_feature_extraction":time.perf_counter()-started,
        "mechanism":(
            "The frozen B-plane compiles language into an operation, but current values remain opaque generation-scoped "
            "handles. J-Space executes typed operations over canonical Pods and carries exact token-sequence value handles "
            "without embedding each future value into persistent neural state. Materialization is delayed until output."
        ),
        "architectural_hypothesis":(
            "Mutable world data should be bound as late as possible. A neural system should learn operators over typed "
            "capabilities, not memorize future entity/value bindings. Late binding makes arbitrary multi-token future values "
            "exactly writable/revocable and reduces the amount of generation-dependent neural state that can become stale."
        ),
        "claim_boundary":(
            "Pointer/copy mechanisms, symbolic execution and late binding are established ideas. R308 tests their joint use "
            "inside CKCA's generation-lifetime contract; it is not a standalone novelty claim or free-form generation result."
        ),
        "dod_status":"NOT_DOD; late-binding multi-token architecture gate",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode();report["report_sha256"]=hashlib.sha256(canonical).hexdigest();OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("heldout_operation_routing_accuracy","late_alias_operation_routing_accuracy","initial_runtime_exactness","online_post_training_update_exactness","old_jstate_generation_invalidation_rate","alias_to_canonical_pod_identity_rate","mean_future_value_token_length","payload_ratio_vs_one_bf16_hidden_vector","median_runtime_update_ns","p99_runtime_update_ns")},indent=2))
    if held_route<.97 or alias_route<.97:return 2
    if report["initial_runtime_exactness"]<.99 or report["online_post_training_update_exactness"]<.99:return 3
    if report["old_jstate_generation_invalidation_rate"]!=1.0:return 4
    if report["alias_to_canonical_pod_identity_rate"]!=1.0:return 5
    return 0

if __name__=="__main__":raise SystemExit(main())
