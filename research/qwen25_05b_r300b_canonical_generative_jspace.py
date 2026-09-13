from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file, candidate_token
from research.qwen25_05b_r300_generative_jspace_adapter import (
    MODEL_ID, REVISION, EXPECTED_WEIGHTS, COLORS, OPS, TARGET_WORDS,
    expected, extract,
)

OUT = Path(os.environ.get("SO_R300B_REPORT", "ci-qwen-r300b/report.json"))


class CanonicalJBridge(nn.Module):
    """Produces an absolute canonical J-Space hidden state from operation + Port.

    The raw alias/entity-specific base hidden state is used only to select a reusable
    operation contract. Entity identity itself is resolved by the Symlink/Port plane.
    """

    def __init__(self, d: int, width: int = 96):
        super().__init__()
        self.op = nn.Linear(d, width, bias=False)
        self.port = nn.Linear(d, width, bias=False)
        self.core = nn.Sequential(
            nn.LayerNorm(width * 2 + 1),
            nn.Linear(width * 2 + 1, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Linear(width, d, bias=False),
        )
        self.out_norm = nn.LayerNorm(d, elementwise_affine=True)

    def forward(self, op_state, port, live, target_norm: float):
        x = torch.cat([self.op(op_state), self.port(port), live.float().unsqueeze(-1)], dim=-1)
        z = self.out_norm(self.core(x))
        # LayerNorm fixes direction/shape; match the frozen model's normal final-state norm.
        return z * (target_norm / (z.norm(dim=-1, keepdim=True) + 1e-8))


def cosine_route(q: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    cn = centroids / (centroids.norm(dim=-1, keepdim=True) + 1e-8)
    return (qn @ cn.T).argmax(dim=-1)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    seed = 3002
    random.seed(seed); torch.manual_seed(seed)
    rng = random.Random(seed + 3)

    model_dir = Path(snapshot_download(
        repo_id=MODEL_ID, revision=REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json"],
    ))
    hashes = {n: sha256_file(model_dir / n) for n in EXPECTED_WEIGHTS}; assert hashes == EXPECTED_WEIGHTS
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    tok.pad_token = tok.eos_token; tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=False,
    ).eval(); model.requires_grad_(False)

    target_ids = {w: candidate_token(tok, w) for w in TARGET_WORDS}
    train_entities = [f"TRN-{i:03d}-CJS" for i in range(32)]
    heldout_entities = [f"NEW-{i:03d}-CJS" for i in range(16)]
    late_aliases = [
        f"supplier nickname {i} / late alias" if i % 2 == 0 else f"alias::{i}::unseen-surface-form"
        for i in range(16)
    ]
    pairs = [(op, ent) for ent in train_entities + heldout_entities + late_aliases for op in OPS]
    started = time.perf_counter()
    features, _baseline_logits, hard_negative_ids = extract(model, tok, pairs, batch_size=24, hard_topk=32)

    d = next(iter(features.values())).numel()
    # Operation contracts are centroids over many unrelated training identities.
    centroids = torch.stack([
        torch.stack([features[(op, ent)] for ent in train_entities]).mean(dim=0)
        for op in OPS
    ])
    target_norm = float(torch.stack(list(features.values())).norm(dim=-1).mean())

    def routing_accuracy(entities):
        rows=[]; labels=[]
        for ent in entities:
            for oi,op in enumerate(OPS):
                rows.append(features[(op,ent)]); labels.append(oi)
        pred=cosine_route(torch.stack(rows),centroids).tolist()
        return sum(int(p==y) for p,y in zip(pred,labels))/len(labels), pred, labels

    held_route,_,_=routing_accuracy(heldout_entities)
    alias_route,_,_=routing_accuracy(late_aliases)

    emb = model.get_input_embeddings().weight.detach().float().cpu()
    port_codes=[]
    for color in COLORS:
        ids=tok.encode(color,add_special_tokens=False); port_codes.append(emb[ids].mean(dim=0))
    null=torch.zeros_like(port_codes[0])

    target_set=set(target_ids.values())
    hard=[x for x in sorted(hard_negative_ids) if x not in target_set][:256]
    local_ids=list(target_ids.values())+hard
    local_pos={tid:i for i,tid in enumerate(local_ids)}
    local_rows=model.lm_head.weight.detach()[local_ids].float().cpu()
    candidate_ids=[target_ids[w] for w in TARGET_WORDS]
    candidate_rows=model.lm_head.weight.detach()[candidate_ids].float().cpu()

    bridge=CanonicalJBridge(d)
    opt=torch.optim.AdamW(bridge.parameters(),lr=3e-3,weight_decay=1e-4)
    model_param_ids={id(p) for p in model.parameters()}; opt_ids={id(p) for g in opt.param_groups for p in g["params"]}
    owns_base=bool(model_param_ids & opt_ids)

    # Semantic training states contain no entity identity at all: only operation contract + Port value.
    q=[]; p=[]; live=[]; y=[]; target_vec=[]
    for oi,op in enumerate(OPS):
        for value in list(range(4))+[None]:
            word=expected(op,value)
            q.append(centroids[oi]); p.append(null if value is None else port_codes[value]); live.append(0.0 if value is None else 1.0)
            y.append(local_pos[target_ids[word]]); target_vec.append(model.lm_head.weight.detach()[target_ids[word]].float().cpu())
    q=torch.stack(q);p=torch.stack(p);live=torch.tensor(live);y=torch.tensor(y);target_vec=torch.stack(target_vec)

    gen=torch.Generator().manual_seed(seed+1)
    bridge.train()
    for _ in range(900):
        idx=torch.randint(0,len(y),(128,),generator=gen)
        z=bridge(q[idx],p[idx],live[idx],target_norm)
        logits=z@local_rows.T
        ce=nn.functional.cross_entropy(logits,y[idx])
        cos=1.0-nn.functional.cosine_similarity(z,target_vec[idx],dim=-1).mean()
        loss=ce+0.15*cos
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    bridge.eval()

    def eval_entities(entities):
        raw=[]; ports=[]; lives=[]; expected_words=[]; route_correct=[]
        for ent in entities:
            for oi,op in enumerate(OPS):
                for value in list(range(4))+[None]:
                    raw.append(features[(op,ent)]); ports.append(null if value is None else port_codes[value]);lives.append(0.0 if value is None else 1.0);expected_words.append(expected(op,value));route_correct.append(oi)
        raw=torch.stack(raw); routed=cosine_route(raw,centroids); op_states=centroids[routed]
        with torch.inference_mode():
            z=bridge(op_states,torch.stack(ports),torch.tensor(lives),target_norm)
            pred=(z@candidate_rows.T).argmax(-1).tolist()
        correct=[TARGET_WORDS[int(i)]==w for i,w in zip(pred,expected_words)]
        route_acc=sum(int(int(r)==y) for r,y in zip(routed.tolist(),route_correct))/len(route_correct)
        return sum(correct)/len(correct),route_acc,z,expected_words,routed

    held_acc,held_route2,held_z,held_words,_=eval_entities(heldout_entities)
    alias_acc,alias_route2,alias_z,alias_words,_=eval_entities(late_aliases)

    # Full-vocabulary greedy probes across both unseen entity families.
    combined_z=torch.cat([held_z,alias_z],dim=0);combined_words=held_words+alias_words
    probe_count=min(32,combined_z.shape[0]); idxs=torch.tensor(rng.sample(range(combined_z.shape[0]),probe_count))
    with torch.inference_mode():
        full=model.lm_head(combined_z[idxs].to(model.lm_head.weight.dtype)).float().cpu()
    greedy=full.argmax(-1).tolist(); full_rows=[]
    for row,idx in enumerate(idxs.tolist()):
        word=combined_words[idx];gid=int(greedy[row]);full_rows.append({"expected":word,"greedy_token_id":gid,"greedy_text":tok.decode([gid]),"correct":gid==target_ids[word]})
    full_acc=sum(int(x["correct"]) for x in full_rows)/len(full_rows)

    # Future online writes: routing uses held-out queries, world state only changes Port code.
    uq=[];up=[];ul=[];uw=[]
    for _ in range(10000):
        ent=heldout_entities[rng.randrange(len(heldout_entities))];oi=rng.randrange(len(OPS));op=OPS[oi];raw=rng.randrange(5);value=None if raw==4 else raw
        uq.append(features[(op,ent)]);up.append(null if value is None else port_codes[value]);ul.append(0.0 if value is None else 1.0);uw.append(expected(op,value))
    routes=cosine_route(torch.stack(uq),centroids)
    with torch.inference_mode(): uz=bridge(centroids[routes],torch.stack(up),torch.tensor(ul),target_norm); pred=(uz@candidate_rows.T).argmax(-1).tolist()
    update_acc=sum(int(TARGET_WORDS[int(i)]==w) for i,w in zip(pred,uw))/len(uw)

    report={
        "stage":"R300B-CANONICAL-GENERATIVE-JSPACE",
        "architecture_candidate":"Alias-Invariant Canonical J-Space Execution",
        "model_id":MODEL_ID,"revision":REVISION,"weights_sha256":hashes,"backbone_frozen":all(not p.requires_grad for p in model.parameters()),"optimizer_owns_any_base_parameter":owns_base,
        "bridge_parameters":sum(p.numel() for p in bridge.parameters()),"bridge_training_steps":900,
        "operation_contracts":len(OPS),"train_entities_for_contracts":len(train_entities),"heldout_entities":len(heldout_entities),"late_aliases":len(late_aliases),
        "heldout_operation_route_accuracy":held_route2,"late_alias_operation_route_accuracy":alias_route2,
        "heldout_candidate_token_accuracy":held_acc,"late_alias_candidate_token_accuracy":alias_acc,"full_vocab_greedy_probe_accuracy":full_acc,"full_vocab_probe_cases":len(full_rows),"full_vocab_rows":full_rows,
        "online_world_update_candidate_accuracy":update_acc,"online_world_updates":10000,"optimizer_steps_after_world_updates":0,
        "mechanism":(
            "The B-plane query is canonicalized into an operation contract before Port execution. The canonical J state "
            "therefore no longer carries accidental alias/entity surface variation. The Symlink plane supplies canonical "
            "identity; the Port supplies mutable value; a shared bridge maps operation+Port to a hidden state consumed by "
            "the frozen Qwen LM head."
        ),
        "claim_boundary":(
            "Operation routing is a four-contract prototype gate over synthetic templated questions, not general language "
            "reasoning. The result tests whether canonicalization fixes the alias leakage seen in R300 while preserving "
            "model-native token generation."
        ),
        "dod_status":"NOT_DOD; alias-invariant single-token J-Space generation gate",
        "elapsed_seconds":time.perf_counter()-started,
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode();report["report_sha256"]=hashlib.sha256(canonical).hexdigest();OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("heldout_operation_route_accuracy","late_alias_operation_route_accuracy","heldout_candidate_token_accuracy","late_alias_candidate_token_accuracy","full_vocab_greedy_probe_accuracy","online_world_update_candidate_accuracy","optimizer_owns_any_base_parameter","elapsed_seconds")},indent=2))
    if held_route2<.97 or alias_route2<.97:return 2
    if held_acc<.97 or alias_acc<.97:return 3
    if update_acc<.97:return 4
    if full_acc<.85:return 5
    if owns_base:return 6
    return 0

if __name__=="__main__":raise SystemExit(main())
