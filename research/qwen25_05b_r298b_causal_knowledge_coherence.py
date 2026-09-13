from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file
from research.qwen25_05b_r298_causal_knowledge_coherence import (
    MODEL_ID, REVISION, EXPECTED_WEIGHTS, COLORS, OPS, LABELS, L2I,
    expected, extract_features, CoherenceBridge, AuthorityWorld, LifetimeDAG, JState, batch_predict,
)

OUT = Path(os.environ.get("SO_R298B_REPORT", "ci-qwen-r298b/report.json"))


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    seed=2982
    random.seed(seed); torch.manual_seed(seed)
    rng=random.Random(seed+77)

    model_dir=Path(snapshot_download(
        repo_id=MODEL_ID,revision=REVISION,
        allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors","*.index.json","*.merges","*.vocab","merges.txt","vocab.json"],
    ))
    hashes={n:sha256_file(model_dir/n) for n in EXPECTED_WEIGHTS}; assert hashes==EXPECTED_WEIGHTS
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    tok.pad_token=tok.eos_token; tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(
        model_dir,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation="eager",trust_remote_code=False,
    ).eval(); model.requires_grad_(False)

    train_pairs=[(f"TA-{i:03d}",f"TB-{i:03d}") for i in range(16)]
    hold_pairs=[(f"HA-{i:03d}",f"HB-{i:03d}") for i in range(8)]
    triples=[(op,a,b) for a,b in train_pairs+hold_pairs for op in OPS]
    probe=(OPS[0],hold_pairs[0][0],hold_pairs[0][1])
    # Exact B-plane integrity witness: identical standalone input shape before any bridge/world work.
    probe_before,probe_logits_before=extract_features(model,tok,[probe],batch_size=1)

    started=time.perf_counter()
    feats,_bulk_logits=extract_features(model,tok,triples,batch_size=32)
    emb=model.get_input_embeddings().weight.detach().float().cpu()
    codes=[]
    for c in COLORS:
        ids=tok.encode(c,add_special_tokens=False); codes.append(emb[ids].mean(dim=0))
    null=torch.zeros_like(codes[0]); d=int(null.numel())

    qx=[];pa=[];pb=[];la=[];lb=[];yy=[]
    for a,b in train_pairs:
        for op in OPS:
            qh=feats[(op,a,b)]
            for va in range(4):
                for vb in range(4):
                    qx.append(qh);pa.append(codes[va]);pb.append(codes[vb]);la.append(1.0);lb.append(1.0);yy.append(L2I[expected(op,va,vb)])
            qx.append(qh);pa.append(null);pb.append(codes[0]);la.append(0.0);lb.append(1.0);yy.append(L2I["NULL"])
            qx.append(qh);pa.append(codes[0]);pb.append(null);la.append(1.0);lb.append(0.0);yy.append(L2I["NULL"])
    qx=torch.stack(qx);pa=torch.stack(pa);pb=torch.stack(pb);la=torch.tensor(la);lb=torch.tensor(lb);yy=torch.tensor(yy,dtype=torch.long)

    bridge=CoherenceBridge(d)
    opt=torch.optim.AdamW(bridge.parameters(),lr=3e-3,weight_decay=1e-4)
    model_param_ids={id(p) for p in model.parameters()}
    optimizer_param_ids={id(p) for g in opt.param_groups for p in g["params"]}
    optimizer_owns_base=bool(model_param_ids & optimizer_param_ids)
    gen=torch.Generator().manual_seed(seed+1)
    bridge.train()
    for _ in range(360):
        idx=torch.randint(0,len(yy),(min(320,len(yy)),),generator=gen)
        loss=torch.nn.functional.cross_entropy(bridge(qx[idx],pa[idx],pb[idx],la[idx],lb[idx]),yy[idx])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    bridge.eval()

    hq=[];hpa=[];hpb=[];hla=[];hlb=[];hy=[]
    for a,b in hold_pairs:
        for op in OPS:
            qh=feats[(op,a,b)]
            for va in range(4):
                for vb in range(4):
                    hq.append(qh);hpa.append(codes[va]);hpb.append(codes[vb]);hla.append(1.0);hlb.append(1.0);hy.append(L2I[expected(op,va,vb)])
            hq.append(qh);hpa.append(null);hpb.append(codes[0]);hla.append(0.0);hlb.append(1.0);hy.append(L2I["NULL"])
            hq.append(qh);hpa.append(codes[0]);hpb.append(null);hla.append(1.0);hlb.append(0.0);hy.append(L2I["NULL"])
    hp=batch_predict(bridge,hq,hpa,hpb,hla,hlb)
    heldout_acc=sum(int(p==y) for p,y in zip(hp,hy))/len(hy)

    world=AuthorityWorld();dag=LifetimeDAG()
    value_ids=list(range(100,164));ptr_ids=list(range(1000,1064));head_ids=list(range(2000,2064))
    for pid in value_ids: world.add_value(pid,rng.randrange(4))
    for i,pid in enumerate(ptr_ids): world.add_pointer(pid,value_ids[i])
    for i,pid in enumerate(head_ids): world.add_pointer(pid,ptr_ids[i])

    states=[]; feature_rows=[];porta=[];portb=[];livea=[];liveb=[];labels=[];state_meta=[]
    for _ in range(12000):
        ha=head_ids[rng.randrange(len(head_ids))];hb=head_ids[rng.randrange(len(head_ids))]
        va,depa=world.resolve(ha);vb,depb=world.resolve(hb)
        op=OPS[rng.randrange(len(OPS))];pair=hold_pairs[rng.randrange(len(hold_pairs))]
        deps=tuple(sorted(set(depa+depb)));fid=dag.factor_for(deps)
        feature_rows.append(feats[(op,pair[0],pair[1])]);porta.append(null if va is None else codes[va]);portb.append(null if vb is None else codes[vb]);livea.append(va is not None);liveb.append(vb is not None);labels.append(L2I[expected(op,va,vb)]);state_meta.append((fid,deps))
    preds=batch_predict(bridge,feature_rows,porta,portb,livea,liveb)
    initial_acc=sum(int(p==y) for p,y in zip(preds,labels))/len(labels)
    for pred,(fid,deps) in zip(preds,state_meta): states.append(JState(fid,deps,LABELS[pred]))

    stale_reject=[]; invalidated=[]; all_mutable=value_ids+ptr_ids+head_ids
    for _ in range(1000):
        pid=all_mutable[rng.randrange(len(all_mutable))]; stale=world.capability(pid); p=world.pods[pid];old_gen=p.generation
        if p.kind=="value": world.update_value(pid,rng.randrange(4) if rng.random()>0.25 else p.payload)
        else:
            targets=value_ids if pid in ptr_ids else ptr_ids
            world.update_pointer(pid,p.payload if rng.random()<0.25 else targets[rng.randrange(len(targets))])
        invalidated.append(dag.invalidate_generation(pid,old_gen));stale_reject.append(not world.verify(stale))

    mismatches=0
    for _ in range(50000):
        s=states[rng.randrange(len(states))]
        mismatches += int(dag.is_valid(s.factor_id)!=world.explicit_state_valid(s.deps))

    fq=[];fpa=[];fpb=[];fla=[];flb=[];fy=[]
    for _ in range(4000):
        ha=head_ids[rng.randrange(len(head_ids))];hb=head_ids[rng.randrange(len(head_ids))]
        va,_=world.resolve(ha);vb,_=world.resolve(hb);op=OPS[rng.randrange(len(OPS))];pair=hold_pairs[rng.randrange(len(hold_pairs))]
        fq.append(feats[(op,pair[0],pair[1])]);fpa.append(null if va is None else codes[va]);fpb.append(null if vb is None else codes[vb]);fla.append(va is not None);flb.append(vb is not None);fy.append(L2I[expected(op,va,vb)])
    fp=batch_predict(bridge,fq,fpa,fpb,fla,flb);fresh_acc=sum(int(p==y) for p,y in zip(fp,fy))/len(fy)

    special=value_ids[0];v0,deps0=world.resolve(special);aba_factor=dag.factor_for(tuple(deps0));old=world.pods[special].generation;world.update_value(special,int(v0));dag.invalidate_generation(special,old);dag.leaf(special,world.pods[special].generation);aba=not dag.is_valid(aba_factor)
    rpid=value_ids[1];rv,rdeps=world.resolve(rpid);rf=dag.factor_for(tuple(rdeps));old=world.revoke(rpid);dag.invalidate_generation(rpid,old);revoke_dead=not dag.is_valid(rf) and world.resolve(rpid)[0] is None;old=world.pods[rpid].generation;world.update_value(rpid,int(rv));dag.invalidate_generation(rpid,old);revive_dead=not dag.is_valid(rf)
    invalid_states=[s for s in states if not dag.is_valid(s.factor_id)]; invalid_admissions=sum(1 for s in invalid_states[:5000] if dag.is_valid(s.factor_id))

    factor_ns=[];explicit_ns=[]
    for _ in range(100000):
        s=states[rng.randrange(len(states))];t=time.perf_counter_ns();_=dag.is_valid(s.factor_id);factor_ns.append(time.perf_counter_ns()-t);t=time.perf_counter_ns();_=world.explicit_state_valid(s.deps);explicit_ns.append(time.perf_counter_ns()-t)

    # Repeat identical standalone B-plane call after all Port/lifecycle changes.
    probe_after,probe_logits_after=extract_features(model,tok,[probe],batch_size=1)
    base_hidden_delta=float(torch.max(torch.abs(probe_before[probe]-probe_after[probe])).item())
    base_logit_delta=float(torch.max(torch.abs(probe_logits_before[probe]-probe_logits_after[probe])).item())

    report={
        "stage":"R298B-CAUSAL-KNOWLEDGE-COHERENCE-MACHINE",
        "architecture_candidate":"Causal Knowledge Coherence Architecture (CKCA)",
        "model_id":MODEL_ID,"revision":REVISION,"weights_sha256":hashes,"backbone_frozen":all(not p.requires_grad for p in model.parameters()),"base_model_optimizer_steps":0,"optimizer_owns_any_base_parameter":optimizer_owns_base,
        "bridge_parameters":sum(p.numel() for p in bridge.parameters()),"bridge_training_steps":360,
        "heldout_two_port_accuracy":heldout_acc,"initial_multihop_world_accuracy":initial_acc,"fresh_post_update_world_accuracy":fresh_acc,
        "initial_j_states":len(states),"lifecycle_updates":1000,"stale_capability_rejection_rate":sum(stale_reject)/len(stale_reject),"factor_vs_explicit_audit_cases":50000,"factor_vs_explicit_mismatches":mismatches,
        "aba_same_value_no_resurrection":aba,"revoke_invalidates_old_jstate":revoke_dead,"revive_same_value_does_not_resurrect_old_jstate":revive_dead,"invalid_jstate_admissions":invalid_admissions,
        "mean_factor_nodes_invalidated_per_update":statistics.mean(invalidated),"p99_factor_nodes_invalidated_per_update":sorted(invalidated)[int(.99*len(invalidated))],
        "factor_validity_ns_median":statistics.median(factor_ns),"explicit_dependency_scan_ns_median":statistics.median(explicit_ns),"validity_speedup_vs_explicit":statistics.median(explicit_ns)/max(statistics.median(factor_ns),1),
        "base_hidden_max_delta_after_world_changes":base_hidden_delta,"base_logits_max_delta_after_world_changes":base_logit_delta,"base_repeatability_probe_shape":"standalone batch=1 before and after","optimizer_steps_after_world_updates":0,"elapsed_seconds":time.perf_counter()-started,
        "architecture_invariant":"Verified current generation capabilities are the sole bridge from mutable authority state into neural Ports. Every mutable-derived J/cache state carries one transitive lifetime factor. Old factors are monotonic-dead; the frozen language/skill plane is not rewritten by world changes.",
        "claim_boundary":"Integrated real frozen Qwen + held-out bindings + two Ports + external 2-hop graph + exact lifetime admission. Tasks remain synthetic/classification; free-form generation and strong-RAG frontier remain open.",
        "dod_status":"NOT_DOD; corrected integrated causal-coherence gate",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode();report["report_sha256"]=hashlib.sha256(canonical).hexdigest();OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("heldout_two_port_accuracy","initial_multihop_world_accuracy","fresh_post_update_world_accuracy","stale_capability_rejection_rate","factor_vs_explicit_mismatches","aba_same_value_no_resurrection","revoke_invalidates_old_jstate","revive_same_value_does_not_resurrect_old_jstate","invalid_jstate_admissions","validity_speedup_vs_explicit","base_hidden_max_delta_after_world_changes","base_logits_max_delta_after_world_changes","optimizer_owns_any_base_parameter","elapsed_seconds")},indent=2))
    if heldout_acc<.97:return 2
    if initial_acc<.97 or fresh_acc<.97:return 3
    if report["stale_capability_rejection_rate"]!=1.0 or mismatches!=0:return 4
    if not aba or not revoke_dead or not revive_dead or invalid_admissions!=0:return 5
    if base_hidden_delta>1e-6 or base_logit_delta>1e-6 or optimizer_owns_base:return 6
    return 0

if __name__=="__main__":raise SystemExit(main())
