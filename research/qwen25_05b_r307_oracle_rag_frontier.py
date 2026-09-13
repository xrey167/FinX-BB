from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file, candidate_token
from research.qwen25_05b_r298_causal_knowledge_coherence import (
    MODEL_ID, REVISION, EXPECTED_WEIGHTS, COLORS, OPS, LABELS, L2I,
    expected, qtext, CoherenceBridge, batch_predict,
)

OUT = Path(os.environ.get("SO_R307_REPORT", "ci-qwen-r307/report.json"))


def ckca_prompt(tok, op: str, a: str, b: str) -> str:
    system=(
        "Mutable current values are provided through a separate trusted knowledge Port plane. "
        "Interpret the requested operation; the text does not contain current values."
    )
    return tok.apply_chat_template(
        [{"role":"system","content":system},{"role":"user","content":qtext(op,a,b)}],
        tokenize=False,add_generation_prompt=True,
    )


def rag_prompt(tok, op: str, a: str, b: str, va: int, vb: int) -> str:
    system=(
        "Use the retrieved current facts below as authoritative. Answer the user's operation exactly in the requested one-word format."
    )
    context=f"Retrieved current facts:\n- {a} current color: {COLORS[va]}.\n- {b} current color: {COLORS[vb]}."
    user=context+"\n\n"+qtext(op,a,b)
    return tok.apply_chat_template(
        [{"role":"system","content":system},{"role":"user","content":user}],
        tokenize=False,add_generation_prompt=True,
    )


def extract_features(model,tok,pairs,batch_size=16):
    feats={};token_counts=[]
    started=time.perf_counter()
    for start in range(0,len(pairs),batch_size):
        batch=pairs[start:start+batch_size]
        texts=[ckca_prompt(tok,op,a,b) for op,a,b in batch]
        enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False)
        token_counts.extend(enc.attention_mask.sum(dim=1).tolist())
        with torch.inference_mode(): out=model(**enc,use_cache=False,output_hidden_states=True,return_dict=True)
        idx=out.hidden_states[-1].shape[1]-1
        for i,key in enumerate(batch): feats[key]=out.hidden_states[-1][i,idx].float().cpu()
    return feats,token_counts,time.perf_counter()-started


def rag_logits(model,tok,rows,batch_size=16):
    outputs=[];token_counts=[]
    started=time.perf_counter()
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size]
        texts=[rag_prompt(tok,*x) for x in batch]
        enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False)
        token_counts.extend(enc.attention_mask.sum(dim=1).tolist())
        with torch.inference_mode(): out=model(**enc,use_cache=False,return_dict=True)
        idx=out.logits.shape[1]-1
        outputs.extend(out.logits[:,idx,:].float().cpu())
    return outputs,token_counts,time.perf_counter()-started


def main()->int:
    OUT.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    seed=307
    random.seed(seed);torch.manual_seed(seed)

    md=Path(snapshot_download(
        repo_id=MODEL_ID,revision=REVISION,
        allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors","*.index.json","*.merges","*.vocab","merges.txt","vocab.json"],
    ))
    hashes={n:sha256_file(md/n) for n in EXPECTED_WEIGHTS};assert hashes==EXPECTED_WEIGHTS
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False)
    tok.pad_token=tok.eos_token;tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(
        md,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation="eager",trust_remote_code=False,
    ).eval();model.requires_grad_(False)

    train_pairs=[(f"TR-A{i:02d}",f"TR-B{i:02d}") for i in range(16)]
    hold_pairs=[(f"HO-A{i:02d}",f"HO-B{i:02d}") for i in range(8)]
    unique_queries=[(op,a,b) for a,b in train_pairs+hold_pairs for op in OPS]
    features,base_token_counts,base_feature_seconds=extract_features(model,tok,unique_queries,batch_size=16)

    emb=model.get_input_embeddings().weight.detach().float().cpu()
    codes=[]
    for c in COLORS:
        ids=tok.encode(c,add_special_tokens=False);codes.append(emb[ids].mean(dim=0))
    d=int(codes[0].numel())

    q=[];pa=[];pb=[];la=[];lb=[];y=[]
    for a,b in train_pairs:
        for op in OPS:
            qh=features[(op,a,b)]
            for va in range(4):
                for vb in range(4):
                    q.append(qh);pa.append(codes[va]);pb.append(codes[vb]);la.append(1.0);lb.append(1.0);y.append(L2I[expected(op,va,vb)])
    q=torch.stack(q);pa=torch.stack(pa);pb=torch.stack(pb);la=torch.tensor(la);lb=torch.tensor(lb);y=torch.tensor(y)
    bridge=CoherenceBridge(d)
    opt=torch.optim.AdamW(bridge.parameters(),lr=3e-3,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(seed+1)
    bridge.train()
    train_started=time.perf_counter()
    for _ in range(320):
        idx=torch.randint(0,len(y),(min(320,len(y)),),generator=gen)
        loss=nn.functional.cross_entropy(bridge(q[idx],pa[idx],pb[idx],la[idx],lb[idx]),y[idx])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    bridge.eval();bridge_train_seconds=time.perf_counter()-train_started

    # Held-out test grid, all value combinations. CKCA uses one reusable B-state per operation/entity pair.
    ck_q=[];ck_pa=[];ck_pb=[];ck_la=[];ck_lb=[];truth=[];rag_rows=[]
    for a,b in hold_pairs:
        for op in OPS:
            qh=features[(op,a,b)]
            for va in range(4):
                for vb in range(4):
                    ck_q.append(qh);ck_pa.append(codes[va]);ck_pb.append(codes[vb]);ck_la.append(1.0);ck_lb.append(1.0)
                    truth.append(L2I[expected(op,va,vb)]);rag_rows.append((op,a,b,va,vb))
    side_started=time.perf_counter();ck_pred=batch_predict(bridge,ck_q,ck_pa,ck_pb,ck_la,ck_lb,chunk=2048);side_seconds=time.perf_counter()-side_started
    ckca_acc=sum(int(p==t) for p,t in zip(ck_pred,truth))/len(truth)

    rag_out,rag_token_counts,rag_seconds=rag_logits(model,tok,rag_rows,batch_size=16)
    candidate_labels=("red","blue","green","yellow","yes","no","A","B")
    candidate_ids=[candidate_token(tok,x) for x in candidate_labels]
    rag_pred=[]
    for lg in rag_out:
        idx=int(lg[candidate_ids].argmax().item());rag_pred.append(L2I[candidate_labels[idx]])
    rag_acc=sum(int(p==t) for p,t in zip(rag_pred,truth))/len(truth)

    # Cold CKCA token budget repeats the B query for every case; warm/reused budget amortizes
    # the B state across the 16 mutable value combinations for each operation/entity query.
    hold_query_token_map={key:len(tok(ckca_prompt(tok,*key),add_special_tokens=False).input_ids) for key in [(op,a,b) for a,b in hold_pairs for op in OPS]}
    cold_ckca_tokens=[hold_query_token_map[(op,a,b)] for op,a,b,_va,_vb in rag_rows]
    warm_unique_tokens=list(hold_query_token_map.values())

    report={
        "stage":"R307-ORACLE-RAG-FRONTIER",
        "architecture_candidate":"CKCA versus oracle-retrieval text context",
        "model_id":MODEL_ID,"revision":REVISION,"weights_sha256":hashes,"backbone_frozen":all(not p.requires_grad for p in model.parameters()),
        "benchmark_cases":len(truth),"heldout_entity_pairs":len(hold_pairs),"operations":OPS,"value_combinations_per_query":16,
        "ckca_accuracy":ckca_acc,"oracle_text_rag_accuracy":rag_acc,
        "accuracy_delta_ckca_minus_rag":ckca_acc-rag_acc,
        "mean_oracle_rag_prompt_tokens":statistics.mean(rag_token_counts),
        "mean_ckca_cold_prompt_tokens":statistics.mean(cold_ckca_tokens),
        "mean_ckca_unique_base_prompt_tokens":statistics.mean(warm_unique_tokens),
        "rag_total_prompt_tokens":sum(rag_token_counts),
        "ckca_cold_total_prompt_tokens":sum(cold_ckca_tokens),
        "ckca_warm_unique_base_prompt_tokens":sum(warm_unique_tokens),
        "warm_prompt_token_reduction_vs_rag":1.0-sum(warm_unique_tokens)/sum(rag_token_counts),
        "cold_prompt_token_reduction_vs_rag":1.0-sum(cold_ckca_tokens)/sum(rag_token_counts),
        "oracle_rag_batch_forward_seconds":rag_seconds,
        "ckca_all_query_base_feature_seconds_for_train_and_holdout_unique_queries":base_feature_seconds,
        "ckca_heldout_sidecar_seconds_for_all_cases":side_seconds,
        "bridge_training_seconds_one_time":bridge_train_seconds,
        "bridge_parameters":sum(p.numel() for p in bridge.parameters()),
        "retrieval_latency_in_rag_baseline_seconds":0.0,
        "rag_retrieval_assumption":"oracle: correct current facts are always retrieved; no vector search errors or latency",
        "fairness_notes":(
            "This deliberately gives RAG perfect retrieval and current facts. CKCA receives typed current Port values. "
            "The 0.5B model and synthetic operation grid are not representative of production QA. Warm CKCA token cost "
            "uses reusable B-plane query states across repeated world-value changes and is therefore an amortized scenario."
        ),
        "dod_status":"NOT_DOD; first oracle-RAG comparison only",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode();report["report_sha256"]=hashlib.sha256(canonical).hexdigest();OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
    # Exploratory benchmark: do not require CKCA to win; require both measurements to be valid.
    if ckca_acc<0.90:return 2
    if rag_acc<0.50:return 3
    return 0

if __name__=="__main__":raise SystemExit(main())
