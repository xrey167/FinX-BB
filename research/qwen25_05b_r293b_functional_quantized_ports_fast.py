from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import (
    candidate_token, concat_cache, legacy_cache, query_logits, run_segment, sha256_file, slice_cache,
)
from research.qwen25_05b_r288_shared_latent_port_codec import MODEL_ID, REVISION, EXPECTED_WEIGHTS, VALUE_POOL
from research.qwen25_05b_r290_quantized_residual_ports import mean_cache, quantize_cache
from research.qwen25_05b_r293_functional_quantized_ports import render, top, expected

OUT = Path(os.environ.get("SO_R293B_REPORT", "ci-qwen-r293b/report.json"))
PROBE_OPS = ("read", "alternating", "edge8")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)

    model_dir = Path(snapshot_download(
        repo_id=MODEL_ID, revision=REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json"],
    ))
    hashes={n:sha256_file(model_dir/n) for n in EXPECTED_WEIGHTS}; assert hashes==EXPECTED_WEIGHTS
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    tok.pad_token_id=tok.eos_token_id
    model=AutoModelForCausalLM.from_pretrained(
        model_dir,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation="eager",trust_remote_code=False,
    ).eval(); model.requires_grad_(False)

    layouts={(w,op):render(tok,w,op) for w in VALUE_POOL for op in PROBE_OPS}
    groups={}
    for word in VALUE_POOL:
        try: candidate_token(tok,word)
        except RuntimeError: continue
        static_variants={tuple(layouts[(word,op)]["static_ids"]) for op in PROBE_OPS}
        dyn_variants={tuple(layouts[(word,op)]["dynamic_ids"]) for op in PROBE_OPS}
        if len(static_variants)!=1 or len(dyn_variants)!=1: continue
        key=(next(iter(static_variants)),len(next(iter(dyn_variants))))
        groups.setdefault(key,[]).append(word)
    if not groups: raise RuntimeError("no stable group")
    group_key,eligible=max(groups.items(),key=lambda kv:len(kv[1])); eligible=list(eligible)
    if len(eligible)<12: raise RuntimeError(f"stable group too small {eligible}")

    # Keep only four future words for the expensive real-model functional gate.
    holdout_words=eligible[-4:]
    train_words=eligible[:-4]
    static_ids=list(group_key[0]); D=int(group_key[1]); S=len(static_ids)
    st=torch.tensor([static_ids],dtype=torch.long)
    with torch.inference_mode():
        sout=model(input_ids=st,attention_mask=torch.ones_like(st),use_cache=True,return_dict=True)
    shared=legacy_cache(sout.past_key_values); shared_mask=torch.ones((1,S),dtype=torch.long)

    capsules={}; prefixes={}
    for word in eligible:
        out=run_segment(model,layouts[(word,"read")]["dynamic_ids"],shared,shared_mask)
        pc=legacy_cache(out.past_key_values); prefixes[word]=pc; capsules[word]=slice_cache(pc,S,S+D)
    anchor=mean_cache([capsules[w] for w in train_words])
    prefix_mask=torch.ones((1,S+D),dtype=torch.long)

    compressed={}; storage={}; relerr={}
    for word in holdout_words:
        qc,nbytes,err=quantize_cache(capsules[word],k_bits=3,v_bits=3,group_size=32,anchor=anchor,correction_topk=0)
        compressed[word]=concat_cache(shared,qc); storage[word]=nbytes; relerr[word]=err

    rows=[]; all_match=[]; all_exact=[]; all_comp=[]; all_delta=[]; op_summary={}
    for op in PROBE_OPS:
        om=[]; oe=[]; oc=[]; od=[]
        for word in holdout_words:
            qids=layouts[(word,op)]["query_ids"]
            exact_logits=query_logits(model,qids,prefixes[word],prefix_mask)
            comp_logits=query_logits(model,qids,compressed[word],prefix_mask)
            et=top(tok,exact_logits,op,eligible); ct=top(tok,comp_logits,op,eligible); exp=expected(op,word)
            delta=float(torch.max(torch.abs(exact_logits-comp_logits)).item())
            match=et==ct; eok=et==exp; cok=ct==exp
            all_match.append(match); all_exact.append(eok); all_comp.append(cok); all_delta.append(delta)
            om.append(match); oe.append(eok); oc.append(cok); od.append(delta)
            rows.append({"op":op,"word":word,"expected":exp,"exact_top":et,"compressed_top":ct,"top_match":match,"exact_correct":eok,"compressed_correct":cok,"max_vocab_logit_delta":delta,"mean_tensor_relative_l2_error":relerr[word],"per_fact_bytes":storage[word]})
        op_summary[op]={"top_match_rate":sum(om)/len(om),"exact_semantic_accuracy":sum(oe)/len(oe),"compressed_semantic_accuracy":sum(oc)/len(oc),"max_vocab_logit_delta":max(od)}

    full_bf16_bytes=sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in legacy_cache(capsules[holdout_words[0]]))
    per_fact=max(storage.values())
    report={
        "stage":"R293B-BOUNDED-FUNCTIONAL-QUANTIZED-PORTS",
        "architecture_candidate":"Query-Independent Quantized Temporal Ports",
        "model_id":MODEL_ID,"revision":REVISION,"weights_sha256":hashes,
        "backbone_frozen":all(not p.requires_grad for p in model.parameters()),
        "codec":"train-only shared mean anchor + groupwise residual K3/V3, group=32",
        "per_fact_gradient_steps":0,
        "train_words_for_anchor":train_words,"future_holdout_words":holdout_words,"operations":list(PROBE_OPS),
        "overall_top_match_rate_vs_exact":sum(all_match)/len(all_match),
        "overall_exact_semantic_accuracy":sum(all_exact)/len(all_exact),
        "overall_compressed_semantic_accuracy":sum(all_comp)/len(all_comp),
        "max_vocab_logit_delta":max(all_delta),
        "full_bf16_capsule_bytes":full_bf16_bytes,
        "compressed_per_fact_bytes":per_fact,
        "compressed_ratio_vs_bf16":per_fact/full_bf16_bytes,
        "op_summary":op_summary,"rows":rows,
        "mechanism":"One query-independent compressed Port capsule per fact is reused unchanged across read, set-membership and relabeled classification operations.",
        "claim_boundary":"Bounded real-model functional gate; four future holdout values and three operations. Not arbitrary-task equivalence or novelty.",
        "dod_status":"NOT_DOD; bounded functional compact-Port gate",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode(); report["report_sha256"]=hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"holdout_words":holdout_words,"operations":PROBE_OPS,"top_match":report["overall_top_match_rate_vs_exact"],"exact_acc":report["overall_exact_semantic_accuracy"],"compressed_acc":report["overall_compressed_semantic_accuracy"],"max_delta":report["max_vocab_logit_delta"],"ratio":report["compressed_ratio_vs_bf16"]},indent=2))
    if report["overall_exact_semantic_accuracy"]<0.90:return 2
    if report["overall_compressed_semantic_accuracy"]<0.90:return 3
    if report["overall_top_match_rate_vs_exact"]<0.90:return 4
    return 0

if __name__=="__main__": raise SystemExit(main())
