from __future__ import annotations

import hashlib, json, os, random, statistics, time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="Qwen/Qwen2.5-0.5B"; REV="060db6499f32faf8b98477b0a26969ef7d8b9987"
WEIGHT_SHA="88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
OUT=Path(os.environ.get("SO_R262_REPORT","ci-qwen-r262/report.json"))
ENTITIES=["Luma Harbor","Selune Registry","Pelmora Index","Orinth Vale","Vexa Foundry","Teralis Point","Nemor Gate","Caldrin Works","Orelith Station","Mireva Labs","Solune Archive","Praxen Yard","Velora Depot","Aster Quay","Nivora Hall","Demeris Node"]
CODES=["amber","silver","violet","cobalt","scarlet","ivory","crimson","azure","coral","delta","sigma","omega","orbit","signal","kernel","vertex"]
PROMPT="Database facts: {a} has registry code {va}. {b} has registry code {vb}.\nQuestion: Are the two registry codes exactly the same? Answer exactly yes or no:"
SLOT="Database facts: {a} has registry code X. {b} has registry code Y.\nQuestion: Are the two registry codes exactly the same? Answer exactly yes or no:"


def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def one(tok,s):
    ids=tok.encode(" "+s,add_special_tokens=False); return ids[0] if len(ids)==1 else None

@dataclass(frozen=True)
class Page:
    entity:str; literal:str; token_id:int; generation:int; state:str="SET"

def compile_pair(tok,a:Page,b:Page):
    if a.state!="SET" or b.state!="SET": return None
    full=PROMPT.format(a=a.entity,b=b.entity,va=a.literal,vb=b.literal)
    slot=SLOT.format(a=a.entity,b=b.entity)
    bf=tok(full,return_tensors="pt",add_special_tokens=False); bs=tok(slot,return_tensors="pt",add_special_tokens=False)
    if bf.input_ids.shape!=bs.input_ids.shape: return None
    diff=(bf.input_ids[0]!=bs.input_ids[0]).nonzero().flatten()
    if diff.numel()!=2: return None
    positions=[int(x) for x in diff]
    full_ids=[int(bf.input_ids[0,p]) for p in positions]
    if full_ids != [a.token_id,b.token_id]: return None
    return bf,bs,positions

def run_case(model,tok,a:Page,b:Page,yes_id:int,no_id:int):
    c=compile_pair(tok,a,b)
    if c is None: return None
    bf,bs,pos=c
    emb=model.get_input_embeddings()(bs.input_ids).detach().clone()
    emb[0,pos[0]]=model.get_input_embeddings().weight[a.token_id]
    emb[0,pos[1]]=model.get_input_embeddings().weight[b.token_id]
    t=time.perf_counter_ns()
    with torch.inference_mode(): slot_logits=model(inputs_embeds=emb,attention_mask=bs.attention_mask,return_dict=True).logits[:,-1,:]
    slot_ns=time.perf_counter_ns()-t
    with torch.inference_mode(): full_logits=model(**bf,return_dict=True).logits[:,-1,:]
    with torch.inference_mode(): base_logits=model(**bs,return_dict=True).logits[:,-1,:]
    same=a.literal==b.literal; target=yes_id if same else no_id
    pred=int(slot_logits.argmax(-1).item()); fpred=int(full_logits.argmax(-1).item()); bpred=int(base_logits.argmax(-1).item())
    # binary score is useful even when model emits punctuation/other token as global argmax
    binary_pred=yes_id if float(slot_logits[0,yes_id])>=float(slot_logits[0,no_id]) else no_id
    binary_full=yes_id if float(full_logits[0,yes_id])>=float(full_logits[0,no_id]) else no_id
    binary_base=yes_id if float(base_logits[0,yes_id])>=float(base_logits[0,no_id]) else no_id
    return {
        "same":same,"target":target,"argmax_pred":pred,"argmax_correct":pred==target,
        "binary_pred":binary_pred,"binary_correct":binary_pred==target,
        "full_binary_correct":binary_full==target,"baseline_binary_correct":binary_base==target,
        "slot_full_binary_same":binary_pred==binary_full,
        "max_logit_error_vs_full_text":float((slot_logits-full_logits).abs().max().item()),
        "slot_ns":slot_ns,
        "yes_logit":float(slot_logits[0,yes_id]),"no_logit":float(slot_logits[0,no_id]),
    }

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    md=Path(snapshot_download(repo_id=MODEL_ID,revision=REV,allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors"]))
    assert sha(md/'model.safetensors')==WEIGHT_SHA
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]
    yes_id=one(tok,"yes"); no_id=one(tok,"no")
    if yes_id is None or no_id is None: raise RuntimeError("yes/no must be single tokens")
    usable=[(x,one(tok,x)) for x in CODES]; usable=[x for x in usable if x[1] is not None]
    if len(usable)<8: raise RuntimeError("too few one-token codes")
    rng=random.Random(262)
    pages={e:Page(e,usable[i%len(usable)][0],usable[i%len(usable)][1],1) for i,e in enumerate(ENTITIES)}
    # Make several exact-equality pairs by sharing codes, and several different pairs.
    pairs=[]
    for i in range(0,8,2):
        a=ENTITIES[i]; b=ENTITIES[i+1]; p=pages[a]; pages[b]=Page(b,p.literal,p.token_id,1); pairs.append((a,b))
    for _ in range(28):
        a,b=rng.sample(ENTITIES,2); pairs.append((a,b))
    rows=[]
    for a,b in pairs:
        r=run_case(model,tok,pages[a],pages[b],yes_id,no_id)
        if r is not None:
            r.update({"a":a,"b":b,"a_value":pages[a].literal,"b_value":pages[b].literal})
            rows.append(r)
    if not rows: raise RuntimeError("no valid reasoning cases")
    binary_acc=sum(int(r['binary_correct']) for r in rows)/len(rows)
    full_acc=sum(int(r['full_binary_correct']) for r in rows)/len(rows)
    parity=sum(int(r['slot_full_binary_same']) for r in rows)/len(rows)
    baseline=sum(int(r['baseline_binary_correct']) for r in rows)/len(rows)
    maxerr=max(r['max_logit_error_vs_full_text'] for r in rows)

    # Lifecycle flip: two equal codes in v1, edit one page in v2 so the correct answer becomes no.
    a,b=ENTITIES[0],ENTITIES[1]; v1a,v1b=pages[a],pages[b]
    different=next(x for x in usable if x[0]!=v1a.literal)
    v2a=Page(a,different[0],different[1],2)
    r_v1=run_case(model,tok,v1a,v1b,yes_id,no_id); r_v2=run_case(model,tok,v2a,v1b,yes_id,no_id); r_rb=run_case(model,tok,v1a,v1b,yes_id,no_id)
    masked=Page(a,v2a.literal,v2a.token_id,3,state="MASK")
    mask_authorized=compile_pair(tok,masked,v1b) is not None

    report={
        "stage":"R262-REAL-QWEN-REASONING-OVER-NOVEL-FACTS",
        "model_id":MODEL_ID,"revision":REV,"parameter_count":sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "cases":len(rows),"binary_reasoning_accuracy":binary_acc,"full_text_reference_accuracy":full_acc,
        "neural_slot_vs_full_text_decision_parity":parity,"baseline_placeholder_binary_accuracy":baseline,
        "max_logit_error_vs_full_text":maxerr,"median_query_ms":statistics.median(r['slot_ns'] for r in rows)/1e6,
        "runtime_knowledge_text_tokens":0,"per_fact_gradient_steps":0,
        "lifecycle":{"v1_same_correct":bool(r_v1 and r_v1['binary_correct']),"v2_changed_correct":bool(r_v2 and r_v2['binary_correct']),"rollback_correct":bool(r_rb and r_rb['binary_correct']),"mask_authorized":mask_authorized,"old_snapshot_value_unchanged":v1a.literal==v1b.literal},
        "scientific_scope":"real frozen Qwen reasoning over two post-training typed Fabric values supplied only as neural slots; equality decision is produced by Qwen, not stored as knowledge. Reference surface only, not final compact ABI or full DoD.",
        "dod_status":"NOT_DOD; real-model novel-fact reasoning evidence"
    }
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,indent=2,sort_keys=True))
    if parity<0.999 or maxerr>1e-5: return 2
    if binary_acc<0.85: return 3
    if not (report['lifecycle']['v1_same_correct'] and report['lifecycle']['v2_changed_correct'] and report['lifecycle']['rollback_correct'] and not mask_authorized): return 4
    return 0
if __name__=='__main__': raise SystemExit(main())
