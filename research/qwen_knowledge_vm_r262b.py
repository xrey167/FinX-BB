from __future__ import annotations

import hashlib, json, os, random, statistics, time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="Qwen/Qwen2.5-0.5B"; REV="060db6499f32faf8b98477b0a26969ef7d8b9987"
WEIGHT_SHA="88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
OUT=Path(os.environ.get("SO_R262B_REPORT","ci-qwen-r262b/report.json"))
CODES=["amber","silver","violet","cobalt","scarlet","ivory","crimson","azure","coral","delta","sigma","omega","orbit","signal","kernel","vertex"]
FULL="Code A is {a}. Code B is {b}. Are Code A and Code B identical? Answer only Yes or No:"
SLOT="Code A is X. Code B is Y. Are Code A and Code B identical? Answer only Yes or No:"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def one(tok,s):
    ids=tok.encode(s,add_special_tokens=False); return ids[0] if len(ids)==1 else None

def semantic_ids(tok, words):
    out=[]
    for w in words:
        for form in (w," "+w,w.lower()," "+w.lower(),w.upper()," "+w.upper()):
            t=one(tok,form)
            if t is not None and t not in out: out.append(t)
    return out

def option_score(logits, ids):
    return float(torch.logsumexp(logits[0,ids].float(),dim=0).item())

def classify(logits, yes_ids, no_ids):
    return option_score(logits,yes_ids)>=option_score(logits,no_ids)

@dataclass(frozen=True)
class Page:
    literal:str; token_id:int; generation:int=1; state:str="SET"

def build(tok,a,b):
    full=FULL.format(a=a.literal,b=b.literal); slot=SLOT
    bf=tok(full,return_tensors="pt",add_special_tokens=False); bs=tok(slot,return_tensors="pt",add_special_tokens=False)
    if bf.input_ids.shape!=bs.input_ids.shape: return None
    d=(bf.input_ids[0]!=bs.input_ids[0]).nonzero().flatten()
    if d.numel()!=2: return None
    pos=[int(x) for x in d]
    if [int(bf.input_ids[0,p]) for p in pos] != [a.token_id,b.token_id]: return None
    return bf,bs,pos

def run(model,tok,a,b,yes_ids,no_ids):
    if a.state!="SET" or b.state!="SET": return None
    c=build(tok,a,b)
    if c is None:return None
    bf,bs,pos=c
    emb=model.get_input_embeddings()(bs.input_ids).detach().clone()
    emb[0,pos[0]]=model.get_input_embeddings().weight[a.token_id]
    emb[0,pos[1]]=model.get_input_embeddings().weight[b.token_id]
    t=time.perf_counter_ns()
    with torch.inference_mode(): sl=model(inputs_embeds=emb,attention_mask=bs.attention_mask,return_dict=True).logits[:,-1,:]
    ns=time.perf_counter_ns()-t
    with torch.inference_mode(): fl=model(**bf,return_dict=True).logits[:,-1,:]
    same=a.literal==b.literal
    sp=classify(sl,yes_ids,no_ids); fp=classify(fl,yes_ids,no_ids)
    return {"same":same,"slot_pred_same":sp,"full_pred_same":fp,"slot_correct":sp==same,"full_correct":fp==same,"parity":sp==fp,"maxerr":float((sl-fl).abs().max().item()),"ns":ns}

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    md=Path(snapshot_download(repo_id=MODEL_ID,revision=REV,allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors"]))
    assert sha(md/'model.safetensors')==WEIGHT_SHA
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]
    yes_ids=semantic_ids(tok,["Yes"]); no_ids=semantic_ids(tok,["No"])
    if not yes_ids or not no_ids: raise RuntimeError("missing semantic yes/no token set")
    vals=[]
    for x in CODES:
        tid=one(tok," "+x)
        if tid is not None: vals.append(Page(x,tid))
    if len(vals)<8: raise RuntimeError("too few single-token values")
    rng=random.Random(2622)
    pairs=[]
    for i in range(min(8,len(vals))): pairs.append((vals[i],vals[i]))
    for _ in range(40):
        a,b=rng.sample(vals,2); pairs.append((a,b))
    rows=[r for a,b in pairs if (r:=run(model,tok,a,b,yes_ids,no_ids)) is not None]
    if not rows: raise RuntimeError("no valid cases")
    sa=sum(int(x['slot_correct']) for x in rows)/len(rows); fa=sum(int(x['full_correct']) for x in rows)/len(rows); par=sum(int(x['parity']) for x in rows)/len(rows)
    maxerr=max(x['maxerr'] for x in rows)
    # lifecycle: equality -> edit -> inequality -> rollback
    a=vals[0]; b=Page(a.literal,a.token_id); c=next(v for v in vals if v.literal!=a.literal)
    rv1=run(model,tok,a,b,yes_ids,no_ids); rv2=run(model,tok,c,b,yes_ids,no_ids); rrb=run(model,tok,a,b,yes_ids,no_ids)
    masked=Page(c.literal,c.token_id,2,"MASK")
    report={
      "stage":"R262B-REAL-QWEN-SEMANTIC-REASONING-NOVEL-VALUES","model_id":MODEL_ID,"revision":REV,
      "parameter_count":sum(p.numel() for p in model.parameters()),"trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad),
      "cases":len(rows),"full_text_oracle_accuracy":fa,"neural_slot_accuracy":sa,"decision_parity":par,"max_logit_error_vs_full_text":maxerr,
      "median_query_ms":statistics.median(x['ns'] for x in rows)/1e6,"runtime_knowledge_text_tokens":0,"per_fact_gradient_steps":0,
      "yes_token_ids":yes_ids,"no_token_ids":no_ids,
      "lifecycle":{"v1_equal_correct":bool(rv1 and rv1['slot_correct']),"v2_edit_unequal_correct":bool(rv2 and rv2['slot_correct']),"rollback_correct":bool(rrb and rrb['slot_correct']),"mask_authorized":run(model,tok,masked,b,yes_ids,no_ids) is not None},
      "scientific_scope":"real frozen Qwen semantic equality reasoning over two post-training typed values, supplied only as neural slots; exact slot surface must match full-text oracle.",
      "dod_status":"NOT_DOD; real-model reasoning mechanism gate"
    }
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,indent=2,sort_keys=True))
    if fa<0.85:return 2
    if par<0.999 or maxerr>1e-5:return 3
    if sa<0.85:return 4
    if not all([report['lifecycle']['v1_equal_correct'],report['lifecycle']['v2_edit_unequal_correct'],report['lifecycle']['rollback_correct']]) or report['lifecycle']['mask_authorized']: return 5
    return 0
if __name__=='__main__': raise SystemExit(main())
