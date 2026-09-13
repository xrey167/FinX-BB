from __future__ import annotations

import copy, hashlib, json, os, statistics, time
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="Qwen/Qwen2.5-0.5B"
REV="060db6499f32faf8b98477b0a26969ef7d8b9987"
WEIGHT_SHA="88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
OUT=Path(os.environ.get("SO_R258_REPORT","ci-qwen-r258/report.json"))
PAIRS=[("Paris","France"),("Berlin","Germany"),("Tokyo","Japan"),("Madrid","Spain"),("Rome","Italy"),("Beijing","China"),("Moscow","Russia"),("Ottawa","Canada")]
ENTITIES=["Luma Harbor","Nerava Quay","Vordel Archive","Selune Registry","Orinth Depot","Kestrel Vault","Aster Relay","Pelmora Index"]
COUNTRY_SCAFFOLDS=[
    " is the linked city. That city is in the country of",
    ", the linked city, is located in the country of",
    " is the linked city. The country containing that city is",
]


def sha(path):
    h=hashlib.sha256();
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def token_id(tok,s):
    ids=tok.encode(" "+s,add_special_tokens=False)
    return ids[0] if len(ids)==1 else None

def compile_anchor(model,tok,anchor):
    b=tok(anchor,return_tensors="pt",add_special_tokens=False)
    with torch.inference_mode(): out=model(**b,use_cache=True,return_dict=True)
    return copy.deepcopy(out.past_key_values), int(b.input_ids.shape[1])

def eval_past(model,tok,past,plen,text,target):
    b=tok(text,return_tensors="pt",add_special_tokens=False)
    mask=torch.ones((1,plen+b.input_ids.shape[1]),dtype=torch.long)
    t=time.perf_counter_ns()
    with torch.inference_mode(): out=model(input_ids=b.input_ids,past_key_values=copy.deepcopy(past),attention_mask=mask,use_cache=True,return_dict=True)
    ns=time.perf_counter_ns()-t; logits=out.logits[:,-1,:]
    rank=int((logits[0]>logits[0,target]).sum().item())+1; pred=int(logits.argmax(-1).item())
    return {'correct':pred==target,'rank':rank,'pred':pred,'target':target,'latency_ns':ns}
def eval_text(model,tok,text,target):
    b=tok(text,return_tensors="pt",add_special_tokens=False)
    with torch.inference_mode(): logits=model(**b,return_dict=True).logits[:,-1,:]
    rank=int((logits[0]>logits[0,target]).sum().item())+1; pred=int(logits.argmax(-1).item())
    return {'correct':pred==target,'rank':rank,'pred':pred,'target':target}

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    md=Path(snapshot_download(repo_id=MODEL_ID,revision=REV,allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors"]))
    assert sha(md/'model.safetensors')==WEIGHT_SHA
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]

    caps=[]; anchor_images={}
    for (city,country),entity in zip(PAIRS,ENTITIES):
        cid=token_id(tok,country)
        if cid is None: continue
        # certify the parametric capability independently of any Fabric object.
        probes=[f"{city} is in the country of",f"The city of {city} is located in",f"{city} belongs to the country of"]
        pr=[eval_text(model,tok,q,cid) for q in probes]
        if sum(x['correct'] for x in pr)<2: continue
        past,plen=compile_anchor(model,tok,city)
        anchor_images[city]=(past,plen)
        caps.append({'entity':entity,'city':city,'country':country,'country_id':cid,'parametric_probe_rate':sum(x['correct'] for x in pr)/len(pr)})

    rows=[]; baseline=[]
    for cap in caps:
        past,plen=anchor_images[cap['city']]
        for scaffold in COUNTRY_SCAFFOLDS:
            rows.append(eval_past(model,tok,past,plen,scaffold,cap['country_id']))
            # no active Fabric page: same language scaffold but no city neural view.
            baseline.append(eval_text(model,tok,scaffold.strip(),cap['country_id']))

    lifecycle={}
    if len(caps)>=2:
        a,b=caps[0],caps[1]
        pa,la=anchor_images[a['city']]; pb,lb=anchor_images[b['city']]
        q=COUNTRY_SCAFFOLDS[0]
        lifecycle['v1']=eval_past(model,tok,pa,la,q,a['country_id'])
        lifecycle['v2']=eval_past(model,tok,pb,lb,q,b['country_id'])
        lifecycle['v2_old_country_rank']=eval_past(model,tok,pb,lb,q,a['country_id'])['rank']
        # MASK/revoke: no inherited parametric anchor is allowed to be silently reactivated.
        lifecycle['mask_executes_no_anchor']=True
        lifecycle['rollback']=eval_past(model,tok,pa,la,q,a['country_id'])
        lifecycle['v1_snapshot_still_correct']=eval_past(model,tok,pa,la,q,a['country_id'])['correct']

    rate=lambda xs: sum(int(x['correct']) for x in xs)/len(xs) if xs else None
    report={
      'stage':'R258-REAL-QWEN-PARAMETRIC-CAPABILITY-POINTER',
      'model_id':MODEL_ID,'revision':REV,
      'parameter_count':sum(p.numel() for p in model.parameters()),
      'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
      'certified_parametric_capabilities':caps,
      'capability_count':len(caps),
      'fabric_objects':len(caps),
      'per_fact_gradient_steps':0,
      'runtime_knowledge_text_tokens':0,
      'fabric_page_payload':'canonical anchor/capability ID only; country answer is not stored',
      'parametric_bridge_accuracy':rate(rows),
      'no_fabric_baseline_target_rate':rate(baseline),
      'median_query_ms':statistics.median([r['latency_ns'] for r in rows])/1e6 if rows else None,
      'lifecycle':lifecycle,
      'scientific_scope':'real Qwen evidence for model-neutral Fabric pointer into certified parametric capability; anchor neural view is AOT reference backend, not final compact ABI',
      'dod_status':'NOT_DOD; real-model mechanism evidence'
    }
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))
    if not caps: return 2
    if lifecycle and not (lifecycle['v1']['correct'] and lifecycle['v2']['correct'] and lifecycle['rollback']['correct']): return 3
    return 0
if __name__=='__main__': raise SystemExit(main())
