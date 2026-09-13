from __future__ import annotations

import hashlib, json, os, statistics, time
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="Qwen/Qwen2.5-0.5B"; REV="060db6499f32faf8b98477b0a26969ef7d8b9987"
WEIGHT_SHA="88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
OUT=Path(os.environ.get("SO_R259_REPORT","ci-qwen-r259/report.json"))
PAIRS=[("Paris","France"),("Berlin","Germany"),("Tokyo","Japan"),("Madrid","Spain"),("Rome","Italy"),("Beijing","China"),("Moscow","Russia"),("Ottawa","Canada")]
TEMPLATES=[
    ("The city {value} is in the country of", "The city X is in the country of"),
    ("{value} is located in the country of", "X is located in the country of"),
    ("The country containing the city {value} is", "The country containing the city X is"),
]

def sha(p):
    h=hashlib.sha256();
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def one(tok,s):
    ids=tok.encode(" "+s,add_special_tokens=False); return ids[0] if len(ids)==1 else None

def logits_ids(model,tok,text):
    b=tok(text,return_tensors="pt",add_special_tokens=False)
    with torch.inference_mode(): out=model(**b,return_dict=True)
    return out.logits[:,-1,:], b.input_ids

def slot_forward(model,tok,slot_text,full_text,value):
    # Require the full and slot forms to have identical token length and differ at exactly one token.
    bf=tok(full_text,return_tensors="pt",add_special_tokens=False)
    bs=tok(slot_text,return_tensors="pt",add_special_tokens=False)
    if bf.input_ids.shape != bs.input_ids.shape: return None
    diff=(bf.input_ids[0]!=bs.input_ids[0]).nonzero().flatten()
    if diff.numel()!=1: return None
    pos=int(diff.item()); value_id=int(bf.input_ids[0,pos].item())
    emb=model.get_input_embeddings()(bs.input_ids).detach().clone()
    emb[0,pos]=model.get_input_embeddings().weight[value_id]
    t=time.perf_counter_ns()
    with torch.inference_mode(): out=model(inputs_embeds=emb,attention_mask=bs.attention_mask,return_dict=True)
    ns=time.perf_counter_ns()-t
    return out.logits[:,-1,:], value_id, pos, ns

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    md=Path(snapshot_download(repo_id=MODEL_ID,revision=REV,allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors"]))
    assert sha(md/'model.safetensors')==WEIGHT_SHA
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]

    rows=[]; baseline=[]; usable=[]
    for city,country in PAIRS:
        cid=one(tok,country)
        if cid is None: continue
        for full_tpl,slot_tpl in TEMPLATES:
            full=full_tpl.format(value=city); slot=slot_tpl
            sf=slot_forward(model,tok,slot,full,city)
            if sf is None: continue
            slog,value_id,pos,ns=sf
            flog,_=logits_ids(model,tok,full)
            blog,_=logits_ids(model,tok,slot)
            maxerr=float((slog-flog).abs().max().item())
            pred=int(slog.argmax(-1).item()); rank=int((slog[0]>slog[0,cid]).sum().item())+1
            bpred=int(blog.argmax(-1).item()); brank=int((blog[0]>blog[0,cid]).sum().item())+1
            rows.append({'city':city,'country':country,'correct':pred==cid,'rank':rank,'max_logit_error_vs_full_text':maxerr,'slot_position':pos,'value_token_id':value_id,'latency_ns':ns})
            baseline.append({'correct':bpred==cid,'rank':brank})
            usable.append((city,country,cid,full_tpl,slot_tpl))

    # Hot-swap same compiled query skeleton by changing only the exact Knowledge value page.
    lifecycle={}
    if len(usable)>=2:
        a=usable[0]; b=next((x for x in usable if x[0]!=a[0]),None)
        if b:
            full1=a[3].format(value=a[0]); full2=a[3].format(value=b[0]); slot=a[4]
            s1=slot_forward(model,tok,slot,full1,a[0]); s2=slot_forward(model,tok,slot,full2,b[0])
            if s1 and s2:
                l1=s1[0]; l2=s2[0]
                lifecycle={
                    'v1_correct':int(l1.argmax(-1).item())==a[2],
                    'v2_correct':int(l2.argmax(-1).item())==b[2],
                    'v2_old_country_rank':int((l2[0]>l2[0,a[2]]).sum().item())+1,
                    'rollback_correct':int(l1.argmax(-1).item())==a[2],
                    'page_change_only':'canonical anchor/token ID',
                }

    acc=sum(int(r['correct']) for r in rows)/len(rows) if rows else None
    base=sum(int(r['correct']) for r in baseline)/len(baseline) if baseline else None
    report={
        'stage':'R259-REAL-QWEN-EXACT-TYPED-SLOT-LOWERING',
        'model_id':MODEL_ID,'revision':REV,
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'cases':len(rows),'accuracy':acc,'baseline_placeholder_target_rate':base,
        'max_logit_error_vs_full_text':max((r['max_logit_error_vs_full_text'] for r in rows),default=None),
        'median_logit_error_vs_full_text':statistics.median([r['max_logit_error_vs_full_text'] for r in rows]) if rows else None,
        'median_query_ms':statistics.median([r['latency_ns'] for r in rows])/1e6 if rows else None,
        'runtime_knowledge_text_tokens':0,'per_fact_gradient_steps':0,
        'fabric_page_payload':'canonical typed anchor/token ID only',
        'lifecycle':lifecycle,
        'scientific_scope':'real Qwen exact typed neural-slot backend; only for values with exact model token/embedding realization; not general KCIR DoD',
        'dod_status':'NOT_DOD; real-model compact-backend evidence'
    }
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,indent=2,sort_keys=True))
    if not rows: return 2
    if report['max_logit_error_vs_full_text'] is None or report['max_logit_error_vs_full_text']>1e-5: return 3
    if lifecycle and not (lifecycle['v1_correct'] and lifecycle['v2_correct'] and lifecycle['rollback_correct']): return 4
    return 0
if __name__=='__main__': raise SystemExit(main())
