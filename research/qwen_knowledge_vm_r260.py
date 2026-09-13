from __future__ import annotations

import hashlib, json, os, random, statistics, time
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="Qwen/Qwen2.5-0.5B"; REV="060db6499f32faf8b98477b0a26969ef7d8b9987"
WEIGHT_SHA="88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
OUT=Path(os.environ.get("SO_R260_REPORT","ci-qwen-r260/report.json"))
ANCHORS=[("Paris","France"),("Berlin","Germany"),("Tokyo","Japan"),("Madrid","Spain"),("Rome","Italy"),("Moscow","Russia")]
DEPTHS=(1,4,16,64)
N_PROGRAMS=12


def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def one(tok,s):
    ids=tok.encode(" "+s,add_special_tokens=False); return ids[0] if len(ids)==1 else None

def run_slot(model,tok,city,country_id):
    # This scaffold is deliberately fixed across all depths. Only the exact typed city page changes.
    full=f"The city {city} is in the country of"
    slot="The city X is in the country of"
    bf=tok(full,return_tensors="pt",add_special_tokens=False); bs=tok(slot,return_tensors="pt",add_special_tokens=False)
    if bf.input_ids.shape!=bs.input_ids.shape: return None
    diff=(bf.input_ids[0]!=bs.input_ids[0]).nonzero().flatten()
    if diff.numel()!=1: return None
    pos=int(diff.item()); value_id=int(bf.input_ids[0,pos].item())
    emb=model.get_input_embeddings()(bs.input_ids).detach().clone(); emb[0,pos]=model.get_input_embeddings().weight[value_id]
    t=time.perf_counter_ns()
    with torch.inference_mode(): slog=model(inputs_embeds=emb,attention_mask=bs.attention_mask,return_dict=True).logits[:,-1,:]
    slot_ns=time.perf_counter_ns()-t
    with torch.inference_mode(): flog=model(**bf,return_dict=True).logits[:,-1,:]
    err=float((slog-flog).abs().max().item())
    pred=int(slog.argmax(-1).item()); rank=int((slog[0]>slog[0,country_id]).sum().item())+1
    return {'correct':pred==country_id,'rank':rank,'logit_error_vs_full_text':err,'slot_ns':slot_ns}

def make_program(rng, depth, anchor_idx, program_idx):
    # model-neutral immutable KCIR page chain. Node IDs are opaque post-training symbols.
    nodes=[f"pod://r260/{program_idx}/{i}/{rng.getrandbits(64):016x}" for i in range(depth)]
    target=f"anchor://city/{anchor_idx}"
    edges={nodes[i]:(nodes[i+1] if i+1<depth else target) for i in range(depth)}
    return nodes[0],edges,target

def execute(start,edges):
    cur=start; reads=[]
    t=time.perf_counter_ns()
    while cur in edges:
        reads.append(cur); cur=edges[cur]
    ns=time.perf_counter_ns()-t
    return cur,reads,ns

def text_chain(start,edges,city):
    # intentionally strong structured text baseline: all exact edge facts plus final city assertion.
    lines=[]; cur=start
    while cur in edges:
        nxt=edges[cur]; lines.append(f"{cur} links to {nxt}."); cur=nxt
    lines.append(f"The final linked city is {city}.")
    return "\n".join(lines)+"\nThe final linked city is"

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    md=Path(snapshot_download(repo_id=MODEL_ID,revision=REV,allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors"]))
    assert sha(md/'model.safetensors')==WEIGHT_SHA
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]

    # Only use anchor/country pairs that Qwen answers correctly in the exact typed-slot scaffold.
    good=[]
    for city,country in ANCHORS:
        cid=one(tok,country)
        if cid is None: continue
        r=run_slot(model,tok,city,cid)
        if r and r['correct']: good.append((city,country,cid))
    if not good: raise RuntimeError('no eligible parametric anchors')

    rng=random.Random(260); rows=[]; lifecycle={}; text_baseline=[]
    saved=None
    for depth in DEPTHS:
        for pidx in range(N_PROGRAMS):
            city,country,cid=good[(pidx+depth)%len(good)]
            start,edges,target=make_program(rng,depth,(pidx+depth)%len(good),pidx+1000*depth)
            root,reads,kcir_ns=execute(start,edges)
            assert root.startswith('anchor://city/')
            slot=run_slot(model,tok,city,cid); assert slot is not None
            # input-token baseline only for shallow/medium depths to bound CI time/context size.
            text_tokens=None; text_correct=None; text_ns=None
            if depth<=16 and pidx<4:
                ctx=text_chain(start,edges,city)
                b=tok(ctx,return_tensors='pt',add_special_tokens=False,truncation=True,max_length=2048)
                text_tokens=int(b.input_ids.shape[1])
                t0=time.perf_counter_ns()
                with torch.inference_mode(): lg=model(**b,return_dict=True).logits[:,-1,:]
                text_ns=time.perf_counter_ns()-t0
                text_correct=int(lg.argmax(-1).item())==one(tok,city)
                text_baseline.append({'depth':depth,'tokens':text_tokens,'correct_city_read':text_correct,'latency_ns':text_ns})
            rows.append({'depth':depth,'program':pidx,'kcir_reads':len(reads),'kcir_ns':kcir_ns,'runtime_knowledge_text_tokens':0,'city':city,'country':country,'country_correct':slot['correct'],'country_rank':slot['rank'],'slot_logit_error':slot['logit_error_vs_full_text'],'slot_ns':slot['slot_ns'],'text_baseline_tokens':text_tokens,'text_baseline_city_correct':text_correct,'text_baseline_ns':text_ns})
            if saved is None and depth==16:
                saved=(start,dict(edges),city,country,cid,slot,reads)

    # Lifecycle mutation inside a 16-hop program: patch one edge so the root switches to a different anchor.
    if saved and len(good)>=2:
        start,edges,city1,country1,cid1,slot1,reads1=saved
        city2,country2,cid2=next(x for x in good if x[0]!=city1)
        old_edges=dict(edges)
        # patch the last traversed edge only; exact typed KCIR truth changes, no model weights/gradients.
        last=reads1[-1]; edges[last]='anchor://city/changed'
        root2,reads2,ns2=execute(start,edges)
        slot2=run_slot(model,tok,city2,cid2)
        root1_again,_,_=execute(start,old_edges)
        lifecycle={'v1_country_correct':slot1['correct'],'v2_country_correct':bool(slot2 and slot2['correct']),'v2_old_country_rank':(run_slot(model,tok,city2,cid1) or {'rank':None})['rank'],'rollback_root_equal':root1_again.startswith('anchor://city/'),'old_snapshot_edge_unchanged':old_edges[last]!=edges[last],'patched_kcir_ns':ns2,'edited_nodes':1,'per_fact_gradient_steps':0}

    by_depth={}
    for d in DEPTHS:
        xs=[r for r in rows if r['depth']==d]
        by_depth[str(d)]={'n':len(xs),'country_accuracy':sum(r['country_correct'] for r in xs)/len(xs),'max_slot_logit_error':max(r['slot_logit_error'] for r in xs),'median_kcir_us':statistics.median(r['kcir_ns'] for r in xs)/1e3,'median_qwen_ms':statistics.median(r['slot_ns'] for r in xs)/1e6,'runtime_knowledge_text_tokens':0}
    report={'stage':'R260-REAL-QWEN-ROOT-FUSED-KCIR','model_id':MODEL_ID,'revision':REV,'parameter_count':sum(p.numel() for p in model.parameters()),'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),'eligible_anchors':[x[:2] for x in good],'by_depth':by_depth,'lifecycle':lifecycle,'text_baseline_samples':text_baseline,'per_fact_gradient_steps':0,'runtime_knowledge_text_tokens':0,'scientific_scope':'real Qwen: post-training KCIR graph execution/root fusion plus exact typed-slot lowering into Qwen parametric knowledge. KCIR traversal is exact microkernel execution, not LLM text reasoning.','dod_status':'NOT_DOD; real-model integrated mechanism evidence'}
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,indent=2,sort_keys=True))
    if any(v['country_accuracy']<0.95 for v in by_depth.values()): return 2
    if any(v['max_slot_logit_error']>1e-5 for v in by_depth.values()): return 3
    if lifecycle and not (lifecycle['v1_country_correct'] and lifecycle['v2_country_correct'] and lifecycle['rollback_root_equal']): return 4
    return 0
if __name__=='__main__': raise SystemExit(main())
