from __future__ import annotations

import hashlib, json, os, statistics
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="Qwen/Qwen2.5-0.5B"; REV="060db6499f32faf8b98477b0a26969ef7d8b9987"
WEIGHT_SHA="88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
OUT=Path(os.environ.get("SO_R265_REPORT","ci-qwen-r265/report.json"))
VALUES=["amber","silver","violet","cobalt","scarlet","ivory","crimson","azure","coral","delta","sigma","omega","orbit","signal","kernel","vertex","falcon","raven","tiger","otter","lotus","cedar","maple","quartz","pearl","comet","north","south","east","west","alpha","beta","gamma","theta","lambda","kappa","seven","eight","nine","eleven","twelve","thirty","forty","fifty","sixty","ninety"]
TEMPLATES=[
  ("Record: code = {v}. Question: repeat the code. Answer:","Record: code = X. Question: repeat the code. Answer:"),
  ("Database value: {v}. Use the database value to answer. Value:","Database value: X. Use the database value to answer. Value:"),
  ("The authoritative registry value is {v}. What is the authoritative registry value?", "The authoritative registry value is X. What is the authoritative registry value?"),
]

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def one(tok,s):
    ids=tok.encode(" "+s,add_special_tokens=False);return ids[0] if len(ids)==1 else None

def hidden_forward(model, input_ids=None, inputs_embeds=None, attention_mask=None):
    with torch.inference_mode():
        o=model(input_ids=input_ids,inputs_embeds=inputs_embeds,attention_mask=attention_mask,use_cache=False,output_hidden_states=True,return_dict=True)
    return tuple(x[:, -1, :].detach().float().cpu() for x in o.hidden_states), o.logits[:,-1,:].detach().float().cpu()

def k_for_energy(s, frac):
    e=s.square(); total=float(e.sum().item())
    if total<=0:return 0
    c=torch.cumsum(e,0)/total
    return int((c<frac).sum().item())+1

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    md=Path(snapshot_download(repo_id=MODEL_ID,revision=REV,allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors"]))
    assert sha(md/'model.safetensors')==WEIGHT_SHA
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32)
    model.eval();[p.requires_grad_(False) for p in model.parameters()]
    values=[(v,one(tok,v)) for v in VALUES];values=[x for x in values if x[1] is not None][:32]
    if len(values)<16:raise RuntimeError("too few single-token values")
    layer_deltas=[[] for _ in range(model.config.num_hidden_layers+1)]
    parity_errors=[]; usable=0
    for full_tpl,slot_text in TEMPLATES:
        bs=tok(slot_text,return_tensors="pt",add_special_tokens=False)
        base_h,_=hidden_forward(model,input_ids=bs.input_ids,attention_mask=bs.attention_mask)
        for v,tid in values:
            full=full_tpl.format(v=v);bf=tok(full,return_tensors="pt",add_special_tokens=False)
            if bf.input_ids.shape!=bs.input_ids.shape:continue
            diff=(bf.input_ids[0]!=bs.input_ids[0]).nonzero().flatten()
            if diff.numel()!=1:continue
            pos=int(diff.item())
            if int(bf.input_ids[0,pos])!=tid:continue
            emb=model.get_input_embeddings()(bs.input_ids).detach().clone();emb[0,pos]=model.get_input_embeddings().weight[tid]
            slot_h,slot_logits=hidden_forward(model,inputs_embeds=emb,attention_mask=bs.attention_mask)
            full_h,full_logits=hidden_forward(model,input_ids=bf.input_ids,attention_mask=bf.attention_mask)
            parity_errors.append(float((slot_logits-full_logits).abs().max().item()))
            for li,(h,bh) in enumerate(zip(slot_h,base_h)):
                layer_deltas[li].append((h-bh).squeeze(0))
            usable+=1
    if usable<32:raise RuntimeError(f"too few usable cases: {usable}")
    layers=[]
    for li,rows in enumerate(layer_deltas):
        X=torch.stack(rows,0)
        # Span of factual effect; do not center because zero is meaningful baseline.
        s=torch.linalg.svdvals(X)
        layers.append({
          "layer":li,"samples":int(X.shape[0]),"hidden_dim":int(X.shape[1]),
          "k90":k_for_energy(s,.90),"k95":k_for_energy(s,.95),"k99":k_for_energy(s,.99),"k999":k_for_energy(s,.999),
          "participation_rank":float((s.square().sum()**2/(s.pow(4).sum()+1e-30)).item()),
          "mean_delta_norm":float(X.norm(dim=1).mean().item()),"max_delta_norm":float(X.norm(dim=1).max().item()),
          "top_singular_values":[float(x) for x in s[:12]],
        })
    median_k99=statistics.median(x['k99'] for x in layers[1:]); max_k95=max(x['k95'] for x in layers[1:])
    classification="low_rank_candidate" if median_k99<=32 and max_k95<=32 else "heterogeneous_or_high_rank_reference_surface"
    report={
      "stage":"R265-REAL-QWEN-KNOWLEDGE-RESPONSE-RANK","model_id":MODEL_ID,"revision":REV,
      "parameter_count":sum(p.numel() for p in model.parameters()),"trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad),
      "usable_value_template_cases":usable,"max_neural_slot_logit_error_vs_text":max(parity_errors),
      "median_k99_decoder_layers":median_k99,"max_k95_decoder_layers":max_k95,"classification":classification,"layers":layers,
      "scientific_scope":"diagnostic rank of exact tokenless typed-value effects at the final query position across real frozen Qwen layers. High rank is a valid negative result and routes such semantics to exact/reference lanes rather than forcing compact neural lowering.",
      "dod_status":"NOT_DOD; real-model structural audit"
    }
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
    if report['max_neural_slot_logit_error_vs_text']>1e-5:return 2
    return 0
if __name__=='__main__':raise SystemExit(main())
