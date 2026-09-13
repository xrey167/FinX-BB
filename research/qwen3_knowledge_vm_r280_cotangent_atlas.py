from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
REV = "a08cec3036ee1085a4863a0b730e5d3f1c2f8d04"
WEIGHT_SHA = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
OUT = Path(os.environ.get("SO_R280_REPORT", "ci-qwen-r280/report.json"))
LAM = 0.08
ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)
SHRINKS = (0.25, 0.5, 0.75, 1.0)


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()


def pick_targets(tok,n=96):
    special=set(tok.all_special_ids); out=[]; seen=set()
    for tid in list(range(900,tok.vocab_size,61))+list(range(1400,tok.vocab_size,103)):
        if tid in seen or tid in special: continue
        seen.add(tid); text=tok.decode([tid],clean_up_tokenization_spaces=False); s=text.strip()
        if 3<=len(s)<=12 and re.fullmatch(r"[A-Za-z]+",s) and s.lower() not in {"the","and","for","with","from","that","this"}:
            out.append((tid,text))
            if len(out)>=n: break
    if len(out)<n: raise RuntimeError(f"only {len(out)} targets")
    return out


def batch(tok,prompts):
    return {k:v for k,v in tok(prompts,return_tensors="pt",padding=True,truncation=True).items()}


def top1(logits,ids): return float((logits.argmax(-1).cpu()==ids.cpu()).float().mean())


def margin_score(logits,ids):
    x=logits.float(); idx=torch.arange(x.shape[0]); target=x[idx,ids]
    y=x.clone(); y[idx,ids]=-torch.inf; other=y.max(-1).values
    return float((target-other).mean())


def capture_cotangents(model,tok,prompts,target_ids,taps):
    b=batch(tok,prompts)
    emb=model.get_input_embeddings()(b["input_ids"]).detach().requires_grad_(True)
    caps={i:[] for i in taps}; handles=[]
    for i in taps:
        def mk(idx):
            def h(_m,args):
                z=args[0]
                z.retain_grad(); caps[idx].append(z)
                return None
            return h
        handles.append(model.model.layers[i].register_forward_pre_hook(mk(i)))
    try:
        out=model(inputs_embeds=emb,attention_mask=b.get("attention_mask"),use_cache=False,return_dict=True)
        logits=out.logits[:,-1]
        loss=logits[torch.arange(logits.shape[0]),target_ids].sum()
        loss.backward()
    finally:
        for h in handles: h.remove()
    grads={}
    norms={}
    for i in taps:
        z=caps[i][0]
        g=z.grad[:,-1].detach().float()
        g=g/torch.linalg.vector_norm(g,dim=-1,keepdim=True).clamp_min(1e-12)
        grads[i]=g.cpu()
        norms[i]=float(torch.linalg.vector_norm(z[:,-1].detach().float(),dim=-1).median())
    model.zero_grad(set_to_none=True)
    return grads,norms


def fit_kernel_atlas(head,cal_ids,grads):
    E=head[cal_ids].detach().float().cpu()
    E=E-E.mean(-1,keepdim=True); E=E/torch.linalg.vector_norm(E,dim=-1,keepdim=True).clamp_min(1e-12)
    K=E@E.T
    inv=torch.linalg.inv(K+LAM*torch.eye(K.shape[0]))
    return E,inv,{i:g.clone() for i,g in grads.items()}


def atlas_dirs(head,target_ids,atlas):
    E,inv,Gs=atlas
    Q=head[target_ids].detach().float().cpu(); Q=Q-Q.mean(-1,keepdim=True); Q=Q/torch.linalg.vector_norm(Q,dim=-1,keepdim=True).clamp_min(1e-12)
    weights=(Q@E.T)@inv
    out={}
    for i,G in Gs.items():
        z=weights@G
        z=z/torch.linalg.vector_norm(z,dim=-1,keepdim=True).clamp_min(1e-12)
        out[i]=z
    return out


def direct_dirs(head,target_ids,taps):
    Q=head[target_ids].detach().float().cpu(); Q=Q-Q.mean(-1,keepdim=True); Q=Q/torch.linalg.vector_norm(Q,dim=-1,keepdim=True).clamp_min(1e-12)
    return {i:Q.clone() for i in taps}


def inject(model,b,dirs,taps,norms,coeffs):
    handles=[]
    for li in taps:
        d=(dirs[li]*float(coeffs[li])*norms[li]).to(dtype=model.dtype)
        def mk(delta):
            def h(_m,args):
                hidden=args[0]; x=torch.zeros_like(hidden); x[:,-1]=delta.to(hidden.device,dtype=hidden.dtype)
                return (hidden+x,*args[1:])
            return h
        handles.append(model.model.layers[li].register_forward_pre_hook(mk(d)))
    try:
        with torch.inference_mode(): return model(**b,use_cache=False,return_dict=True).logits[:,-1].detach()
    finally:
        for h in handles: h.remove()


def calibrate_single(model,b,ids,dirs,taps,norms):
    best={}; detail={}
    for li in taps:
        rows=[]
        for a in ALPHAS:
            logits=inject(model,b,dirs,(li,),norms,{li:a})
            rows.append((top1(logits,ids),margin_score(logits,ids),a))
        rows.sort(reverse=True); best[li]=rows[0][2]; detail[str(li)]=[{"alpha":r[2],"rate":r[0],"margin":r[1]} for r in rows]
    return best,detail


def choose_combo(model,b,ids,dirs,taps,norms,single):
    ranked=[]
    for li in taps:
        logits=inject(model,b,dirs,(li,),norms,{li:single[li]})
        ranked.append((top1(logits,ids),margin_score(logits,ids),li))
    ranked.sort(reverse=True); ordered=[x[2] for x in ranked]
    candidates=[]
    for n in range(1,min(4,len(ordered))+1):
        chosen=tuple(sorted(ordered[:n]))
        for s in SHRINKS:
            coeff={li:single[li]*s for li in chosen}
            logits=inject(model,b,dirs,chosen,norms,coeff)
            candidates.append((top1(logits,ids),margin_score(logits,ids),-n,s,chosen,coeff))
    candidates.sort(reverse=True); x=candidates[0]
    return x[4],x[5],[{"rate":c[0],"margin":c[1],"shrink":c[3],"taps":list(c[4])} for c in candidates[:20]]


def eval_panel(model,tok,prompts,ids,dirs,taps,norms,coeffs):
    b=batch(tok,prompts); logits=inject(model,b,dirs,taps,norms,coeffs)
    return {"rate":top1(logits,ids),"margin":margin_score(logits,ids)}


def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    md=Path(snapshot_download(repo_id=MODEL_ID,revision=REV,allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors"]))
    got=sha(md/"model.safetensors")
    if got!=WEIGHT_SHA: raise RuntimeError(f"hash mismatch {got}")
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]
    L=len(model.model.layers); taps=tuple(sorted(set([max(1,L//6),max(2,2*L//6),max(3,3*L//6),max(4,4*L//6),max(5,5*L//6),L-2])))
    targets=pick_targets(tok,96); cal=targets[:64]; held=targets[64:96]
    cal_ids=torch.tensor([x[0] for x in cal]); ho_ids=torch.tensor([x[0] for x in held])
    cal_prompts=[f"Registry code for Calib Archive {i}:" for i in range(len(cal))]
    ho_a=[f"Registry code for Heldout Depot {i}:" for i in range(len(held))]
    ho_b=[f"Which registry code belongs to Heldout Depot {i}?" for i in range(len(held))]

    t0=time.perf_counter_ns(); grads,norms=capture_cotangents(model,tok,cal_prompts,cal_ids,taps); grad_ms=(time.perf_counter_ns()-t0)/1e6
    head=model.get_output_embeddings().weight.detach()
    atlas=fit_kernel_atlas(head,cal_ids,grads)
    cal_dirs=grads
    cal_b=batch(tok,cal_prompts)
    single, single_detail=calibrate_single(model,cal_b,cal_ids,cal_dirs,taps,norms)
    chosen,coeffs,combo_detail=choose_combo(model,cal_b,cal_ids,cal_dirs,taps,norms,single)

    held_dirs=atlas_dirs(head,ho_ids,atlas)
    direct=direct_dirs(head,ho_ids,taps)
    held_a=eval_panel(model,tok,ho_a,ho_ids,held_dirs,chosen,norms,coeffs)
    held_b=eval_panel(model,tok,ho_b,ho_ids,held_dirs,chosen,norms,coeffs)
    direct_a=eval_panel(model,tok,ho_a,ho_ids,direct,chosen,norms,coeffs)
    swap=torch.flip(ho_ids,dims=[0]); swap_dirs=atlas_dirs(head,swap,atlas)
    swap_result=eval_panel(model,tok,ho_a,swap,swap_dirs,chosen,norms,coeffs)

    b=batch(tok,ho_a)
    with torch.inference_mode(): x0=model(**b,use_cache=False,return_dict=True).logits[:,-1]; x1=model(**b,use_cache=False,return_dict=True).logits[:,-1]
    report={
      "stage":"R280-QWEN3-COTANGENT-KNOWLEDGE-ABI-ATLAS","model_id":MODEL_ID,"revision":REV,"weight_sha256":got,
      "parameter_count":sum(p.numel() for p in model.parameters()),"trainable_parameter_count":0,"per_fact_gradient_steps":0,"runtime_knowledge_text_tokens":0,
      "global_calibration_target_count":len(cal),"heldout_target_count":len(held),"candidate_taps":list(taps),"chosen_taps":list(chosen),"chosen_global_coefficients":{str(k):v for k,v in coeffs.items()},
      "cotangent_capture_ms":grad_ms,"layer_hidden_norms":norms,"single_tap_calibration":single_detail,"combo_calibration_top20":combo_detail,
      "heldout_paraphrase_a":held_a,"heldout_paraphrase_b":held_b,"heldout_mean_rate":(held_a["rate"]+held_b["rate"])/2,
      "heldout_direct_lm_head_same_coeff_rate":direct_a["rate"],"hot_swap":swap_result,"no_program_max_abs_logit_error":float((x0-x1).abs().max()),
      "scientific_scope":"Real frozen Qwen3-0.6B. A one-time model-global calibration basis (64 target tokens) extracts local target cotangents at six layers in one backward pass. A kernel atlas maps frozen LM-head semantics to local layer directions. No heldout fact receives a gradient or adapter; heldout target IDs are converted to layer-local directions only through the global atlas. Global tap coefficients are selected only on calibration targets.",
      "dod_status":"NOT_DOD; real layerwise model-ABI calibration mechanism gate"
    }
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2,sort_keys=True))
    if report["heldout_mean_rate"]<0.90: return 2
    if swap_result["rate"]<0.90: return 3
    if report["no_program_max_abs_logit_error"]!=0.0: return 4
    if report["heldout_mean_rate"]<=report["heldout_direct_lm_head_same_coeff_rate"]: return 5
    return 0

if __name__=="__main__": raise SystemExit(main())
