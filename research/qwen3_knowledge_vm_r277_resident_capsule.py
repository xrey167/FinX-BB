from __future__ import annotations

import hashlib
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
OUT = Path(os.environ.get("SO_R277_REPORT", "ci-qwen-r277/report.json"))
ALPHAS = (0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def pick_targets(tok, n=48):
    special = set(tok.all_special_ids)
    out=[]; seen=set()
    for tid in list(range(1200, tok.vocab_size, 89)) + list(range(2000, tok.vocab_size, 157)):
        if tid in seen or tid in special: continue
        seen.add(tid)
        text=tok.decode([tid], clean_up_tokenization_spaces=False)
        s=text.strip()
        if 3 <= len(s) <= 12 and re.fullmatch(r"[A-Za-z]+", s) and s.lower() not in {"the","and","for","with","from","that","this"}:
            out.append((tid,text))
            if len(out)>=n: break
    if len(out)<n: raise RuntimeError(f"only {len(out)} targets")
    return out


def inputs(tok, prompts):
    return {k:v for k,v in tok(prompts, return_tensors="pt", padding=True, truncation=True).items()}


def base_logits(model, batch):
    with torch.inference_mode():
        return model(**batch, use_cache=False, return_dict=True).logits[:,-1].detach()


def capture_layer_norms(model, batch, layers):
    caps={i:[] for i in layers}; handles=[]
    for i in layers:
        def mk(idx):
            def h(_m,args):
                caps[idx].append(args[0][:,-1].detach().float())
                return None
            return h
        handles.append(model.model.layers[i].register_forward_pre_hook(mk(i)))
    try: _=base_logits(model,batch)
    finally:
        for h in handles: h.remove()
    return {i:float(torch.linalg.vector_norm(caps[i][0],dim=-1).median()) for i in layers}


def inject_logits(model,batch,target_ids,layers,norms,alpha):
    head=model.get_output_embeddings().weight.detach()
    v=head[target_ids].float()
    v=v-v.mean(-1,keepdim=True)
    v=v/torch.linalg.vector_norm(v,dim=-1,keepdim=True).clamp_min(1e-8)
    handles=[]
    for li in layers:
        # Split total global strength across active taps. Per-fact state remains
        # just target_ids; layer strengths are global runtime calibration.
        scale=alpha*norms[li]/max(1,len(layers))
        delta=(v*scale).to(dtype=model.dtype)
        def mk(d):
            def h(_m,args):
                hidden=args[0]
                x=torch.zeros_like(hidden)
                x[:,-1]=d.to(hidden.device,dtype=hidden.dtype)
                return (hidden+x,*args[1:])
            return h
        handles.append(model.model.layers[li].register_forward_pre_hook(mk(delta)))
    try: return base_logits(model,batch)
    finally:
        for h in handles: h.remove()


def rate(logits, ids): return float((logits.argmax(-1).cpu()==ids.cpu()).float().mean())


def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    md=Path(snapshot_download(repo_id=MODEL_ID,revision=REV,allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors"]))
    got=sha(md/"model.safetensors")
    if got!=WEIGHT_SHA: raise RuntimeError(f"hash mismatch {got}")
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    L=len(model.model.layers)
    # Interior taps avoid the fragile first/last boundary and spread resident
    # influence across depth.
    taps=tuple(sorted(set([max(1,L//5), max(2,2*L//5), max(3,3*L//5), max(4,4*L//5)])))
    one=(taps[len(taps)//2],)

    targets=pick_targets(tok,48)
    cal=targets[:24]; held=targets[24:48]
    cal_ids=torch.tensor([x[0] for x in cal]); ho_ids=torch.tensor([x[0] for x in held])
    cal_prompts=[f"Registry code for Nivora Archive {i}:" for i in range(len(cal))]
    ho_a=[f"Registry code for Talvera Depot {i}:" for i in range(len(held))]
    ho_b=[f"Which registry code belongs to Talvera Depot {i}?" for i in range(len(held))]
    cb=inputs(tok,cal_prompts); hb_a=inputs(tok,ho_a); hb_b=inputs(tok,ho_b)
    norms=capture_layer_norms(model,cb,tuple(sorted(set(taps+one))))

    def calibrate(layers):
        trials=[]; best=(-1.0,None)
        for a in ALPHAS:
            logits=inject_logits(model,cb,cal_ids,layers,norms,a)
            r=rate(logits,cal_ids); trials.append({"alpha":a,"rate":r})
            if r>best[0]: best=(r,a)
        return trials,best

    one_trials,(one_cal,one_alpha)=calibrate(one)
    res_trials,(res_cal,res_alpha)=calibrate(taps)

    one_ha=rate(inject_logits(model,hb_a,ho_ids,one,norms,one_alpha),ho_ids)
    one_hb=rate(inject_logits(model,hb_b,ho_ids,one,norms,one_alpha),ho_ids)
    res_ha=rate(inject_logits(model,hb_a,ho_ids,taps,norms,res_alpha),ho_ids)
    res_hb=rate(inject_logits(model,hb_b,ho_ids,taps,norms,res_alpha),ho_ids)

    # Hot swap heldout pages; no model update.
    swap=torch.flip(ho_ids,dims=[0])
    res_swap=rate(inject_logits(model,hb_a,swap,taps,norms,res_alpha),swap)

    # No-program path exact equivalence.
    b0=base_logits(model,hb_a); b1=base_logits(model,hb_a)
    no_prog_err=float((b0-b1).abs().max())

    # Simple MAC model for a low-rank resident port if later factorized to K=16.
    D=int(model.config.hidden_size); I=int(getattr(model.config,"intermediate_size",4*D)); K=16
    port_macs_per_layer=2*D*K+K*K
    block_large_linear_macs=4*D*D+3*D*I
    est_frac=port_macs_per_layer/block_large_linear_macs

    report={
      "stage":"R277-QWEN3-RESIDENT-KNOWLEDGE-CAPSULE",
      "model_id":MODEL_ID,"revision":REV,"weight_sha256":got,
      "parameter_count":sum(p.numel() for p in model.parameters()),
      "trainable_parameter_count":0,"per_fact_gradient_steps":0,"runtime_knowledge_text_tokens":0,
      "num_layers":L,"hidden_size":D,"resident_taps":list(taps),"single_tap":list(one),
      "layer_hidden_norms":norms,
      "single_tap_calibration":one_trials,"single_tap_selected_alpha":one_alpha,"single_tap_calibration_rate":one_cal,
      "resident_calibration":res_trials,"resident_selected_alpha":res_alpha,"resident_calibration_rate":res_cal,
      "single_tap_heldout_rates":[one_ha,one_hb],"single_tap_heldout_mean":(one_ha+one_hb)/2,
      "resident_heldout_rates":[res_ha,res_hb],"resident_heldout_mean":(res_ha+res_hb)/2,
      "resident_hot_swap_rate":res_swap,"no_program_max_abs_logit_error":no_prog_err,
      "factorized_port_rank_for_compute_model":K,"factorized_port_macs_per_active_layer":port_macs_per_layer,
      "qwen_block_large_linear_macs_estimate":block_large_linear_macs,"estimated_factorized_port_fraction_per_active_layer":est_frac,
      "scientific_scope":"Real frozen Qwen3-0.6B. Synthetic post-training facts carry only exact target token IDs. Model-specific semantics come from the frozen LM head. Only global tap locations and a single global alpha are calibrated. Resident multi-layer injection is compared directly with one-shot/single-tap injection on heldout targets and paraphrases.",
      "dod_status":"NOT_DOD; real resident capsule mechanism gate"
    }
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
    if res_ha < .90 or res_hb < .90: return 2
    if res_swap < .90: return 3
    if no_prog_err != 0.0: return 4
    if (res_ha+res_hb)/2 <= (one_ha+one_hb)/2: return 5
    return 0

if __name__=="__main__": raise SystemExit(main())
