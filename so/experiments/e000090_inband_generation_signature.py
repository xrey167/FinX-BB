"""E-000090 -- In-Band Neural Generation Signature screening experiment.

Preregistered in docs/novelty/e000090-inband-neural-generation-signature.md.
This is a falsification screen, not a novelty claim.

A deterministic 8-bit generation code is injected at memory broadcast sites in addition to a
material content payload. A one-time per-backbone linear decoder is calibrated on some generations
and prompt/content combinations, then evaluated on held-out generations and held-out combinations
using only the *final hidden state*. External generation metadata is intentionally unavailable to the
decoder. The same hidden tensors are serialized/deserialized before a second decode.

Strong baseline: an ordinary external generation tag is exact when correctly paired; under the
registered metadata-swap fault it follows the wrong tag by construction. That baseline remains the
preferred normal path. The in-band mechanism earns value only as a redundancy/fault-detection path.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from so.llm_adapter import transformer_blocks

BITS = 8
GENERATIONS = tuple(range(24))          # 16 calibration, 8 held-out
TRAIN_GENS = tuple(range(16))
TEST_GENS = tuple(range(16, 24))
SIGNATURE_RMS_ARMS = (0.02, 0.05, 0.10, 0.20)
CONTENT_RMS = 4.0
RIDGE = 1e-2


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden,) + tuple(output[1:])
    return hidden


def _unit_rms(v: torch.Tensor) -> torch.Tensor:
    return v / (v.float().pow(2).mean().sqrt().clamp_min(1e-8))


def _content_payload(model, token_id: int, rms: float) -> torch.Tensor:
    emb = model.get_output_embeddings()
    if emb is None:
        emb = model.get_input_embeddings()
    return _unit_rms(emb.weight[int(token_id)].detach().float().clone()) * float(rms)


def _basis(d: int, seed: int = 90090) -> torch.Tensor:
    """Deterministic approximately orthonormal generation-code basis, one-time per hidden width."""
    g = torch.Generator().manual_seed(seed + d)
    m = torch.randn(d, BITS, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(m, mode='reduced')
    return q.T.float()  # (BITS,d), Euclidean unit vectors


def _bits(generation: int) -> torch.Tensor:
    return torch.tensor([1.0 if ((generation >> i) & 1) else -1.0 for i in range(BITS)])


def _signature(basis: torch.Tensor, generation: int, rms: float) -> torch.Tensor:
    raw = _bits(generation) @ basis
    return _unit_rms(raw) * float(rms)


def _forward(model, blocks, read_layers: Tuple[int, int], ids: torch.Tensor,
             content: torch.Tensor | None, signature: torch.Tensor | None):
    handles=[]
    if content is not None or signature is not None:
        delta = torch.zeros(model.get_input_embeddings().weight.shape[1], dtype=torch.float32)
        if content is not None: delta = delta + content.float()
        if signature is not None: delta = delta + signature.float()
        for layer in read_layers:
            def hook(module, inputs, output, delta=delta):
                h=_hidden(output); h2=h.clone(); h2[:,-1,:]=h2[:,-1,:]+delta.to(h.device,h.dtype)
                return _replace_hidden(output,h2)
            handles.append(blocks[layer].register_forward_hook(hook))
    try:
        with torch.no_grad():
            out=model(input_ids=ids,output_hidden_states=True,use_cache=True)
    finally:
        for h in handles: h.remove()
    hidden=out.hidden_states[-1][:,-1,:].detach().float().cpu()
    logits=out.logits[:,-1,:].detach().float().cpu()
    if out.past_key_values is None: raise RuntimeError('no KV cache returned')
    return hidden,logits,out.past_key_values


def _kl(ref: torch.Tensor, obs: torch.Tensor) -> float:
    lp=torch.log_softmax(ref.double(),-1); lo=torch.log_softmax(obs.double(),-1); p=lp.exp()
    return float((p*(lp-lo)).sum(-1).mean())


def _serialize_roundtrip(x: torch.Tensor) -> torch.Tensor:
    b=io.BytesIO(); torch.save(x,b); b.seek(0); return torch.load(b,map_location='cpu',weights_only=True)


def _fit_dual_ridge(X: np.ndarray, Y: np.ndarray, ridge: float=RIDGE):
    """Fit multi-output linear ridge with bias by the dual form; returns mean/scale and weights."""
    mu=X.mean(0,keepdims=True); sd=X.std(0,keepdims=True)+1e-6; Z=(X-mu)/sd
    Z=np.concatenate([Z,np.ones((Z.shape[0],1))],axis=1)
    K=Z@Z.T + ridge*np.eye(Z.shape[0])
    alpha=np.linalg.solve(K,Y)
    W=Z.T@alpha
    return mu,sd,W


def _decode(X: np.ndarray, fit) -> np.ndarray:
    mu,sd,W=fit; Z=(X-mu)/sd; Z=np.concatenate([Z,np.ones((Z.shape[0],1))],axis=1)
    return np.where(Z@W>=0,1.0,-1.0)


def _gen_from_bits(bits: np.ndarray) -> np.ndarray:
    out=[]
    for row in bits:
        g=0
        for i,b in enumerate(row):
            if b>0: g |= (1<<i)
        out.append(g)
    return np.asarray(out,dtype=np.int64)


def run(model_name: str, seed: int=0) -> Dict[str,object]:
    torch.manual_seed(seed); torch.set_num_threads(max(1,min(2,torch.get_num_threads())))
    model=AutoModelForCausalLM.from_pretrained(model_name); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    blocks=transformer_blocks(model); d=int(model.get_input_embeddings().weight.shape[1]); vocab=int(model.get_input_embeddings().weight.shape[0])
    read_layers=(1,min(3,len(blocks)-2)) if len(blocks)>=5 else (0,len(blocks)-2)
    if read_layers[0]==read_layers[1]: read_layers=(0,len(blocks)-2)
    basis=_basis(d)

    # Six synthetic Pod/prompt combinations. Last two are held out from detector calibration.
    prompts=[]
    for j in range(6):
        toks=[(17+13*j)%vocab,(103+17*j)%vocab,(227+19*j)%vocab,(331+23*j)%vocab,(443+29*j)%vocab]
        prompts.append(torch.tensor([toks],dtype=torch.long))
    content_ids=[(42+37*j)%vocab for j in range(6)]

    bypass_logits=[]; unsigned_logits=[]; unsigned_hidden=[]
    for j in range(6):
        content=_content_payload(model,content_ids[j],CONTENT_RMS)
        hb,lb,_=_forward(model,blocks,read_layers,prompts[j],None,None)
        hu,lu,_=_forward(model,blocks,read_layers,prompts[j],content,None)
        bypass_logits.append(lb); unsigned_logits.append(lu); unsigned_hidden.append(hu)
    material_effect=max(float((u-b).abs().max()) for u,b in zip(unsigned_logits,bypass_logits))

    arms=[]
    for sig_rms in SIGNATURE_RMS_ARMS:
        Xtr=[];Ytr=[]; Xte=[];Yte=[]; meta_te=[]; signed_logits_te=[]; unsigned_logits_te=[]
        # calibration: generations 0..15 x prompt/pod combinations 0..3
        for g in TRAIN_GENS:
            y=_bits(g).numpy()
            for j in range(4):
                content=_content_payload(model,content_ids[j],CONTENT_RMS); sig=_signature(basis,g,sig_rms)
                h,l,_=_forward(model,blocks,read_layers,prompts[j],content,sig)
                Xtr.append(h.numpy()[0]);Ytr.append(y)
        # test: unseen generations 16..23 x held-out prompt/pod combinations 4..5
        for g in TEST_GENS:
            y=_bits(g).numpy()
            for j in (4,5):
                content=_content_payload(model,content_ids[j],CONTENT_RMS); sig=_signature(basis,g,sig_rms)
                h,l,_=_forward(model,blocks,read_layers,prompts[j],content,sig)
                Xte.append(h.numpy()[0]);Yte.append(y);meta_te.append((g,j));signed_logits_te.append(l);unsigned_logits_te.append(unsigned_logits[j])
        Xtr=np.asarray(Xtr);Ytr=np.asarray(Ytr);Xte=np.asarray(Xte);Yte=np.asarray(Yte)
        fit=_fit_dual_ridge(Xtr,Ytr)
        pred_bits=_decode(Xte,fit); pred_gen=_gen_from_bits(pred_bits); true_gen=np.asarray([g for g,j in meta_te])
        exact_acc=float((pred_gen==true_gen).mean()); bit_acc=float((pred_bits==Yte).mean())
        # serialization path contains tensor only; no generation side metadata reaches the decoder.
        Xrt=np.asarray([_serialize_roundtrip(torch.tensor(x)).numpy() for x in Xte])
        rt_gen=_gen_from_bits(_decode(Xrt,fit)); rt_acc=float((rt_gen==true_gen).mean())
        # stale/current discrimination: authority current = each held-out generation, stale = previous generation.
        # Build paired hidden states at the same held-out prompt; detector must distinguish current g from g-1.
        stale_correct=0; stale_total=0; false_current=0
        for g in TEST_GENS:
            if g==0: continue
            for j in (4,5):
                content=_content_payload(model,content_ids[j],CONTENT_RMS)
                hs,_,_=_forward(model,blocks,read_layers,prompts[j],content,_signature(basis,g-1,sig_rms))
                hc,_,_=_forward(model,blocks,read_layers,prompts[j],content,_signature(basis,g,sig_rms))
                dec=_gen_from_bits(_decode(np.vstack([hs.numpy()[0],hc.numpy()[0]]),fit))
                stale_correct += int(dec[0]==g-1) + int(dec[1]==g)
                false_current += int(dec[0]==g)
                stale_total += 2
        stale_current_acc=stale_correct/max(stale_total,1); false_current_rate=false_current/max(stale_total/2,1)
        kls=[_kl(u,s) for u,s in zip(unsigned_logits_te,signed_logits_te)]
        agrees=[int(int(u.argmax(-1)[0])==int(s.argmax(-1)[0])) for u,s in zip(unsigned_logits_te,signed_logits_te)]
        max_kl=max(kls); top1_agree=float(np.mean(agrees))
        # Metadata-swap fault baseline: every pair receives the other sample's generation tag.
        ext_correct=0; ext_total=0
        for ix,(g,j) in enumerate(meta_te):
            swapped=meta_te[ix^1][0] if ix%2==0 else meta_te[ix-1][0]
            ext_correct+=int(swapped==g);ext_total+=1
        external_swapped_accuracy=ext_correct/max(ext_total,1)
        checks={
            'generation_exact_acc_ge_099':exact_acc>=.99,
            'roundtrip_acc_ge_099':rt_acc>=.99,
            'stale_current_acc_ge_099':stale_current_acc>=.99,
            'false_current_le_001':false_current_rate<=.01,
            'signature_kl_le_005':max_kl<=.05,
            'signature_top1_agree_ge_098':top1_agree>=.98,
            'material_content_effect':material_effect>1e-4,
        }
        arms.append({'signature_rms':sig_rms,'generation_exact_accuracy':exact_acc,'bit_accuracy':bit_acc,
                     'roundtrip_accuracy':rt_acc,'stale_current_accuracy':stale_current_acc,'false_current_rate':false_current_rate,
                     'max_signature_vs_unsigned_kl_nats':max_kl,'top1_agreement':top1_agree,
                     'external_metadata_swap_accuracy':external_swapped_accuracy,'checks':checks,'pass':all(checks.values())})

    passing=[a for a in arms if a['pass']]
    return {'model':model_name,'seed':seed,'hidden_width':d,'read_layers':list(read_layers),'material_content_effect_maxabs':material_effect,
            'arms':arms,'any_arm_pass':bool(passing),'passing_signature_rms':[a['signature_rms'] for a in passing],
            'not_claimed':'activation watermarking, residual marking, version metadata, provenance, probes, ECC or cryptographic security individually'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--results-dir',default='so/results')
    a=ap.parse_args();rec={'experiment':'E-000090','title':'In-Band Neural Generation Signature','row':run(a.model,a.seed)}
    p=Path(a.results_dir);p.mkdir(parents=True,exist_ok=True);fn='e000090_'+a.model.replace('/','_')+'.json';(p/fn).write_text(json.dumps(rec,indent=2),encoding='utf-8')
    print(json.dumps(rec,indent=2))
if __name__=='__main__': main()
