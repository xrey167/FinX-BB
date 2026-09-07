"""E-000092 -- Generation-Keyed Neural Addressing functional kill screen.

Controlled frozen-backbone experiment.  This is an existence/utility screen, not a novelty claim.
Stale and current rows for the SAME Pod coexist physically.  Only the addressing rule changes.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM

from so.llm_adapter import transformer_blocks


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _replace(output, h):
    return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h


def _capture_query(model, blocks, layer: int, ids: torch.Tensor) -> torch.Tensor:
    box={}
    def hook(module, inputs, output):
        box['h']=_hidden(output)[:, -1, :].detach().float().clone()
    hd=blocks[layer].register_forward_hook(hook)
    try:
        with torch.no_grad(): model(input_ids=ids, use_cache=False)
    finally: hd.remove()
    return box['h'][0]


def _payload(model, token_id: int, rms: float=2.0) -> torch.Tensor:
    emb=model.get_output_embeddings() or model.get_input_embeddings()
    v=emb.weight[int(token_id)].detach().float().clone()
    return v * (rms / float(v.pow(2).mean().sqrt().clamp_min(1e-8)))


def _run_with_payload(model, blocks, layer: int, ids: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
    def hook(module, inputs, output):
        h=_hidden(output); h2=h.clone(); h2[:, -1, :]=h2[:, -1, :]+payload.to(h.device,h.dtype)
        return _replace(output,h2)
    hd=blocks[layer].register_forward_hook(hook)
    try:
        with torch.no_grad(): out=model(input_ids=ids, use_cache=False)
    finally: hd.remove()
    return out.logits[:, -1, :].detach().float()


def _kl(ref, obs)->float:
    p=torch.softmax(ref,-1)
    return float((p*(torch.log_softmax(ref,-1)-torch.log_softmax(obs,-1))).sum(-1).mean())


def _basis(d:int, seed:int)->Tuple[torch.Tensor,torch.Tensor]:
    g=torch.Generator().manual_seed(seed)
    a=torch.randn(d,generator=g); a=a/a.norm()
    b=torch.randn(d,generator=g); b=b-a*torch.dot(a,b); b=b/b.norm()
    return a,b


def _semantic(q:torch.Tensor,c0:torch.Tensor,c1:torch.Tensor)->torch.Tensor:
    s=q-q.dot(c0)*c0-q.dot(c1)*c1
    return s/(s.norm()+1e-9)


def _select(q:torch.Tensor, keys:torch.Tensor)->Tuple[int,torch.Tensor]:
    scores=q@keys.T
    p=torch.softmax(scores*8.0,-1)
    return int(torch.argmax(scores)),p


def run(model_name:str, seed:int, n_prompts:int, alphas:List[float])->Dict:
    torch.manual_seed(seed); torch.set_num_threads(2)
    model=AutoModelForCausalLM.from_pretrained(model_name); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    blocks=transformer_blocks(model); qlayer=1; inject_layer=min(3,len(blocks)-2)
    d=int(model.get_input_embeddings().weight.shape[1]); vocab=int(model.get_input_embeddings().weight.shape[0])
    cold,cnew=_basis(d,92000+seed)
    rows=[]
    for alpha in alphas:
        current_ok=stale_sel=unrel_ok=0; ambiguous_wrong=0; material=0; kls=[]; top=[]; metadata_swap_ok=0
        current_probs=[]; stale_probs=[]
        for i in range(n_prompts):
            ids=torch.tensor([[17%vocab,(103+i*7)%vocab,(227+i*11)%vocab,(331+i*13)%vocab,(443+i*17)%vocab]],dtype=torch.long)
            qraw=_capture_query(model,blocks,qlayer,ids); s=_semantic(qraw,cold,cnew)
            # Same Pod semantic identity; only generation code differs.
            k_old=torch.nn.functional.normalize(s+alpha*cold,dim=0)
            k_new=torch.nn.functional.normalize(s+alpha*cnew,dim=0)
            q_cur=torch.nn.functional.normalize(s+alpha*cnew,dim=0)
            # Unrelated Pod: independent semantic direction, current generation.
            u=torch.roll(s,shifts=max(1,d//7)); u=_semantic(u,cold,cnew)
            k_u=torch.nn.functional.normalize(u+alpha*cnew,dim=0)
            keys=torch.stack([k_old,k_new,k_u])
            sel,p=_select(q_cur,keys); current_ok+=int(sel==1); stale_sel+=int(sel==0); current_probs.append(float(p[1])); stale_probs.append(float(p[0]))
            # Metadata is intentionally swapped/deleted; in-band selection does not consume it.
            fake_metadata={'row0_generation':'current','row1_generation':'stale'}
            _=fake_metadata; metadata_swap_ok+=int(sel==1)
            # No-generation baseline has identical stale/current semantic keys. stale row is first -> wrong tie winner.
            sel0,_=_select(s,torch.stack([s,s,u])); ambiguous_wrong+=int(sel0==0)
            old_payload=_payload(model,(42+i)%vocab); new_payload=_payload(model,(314+i)%vocab)
            gold=_run_with_payload(model,blocks,inject_layer,ids,new_payload)
            chosen=[old_payload,new_payload,_payload(model,(271+i)%vocab)][sel]
            guarded=_run_with_payload(model,blocks,inject_layer,ids,chosen)
            stale=_run_with_payload(model,blocks,inject_layer,ids,old_payload)
            kls.append(_kl(gold,guarded)); top.append(float((gold.argmax(-1)==guarded.argmax(-1)).float().mean()))
            material+=int(float((stale-gold).abs().max())>1e-4)
            # unrelated query must still choose unrelated pod
            q_u=torch.nn.functional.normalize(u+alpha*cnew,dim=0); su,_=_select(q_u,keys); unrel_ok+=int(su==2)
        rec={
          'alpha':alpha,'current_selection_rate':current_ok/n_prompts,'stale_selection_rate':stale_sel/n_prompts,
          'mean_current_routing_mass':sum(current_probs)/n_prompts,'mean_stale_routing_mass':sum(stale_probs)/n_prompts,
          'metadata_swap_selection_rate':metadata_swap_ok/n_prompts,'ambiguous_baseline_stale_rate':ambiguous_wrong/n_prompts,
          'unrelated_selection_rate':unrel_ok/n_prompts,'material_stale_effect_rate':material/n_prompts,
          'max_guarded_vs_gold_kl_nats':max(kls),'top1_agreement':sum(top)/n_prompts,
        }
        rec['checks']={
          'current_selection_ge_099':rec['current_selection_rate']>=.99,
          'stale_selection_le_001':rec['stale_selection_rate']<=.01,
          'metadata_swap_no_effect':rec['metadata_swap_selection_rate']>=.99,
          'unrelated_selection_ge_099':rec['unrelated_selection_rate']>=.99,
          'material_stale_control_ge_095':rec['material_stale_effect_rate']>=.95,
          'guarded_kl_le_005':rec['max_guarded_vs_gold_kl_nats']<=.05,
          'top1_ge_098':rec['top1_agreement']>=.98,
        }
        rec['pass']=all(rec['checks'].values()); rows.append(rec)
    return {'model':model_name,'seed':seed,'hidden_width':d,'rows':rows,'any_pass':any(r['pass'] for r in rows),
      'passing_alphas':[r['alpha'] for r in rows if r['pass']],
      'strong_baseline_note':'A correctly co-located sidecar can mask/delete the stale row and may be guarantee-equivalent at lower complexity; Phase A does not establish novelty.'}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--n-prompts',type=int,default=32)
    ap.add_argument('--alphas',type=float,nargs='*',default=[.05,.1,.2,.4]); ap.add_argument('--results-dir',default='so/results'); a=ap.parse_args()
    rec={'experiment':'E-000092','title':'Generation-Keyed Neural Addressing','row':run(a.model,a.seed,a.n_prompts,a.alphas)}
    p=Path(a.results_dir);p.mkdir(parents=True,exist_ok=True); name=a.model.replace('/','_');(p/f'e000092_{name}.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
    print(json.dumps(rec,indent=2))
if __name__=='__main__': main()
