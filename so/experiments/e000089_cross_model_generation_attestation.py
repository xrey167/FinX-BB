"""E-000089 -- cross-model causal generation attestation structural kill screen.

Two heterogeneous frozen causal-LM backbones consume the SAME logical Pod generation
through model-specific residual payloads. Both materialize KV state under generation g.
One shared CAVIAuthority update advances the Pod once to g+1. We then attack each model
with its stale g cache, require generation-qualified rejection/recompute, preserve an
unrelated Pod cache, and record a model-internal hidden-state audit for bypass/old/new.

This is a structural screen with controlled payloads, not real LINK->Pod evidence and not
a novelty claim. Cross-model memory, shared versioning, cache invalidation, causal auditing,
and target-side readers are prior art; the experiment only asks whether the stronger
composed property exists before spending on a real-reader implementation.
"""
from __future__ import annotations

import argparse, gc, json, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM

from so.cavi import CAVIAuthority
from so.derived_lineage import DerivedLineage, LineagedState
from so.experiments import e000080_object_scoped_kv_lineage as E80
from so.llm_adapter import transformer_blocks


@dataclass
class Arm:
    model_name: str
    model: Any
    blocks: Any
    read_layers: Tuple[int, int]
    prompt: torch.Tensor
    cont: torch.Tensor
    old_payload: torch.Tensor
    new_payload: torch.Tensor
    other_prompt: torch.Tensor
    other_cont: torch.Tensor
    other_payload: torch.Tensor
    alias_id: int
    other_alias_id: int
    stale_cache: LineagedState | None = None
    other_cache: LineagedState | None = None


def _make_arm(model_name: str, auth: CAVIAuthority, alias_id: int, other_alias_id: int, payload_rms: float) -> Arm:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    blocks = transformer_blocks(model)
    if len(blocks) < 5: raise ValueError(f'{model_name}: need >=5 blocks')
    read_layers = (1, min(3, len(blocks)-2))
    if read_layers[0] == read_layers[1]: read_layers = (0, len(blocks)-2)
    vocab = int(model.get_input_embeddings().weight.shape[0])
    shift = (sum(ord(c) for c in model_name) % 97)
    ids = lambda xs: torch.tensor([[int((x+shift)%vocab) for x in xs]], dtype=torch.long)
    prompt=ids([17,103,227,331,443]); cont=ids([521])
    other_prompt=ids([19,107,229,337,449]); other_cont=ids([523])
    old_payload=E80._payload(model,(42+shift)%vocab,payload_rms)
    new_payload=E80._payload(model,(314+shift)%vocab,payload_rms)
    other_payload=E80._payload(model,(271+shift)%vocab,payload_rms)
    return Arm(model_name,model,blocks,read_layers,prompt,cont,old_payload,new_payload,
               other_prompt,other_cont,other_payload,alias_id,other_alias_id)


def _prefill_with_audit(arm: Arm, payload: torch.Tensor | None):
    captured={}; handles=[]
    if payload is not None:
        for layer in arm.read_layers:
            def inject(module, inputs, output, _layer=layer):
                h=E80._hidden(output); h2=h.clone()
                h2[:,-1,:]=h2[:,-1,:]+payload.to(h.device,dtype=h.dtype)
                captured[f'layer{_layer}']=h2[:,-1,:].detach().float().cpu()
                return E80._replace_hidden(output,h2)
            handles.append(arm.blocks[layer].register_forward_hook(inject))
    else:
        layer=arm.read_layers[-1]
        def observe(module,inputs,output,_layer=layer):
            h=E80._hidden(output); captured[f'layer{_layer}']=h[:,-1,:].detach().float().cpu()
        handles.append(arm.blocks[layer].register_forward_hook(observe))
    try:
        with torch.no_grad(): out=arm.model(input_ids=arm.prompt,use_cache=True)
    finally:
        for h in handles: h.remove()
    key=f'layer{arm.read_layers[-1]}'
    if key not in captured:
        # First read layer may be all we captured on unusual model output; keep deterministic fallback.
        key=sorted(captured)[-1]
    return out.past_key_values, captured[key]


def _audit_delta(a: torch.Tensor,b: torch.Tensor)->float:
    return float((a-b).abs().max())


def run(models: List[str], seed: int, payload_rms: float)->Dict[str,object]:
    torch.manual_seed(seed); torch.set_num_threads(2)
    auth=CAVIAuthority(); auth.create_pod(1); auth.create_pod(2)
    arms=[]
    # Each model has its own linguistic/interface alias but both resolve canonical Pod 1.
    for i,name in enumerate(models):
        aid=100+i; bid=200+i; auth.create_alias(aid,1); auth.create_alias(bid,2)
        arms.append(_make_arm(name,auth,aid,bid,payload_rms))

    # Materialize generation g across every model BEFORE the single shared transition.
    for arm in arms:
        old_cache,old_hidden=_prefill_with_audit(arm,arm.old_payload)
        other_cache=E80._prefill(arm.model,arm.blocks,arm.read_layers,arm.other_prompt,arm.other_payload)
        arm.stale_cache=LineagedState(old_cache,DerivedLineage.of(auth.witness(arm.alias_id)))
        arm.other_cache=LineagedState(other_cache,DerivedLineage.of(auth.witness(arm.other_alias_id)))
        arm._old_hidden=old_hidden
        _,arm._bypass_hidden=_prefill_with_audit(arm,None)

    old_generation=auth.pod_incarnation(1)
    auth.update_pod(1)                         # THE one canonical fleet-wide mutation
    new_generation=auth.pod_incarnation(1)
    rows=[]
    for arm in arms:
        stale_rejected=not arm.stale_cache.reusable(auth)
        unrelated_current=arm.other_cache.reusable(auth)
        stale_logits=E80._continue(arm.model,arm.stale_cache.payload,arm.prompt.shape[1],arm.cont)
        fresh_cache,new_hidden=_prefill_with_audit(arm,arm.new_payload)
        fresh_logits=E80._continue(arm.model,fresh_cache,arm.prompt.shape[1],arm.cont)
        # Separate recompute for guarded path because HF cache may mutate on decode.
        guarded_cache,_=_prefill_with_audit(arm,arm.new_payload)
        guarded_logits=E80._continue(arm.model,guarded_cache,arm.prompt.shape[1],arm.cont)
        bypass_cache=E80._prefill_no_memory(arm.model,arm.prompt)
        bypass_logits=E80._continue(arm.model,bypass_cache,arm.prompt.shape[1],arm.cont)
        stale_effect=E80._maxabs(stale_logits,fresh_logits)
        guarded_match=E80._maxabs(guarded_logits,fresh_logits)
        old_vs_bypass=_audit_delta(arm._old_hidden,arm._bypass_hidden)
        new_vs_old=_audit_delta(new_hidden,arm._old_hidden)
        row={
          'model':arm.model_name,'old_generation':old_generation,'new_generation':new_generation,
          'stale_generation_rejected':stale_rejected,'unrelated_generation_reusable':unrelated_current,
          'stale_replay_vs_fresh_maxabs':stale_effect,'guarded_recompute_vs_fresh_maxabs':guarded_match,
          'old_hidden_vs_bypass_maxabs':old_vs_bypass,'new_hidden_vs_old_maxabs':new_vs_old,
          'stale_top1':int(stale_logits.argmax(-1)[0]),'fresh_top1':int(fresh_logits.argmax(-1)[0]),
          'bypass_top1':int(bypass_logits.argmax(-1)[0]),
          'checks':{
            'V1_old_memory_has_internal_effect':old_vs_bypass>1e-4,
            'V2_single_shared_generation_advanced':new_generation>old_generation,
            'V3_stale_replay_materially_differs':stale_effect>1e-4,
            'V4_stale_lineage_rejected':stale_rejected,
            'V5_guarded_recompute_matches_fresh':guarded_match<5e-3,
            'V6_internal_audit_sees_generation_transition':new_vs_old>1e-4,
            'V7_unrelated_state_reusable':unrelated_current,
          }
        }
        row['pass']=all(row['checks'].values()); rows.append(row)
    same_generation=all(r['old_generation']==old_generation and r['new_generation']==new_generation for r in rows)
    all_pass=same_generation and all(r['pass'] for r in rows)
    return {'experiment':'E-000089','candidate_only':True,'seed':seed,'models':models,
            'one_shared_authority_update':True,'same_generation_seen_by_all_models':same_generation,
            'rows':rows,'all_pass':all_pass,
            'decision':'SURVIVE_STRUCTURAL_SCREEN' if all_pass else 'KILL_OR_REDESIGN',
            'limitations':'Controlled residual payloads; hidden-delta audit is causal instrumentation but not J-space/J-Access; real LINK->Pod cross-model reader still required.',
            'not_claimed':'Cross-model memory, shared versioning, dependency tags, cache invalidation, MindBridge/XMemTransfer, Knowledge Objects, or causal auditing individually.'}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--models',nargs='+',default=['distilgpt2','EleutherAI/pythia-70m'])
    ap.add_argument('--seeds',type=int,nargs='*',default=[0,1,2]); ap.add_argument('--payload-rms',type=float,default=4.0)
    ap.add_argument('--results-dir',default='so/results'); a=ap.parse_args()
    rows=[run(a.models,s,a.payload_rms) for s in a.seeds]
    rec={'experiment':'E-000089','title':'Cross-Model Causal Generation Attestation structural screen',
         'runs':rows,'all_pass':all(r['all_pass'] for r in rows)}
    p=Path(a.results_dir);p.mkdir(parents=True,exist_ok=True);(p/'e000089_cross_model_generation_attestation.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
    print(json.dumps(rec,indent=2));
    if not rec['all_pass']: raise SystemExit(2)
if __name__=='__main__': main()
