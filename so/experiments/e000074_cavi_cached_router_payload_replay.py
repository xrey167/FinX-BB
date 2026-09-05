"""E-000074 -- stale cached router/resolved-payload replay after CAVI invalidation.

E-000073 attacks a very downstream cached post-read hidden state.  This experiment moves the attack
upstream to the neural-memory machinery itself.  During a correct live symlink read we serialize the
actual routing distributions returned by KnowledgeAdapterLM plus the encoded Bank values.  From those
cached router outputs we reconstruct each read layer's final resolved value (after the LINK dereference
slot).  After the alias is relinked, the old Bank need not be passed to model.forward at all: the cached
resolved values are injected directly at the adapter read sites.

This tests whether "validate the Bank" is enough.  It is not enough if post-authorization router output
or resolved payload becomes a bearer capability.  Baselines are no check, cached commit-time auth and
pod-only incarnation equality.  Full CAVI revalidates the original alias+pod ResolveWitness at the
actual resolved-payload injection.  Alias relink keeps the old pod live/current, so a pod-only version
check is deliberately unable to distinguish the stale pointer.

No J-space/J-lens signal is used for routing, training, gating or authorization.  This is a candidate
structural falsification test, not a novelty claim.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from so.cavi import NeuralConsumptionGuard
from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments.e000070_cavi_live_symlink_boundary import _authority_and_manifest, _text
from so.experiments.e000071_cavi_read_hook_race import _manifest_from_live
from so.llm_adapter import AdapterConfig, transformer_blocks


def _roundtrip(obj):
    b=io.BytesIO(); torch.save(obj,b); b.seek(0)
    return torch.load(b,map_location='cpu',weights_only=True)


def _live_forward(gk, bank, text):
    ids,am,last=E8.encode_texts(gk.tok,[text])
    with torch.no_grad():
        cand,full,routing,_=gk.model(bank.tensors(),ids,am,last)
    if routing is None:
        raise RuntimeError('routing was not returned')
    return int(cand.argmax(-1)[0]),full.detach().cpu(),routing.detach().cpu(),ids,am,last


def _resolved_values_from_cached_routing(gk, bank, routing:torch.Tensor)->List[torch.Tensor]:
    """Reconstruct the exact final `val` consumed at every read layer for n_deref=1.

    The routing tensor is ordered (resolve,deref) per read layer by KnowledgeAdapterLM.  Values are
    from the serialized Bank snapshot.  Returned vectors are post-dereference but pre-o_proj/RMS scale.
    """
    if gk.model.cfg.n_deref != 1:
        raise ValueError('E-000074 is preregistered for exactly one dereference slot')
    with torch.no_grad():
        enc=gk.model.encode_bank(bank.tensors())
        base_values=enc['values'].detach()
        out=[]
        for i,_layer in enumerate(gk.model.cfg.read_layers):
            values=torch.cat([base_values,gk.model.null_value[i][None]],dim=0)
            p=routing[0,2*i].to(values.device,dtype=values.dtype)
            pd=routing[0,2*i+1].to(values.device,dtype=values.dtype)
            val1=p @ values
            val2=pd[:-1] @ values[:-1] + pd[-1] * val1
            out.append(val2.detach().cpu())
    return out


def _bypass(gk,text):
    ids,am,last=E8.encode_texts(gk.tok,[text])
    with torch.no_grad():
        cand,full,_,_=gk.model(None,ids,am,last)
    return int(cand.argmax(-1)[0]),full.detach().cpu()


def _replay_resolved(gk,text,vals:List[torch.Tensor],*,allow_fn=None,lock=None,mutate_pre=None):
    ids,am,last=E8.encode_texts(gk.tok,[text])
    blocks=transformer_blocks(gk.model.lm)
    state={'injected_layers':0,'mutated':False}
    handles=[]

    if mutate_pre is not None:
        first=int(gk.model.cfg.read_layers[0])
        def mut(module,inputs):
            if not state['mutated']:
                mutate_pre(); state['mutated']=True
        handles.append(blocks[first].register_forward_pre_hook(mut))

    for i,layer in enumerate(gk.model.cfg.read_layers):
        cached=vals[i]
        def hook(module,inputs,output,i=i,layer=layer,cached=cached):
            def apply():
                if allow_fn is not None and not bool(allow_fn()):
                    return None
                h=output[0] if isinstance(output,tuple) else output
                ar=torch.arange(h.shape[0],device=h.device)
                hl=h[ar,last]
                v=cached.to(h.device,dtype=h.dtype)[None].expand(h.shape[0],-1)
                read=gk.model.o_proj[str(layer)](v)
                rms_h=hl.detach().pow(2).mean(-1,keepdim=True).sqrt()
                rms_r=read.pow(2).mean(-1,keepdim=True).sqrt().clamp_min(1e-3*rms_h+1e-6)
                read=read*(rms_h/rms_r)*gk.model.inject_gain[i]
                h2=h.clone(); h2[ar,last]=h2[ar,last]+read
                state['injected_layers']+=1
                return (h2,)+tuple(output[1:]) if isinstance(output,tuple) else h2
            if lock is None: return apply()
            with lock: return apply()
        handles.append(blocks[int(layer)].register_forward_hook(hook))
    try:
        with torch.no_grad():
            cand,full,_,_=gk.model(None,ids,am,last)
    finally:
        for h in handles: h.remove()
    return int(cand.argmax(-1)[0]),full.detach().cpu(),state


def _case(gk,centre,seed,groups,offset):
    rng=np.random.default_rng(offset+seed)
    world,spec=E15.sample_alias_world(rng,180,groups,2,gk.n_entities,4,E20.N_TRAIN_TEMPLATES)
    store,kids=E15.load_arm(world,spec,centre,offset+1000+seed,symlink=True)
    bank=bank_from_store(store); auth,manifest=_authority_and_manifest(store,bank)
    target,aliases=spec.groups[0]; target2,_=spec.groups[1]
    aid=kids[aliases[0]]; p1=kids[target]; p2=kids[target2]; text=_text(gk,aliases[0],9)
    return world,spec,store,kids,bank,auth,manifest,aid,p1,p2,text


def run(seed:int,steps:int,groups:int)->Dict[str,object]:
    torch.manual_seed(seed)
    cfg=AdapterConfig(status_gated=True,use_links=True,n_deref=1)
    gk=E8.GPT2Knowledge(cfg)
    tr=E20.train_adapter_links(gk,seed,steps,n_groups=max(groups,24),verbose=True)
    gk.model.eval(); centre=np.asarray(tr['centre'])

    world,spec,store,kids,bank,auth,manifest,aid,p1,p2,text=_case(gk,centre,seed,groups,101000)
    old_truth=int(world.index[spec.groups[0][0]]); new_truth=int(world.index[spec.groups[1][0]])
    fresh_pred,fresh_logits,routing,_,_,_=_live_forward(gk,bank,text)
    routing=_roundtrip(routing)
    vals=_roundtrip(_resolved_values_from_cached_routing(gk,bank,routing))
    witness=auth.witness(aid); commit_ok=auth.validate_witness(witness)
    bypass_pred,bypass_logits=_bypass(gk,text)

    # Relink A->Q, while P remains current/live.  Old serialized Bank/router/payload material is stale.
    store.relink(aid,p2); auth.relink_alias(aid,p2)
    none_pred,none_logits,none_state=_replay_resolved(gk,text,vals)
    commit_pred,commit_logits,commit_state=_replay_resolved(gk,text,vals,allow_fn=lambda:commit_ok)
    pod_pred,pod_logits,pod_state=_replay_resolved(gk,text,vals,allow_fn=lambda:auth.validate_pod_only(witness),lock=auth.lock)
    cavi_pred,cavi_logits,cavi_state=_replay_resolved(gk,text,vals,allow_fn=lambda:auth.validate_witness(witness),lock=auth.lock)

    fresh_bank=bank_from_store(store); fm=_manifest_from_live(auth,store,fresh_bank)
    with NeuralConsumptionGuard(gk.model,lambda:auth.row_mask(fm,full=True),lock=auth.lock):
        fresh_after_pred,fresh_after_logits,_,_,_,_=_live_forward(gk,fresh_bank,text)

    # In-forward race on an independent case. Commit authorization is cached before mutation.
    wr,spr,sr,kr,br,ar,mr,raid,rp1,rp2,rtxt=_case(gk,centre,seed,6,103000)
    _,_,rrouting,_,_,_=_live_forward(gk,br,rtxt)
    rvals=_resolved_values_from_cached_routing(gk,br,rrouting)
    rw=ar.witness(raid); rcommit=ar.validate_witness(rw)
    def mut_commit(): ar.relink_alias(raid,rp2)
    _,rclog,rcstate=_replay_resolved(gk,rtxt,rvals,allow_fn=lambda:rcommit,mutate_pre=mut_commit)

    wc,spc,sc,kc,bc,ac,mc,caid,cp1,cp2,ctxt=_case(gk,centre,seed,6,105000)
    _,_,crouting,_,_,_=_live_forward(gk,bc,ctxt)
    cvals=_resolved_values_from_cached_routing(gk,bc,crouting); cw=ac.witness(caid)
    _,cbl=_bypass(gk,ctxt)
    def mut_cavi(): ac.relink_alias(caid,cp2)
    _,rflog,rfstate=_replay_resolved(gk,ctxt,cvals,allow_fn=lambda:ac.validate_witness(cw),lock=ac.lock,mutate_pre=mut_cavi)

    metrics={
      'fresh_before_correct':fresh_pred==old_truth,
      'old_full_witness_rejected':not auth.validate_witness(witness),
      'old_pod_witness_valid':auth.validate_pod_only(witness),
      'unguarded_injected_layers':none_state['injected_layers'],
      'unguarded_vs_fresh_maxabs':float((none_logits-fresh_logits).abs().max()),
      'unguarded_vs_bypass_maxabs':float((none_logits-bypass_logits).abs().max()),
      'commit_injected_layers':commit_state['injected_layers'],
      'commit_vs_unguarded_maxabs':float((commit_logits-none_logits).abs().max()),
      'pod_injected_layers':pod_state['injected_layers'],
      'pod_vs_unguarded_maxabs':float((pod_logits-none_logits).abs().max()),
      'cavi_injected_layers':cavi_state['injected_layers'],
      'cavi_vs_bypass_maxabs':float((cavi_logits-bypass_logits).abs().max()),
      'cavi_vs_unguarded_maxabs':float((cavi_logits-none_logits).abs().max()),
      'fresh_after_pred':fresh_after_pred,'new_truth':new_truth,'fresh_after_correct':fresh_after_pred==new_truth,
      'fresh_after_vs_stale_maxabs':float((fresh_after_logits-none_logits).abs().max()),
      'race_commit_mutated':rcstate['mutated'],'race_commit_injected_layers':rcstate['injected_layers'],
      'race_cavi_mutated':rfstate['mutated'],'race_cavi_injected_layers':rfstate['injected_layers'],
      'race_cavi_vs_bypass_maxabs':float((rflog-cbl).abs().max()),
    }
    nread=len(cfg.read_layers)
    checks={
      'real_symlink_capability_before':metrics['fresh_before_correct'],
      'alias_relink_full_not_pod_only':metrics['old_full_witness_rejected'] and metrics['old_pod_witness_valid'],
      'cached_router_payload_replays_without_bank':metrics['unguarded_injected_layers']==nread and metrics['unguarded_vs_fresh_maxabs']<=2e-5,
      'replay_is_not_bypass':metrics['unguarded_vs_bypass_maxabs']>1e-6,
      'commit_time_auth_is_insufficient':metrics['commit_injected_layers']==nread and metrics['commit_vs_unguarded_maxabs']<=1e-7,
      'pod_version_check_is_insufficient':metrics['pod_injected_layers']==nread and metrics['pod_vs_unguarded_maxabs']<=1e-7,
      'full_cavi_rejects_cached_router_payload':metrics['cavi_injected_layers']==0 and metrics['cavi_vs_bypass_maxabs']<=1e-7,
      'cavi_rejection_changes_replay':metrics['cavi_vs_unguarded_maxabs']>1e-6,
      'fresh_current_generation_retains_capability':metrics['fresh_after_correct'],
      'commit_time_loses_replay_race':metrics['race_commit_mutated'] and metrics['race_commit_injected_layers']==nread,
      'cavi_closes_replay_race':metrics['race_cavi_mutated'] and metrics['race_cavi_injected_layers']==0 and metrics['race_cavi_vs_bypass_maxabs']<=1e-7,
    }
    return {'seed':seed,'steps':steps,'candidate_only':True,'screening_pass':all(checks.values()),'checks':checks,'metrics':metrics,
            'not_claimed':'routing, caches, versions, freshness, commit authorization, pointers or capabilities individually; J-space not used'}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,nargs='*',default=[0]); ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--groups',type=int,default=16); ap.add_argument('--threads',type=int,default=2); ap.add_argument('--results-dir',default='so/results')
    a=ap.parse_args(); torch.set_num_threads(a.threads)
    rows=[run(s,a.steps,a.groups) for s in a.seeds]
    rec={'experiment':'E-000074','title':'CAVI cached router/resolved-payload replay','all_screening_pass':all(r['screening_pass'] for r in rows),'rows':rows}
    p=Path(a.results_dir);p.mkdir(parents=True,exist_ok=True);(p/'e000074_cavi_cached_router_payload_replay.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
    print(json.dumps(rec,indent=2))
    if not rec['all_screening_pass']: raise SystemExit(2)
if __name__=='__main__': main()
