"""Adversarial provenance audit on immutable recorded readers, without training.

Threat model: the caller may recombine serialized Bank, manifest, resolved-value,
and witness fields, but cannot mutate the independent authority except through
recorded lifecycle transitions. No arbitrary hook removal or authority rollback.
The >=.95 real-reader prerequisite is enforced on three worlds/four held-out
phrases BEFORE interpreting the neural attacks. No novelty is claimed for
content authentication, typed artifacts, or dependency tracking.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from so.cavi import CAVIAuthority, RowManifest
from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000079_joint_contract as E79
from so.experiments.cavi_capability_continuation import digest, fresh_audit
from so.experiments.e000070_cavi_live_symlink_boundary import _authority_and_manifest, _text
from so.experiments.e000071_cavi_read_hook_race import _manifest_from_live
from so.experiments.e000074_cavi_cached_router_payload_replay import (
    _live_forward, _resolved_values_from_cached_routing, _replay_resolved)
from so.llm_adapter import AdapterConfig


def world_case(gk, centre, seed):
    w, spec = E15.sample_alias_world(np.random.default_rng(70000+seed),180,100,2,
                                     gk.n_entities,4,E20.N_TRAIN_TEMPLATES)
    store, kids = E15.load_arm(w,spec,centre,71000+seed,symlink=True)
    bank = bank_from_store(store)
    auth, manifest = _authority_and_manifest(store, bank)
    return w, spec, store, kids, bank, auth, manifest


def splice_bank(gk, centre, seed):
    w, spec, store, kids, old, auth, manifest0 = world_case(gk,centre,seed)
    keys = spec.alias_keys
    texts = [_text(gk,k,9) for k in keys]
    truth0 = np.array([w.index[k] for k in keys])
    p0, l0 = E79.infer(gk,old,texts,auth,manifest0)
    # Legitimate coordinated UPDATE, while preserving the canonical identities.
    with auth.lock:
        for target, _ in spec.groups:
            store.update(kids[target],(int(w.index[target])+17)%gk.n_entities)
            auth.update_pod(kids[target])
    new = bank_from_store(store)
    manifest1 = _manifest_from_live(auth,store,new)
    if not np.array_equal(old.kid,new.kid): raise AssertionError('row order changed')
    truth1 = (truth0+17)%gk.n_entities
    pn, ln = E79.infer(gk,new,texts,auth,manifest1)
    pd, ld = E79.infer(gk,old,texts,auth,manifest0)
    # The attacker does NOT alter live state: it combines two serialized fields.
    ps, ls = E79.infer(gk,old,texts,auth,manifest1)
    m = {'n':len(keys),'fresh_before_accuracy':float((p0==truth0).mean()),
         'fresh_after_accuracy':float((pn==truth1).mean()),
         'old_manifest_accepted_rows':int(auth.row_mask(manifest0).sum()),
         'new_manifest_accepted_rows':int(auth.row_mask(manifest1).sum()),
         'unspliced_stale_old_answer_rate':float((pd==truth0).mean()),
         'spliced_stale_old_answer_rate':float((ps==truth0).mean()),
         'spliced_new_answer_rate':float((ps==truth1).mean()),
         'spliced_vs_original_maxabs':float((ls-l0).abs().max()),
         'spliced_vs_current_maxabs':float((ls-ln).abs().max()),
         'spliced_vs_unspliced_maxabs':float((ls-ld).abs().max()),
         'rows': [{'old':int(o),'new':int(n),'fresh':int(f),'ordinary_stale':int(d),'spliced':int(s)}
                  for o,n,f,d,s in zip(truth0,truth1,pn,pd,ps)]}
    m['attack_valid'] = m['fresh_before_accuracy']>=.95 and m['fresh_after_accuracy']>=.95
    m['counterexample'] = (m['attack_valid'] and m['spliced_stale_old_answer_rate']>=.95
                           and m['spliced_vs_original_maxabs']<=1e-5
                           and m['spliced_vs_current_maxabs']>1e-4)
    return m


def splice_resolved(gk, centre, seed):
    w, spec, store, kids, bank, auth, manifest = world_case(gk,centre,seed)
    target, aliases = spec.groups[0]
    other, _ = spec.groups[1]
    key = aliases[0]; aid=kids[key]; pnew=kids[other]
    text = _text(gk,key,9)
    old_truth=int(w.index[target]); new_truth=int(w.index[other])
    pred, logits, routing, _, _, _ = _live_forward(gk,bank,text)
    vals = _resolved_values_from_cached_routing(gk,bank,routing)
    original_witness = auth.witness(aid)
    with auth.lock:
        store.relink(aid,pnew); auth.relink_alias(aid,pnew)
    fresh=bank_from_store(store); fm=_manifest_from_live(auth,store,fresh)
    current,_=E79.infer(gk,fresh,[text],auth,fm)
    fresh_witness=auth.witness(aid)
    denied, _, ds = _replay_resolved(gk,text,vals,
             allow_fn=lambda:auth.validate_witness(original_witness),lock=auth.lock)
    replay, replay_logits, state = _replay_resolved(gk,text,vals,
             allow_fn=lambda:auth.validate_witness(fresh_witness),lock=auth.lock)
    valid = pred==old_truth and int(current[0])==new_truth and old_truth!=new_truth
    return {'attack_valid':bool(valid),'old_truth':old_truth,'new_truth':new_truth,
            'fresh_before':pred,'fresh_after':int(current[0]),'replayed':replay,
            'original_witness_rejected':not auth.validate_witness(original_witness),
            'fresh_witness_accepted':auth.validate_witness(fresh_witness),
            'original_witness_injected_layers':ds['injected_layers'],
            'spliced_witness_injected_layers':state['injected_layers'],
            'replay_vs_original_maxabs':float((replay_logits-logits).abs().max()),
            'counterexample':bool(valid and replay==old_truth and ds['injected_layers']==0
                                  and state['injected_layers']==len(gk.model.cfg.read_layers))}


def incomplete_lineage(gk,centre,seed):
    w,spec,store,kids,bank,auth,manifest=world_case(gk,centre,seed)
    target, aliases=spec.groups[0]; key=aliases[0]; aid=kids[key]
    text=_text(gk,key,9)
    pred,logits,routing,_,_,_=_live_forward(gk,bank,text)
    vals=_resolved_values_from_cached_routing(gk,bank,routing)
    witness=auth.witness(aid)
    # Pick the largest EXACT nonzero direct value coefficient outside the
    # claimed canonical pod, using only the pre-mutation recorded routing.
    enc=gk.model.encode_bank(bank.tensors())
    strength=torch.zeros(bank.size)
    for i in range(len(gk.model.cfg.read_layers)):
        p=routing[0,2*i]; pd=routing[0,2*i+1]
        strength += (pd[:-1] + pd[-1]*p[:-1]).abs()
    candidates=[]
    for other,_ in spec.groups[1:]:
        pos=int(np.flatnonzero(bank.kid==kids[other])[0])
        if int(w.index[other])!=int(w.index[target]): candidates.append((float(strength[pos]),other,pos))
    coefficient,other,pos=max(candidates,key=lambda x:x[0])
    prevalue=enc['values'][pos].detach().clone()
    with auth.lock:
        store.shred(kids[other]); auth.shred_pod(kids[other])
    fresh=bank_from_store(store); fm=_manifest_from_live(auth,store,fresh)
    postpred,postlog=E79.infer(gk,fresh,[text],auth,fm)
    replay,replaylog,state=_replay_resolved(gk,text,vals,
             allow_fn=lambda:auth.validate_witness(witness),lock=auth.lock)
    # A value-only intervention: remove precisely the deleted row's linear
    # contribution from the cached mixtures, without changing any other row,
    # routing coefficient, prompt, or authority state.
    reduced=[]
    for i,v in enumerate(vals):
        p=routing[0,2*i]; pd=routing[0,2*i+1]
        c=pd[pos]+pd[-1]*p[pos]
        reduced.append(v-c*prevalue)
    _,ablated,_=_replay_resolved(gk,text,reduced,
             allow_fn=lambda:auth.validate_witness(witness),lock=auth.lock)
    truth=int(w.index[target])
    valid=(pred==truth and int(postpred[0])==truth and coefficient>0)
    delta=float((replaylog-ablated).abs().max())
    return {'attack_valid':bool(valid),'queried_pod':int(kids[target]),'deleted_other_pod':int(kids[other]),
            'deleted_row_coefficient_across_reads':coefficient,
            'single_queried_witness_still_valid':auth.validate_witness(witness),
            'deleted_row_rejected_by_fresh_mask':not bool(auth.row_mask(manifest)[pos]),
            'fresh_before':pred,'fresh_after':int(postpred[0]),'replayed':replay,
            'truth':truth,'injected_layers':state['injected_layers'],
            'replay_vs_original_maxabs':float((replaylog-logits).abs().max()),
            'replay_vs_current_maxabs':float((replaylog-postlog).abs().max()),
            'deleted_value_ablation_logit_maxabs':delta,
            'counterexample':bool(valid and auth.validate_witness(witness) and delta>1e-5),
            'qualification':'Nonzero stale causal dependence, not by itself an answer-level deleted-object leakage rate.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    os.environ['SO_BOS']='1';torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    seed=int(ck['seed']); cfg=AdapterConfig(**ck['config'])
    gk=E8.GPT2Knowledge(cfg,model_name=ck['model_name'])
    current=E8.adapter_state(gk.model)
    if set(current)!=set(ck['adapter']):raise ValueError('checkpoint key mismatch')
    for k in ('candidate_ids','entity_token_ids'):
        if not torch.equal(current[k],ck['adapter'][k]):raise ValueError('token identity mismatch')
    result=gk.model.load_state_dict(ck['adapter'],strict=False)
    if result.unexpected_keys or any(not k.startswith('lm.') for k in result.missing_keys):
        raise ValueError('adapter state omitted')
    gk.model.eval();centre=np.asarray(ck['centre'])
    a.output.mkdir(parents=True,exist_ok=False)
    record={'experiment':'CAVI-PROVENANCE-FALSIFICATION','seed':seed,'model_name':ck['model_name'],
            'checkpoint_sha256':digest(a.checkpoint),'source_commit':os.getenv('GITHUB_SHA'),
            'parent_source':ck.get('provenance',{}).get('source_commit',ck.get('source_commit')),
            'breakthrough':False,'reader':fresh_audit(gk,centre,seed)}
    (a.output/'result.json').write_text(json.dumps(record,indent=2))
    if not record['reader']['valid_reader']:
        record['attacks']={'status':'SKIPPED_INVALID_READER'}
    else:
        record['attacks']={'bank_manifest_splice':splice_bank(gk,centre,seed),
                          'resolved_value_witness_splice':splice_resolved(gk,centre,seed),
                          'single_witness_dependency_omission':incomplete_lineage(gk,centre,seed)}
    (a.output/'result.json').write_text(json.dumps(record,indent=2))
    compact={**record,'attacks':{}}
    for k,v in record['attacks'].items():
        compact['attacks'][k]={kk:vv for kk,vv in v.items() if kk!='rows'} if isinstance(v,dict) else v
    print(json.dumps(compact,indent=2),flush=True)
    if not record['reader']['valid_reader']:raise SystemExit(2)
    if any(v.get('counterexample',False) for v in record['attacks'].values()):raise SystemExit(3)

if __name__=='__main__':main()
