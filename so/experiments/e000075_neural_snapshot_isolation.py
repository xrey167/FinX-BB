"""E-000075 -- torn neural-memory read across multiple adapter layers.

CAVI's current NeuralConsumptionGuard validates and locks each read layer independently.  With two
read layers, a mutation can therefore linearize between them: layer 1 may consume generation G while
layer 2 consumes/rejects against generation G+1.  That is a torn neural read inside one model forward.

This experiment attacks exactly that seam on the trained symlink GPT-2 adapter.  It compares:

  per_layer -- current CAVI guard.  A relink is scheduled after the first read and before the second.
               It should be able to commit between layers, producing a mixed-generation forward.
  snapshot  -- a forward-snapshot guard holds the same authority lock from the first read pre-hook
               until the last read post-hook and uses one authority mask for all read sites.  The
               relink must remain blocked until the memory-reading portion of the forward completes.

Snapshot isolation, locks and linearizability are established systems ideas and are NOT claimed.
The candidate neural-specific property is that a single inference consuming one logical knowledge
object must not observe different authority incarnations at different neural read sites.
"""
from __future__ import annotations

import argparse, json, threading, time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from so.cavi import NeuralConsumptionGuard
from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments.e000070_cavi_live_symlink_boundary import _authority_and_manifest, _row, _run_logits, _text
from so.llm_adapter import AdapterConfig, transformer_blocks


class ForwardSnapshotGuard:
    """Reference baseline: one authority snapshot across all adapter read sites in a forward."""
    def __init__(self, adapter, mask_fn, lock):
        self.adapter=adapter; self.mask_fn=mask_fn; self.lock=lock
        self.blocks=transformer_blocks(adapter.lm)
        self.first=int(adapter.cfg.read_layers[0]); self.last=int(adapter.cfg.read_layers[-1])
        self.held=False; self.pre=self.blocks[self.first].register_forward_pre_hook(self._begin)
        try: self.post=self.blocks[self.last].register_forward_hook(self._end,always_call=True)
        except TypeError: self.post=self.blocks[self.last].register_forward_hook(self._end)
    def _begin(self,module,inputs):
        ctx=self.adapter._ctx
        if ctx is None: return None
        self.lock.acquire(); self.held=True
        try:
            base=ctx.get('_cavi_base_allowed')
            if base is None:
                base=ctx['allowed'].clone(); ctx['_cavi_base_allowed']=base
            live=self.mask_fn()
            if not torch.is_tensor(live): live=torch.as_tensor(live,dtype=torch.bool,device=base.device)
            else: live=live.to(device=base.device,dtype=torch.bool)
            ctx['allowed']=base & live
        except Exception:
            self.held=False; self.lock.release(); raise
        return None
    def _end(self,module,inputs,output):
        if self.held:
            self.held=False; self.lock.release()
        return None
    def close(self): self.pre.remove(); self.post.remove()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()


def _case(gk,centre,seed,groups,template,offset):
    rng=np.random.default_rng(offset+seed)
    world,spec=E15.sample_alias_world(rng,180,groups,2,gk.n_entities,4,E20.N_TRAIN_TEMPLATES)
    store,kids=E15.load_arm(world,spec,centre,offset+1000+seed,symlink=True)
    bank=bank_from_store(store); auth,manifest=_authority_and_manifest(store,bank)
    target,aliases=spec.groups[0]; target2,_=spec.groups[1]
    aid=kids[aliases[0]]; p2=kids[target2]; apos=_row(bank,aid); txt=_text(gk,aliases[0],template)
    return world,store,bank,auth,manifest,aid,p2,apos,txt,int(world.index[target])


def _arm(gk,centre,seed,groups,template,*,snapshot:bool)->Dict[str,object]:
    world,store,bank,auth,manifest,aid,p2,apos,text,truth=_case(gk,centre,seed,groups,template,120000+(10000 if snapshot else 0))
    fresh_pred,fresh_logits,_=_run_logits(gk,bank,text)
    commit=auth.row_mask(manifest,full=True)
    reject=np.ones(bank.size,dtype=bool); reject[apos]=False
    _,reject_logits,_=_run_logits(gk,bank,text,reject)

    reads=list(map(int,gk.model.cfg.read_layers))
    if len(reads)<2: raise RuntimeError('E-000075 requires >=2 read layers')
    blocks=transformer_blocks(gk.model.lm)
    started=threading.Event(); finished=threading.Event(); state={'err':None}
    def mutate():
        started.set()
        try:
            store.relink(aid,p2); auth.relink_alias(aid,p2)
        except Exception as e: state['err']=repr(e)
        finished.set()
    coordination={'finished_between':False,'thread':None}
    # Register on the second read block BEFORE the guard so, in per-layer mode, mutation occurs before
    # that layer's fresh validation.  In snapshot mode the forward-wide lock is already held from layer 1.
    def between(module,inputs):
        th=threading.Thread(target=mutate,daemon=True); coordination['thread']=th; th.start()
        if not started.wait(1.0): raise RuntimeError('mutator did not start')
        coordination['finished_between']=finished.wait(0.10)
    h=blocks[reads[1]].register_forward_pre_hook(between)
    try:
        guard=(ForwardSnapshotGuard(gk.model,lambda:auth.row_mask(manifest,full=True),auth.lock)
               if snapshot else NeuralConsumptionGuard(gk.model,lambda:auth.row_mask(manifest,full=True),lock=auth.lock))
        with guard:
            race_pred,race_logits,_=_run_logits(gk,bank,text,commit)
    finally:
        h.remove()
    th=coordination['thread']
    if th is not None: th.join(timeout=2.0)
    if th is not None and th.is_alive(): raise RuntimeError('mutator remained blocked after forward')
    if state['err'] is not None: raise RuntimeError(state['err'])

    # all-old reference after mutation: cached commit mask intentionally permits the old serialized row.
    _,old_logits,_=_run_logits(gk,bank,text,commit)
    return {
      'snapshot':snapshot,
      'fresh_before_correct':fresh_pred==truth,
      'mutation_finished_between_reads':bool(coordination['finished_between']),
      'mutation_finished_after_forward':bool(finished.is_set()),
      'race_pred':int(race_pred),
      'race_vs_all_old_maxabs':float((race_logits-old_logits).abs().max()),
      'race_vs_all_reject_maxabs':float((race_logits-reject_logits).abs().max()),
      'old_vs_reject_maxabs':float((old_logits-reject_logits).abs().max()),
    }


def run(seed:int,steps:int,groups:int,template:int)->Dict[str,object]:
    torch.manual_seed(seed)
    gk=E8.GPT2Knowledge(AdapterConfig(status_gated=True,use_links=True,n_deref=E20.N_DEREF))
    tr=E20.train_adapter_links(gk,seed,steps,n_groups=max(groups,24),verbose=True); gk.model.eval()
    centre=np.asarray(tr['centre'])
    per=_arm(gk,centre,seed,groups,template,snapshot=False)
    snap=_arm(gk,centre,seed,groups,template,snapshot=True)
    checks={
      'real_symlink_capability':per['fresh_before_correct'] and snap['fresh_before_correct'],
      'per_layer_allows_inter_read_commit':per['mutation_finished_between_reads'],
      'per_layer_is_torn_not_all_old':per['race_vs_all_old_maxabs']>1e-6,
      'per_layer_is_torn_not_all_reject':per['race_vs_all_reject_maxabs']>1e-6,
      'snapshot_blocks_inter_read_commit':not snap['mutation_finished_between_reads'],
      'snapshot_allows_mutation_after_forward':snap['mutation_finished_after_forward'],
      'snapshot_consumes_one_generation':snap['race_vs_all_old_maxabs']<=1e-7,
      'memory_path_material':snap['old_vs_reject_maxabs']>1e-6,
    }
    return {'seed':seed,'screening_pass':all(checks.values()),'checks':checks,'per_layer':per,'snapshot':snap,
            'interpretation':'Positive result shows per-read atomicity is insufficient for a multi-read neural forward; one authority snapshot across the memory-consuming forward region prevents torn-generation execution.',
            'not_claimed':'snapshot isolation, locks, MVCC, symlinks, versioning, or linearizability individually'}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,nargs='*',default=[0]); ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--groups',type=int,default=16); ap.add_argument('--template',type=int,default=9); ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--results-dir',default='so/results'); a=ap.parse_args(); torch.set_num_threads(a.threads)
    rows=[run(s,a.steps,a.groups,a.template) for s in a.seeds]
    out={'experiment':'E-000075','title':'Neural memory snapshot isolation across read layers','candidate_only':True,
         'all_screening_pass':all(r['screening_pass'] for r in rows),'rows':rows}
    p=Path(a.results_dir); p.mkdir(parents=True,exist_ok=True)
    (p/'e000075_neural_snapshot_isolation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
    if not out['all_screening_pass']: raise SystemExit(2)
if __name__=='__main__': main()
