"""E-000057 — make the intended router semantics explicit: MATCH / UNKNOWN / BYPASS.

The current two-channel adapter still composes two soft gates independently: relevance scales only the
null contribution while match scales only the cell contribution.  Therefore unrelated prose can still
receive a cell payload if it accidentally matches a key, and a relevant broken question can receive only
a fractional UNKNOWN because the null softmax weight is small.  This experiment implements the actual
three-state semantics at inference without changing the frozen LM or store:

  irrelevant text                  -> BYPASS, exactly zero adapter write
  relevant question + key match    -> MATCH, cell payload only
  relevant question + no key match -> UNKNOWN, full null/unknown payload

Two arms separate policy from training: 'eval_only' trains the recorded E-000052 soft architecture and
switches to the exact mux only for evaluation; 'soft_train' trains a differentiable three-state analogue
before the same hard evaluation.  Any positive grid cell is diagnostic only and must be selected on a
separate calibration split before multi-seed / attack validation.
"""
from __future__ import annotations

import argparse, copy, json, os, time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.llm_adapter import AdapterConfig, KnowledgeAdapterLM

BARS={
 "train/active_correct":(">=",.95), "heldout/active_correct":(">=",.95),
 "revoke_train_min":(">=",.95), "revoke_heldout_min":(">=",.95), "shred_heldout_min":(">=",.95),
 "heldout/revoked_deleted_object":("<=",.02), "broken1_unknown":(">=",.90),
 "generic/kl_to_base":("<=",.05),
}
REL_TAUS=(.30,.50,.70)
MATCH_OFFSETS=(-.05,0.0,.05)

class ThreeStateAdapter(KnowledgeAdapterLM):
    """Experiment-local hook; restricted to the n_deref=0 architecture used here."""
    def __init__(self,*a,**kw):
        self.tri_mode='parent'; self.rel_tau=.5; self.match_offset=0.0
        super().__init__(*a,**kw)

    def _make_hook(self, read_index:int, layer:int):
        def hook(module, inputs, output):
            if self._ctx is None: return None
            if self.cfg.n_deref != 0: raise RuntimeError('E-000057 is defined only for n_deref=0')
            h=output[0] if isinstance(output,tuple) else output
            ctx=self._ctx; B=h.shape[0]; ar=torch.arange(B,device=h.device)
            hl=h[ar,ctx['last_idx']]
            q=self.q_proj[str(layer)](self.q_ln[str(layer)](hl)); ctx.setdefault('query',[]).append(q)
            keys=torch.cat([ctx['keys'],self.null_key[read_index][None]])
            values=torch.cat([ctx['values'],self.null_value[read_index][None]])
            allowed=torch.cat([ctx['allowed'],torch.ones(1,dtype=torch.bool,device=h.device)])
            scores=(q@keys.t())*(self.scale/self.cfg.d_key**0.5); scores=scores.masked_fill(~allowed[None],float('-inf'))
            p=torch.softmax(scores,dim=-1); ctx['routing'].append(p)
            val=p@values; w_null=p[:,-1:]
            null_c=w_null*values[-1][None]; cell_c=val-null_c

            rel=torch.sigmoid(self.query_relevance[str(layer)](hl)).squeeze(-1)
            ctx.setdefault('relevance',[]).append(rel.detach())
            cells=ctx['keys']; qn=q/(q.norm(dim=-1,keepdim=True)+1e-6)
            kn=cells/(cells.norm(dim=-1,keepdim=True)+1e-6)
            cos=(qn@kn.t()).masked_fill(~ctx['allowed'][None],-1.0)
            cos_max=cos.max(dim=-1).values if cos.shape[-1] else torch.full((B,),-1.0,device=h.device)
            m=torch.sigmoid((cos_max-self.match_tau[read_index])*self.match_temp[read_index].abs())
            ctx.setdefault('match',[]).append(m.detach())

            if self.tri_mode=='parent':
                # Exact E-000022/E-000052 behaviour.
                null_c=null_c*rel[:,None]
                cell_c=cell_c*m[:,None]
            elif self.tri_mode=='soft':
                # Differentiable three-state approximation for training.
                null_c=null_c*rel[:,None]*(1.0-m[:,None])
                cell_c=cell_c*rel[:,None]*m[:,None]
            elif self.tri_mode=='hard':
                rh=(rel>=float(self.rel_tau)).to(val.dtype)
                mt=self.match_tau[read_index]+float(self.match_offset)
                mh=(cos_max>=mt).to(val.dtype)
                # Exact bypass for prose; full UNKNOWN payload for a relevant no-match question.
                cell_c=cell_c*(rh*mh)[:,None]
                null_c=values[-1][None]*(rh*(1.0-mh))[:,None]
            else:
                raise ValueError(self.tri_mode)

            read=self.o_proj[str(layer)](cell_c+null_c)
            rms_h=hl.detach().pow(2).mean(-1,keepdim=True).sqrt()
            if self.cfg.fallback=='prior':
                read=read*rms_h*self.inject_gain[read_index]
            else:
                ref=self.o_proj[str(layer)](val)
                rms_r=ref.pow(2).mean(-1,keepdim=True).sqrt().clamp_min(1e-3*rms_h+1e-6)
                read=read*(rms_h/rms_r)*self.inject_gain[read_index]
            delta=torch.zeros_like(h); delta[ar,ctx['last_idx']]=read
            h2=h+delta
            return (h2,)+tuple(output[1:]) if isinstance(output,tuple) else h2
        return hook


def crit(m:Dict[str,float])->Dict:
    out={}
    for k,(op,b) in BARS.items():
        v=float(m.get(k,float('nan'))); ok=v>=b if op=='>=' else v<=b
        out[k]={'value':v,'op':op,'bar':b,'pass':bool(ok)}
    return out

def evaluate(gk,centre,seed):
    m=E17.evaluate_templates(gk, 9700+seed, centre, E18.N_TRAIN_TEMPLATES)
    return {k:float(v) for k,v in m.items() if isinstance(v,(int,float,bool))}

def run(arm:str,seed:int,steps:int,threads:int,outdir:str):
    if threads: torch.set_num_threads(threads)
    os.environ['SO_BOS']='1'
    old=E8.KnowledgeAdapterLM; E8.KnowledgeAdapterLM=ThreeStateAdapter
    try:
        cfg=AdapterConfig(status_gated=True,match_gate=True,two_channel_null=True)
        gk=E8.GPT2Knowledge(cfg)
    finally:
        E8.KnowledgeAdapterLM=old
    model=gk.model; model.tri_mode='soft' if arm=='soft_train' else 'parent'; t0=time.time()
    out=E18.train_arm(gk,seed,steps,generic_share=.25); centre=np.asarray(out['centre'])
    trained_state=copy.deepcopy(model.state_dict())
    base=evaluate(gk,centre,seed); rows=[]
    model.tri_mode='hard'
    for rt in REL_TAUS:
        for mo in MATCH_OFFSETS:
            model.load_state_dict(trained_state); model.tri_mode='hard'; model.rel_tau=rt; model.match_offset=mo
            m=evaluate(gk,centre,seed); c=crit(m); passed=all(x['pass'] for x in c.values())
            row={'rel_tau':rt,'match_offset':mo,'metrics':m,'criteria':c,'screening_pass':passed}; rows.append(row)
            print({'arm':arm,'rel_tau':rt,'match_offset':mo,'screening_pass':passed,
                   **{k:round(c[k]['value'],4) for k in BARS}},flush=True)
    rec={'experiment':'E-000057','candidate_only':True,'arm':arm,'seed':seed,'steps':steps,
         'base_metrics':base,'rows':rows,'any_screening_pass':any(r['screening_pass'] for r in rows),
         'adapter':cfg.to_dict(),'seconds':time.time()-t0}
    p=Path(outdir);p.mkdir(parents=True,exist_ok=True)
    (p/f'e000057-{arm}-s{seed}.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
    return rec

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['eval_only','soft_train'],required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--threads',type=int,default=2);ap.add_argument('--results-dir',default='ci-e57')
    a=ap.parse_args();run(a.arm,a.seed,a.steps,a.threads,a.results_dir)
