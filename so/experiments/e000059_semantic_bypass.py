"""E-000059 — a certified no-op path for text that the knowledge layer should not touch.

The remaining locality target is unusually strict (<= .05 nats): even a tiny residual write on prose can
fail it.  This experiment tests a structural upper bound rather than another loss weight.  A small binary
head reads the frozen GPT-2 state from a bankless pass and decides whether the mutable knowledge layer is
relevant.  If not, the adapter is bypassed and the exact frozen-model output is returned; if relevant, an
E-000057 three-state adapter handles MATCH versus UNKNOWN.  Thus locality is exact for correctly bypassed
text, while broken questions still receive the memory's UNKNOWN path.

This is intentionally a two-pass screening architecture.  A positive result would establish that the
remaining seam is conditional execution, not knowledge representation; it would then need a one-pass
integration, multi-seed replication and the attack battery before any breakthrough claim.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.experiments.e000057_three_state_router import ThreeStateAdapter
from so.llm_adapter import AdapterConfig

BARS={
 "train/active_correct":(">=",.95), "heldout/active_correct":(">=",.95),
 "revoke_train_min":(">=",.95), "revoke_heldout_min":(">=",.95), "shred_heldout_min":(">=",.95),
 "heldout/revoked_deleted_object":("<=",.02), "broken1_unknown":(">=",.90), "generic/kl_to_base":("<=",.05),
}
MATCH_OFFSETS=(-.05,0.0,.05)

class SemanticBypass(nn.Module):
    def __init__(self, inner:ThreeStateAdapter, head:nn.Linear, tau:float=.5):
        super().__init__(); self.inner=inner; self.head=head; self.tau=tau
        for p in self.head.parameters(): p.requires_grad_(False)
    @property
    def cfg(self): return self.inner.cfg
    def forward(self,bank,input_ids,attention_mask,last_idx,cell_mask=None):
        if bank is None:
            return self.inner(None,input_ids,attention_mask,last_idx,cell_mask=cell_mask)
        with torch.no_grad():
            cb,fb,_,hb=self.inner(None,input_ids,attention_mask,last_idx,cell_mask=cell_mask)
            rel=torch.sigmoid(self.head(hb)).squeeze(-1)>=self.tau
            ca,fa,ra,ha=self.inner(bank,input_ids,attention_mask,last_idx,cell_mask=cell_mask)
            m=rel[:,None]
            c=torch.where(m,ca,cb); f=torch.where(m,fa,fb); h=torch.where(m,ha,hb)
        return c,f,ra,h

@torch.no_grad()
def frozen_hidden(inner,tok,texts:List[str],batch:int=64):
    xs=[]
    for i in range(0,len(texts),batch):
        ids,am,last=E8.encode_texts(tok,texts[i:i+batch]); _,_,_,h=inner(None,ids,am,last); xs.append(h)
    return torch.cat(xs)

def train_classifier(gk,seed:int,n_each:int=2048,epochs:int=250):
    rng=np.random.default_rng(25000+seed); q=[]; p=[]
    for _ in range(n_each):
        r=int(rng.integers(0,4)); t=int(rng.integers(0,E18.N_TRAIN_TEMPLATES)); s=gk.names[int(rng.integers(0,gk.n_entities))]
        q.append(E17.TEMPLATES12[r][t].format(s=s))
        p.append(E18.TRAIN_GENERIC[int(rng.integers(0,len(E18.TRAIN_GENERIC)))].format(s=gk.names[int(rng.integers(0,gk.n_entities))]))
    X=torch.cat([frozen_hidden(gk.model,gk.tok,q),frozen_hidden(gk.model,gk.tok,p)]); y=torch.cat([torch.ones(n_each),torch.zeros(n_each)])
    perm=torch.as_tensor(rng.permutation(len(y))); X=X[perm]; y=y[perm]; split=int(.8*len(y)); Xt,Xv=X[:split],X[split:]; yt,yv=y[:split],y[split:]
    head=nn.Linear(X.shape[1],1); opt=torch.optim.AdamW(head.parameters(),lr=3e-3,weight_decay=.01)
    for e in range(epochs):
        z=head(Xt).squeeze(-1); loss=F.binary_cross_entropy_with_logits(z,yt); opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        val=float(((head(Xv).squeeze(-1)>0)==yv.bool()).float().mean()); train=float(((head(Xt).squeeze(-1)>0)==yt.bool()).float().mean())
    head.eval(); return head,{'train_acc':train,'val_acc':val,'n_train':split,'n_val':len(y)-split}

def checks(m):
    c={}
    for k,(op,b) in BARS.items():
        v=float(m.get(k,float('nan')));ok=v>=b if op=='>=' else v<=b;c[k]={'value':v,'op':op,'bar':b,'pass':bool(ok)}
    return c

def run(seed,steps,threads,outdir):
    if threads:torch.set_num_threads(threads)
    os.environ['SO_BOS']='1';old=E8.KnowledgeAdapterLM;E8.KnowledgeAdapterLM=ThreeStateAdapter
    try:
        cfg=AdapterConfig(status_gated=True,match_gate=True,two_channel_null=True);gk=E8.GPT2Knowledge(cfg)
    finally:E8.KnowledgeAdapterLM=old
    inner=gk.model;inner.tri_mode='parent';t0=time.time();out=E18.train_arm(gk,seed,steps,generic_share=.25);centre=np.asarray(out['centre'])
    head,clf=train_classifier(gk,seed);rows=[]
    # External relevance makes the internal relevance decision redundant: all examples reaching the adapter are treated as relevant.
    inner.tri_mode='hard';inner.rel_tau=0.0
    for mo in MATCH_OFFSETS:
        inner.match_offset=mo;gk.model=SemanticBypass(inner,head,.5)
        m=E17.evaluate_templates(gk,11900+seed,centre,E18.N_TRAIN_TEMPLATES);mm={k:float(v) for k,v in m.items() if isinstance(v,(int,float,bool))};c=checks(mm)
        row={'match_offset':mo,'metrics':mm,'criteria':c,'screening_pass':all(x['pass'] for x in c.values())};rows.append(row)
        print({'match_offset':mo,'classifier':clf,'screening_pass':row['screening_pass'],**{k:round(c[k]['value'],4) for k in BARS}},flush=True)
        gk.model=inner
    rec={'experiment':'E-000059','candidate_only':True,'two_pass':True,'seed':seed,'steps':steps,'classifier':clf,'rows':rows,
         'any_screening_pass':any(r['screening_pass'] for r in rows),'seconds':time.time()-t0}
    p=Path(outdir);p.mkdir(parents=True,exist_ok=True);(p/f'e000059-s{seed}.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');return rec

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=0);ap.add_argument('--steps',type=int,default=1200);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--results-dir',default='ci-e59')
    a=ap.parse_args();run(a.seed,a.steps,a.threads,a.results_dir)
