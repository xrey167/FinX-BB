"""Exploratory performance seam: clean-prefix, write-only scope.

Unlike E72's execution-state-machine transplant, an accepted query runs the
unaltered learned resolver/dereference/write path. A single scope decision made
before the first write controls all memory writes. Scope is ordinary prior art,
not a novelty claim. No held-out templates select a threshold or train the head.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from so.llm_adapter import KnowledgeAdapterLM, AdapterConfig, transformer_blocks
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.experiments import e000079_joint_contract as E79
from so.experiments.cavi_capability_continuation import digest, fresh_audit
from so.experiments.cavi_provenance_audit import world_case
from so.experiments.e000070_cavi_live_symlink_boundary import _text

class WriteScopeAdapter(KnowledgeAdapterLM):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.scope_head=nn.Sequential(nn.LayerNorm(self.d,elementwise_affine=False),
                                     nn.Linear(self.d,64),nn.GELU(),nn.Linear(64,1))
        self.scope_enabled=True
        self.scope_force=None

    def _make_hook(self,i,layer):
        original=super()._make_hook(i,layer)
        def hook(module,inputs,output):
            ctx=self._ctx
            if ctx is None or not self.scope_enabled:
                return original(module,inputs,output)
            h=output[0] if isinstance(output,tuple) else output
            ar=torch.arange(h.shape[0],device=h.device)
            if i==0:
                # Before any memory has entered the residual stream.
                feature=h[ar,ctx['last_idx']]
                if self.scope_force is None:
                    keep=self.scope_head(feature).squeeze(-1)>=0
                else:
                    keep=torch.full((h.shape[0],),bool(self.scope_force),device=h.device)
                ctx['_write_scope_keep']=keep
            keep=ctx['_write_scope_keep']
            if not bool(keep.any()):return None
            changed=original(module,inputs,output)
            if changed is None or bool(keep.all()):return changed
            h2=changed[0] if isinstance(changed,tuple) else changed
            mixed=torch.where(keep[:,None,None],h2,h)
            return (mixed,)+tuple(changed[1:]) if isinstance(changed,tuple) else mixed
        return hook

@torch.no_grad()
def features(gk,texts):
    blocks=transformer_blocks(gk.model.lm);captured=[];last_box=[]
    def capture(module,inputs,output):
        h=output[0] if isinstance(output,tuple) else output
        ar=torch.arange(h.shape[0],device=h.device)
        captured.append(h[ar,last_box[0]].detach().cpu())
    handle=blocks[gk.model.cfg.read_layers[0]].register_forward_hook(capture)
    try:
        for start in range(0,len(texts),32):
            ids,am,last=E8.encode_texts(gk.tok,texts[start:start+32]);last_box[:]=[last]
            # Direct clean backbone call: no vocabulary head and no memory context.
            if gk.model._ctx is not None:raise RuntimeError('feature calibration saw memory')
            gk.model.lm.base_model(input_ids=ids,attention_mask=am,use_cache=False,return_dict=True)
    finally:handle.remove()
    return torch.cat(captured)

def parameter_digest(model):
    h=hashlib.sha256()
    for name,t in sorted(E8.adapter_state(model).items()):
        if name.startswith('scope_head.'):continue
        h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def train_scope(gk,seed):
    rng=np.random.default_rng(881000+seed)
    texts=[];labels=[]
    for _ in range(2048):
        s=int(rng.integers(gk.n_entities));r=int(rng.integers(4));t=int(rng.integers(8))
        texts.append(E17.TEMPLATES12[r][t].format(s=gk.names[s]));labels.append(1.)
    extra=('What is two plus two?','How does a rainbow form?',
           'Explain the difference between a list and a tuple.',
           'Write a brief poem about a quiet lake.',
           'Why does water freeze?','How can a program sort numbers?')
    for i in range(2048):
        name=gk.names[int(rng.integers(gk.n_entities))]
        text=(E18.TRAIN_GENERIC[int(rng.integers(len(E18.TRAIN_GENERIC)))].format(s=name)
              if i%4 else extra[int(rng.integers(len(extra)))])
        texts.append(text);labels.append(0.)
    data_sha=hashlib.sha256(json.dumps(list(zip(texts,labels)),separators=(',',':')).encode()).hexdigest()
    x=features(gk,texts);y=torch.tensor(labels)
    for p in gk.model.parameters():p.requires_grad_(False)
    for p in gk.model.scope_head.parameters():p.requires_grad_(True)
    opt=torch.optim.AdamW(gk.model.scope_head.parameters(),lr=1e-3,weight_decay=.01)
    generator=torch.Generator().manual_seed(882000+seed)
    history=[]
    for step in range(500):
        ix=torch.randint(len(x),(128,),generator=generator)
        logits=gk.model.scope_head(x[ix]).squeeze(-1)
        loss=nn.functional.binary_cross_entropy_with_logits(logits,y[ix])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%100==0:history.append({'step':step,'loss':float(loss.detach())})
    with torch.no_grad():acc=float(((gk.model.scope_head(x).squeeze(-1)>=0)==y.bool()).float().mean())
    return {'training_data_sha256':data_sha,'examples':len(x),'steps':500,'threshold_logit':0.,
            'train_accuracy':acc,'history':history,'heldout_templates_used_for_training':False}

@torch.no_grad()
def preflight(gk,centre,seed):
    _,spec,_,_,bank,auth,manifest=world_case(gk,centre,seed)
    texts=[_text(gk,k,9) for k in spec.alias_keys[:8]]
    gk.model.scope_enabled=False
    p0,l0=E79.infer(gk,bank,texts,auth,manifest)
    gk.model.scope_enabled=True;gk.model.scope_force=True
    p1,l1=E79.infer(gk,bank,texts,auth,manifest)
    torch.testing.assert_close(l0,l1,rtol=0,atol=0)
    gk.model.scope_force=False
    _,off=E79.infer(gk,bank,texts,auth,manifest)
    _,base=E79.infer(gk,None,texts)
    torch.testing.assert_close(off,base,rtol=0,atol=0)
    gk.model.scope_force=None
    return {'forced_on_vs_original_maxabs':float((l0-l1).abs().max()),
            'forced_off_vs_base_maxabs':float((off-base).abs().max()),
            'qualification':'Mechanism equality, not learned out-of-domain scope accuracy.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(2);os.environ['SO_BOS']='1'
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);seed=int(ck['seed'])
    torch.manual_seed(883000+seed)
    old_factory=E8.KnowledgeAdapterLM;E8.KnowledgeAdapterLM=WriteScopeAdapter
    try:gk=E8.GPT2Knowledge(AdapterConfig(**ck['config']),model_name=ck['model_name'])
    finally:E8.KnowledgeAdapterLM=old_factory
    current=E8.adapter_state(gk.model)
    if {k for k in current if not k.startswith('scope_head.')}!=set(ck['adapter']):
        raise ValueError('parent adapter keys differ')
    for name in ('candidate_ids','entity_token_ids'):
        if not torch.equal(current[name],ck['adapter'][name]):raise ValueError('token identity mismatch')
    status=gk.model.load_state_dict(ck['adapter'],strict=False)
    if status.unexpected_keys or any(not (n.startswith('lm.') or n.startswith('scope_head.')) for n in status.missing_keys):
        raise ValueError('unexpected missing weights')
    gk.model.eval();centre=np.asarray(ck['centre']);a.output.mkdir(parents=True,exist_ok=False)
    record={'experiment':'CAVI-WRITE-ONLY-SCOPE-PILOT','seed':seed,'candidate_only':True,'breakthrough':False,
            'source_commit':os.getenv('GITHUB_SHA'),'parent_checkpoint_sha256':digest(a.checkpoint),
            'preflight':preflight(gk,centre,seed),'frozen_reader_digest_before':parameter_digest(gk.model)}
    gk.model.scope_enabled=False
    record['before']=E79.evaluate(gk,centre,seed,100)
    (a.output/'before.json').write_text(json.dumps(record,indent=2))
    record['training']=train_scope(gk,seed)
    record['frozen_reader_digest_after']=parameter_digest(gk.model)
    if record['frozen_reader_digest_before']!=record['frozen_reader_digest_after']:
        raise AssertionError('scope training changed the memory reader')
    gk.model.scope_enabled=True;gk.model.eval()
    torch.save({'adapter':E8.adapter_state(gk.model),'scope':gk.model.scope_head.state_dict(),
                'centre':centre.tolist(),'seed':seed,'config':ck['config'],'model_name':ck['model_name'],
                'provenance':record},a.output/'scoped_reader.pt')
    record['checkpoint_sha256']=digest(a.output/'scoped_reader.pt')
    record['fresh']=fresh_audit(gk,centre,seed)
    record['after']=E79.evaluate(gk,centre,seed,100)
    record['not_established']=['full missing-key battery','REVOKE/full leakage battery',
          'independent J-space audit','full adversarial battery','multiple backbones','novelty']
    (a.output/'result.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)
    if not record['fresh']['valid_reader'] or not record['after']['screening_pass']:raise SystemExit(2)

if __name__=='__main__':main()
