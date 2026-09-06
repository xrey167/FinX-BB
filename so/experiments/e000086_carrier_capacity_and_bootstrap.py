"""E-000086 -- handle-mixture readout on frozen GPT-2. RESULTS SUPERSEDED; READ THIS FIRST.

This experiment as first written scored the RAW boundary residual against the handle table. The raw
residual is dominated by the prompt, which is identical across identities, so its numbers measured the
readout and not the channel.  Two conclusions were published from them and BOTH ARE WITHDRAWN:

  * a "capacity ceiling" of 0.3799 at 700 identities -- with the no-memory baseline subtracted it is
    0.9873, so there is no capacity ceiling at realistic bank sizes;
  * a "bootstrap gap" in which a cold router conveys nothing -- the carrier in fact tolerates five
    percent of the routing mass on the correct row (0.9658) and only collapses below about one
    percent, which arm E's router clears early.

Corrected numbers, 700-row bank, baseline subtracted, chance 0.0014: one-hot 0.9873, 0.50 mass 0.9873,
0.10 mass 0.9795, 0.05 mass 0.9658, 0.01 mass 0.1377, cold start 0.0381.

The cause of arm E's end-to-end failure is therefore OPEN.  Anyone extending this should instrument a
trained checkpoint rather than synthetic injections: three explanations derived this way were artefacts
of the measurement.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from so.llm_adapter import AdapterConfig, KnowledgeAdapterLM, transformer_blocks
from so.experiments.e000085_reference_transport import PROMPTS, _hidden
torch.set_num_threads(2); torch.manual_seed(0)
tok=AutoTokenizer.from_pretrained('gpt2'); tok.pad_token=tok.eos_token; tok.padding_side='right'
lm=AutoModelForCausalLM.from_pretrained('gpt2').float().eval()
for p in lm.parameters(): p.requires_grad_(False)
blocks=transformer_blocks(lm); nb=len(blocks)
ad=KnowledgeAdapterLM(lm,AdapterConfig(read_layers=(8,10),write_layer=nb-1,reference_carrier=True),list(range(10,18)),11)
N=700                                   # a realistic bank size
H=ad.handles_for(torch.arange(N)); H=H/H.norm(dim=-1,keepdim=True)
enc=tok(PROMPTS,return_tensors='pt',padding=True); last=enc['attention_mask'].sum(-1)-1; ar=torch.arange(len(PROMPTS))
st={'v':None}
def mk():
    def hook(m,i,o):
        if st['v'] is None: return None
        h=_hidden(o); hl=h[ar,last]; rms=hl.pow(2).mean(-1,keepdim=True).sqrt()
        v=st['v'][None,:].expand(h.shape[0],-1); v=v/v.pow(2).mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
        d=torch.zeros_like(h); d[ar,last]=v*rms
        h2=h+d
        return (h2,)+tuple(o[1:]) if isinstance(o,tuple) else h2
    return hook
hs=[blocks[l].register_forward_hook(mk()) for l in (8,10)]
def boundary(V):
    out=[]
    with torch.no_grad():
        for i in range(V.shape[0]):
            st['v']=V[i]
            out.append(lm(input_ids=enc['input_ids'],attention_mask=enc['attention_mask'],output_hidden_states=True).hidden_states[-1][ar,last].clone())
    return torch.stack(out)
def ident(V_inj, targets):
    D=boundary(V_inj)
    Dn=D/D.norm(dim=-1,keepdim=True).clamp_min(1e-9)
    S=torch.einsum('npd,md->npm',Dn,H)
    return float((S.argmax(-1)==targets[:,None]).float().mean())
M=64
tgt=torch.arange(M)
print('bank of %d handles; store-supplied readout, no learning; chance %.5f' % (N,1.0/N))
print('  ONE-HOT routing (the ceiling)            top-1 %.4f' % ident(H[:M], tgt))
for p_top in (0.9,0.7,0.5,0.3):
    V=[]
    for j in range(M):
        w=torch.full((N,),(1-p_top)/(N-1)); w[j]=p_top
        v=w@H; V.append(v/v.norm())
    print('  routing mass %.1f on the right row       top-1 %.4f' % (p_top, ident(torch.stack(V), tgt)))
# and what an untrained/diffuse router actually produces
w=torch.softmax(torch.randn(M,N)*0.5,dim=-1)
V=torch.stack([ (w[j]@H)/ (w[j]@H).norm() for j in range(M)])
print('  near-uniform routing (start of training) top-1 %.4f   (max mass %.4f)' % (ident(V,tgt), float(w.max())))
for h in hs: h.remove()
