from __future__ import annotations

import hashlib, json, os, random, statistics, time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="Qwen/Qwen3-0.6B"
REV="a08cec3036ee1085a4863a0b730e5d3f1c2f8d04"
EXPECTED_WEIGHT_SHA="f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
OUT=Path(os.environ.get("SO_R267_REPORT","ci-qwen-r267/report.json"))
N=100_000
RELATIONS=("code","score","owner","city")
CITIES=("Paris","Berlin","Rome","Madrid","Vienna","Prague","Warsaw","Lisbon")
TRAIN_TEMPLATES={
 "code":["What registry code belongs to {e}?","Give the registry identifier for {e}.","Which code is registered for {e}?"],
 "score":["What priority score belongs to {e}?","Give the priority rating for {e}.","Which score is assigned to {e}?"],
 "owner":["Who owns {e}?","Give the owner of {e}.","Which owner controls {e}?"],
 "city":["Which city is {e} located in?","Give the city location of {e}.","Where is {e} located, by city?"]}
TEST_TEMPLATES={
 "code":["Tell me {e}'s registry code.","I need the registered code of {e}.","For {e}, what is the registry identifier?"],
 "score":["Tell me {e}'s priority score.","How is {e} rated for priority?","For {e}, what score is on record?"],
 "owner":["Who is the owner behind {e}?","Which party owns {e}?","For {e}, identify its owner."],
 "city":["In what city can {e} be found?","Name the city containing {e}.","What city is {e} based in?"]}
TRAIN_ENTITIES=["Arven Delta","Belora Stack","Cinder Quay","Dovren Lab","Ester Node","Falrin Gate","Gavora Hub","Hesper Dock","Ivera Point","Jalen Yard","Korvin Hall","Lester Forge"]

def sha_file(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def entity(i):return f"ZXQ-{i:06d}"
def code(i):return f"C{i:06d}"
def owner(i):return entity((i*7919+17)%N)
def read(i,rel):
 if rel=="code":return code(i)
 if rel=="score":return (i*37)%1000
 if rel=="owner":return owner(i)
 if rel=="city":return CITIES[i%len(CITIES)]
 raise KeyError(rel)
def canonical_digest():
 h=hashlib.sha256()
 for i in range(N):
  h.update(f"{i}|{code(i)}|{(i*37)%1000}|{owner(i)}|{CITIES[i%len(CITIES)]}\n".encode())
 return h.hexdigest()
def prompts(templates,entities):
 xs=[];ys=[]
 for yi,r in enumerate(RELATIONS):
  for t in templates[r]:
   for e in entities:
    xs.append("Query: "+t.format(e=e)+"\nRequested knowledge field:");ys.append(yi)
 return xs,torch.tensor(ys,dtype=torch.long)
def encode(model,tok,texts,b=8):
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 out=[]
 for i in range(0,len(texts),b):
  bb=tok(texts[i:i+b],return_tensors="pt",padding=True,add_special_tokens=False)
  with torch.inference_mode():hs=model(**bb,output_hidden_states=True,use_cache=False,return_dict=True).hidden_states[-1]
  last=bb.attention_mask.sum(1)-1;z=hs[torch.arange(hs.shape[0]),last].float().cpu();out.append(F.normalize(z,dim=1))
 return torch.cat(out)
def ridge(X,y,c,l=.05):
 Xa=torch.cat([X,torch.ones(X.shape[0],1)],1).double();Y=F.one_hot(y,c).double();K=Xa@Xa.T+l*torch.eye(Xa.shape[0],dtype=torch.double);return Xa.T@torch.linalg.solve(K,Y)
def pred(X,W):return (torch.cat([X,torch.ones(X.shape[0],1)],1).double()@W).argmax(1)
def resolve(s):
 if len(s)==10 and s.startswith("ZXQ-") and s[4:].isdigit():
  i=int(s[4:]);return i if 0<=i<N else None
 return None

def main():
 OUT.parent.mkdir(parents=True,exist_ok=True)
 md=Path(snapshot_download(repo_id=MODEL_ID,revision=REV,allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors"]))
 weight=md/'model.safetensors'; got=sha_file(weight)
 if got!=EXPECTED_WEIGHT_SHA:raise RuntimeError(f"weight sha mismatch {got}")
 tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32)
 model.eval();[p.requires_grad_(False) for p in model.parameters()]
 tx,ty=prompts(TRAIN_TEMPLATES,TRAIN_ENTITIES);X=encode(model,tok,tx);W=ridge(X,ty,len(RELATIONS))
 rng=random.Random(267);ids=rng.sample(range(N),384);qx=[];qy=[];meta=[]
 for j,i in enumerate(ids):
  r=RELATIONS[j%4];t=TEST_TEMPLATES[r][j%3];qx.append("Query: "+t.format(e=entity(i))+"\nRequested knowledge field:");qy.append(RELATIONS.index(r));meta.append((i,r))
 t0=time.perf_counter_ns();Xt=encode(model,tok,qx);p=pred(Xt,W);planner_ns=time.perf_counter_ns()-t0
 acc=sum(int(int(p[i])==qy[i]) for i in range(len(qy)))/len(qy)
 mmu=[];correct=0
 for i,r in meta:
  t=time.perf_counter_ns();rid=resolve(entity(i));v=read(rid,r) if rid is not None else None;mmu.append(time.perf_counter_ns()-t);correct+=int(v==read(i,r))
 report={
  "stage":"R267-QWEN3-0.6B-CROSS-MODEL-100K","model_id":MODEL_ID,"revision":REV,"weight_sha256":got,
  "parameter_count":sum(x.numel() for x in model.parameters()),"trainable_parameter_count":sum(x.numel() for x in model.parameters() if x.requires_grad),
  "hidden_size":int(model.config.hidden_size),"num_hidden_layers":int(model.config.num_hidden_layers),
  "knowledge_objects":N,"canonical_snapshot_sha256":canonical_digest(),"fact_rewrites_for_model":0,"per_fact_gradient_steps":0,"runtime_knowledge_text_tokens":0,
  "global_planner_parameters":int(W.numel()),"heldout_entity_and_template_accuracy":acc,"planner_queries":len(qx),"planner_ms_total":planner_ns/1e6,
  "mmu_exact_accuracy":correct/len(meta),"mmu_median_us":statistics.median(mmu)/1e3,
  "scientific_scope":"real Qwen3-0.6B backend over the same deterministic model-neutral 100k KCIR snapshot family used by R264; only a small global language->schema planner is calibrated. Facts remain model-neutral and gradient-free.",
  "dod_status":"NOT_DOD; real >=0.6B second-backbone/cross-model mechanism gate"
 }
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
 if acc<.95:return 2
 if report['mmu_exact_accuracy']!=1.0:return 3
 return 0
if __name__=='__main__':raise SystemExit(main())
