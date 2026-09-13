from __future__ import annotations

import hashlib, json, os, statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="Qwen/Qwen2.5-0.5B"; REV="060db6499f32faf8b98477b0a26969ef7d8b9987"
WEIGHT_SHA="88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
OUT=Path(os.environ.get("SO_R263_REPORT","ci-qwen-r263/report.json"))
RELATIONS=["code","score","owner","city"]
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
TEST_ENTITIES=["Mavora Relay","Neris Vault","Orren Spire","Pavela Works","Quorin Field","Rester Archive","Solen Campus","Tivera Port","Ulden Keep","Varex Court","Wylin Annex","Zorin Station"]

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()

def prompts(templates,entities):
 xs=[]; ys=[]; meta=[]
 for yi,r in enumerate(RELATIONS):
  for ti,t in enumerate(templates[r]):
   for e in entities:
    xs.append("Query: "+t.format(e=e)+"\nRequested knowledge field:")
    ys.append(yi);meta.append((r,ti,e))
 return xs,torch.tensor(ys,dtype=torch.long),meta

def encode_hidden(model,tok,texts,batch_size=8):
 if tok.pad_token_id is None: tok.pad_token=tok.eos_token
 out=[]
 for i in range(0,len(texts),batch_size):
  bb=tok(texts[i:i+batch_size],return_tensors="pt",padding=True,add_special_tokens=False)
  with torch.inference_mode(): h=model(**bb,output_hidden_states=True,use_cache=False,return_dict=True).hidden_states[-1]
  last=bb.attention_mask.sum(dim=1)-1
  z=h[torch.arange(h.shape[0]),last].float().cpu()
  out.append(F.normalize(z,dim=1))
 return torch.cat(out,0)

def fit_ridge(X,y,ncls,l2=0.05):
 # Add bias and use dual ridge solve: W=X^T(XX^T+lI)^-1Y.
 Xa=torch.cat([X,torch.ones(X.shape[0],1)],1).double();Y=F.one_hot(y,ncls).double()
 K=Xa@Xa.T + l2*torch.eye(Xa.shape[0],dtype=torch.double)
 A=torch.linalg.solve(K,Y)
 return Xa.T@A

def predict(X,W):
 Xa=torch.cat([X,torch.ones(X.shape[0],1)],1).double();return (Xa@W).argmax(1)

def main():
 OUT.parent.mkdir(parents=True,exist_ok=True)
 md=Path(snapshot_download(repo_id=MODEL_ID,revision=REV,allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors"]))
 assert sha(md/'model.safetensors')==WEIGHT_SHA
 tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32)
 model.eval();[p.requires_grad_(False) for p in model.parameters()]
 tr_x,tr_y,tr_meta=prompts(TRAIN_TEMPLATES,TRAIN_ENTITIES);te_x,te_y,te_meta=prompts(TEST_TEMPLATES,TEST_ENTITIES)
 X=encode_hidden(model,tok,tr_x);Xt=encode_hidden(model,tok,te_x)
 W=fit_ridge(X,tr_y,len(RELATIONS));ptr=predict(X,W);pte=predict(Xt,W)
 tr_acc=float((ptr==tr_y).float().mean().item());te_acc=float((pte==te_y).float().mean().item())
 per_rel={};per_tpl={}
 for ri,r in enumerate(RELATIONS):
  idx=[i for i,(rr,_,_) in enumerate(te_meta) if rr==r];per_rel[r]=sum(int(pte[i]==te_y[i]) for i in idx)/len(idx)
  for ti in range(len(TEST_TEMPLATES[r])):
   idx=[i for i,(rr,tt,_) in enumerate(te_meta) if rr==r and tt==ti];per_tpl[f"{r}:{ti}"]=sum(int(pte[i]==te_y[i]) for i in idx)/len(idx)
 # Entity namespace is disjoint by construction. Planner output space is fixed at schemas, never facts/entities.
 report={
  "stage":"R263-REAL-QWEN-OPEN-WORLD-KCIR-PLANNER","model_id":MODEL_ID,"revision":REV,
  "parameter_count":sum(p.numel() for p in model.parameters()),"backbone_trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad),
  "global_planner_parameters":int(W.numel()),"per_fact_gradient_steps":0,"knowledge_id_head_size":0,
  "train_prompts":len(tr_x),"test_prompts":len(te_x),"train_entity_count":len(TRAIN_ENTITIES),"test_entity_count":len(TEST_ENTITIES),
  "entity_names_disjoint":set(TRAIN_ENTITIES).isdisjoint(TEST_ENTITIES),"templates_disjoint":all(set(TRAIN_TEMPLATES[r]).isdisjoint(TEST_TEMPLATES[r]) for r in RELATIONS),
  "train_accuracy":tr_acc,"heldout_entity_and_template_accuracy":te_acc,"per_relation_accuracy":per_rel,"per_heldout_template_accuracy":per_tpl,
  "runtime_knowledge_text_tokens":0,
  "scientific_scope":"real frozen Qwen language-to-KCIR schema planning with a small globally calibrated ridge head; held-out paraphrase templates and completely disjoint post-calibration entity names. Entity IDs remain copied/resolved pointers, not classifier classes.",
  "dod_status":"NOT_DOD; real-model natural-language planner mechanism gate"
 }
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
 if te_acc<0.95:return 2
 if min(per_rel.values())<0.90:return 3
 if min(per_tpl.values())<0.80:return 4
 return 0
if __name__=='__main__':raise SystemExit(main())
