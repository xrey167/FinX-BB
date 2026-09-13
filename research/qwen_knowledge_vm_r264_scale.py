from __future__ import annotations

import hashlib, json, os, random, statistics, time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="Qwen/Qwen2.5-0.5B"; REV="060db6499f32faf8b98477b0a26969ef7d8b9987"
WEIGHT_SHA="88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
OUT=Path(os.environ.get("SO_R264_REPORT","ci-qwen-r264/report.json"))
N_OBJECTS=100_000
RELATIONS=("code","score","owner","city")
CITIES=("Paris","Berlin","Rome","Madrid","Vienna","Prague","Warsaw","Lisbon")
COUNTRIES=("France","Germany","Italy","Spain","Austria","Czechia","Poland","Portugal")
CITY_TO_COUNTRY=dict(zip(CITIES,COUNTRIES))
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


def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()

def entity(i:int)->str:return f"ZXQ-{i:06d}"
def code(i:int)->str:return f"C{i:06d}"
def owner(i:int)->str:return entity((i*7919+17)%N_OBJECTS)

def prompts(templates,entities):
 xs=[];ys=[]
 for yi,r in enumerate(RELATIONS):
  for t in templates[r]:
   for e in entities:
    xs.append("Query: "+t.format(e=e)+"\nRequested knowledge field:");ys.append(yi)
 return xs,torch.tensor(ys,dtype=torch.long)

def encode_hidden(model,tok,texts,batch_size=8):
 if tok.pad_token_id is None: tok.pad_token=tok.eos_token
 out=[]
 for i in range(0,len(texts),batch_size):
  bb=tok(texts[i:i+batch_size],return_tensors="pt",padding=True,add_special_tokens=False)
  with torch.inference_mode(): h=model(**bb,output_hidden_states=True,use_cache=False,return_dict=True).hidden_states[-1]
  last=bb.attention_mask.sum(dim=1)-1;z=h[torch.arange(h.shape[0]),last].float().cpu();out.append(F.normalize(z,dim=1))
 return torch.cat(out,0)

def fit_ridge(X,y,ncls,l2=.05):
 Xa=torch.cat([X,torch.ones(X.shape[0],1)],1).double();Y=F.one_hot(y,ncls).double();K=Xa@Xa.T+l2*torch.eye(Xa.shape[0],dtype=torch.double);return Xa.T@torch.linalg.solve(K,Y)
def predict(X,W):return (torch.cat([X,torch.ones(X.shape[0],1)],1).double()@W).argmax(1)

@dataclass(frozen=True)
class Snapshot:
 revision:int
 code_overrides:dict[int,str]
 score_overrides:dict[int,int]
 owner_overrides:dict[int,str]
 city_overrides:dict[int,str]
 masked:frozenset[int]

def read(snapshot:Snapshot,i:int,rel:str):
 if i in snapshot.masked:return None
 if rel=="code":return snapshot.code_overrides.get(i,code(i))
 if rel=="score":return snapshot.score_overrides.get(i,(i*37)%1000)
 if rel=="owner":return snapshot.owner_overrides.get(i,owner(i))
 if rel=="city":return snapshot.city_overrides.get(i,CITIES[i%len(CITIES)])
 raise KeyError(rel)

def exact_alias_resolve(alias:str):
 if len(alias)==10 and alias.startswith("ZXQ-") and alias[4:].isdigit():
  i=int(alias[4:]);return i if 0<=i<N_OBJECTS else None
 return None

def rag_fact_text(e:str,rel:str,value)->str:
 labels={"code":"registry code","score":"priority score","owner":"owner","city":"city"}
 return f"Authoritative fact: the {labels[rel]} for {e} is {value}.\n"

def country_prompt(entity_name:str,city_placeholder="X"):
 return f"{entity_name} is located in {city_placeholder}. Which country is that city in? Answer only the country:"

def country_full_prompt(entity_name:str,city:str):return country_prompt(entity_name,city)

def compile_city_slot(model,tok,entity_name:str,city:str):
 full=country_full_prompt(entity_name,city);slot=country_prompt(entity_name,"X")
 bf=tok(full,return_tensors="pt",add_special_tokens=False);bs=tok(slot,return_tensors="pt",add_special_tokens=False)
 if bf.input_ids.shape!=bs.input_ids.shape:return None
 d=(bf.input_ids[0]!=bs.input_ids[0]).nonzero().flatten()
 if d.numel()!=1:return None
 pos=int(d.item());emb=model.get_input_embeddings()(bs.input_ids).detach().clone();emb[0,pos]=model.get_input_embeddings().weight[int(bf.input_ids[0,pos])]
 return bf,bs,emb

def full_next(model,batch):
 t=time.perf_counter_ns()
 with torch.inference_mode():log=model(**batch,return_dict=True).logits[:,-1,:]
 return log,time.perf_counter_ns()-t

def slot_next(model,bs,emb):
 t=time.perf_counter_ns()
 with torch.inference_mode():log=model(inputs_embeds=emb,attention_mask=bs.attention_mask,return_dict=True).logits[:,-1,:]
 return log,time.perf_counter_ns()-t

def token_set(tok,text):
 ids=[]
 for s in (text," "+text):
  x=tok.encode(s,add_special_tokens=False)
  if len(x)==1 and x[0] not in ids:ids.append(x[0])
 return ids

def semantic_match(logits,ids):return max(float(logits[0,i]) for i in ids) if ids else -1e30

def main():
 OUT.parent.mkdir(parents=True,exist_ok=True)
 md=Path(snapshot_download(repo_id=MODEL_ID,revision=REV,allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors"]));assert sha(md/'model.safetensors')==WEIGHT_SHA
 tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False);model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32);model.eval();[p.requires_grad_(False) for p in model.parameters()]
 # Global planner calibration once, on names disjoint from the 100k installed store.
 tx,ty=prompts(TRAIN_TEMPLATES,TRAIN_ENTITIES);X=encode_hidden(model,tok,tx);W=fit_ridge(X,ty,len(RELATIONS))
 rng=random.Random(264)
 query_ids=rng.sample(range(N_OBJECTS),384);queries=[];labels=[];meta=[]
 for j,i in enumerate(query_ids):
  rel=RELATIONS[j%len(RELATIONS)];tpl=TEST_TEMPLATES[rel][j%len(TEST_TEMPLATES[rel])];queries.append("Query: "+tpl.format(e=entity(i))+"\nRequested knowledge field:");labels.append(RELATIONS.index(rel));meta.append((i,rel))
 t0=time.perf_counter_ns();Xt=encode_hidden(model,tok,queries,batch_size=8);pp=predict(Xt,W);planner_ns=time.perf_counter_ns()-t0
 planner_acc=sum(int(int(pp[j])==labels[j]) for j in range(len(labels)))/len(labels)
 # Exact MMU execution and direct Oracle-RAG latency on same Qwen CPU runtime.
 snap1=Snapshot(1,{}, {}, {}, {}, frozenset())
 vm_values=[];vm_ns=[];rag_ns=[];rag_tokens=[]
 for j,(i,rel) in enumerate(meta[:96]):
  e=entity(i);t=time.perf_counter_ns();rid=exact_alias_resolve(e);v=read(snap1,rid,rel) if rid is not None else None;vm_ns.append(time.perf_counter_ns()-t);vm_values.append(v)
  # Privileged Oracle-RAG: perfect address, perfect fact, no vector search/rerank. It still pays knowledge tokens + Qwen read.
  q=TEST_TEMPLATES[rel][j%len(TEST_TEMPLATES[rel])].format(e=e);fact=rag_fact_text(e,rel,v);rb=tok(fact+"Question: "+q+"\nAnswer:",return_tensors="pt",add_special_tokens=False);_,ns=full_next(model,rb);rag_ns.append(ns);rag_tokens.append(int(rb.input_ids.numel()))
 # Parametric bridge on city facts: exact Fabric city anchor lowered as tokenless slot, Qwen supplies city->country capability.
 bridge=[]
 for i in query_ids[:64]:
  e=entity(i);city=read(snap1,i,"city");c=compile_city_slot(model,tok,e,city)
  if c is None:continue
  bf,bs,emb=c;sl,sns=slot_next(model,bs,emb);fl,fns=full_next(model,bf);target=CITY_TO_COUNTRY[city];ids=token_set(tok,target)
  if not ids:continue
  pred=max(COUNTRIES,key=lambda x:semantic_match(sl,token_set(tok,x)));fp=max(COUNTRIES,key=lambda x:semantic_match(fl,token_set(tok,x)))
  bridge.append({"correct":pred==target,"parity":pred==fp,"maxerr":float((sl-fl).abs().max().item()),"slot_ns":sns,"full_ns":fns})
 # v2: 1000 edits + 500 masks; old snapshot retained.
 edit_ids=set(query_ids[:min(40,len(query_ids))]);extra=set(rng.sample([x for x in range(N_OBJECTS) if x not in edit_ids],960));edit_ids|=extra
 mask_ids=set(rng.sample([x for x in range(N_OBJECTS) if x not in edit_ids],500))
 code_ov={i:"R"+code(i)[1:] for i in edit_ids};city_ov={i:CITIES[(i+3)%len(CITIES)] for i in list(edit_ids)[:250]}
 snap2=Snapshot(2,code_ov,{}, {},city_ov,frozenset(mask_ids))
 lifecycle={"edits_checked":0,"masks_checked":0,"old_snapshot_mismatches":0,"new_snapshot_mismatches":0}
 for i in list(edit_ids)[:128]:
  old=read(snap1,i,"code");new=read(snap2,i,"code");lifecycle["edits_checked"]+=1;lifecycle["old_snapshot_mismatches"]+=int(old!=code(i));lifecycle["new_snapshot_mismatches"]+=int(new!=code_ov[i])
 for i in list(mask_ids)[:128]:lifecycle["masks_checked"]+=1;lifecycle["new_snapshot_mismatches"]+=int(read(snap2,i,"city") is not None);lifecycle["old_snapshot_mismatches"]+=int(read(snap1,i,"city") is None)
 # Storage is model-neutral source representation; no per-fact learned parameters.
 estimated_source_bytes=N_OBJECTS*(8+4+8+2) # compact ids/code seed/owner id/city id estimate
 report={
  "stage":"R264-REAL-QWEN-100K-KNOWLEDGE-VM-SCALE","model_id":MODEL_ID,"revision":REV,"parameter_count":sum(p.numel() for p in model.parameters()),"backbone_trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
  "knowledge_objects":N_OBJECTS,"per_fact_gradient_steps":0,"runtime_knowledge_text_tokens_vm":0,"planner_global_parameters":int(W.numel()),"planner_heldout_open_world_accuracy":planner_acc,"planner_queries":len(queries),"planner_ms_total":planner_ns/1e6,
  "mmu_direct_reads":len(vm_ns),"mmu_median_us":statistics.median(vm_ns)/1e3,"estimated_model_neutral_source_bytes":estimated_source_bytes,
  "oracle_rag":{"perfect_address":True,"vector_search_cost_included":False,"rerank_cost_included":False,"median_qwen_read_ms":statistics.median(rag_ns)/1e6,"median_prompt_tokens_including_fact":statistics.median(rag_tokens)},
  "parametric_bridge":{"cases":len(bridge),"accuracy":sum(int(x['correct']) for x in bridge)/len(bridge) if bridge else None,"decision_parity_vs_full_text":sum(int(x['parity']) for x in bridge)/len(bridge) if bridge else None,"max_logit_error_vs_full_text":max((x['maxerr'] for x in bridge),default=None),"median_slot_ms":statistics.median([x['slot_ns'] for x in bridge])/1e6 if bridge else None,"median_full_text_ms":statistics.median([x['full_ns'] for x in bridge])/1e6 if bridge else None},
  "lifecycle":lifecycle,"snapshots_coexist":True,
  "scientific_scope":"real frozen Qwen 0.5B with a 100k model-neutral typed knowledge store, tiny global language->KCIR planner, exact O(1) alias/MMU, typed/COPY lane, parametric city->country bridge, and privileged Oracle-RAG timing baseline. Direct literal truth is emitted by exact typed lane, not by asking Qwen to reconstruct arbitrary bytes.",
  "dod_status":"NOT_DOD; real-model 100k scale mechanism evidence. Need stronger multi-hop/NL end-to-end, second backbone and production-grade RAG benchmark."
 }
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
 if planner_acc<.95:return 2
 if bridge and (report['parametric_bridge']['accuracy']<.90 or report['parametric_bridge']['decision_parity_vs_full_text']<.99):return 3
 if lifecycle['old_snapshot_mismatches'] or lifecycle['new_snapshot_mismatches']:return 4
 return 0
if __name__=='__main__':raise SystemExit(main())
