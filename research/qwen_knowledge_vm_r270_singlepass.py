from __future__ import annotations
import hashlib,json,os,statistics,time
from pathlib import Path
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL_ID='Qwen/Qwen2.5-0.5B';REV='060db6499f32faf8b98477b0a26969ef7d8b9987';WEIGHT_SHA='88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342'
OUT=Path(os.environ.get('SO_R270_REPORT','ci-qwen-r270/report.json'));REL=('code','score','owner','city')
TR={'code':['What registry code belongs to {e}?','Give the registry identifier for {e}.','Which code is registered for {e}?'],'score':['What priority score belongs to {e}?','Give the priority rating for {e}.','Which score is assigned to {e}?'],'owner':['Who owns {e}?','Give the owner of {e}.','Which owner controls {e}?'],'city':['Which city is {e} located in?','Give the city location of {e}.','Where is {e} located, by city?']}
TE={'code':["Tell me {e}'s registry code.",'I need the registered code of {e}.','For {e}, what is the registry identifier?'],'score':["Tell me {e}'s priority score.",'How is {e} rated for priority?','For {e}, what score is on record?'],'owner':['Who is the owner behind {e}?','Which party owns {e}?','For {e}, identify its owner.'],'city':['In what city can {e} be found?','Name the city containing {e}.','What city is {e} based in?']}
TRAIN=['Arven Delta','Belora Stack','Cinder Quay','Dovren Lab','Ester Node','Falrin Gate','Gavora Hub','Hesper Dock','Ivera Point','Jalen Yard','Korvin Hall','Lester Forge'];TEST=['Mavora Relay','Neris Vault','Orren Spire','Pavela Works','Quorin Field','Rester Archive','Solen Campus','Tivera Port','Ulden Keep','Varex Court','Wylin Annex','Zorin Station']
def sha(p):
 h=hashlib.sha256();f=open(p,'rb')
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 f.close();return h.hexdigest()
def data(T,E):
 xs=[];ys=[]
 for y,r in enumerate(REL):
  for t in T[r]:
   for e in E:xs.append(t.format(e=e).replace(e,'ENTITY'));ys.append(y)
 return xs,torch.tensor(ys)
def emb_features(model,tok,texts):
 rows=[];t0=time.perf_counter_ns()
 with torch.inference_mode():
  for s in texts:
   ids=tok(s,return_tensors='pt',add_special_tokens=False).input_ids;v=model.get_input_embeddings()(ids)[0].float();rows.append(F.normalize(v.mean(0),dim=0).cpu())
 return torch.stack(rows),time.perf_counter_ns()-t0
def ridge(X,y,c,l=.02):
 Xa=torch.cat([X,torch.ones(X.shape[0],1)],1).double();Y=F.one_hot(y,c).double();K=Xa@Xa.T+l*torch.eye(Xa.shape[0],dtype=torch.double);return Xa.T@torch.linalg.solve(K,Y)
def pred(X,W):return (torch.cat([X,torch.ones(X.shape[0],1)],1).double()@W).argmax(1)
def main():
 OUT.parent.mkdir(parents=True,exist_ok=True);md=Path(snapshot_download(repo_id=MODEL_ID,revision=REV,allow_patterns=['*.json','*.txt','*.model','*.tiktoken','*.safetensors']));assert sha(md/'model.safetensors')==WEIGHT_SHA
 tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False);model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32);model.eval();[p.requires_grad_(False) for p in model.parameters()]
 tx,ty=data(TR,TRAIN);qx,qy=data(TE,TEST);X,tr_ns=emb_features(model,tok,tx);W=ridge(X,ty,len(REL));Q,te_ns=emb_features(model,tok,qx);p=pred(Q,W);acc=float((p==qy).float().mean())
 per={r:sum(int(p[i]==qy[i]) for i in range(len(qy)) if qy[i]==j)/sum(int(qy[i]==j) for i in range(len(qy))) for j,r in enumerate(REL)}
 report={'stage':'R270-REAL-QWEN-EMBEDDING-SCHEMA-MICROKERNEL','model_id':MODEL_ID,'revision':REV,'parameter_count':sum(x.numel() for x in model.parameters()),'backbone_trainable_parameters':0,'global_router_parameters':int(W.numel()),'train_entity_names_disjoint':set(TRAIN).isdisjoint(TEST),'heldout_templates_disjoint':all(set(TR[r]).isdisjoint(TE[r]) for r in REL),'heldout_accuracy':acc,'per_relation_accuracy':per,'planner_feature_extractor':'frozen input embeddings mean after exact entity span replacement','transformer_layers_used_for_planner':0,'train_feature_ms_total':tr_ns/1e6,'test_feature_ms_total':te_ns/1e6,'test_queries':len(qx),'median_equivalent_test_feature_us':te_ns/len(qx)/1e3,'per_fact_gradient_steps':0,'knowledge_id_head_size':0,'scientific_scope':'real Qwen embedding table used only as cheap global schema microkernel; no transformer forward is required to select KCIR relation schema. Entity identity remains an exact Symlink pointer.','dod_status':'NOT_DOD; removes the second-LLM-forward planner bottleneck if OOD accuracy holds.'}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
 if acc<.95:return 2
 if min(per.values())<.90:return 3
 return 0
if __name__=='__main__':raise SystemExit(main())