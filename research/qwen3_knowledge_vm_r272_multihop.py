from __future__ import annotations
import hashlib,json,os,random,re,statistics,time
from pathlib import Path
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL_ID='Qwen/Qwen3-0.6B';REV='a08cec3036ee1085a4863a0b730e5d3f1c2f8d04';WEIGHT_SHA='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b';OUT=Path(os.environ.get('SO_R272_REPORT','ci-qwen-r272/report.json'));N=100000;DEPTHS=(1,4,16,64);RELS=('next','parent','ownerlink','supply')
TRAIN={r:[f'From {{e}}, follow the {r} relation {{n}} times and tell me the country of the final node.',f'Start at {{e}}; traverse {r} exactly {{n}} hops. What country is the destination in?'] for r in RELS}
TEST={r:[f'Beginning with {{e}}, take {{n}} successive {r} links. Name the country containing the endpoint.',f'After {{n}} {r} steps from {{e}}, which country contains the resulting node?'] for r in RELS}
TRAIN_E=['Arven Delta','Belora Stack','Cinder Quay','Dovren Lab','Ester Node','Falrin Gate','Gavora Hub','Hesper Dock'];CITIES=('Paris','Berlin','Rome','Madrid','Vienna','Prague','Warsaw','Lisbon');COUNTRIES=('France','Germany','Italy','Spain','Austria','Czechia','Poland','Portugal');C2C=dict(zip(CITIES,COUNTRIES))
def sha(p):
 h=hashlib.sha256();f=open(p,'rb')
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 f.close();return h.hexdigest()
def ent(i):return f'ZXQ-{i:06d}'
def norm_query(s,e,n):return re.sub(r'\b\d+\b','COUNT',s.replace(e,'ENTITY'))
def make_data(T,entities,depths):
 xs=[];ys=[]
 for y,r in enumerate(RELS):
  for t in T[r]:
   for e in entities:
    for n in depths:xs.append(norm_query(t.format(e=e,n=n),e,n));ys.append(y)
 return xs,torch.tensor(ys)
def emb_feat(model,tok,texts):
 out=[];t=time.perf_counter_ns()
 with torch.inference_mode():
  for s in texts:
   ids=tok(s,return_tensors='pt',add_special_tokens=False).input_ids;v=model.get_input_embeddings()(ids)[0].float();out.append(F.normalize(v.mean(0),dim=0).cpu())
 return torch.stack(out),time.perf_counter_ns()-t
def ridge(X,y,c,l=.02):
 A=torch.cat([X,torch.ones(X.shape[0],1)],1).double();Y=F.one_hot(y,c).double();return A.T@torch.linalg.solve(A@A.T+l*torch.eye(A.shape[0],dtype=torch.double),Y)
def pred(X,W):return (torch.cat([X,torch.ones(X.shape[0],1)],1).double()@W).argmax(1)
def edge(i,r):
 a=(1664525,1103515245,22695477,214013)[RELS.index(r)];b=(1013904223,12345,1,2531011)[RELS.index(r)];return (a*i+b)%N
def exec_chain(i,r,n):
 t=time.perf_counter_ns();x=i
 for _ in range(n):x=edge(x,r)
 return x,time.perf_counter_ns()-t
def token_ids(tok,x):
 out=[]
 for s in (x,' '+x):
  z=tok.encode(s,add_special_tokens=False)
  if len(z)==1 and z[0] not in out:out.append(z[0])
 return out
def choose(tok,log):
 scores={c:max([float(log[0,i]) for i in token_ids(tok,c)] or [-1e30]) for c in COUNTRIES};return max(scores,key=scores.get)
def country_slot(model,tok,city):
 full=f'The city {city} is in the country of';slot='The city X is in the country of';bf=tok(full,return_tensors='pt',add_special_tokens=False);bs=tok(slot,return_tensors='pt',add_special_tokens=False)
 if bf.input_ids.shape!=bs.input_ids.shape:return None
 d=(bf.input_ids[0]!=bs.input_ids[0]).nonzero().flatten()
 if d.numel()!=1:return None
 p=int(d.item());emb=model.get_input_embeddings()(bs.input_ids).detach().clone();emb[0,p]=model.get_input_embeddings().weight[int(bf.input_ids[0,p])];t=time.perf_counter_ns()
 with torch.inference_mode():lg=model(inputs_embeds=emb,attention_mask=bs.attention_mask,return_dict=True).logits[:,-1,:]
 return lg,time.perf_counter_ns()-t
def main():
 OUT.parent.mkdir(parents=True,exist_ok=True);md=Path(snapshot_download(repo_id=MODEL_ID,revision=REV,allow_patterns=['*.json','*.txt','*.model','*.tiktoken','*.safetensors']));assert sha(md/'model.safetensors')==WEIGHT_SHA
 tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False);model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32);model.eval();[p.requires_grad_(False) for p in model.parameters()]
 tx,ty=make_data(TRAIN,TRAIN_E,(1,4,16));X,_=emb_feat(model,tok,tx);W=ridge(X,ty,len(RELS))
 rng=random.Random(272);test_ids=rng.sample(range(N),128);qx=[];qy=[];meta=[]
 for j,i in enumerate(test_ids):
  r=RELS[j%len(RELS)];n=DEPTHS[j%len(DEPTHS)];t=TEST[r][j%len(TEST[r])];q=t.format(e=ent(i),n=n);qx.append(norm_query(q,ent(i),n));qy.append(RELS.index(r));meta.append((i,r,n,q))
 Q,pns=emb_feat(model,tok,qx);pp=pred(Q,W);pacc=float((pp==torch.tensor(qy)).float().mean())
 rows=[]
 for j,(i,r,n,q) in enumerate(meta):
  if int(pp[j])!=qy[j]:continue
  dst,kns=exec_chain(i,r,n);city=CITIES[dst%len(CITIES)];z=country_slot(model,tok,city)
  if z is None:continue
  lg,qns=z;target=C2C[city];rows.append({'depth':n,'correct':choose(tok,lg)==target,'kcir_ns':kns,'qwen_ns':qns})
 by={}
 for d in DEPTHS:
  xs=[x for x in rows if x['depth']==d];by[str(d)]={'n':len(xs),'accuracy':sum(int(x['correct']) for x in xs)/len(xs) if xs else None,'median_kcir_us':statistics.median([x['kcir_ns'] for x in xs])/1e3 if xs else None,'median_qwen_ms':statistics.median([x['qwen_ns'] for x in xs])/1e6 if xs else None}
 report={'stage':'R272-QWEN3-0.6B-NL-FUSED-MULTIHOP','model_id':MODEL_ID,'revision':REV,'parameter_count':sum(p.numel() for p in model.parameters()),'trainable_parameter_count':0,'knowledge_nodes':N,'knowledge_relation_pages':N*len(RELS),'planner_global_parameters':int(W.numel()),'heldout_nl_program_accuracy':pacc,'planner_transformer_layers_used':0,'planner_feature_ms_total':pns/1e6,'by_depth':by,'runtime_knowledge_text_tokens':0,'per_fact_gradient_steps':0,'scientific_scope':'real Qwen3 0.6B: held-out natural language -> cheap embedding-only KCIR relation plan + exact depth -> root-fused post-training graph execution -> typed city -> Qwen parametric country. Raw hops never enter the prompt.','dod_status':'NOT_DOD; real >=0.6B natural-language multi-hop system gate'}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
 if pacc<.95:return 2
 if any(by[str(d)]['accuracy'] is None or by[str(d)]['accuracy']<.90 for d in DEPTHS):return 3
 return 0
if __name__=='__main__':raise SystemExit(main())