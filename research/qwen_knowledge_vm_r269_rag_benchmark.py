from __future__ import annotations
import hashlib,json,os,random,statistics,time
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL_ID='Qwen/Qwen2.5-0.5B';REV='060db6499f32faf8b98477b0a26969ef7d8b9987';WEIGHT_SHA='88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342'
OUT=Path(os.environ.get('SO_R269_REPORT','ci-qwen-r269/report.json'));DEPTHS=(1,4,16,64);ANCHORS=(('Paris','France'),('Berlin','Germany'),('Rome','Italy'),('Madrid','Spain'),('Vienna','Austria'))
def sha(p):
 h=hashlib.sha256();f=open(p,'rb')
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 f.close();return h.hexdigest()
def one(tok,s):
 z=tok.encode(' '+s,add_special_tokens=False);return z[0] if len(z)==1 else None
def slot(model,tok,city):
 full=f'The city {city} is in the country of';base='The city X is in the country of';bf=tok(full,return_tensors='pt',add_special_tokens=False);bs=tok(base,return_tensors='pt',add_special_tokens=False)
 if bf.input_ids.shape!=bs.input_ids.shape:return None
 d=(bf.input_ids[0]!=bs.input_ids[0]).nonzero().flatten()
 if d.numel()!=1:return None
 p=int(d.item());emb=model.get_input_embeddings()(bs.input_ids).detach().clone();emb[0,p]=model.get_input_embeddings().weight[int(bf.input_ids[0,p])]
 t=time.perf_counter_ns();
 with torch.inference_mode():lg=model(inputs_embeds=emb,attention_mask=bs.attention_mask,return_dict=True).logits[:,-1,:]
 return lg,time.perf_counter_ns()-t,int(bs.input_ids.numel()),bf
def mk(rng,d,k):
 ns=[f'pod://r269/{k}/{j}/{rng.getrandbits(64):016x}' for j in range(d)];edges={ns[j]:(ns[j+1] if j+1<d else 'anchor') for j in range(d)};return ns[0],edges
def execp(s,e):
 t=time.perf_counter_ns();c=s
 while c in e:c=e[c]
 return time.perf_counter_ns()-t
def rag_context(s,e,city,compact=False):
 if compact:return f'The final linked city is {city}.\nThe city {city} is in the country of'
 lines=[];c=s
 while c in e:
  n=e[c];lines.append(f'{c} links to {n}.');c=n
 lines.append(f'The final linked city is {city}.');lines.append(f'The city {city} is in the country of');return '\n'.join(lines)
def fwd(model,b):
 t=time.perf_counter_ns();
 with torch.inference_mode():lg=model(**b,return_dict=True).logits[:,-1,:]
 return lg,time.perf_counter_ns()-t
def main():
 OUT.parent.mkdir(parents=True,exist_ok=True);md=Path(snapshot_download(repo_id=MODEL_ID,revision=REV,allow_patterns=['*.json','*.txt','*.model','*.tiktoken','*.safetensors']));assert sha(md/'model.safetensors')==WEIGHT_SHA
 tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False);model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32);model.eval();[p.requires_grad_(False) for p in model.parameters()]
 good=[]
 for c,n in ANCHORS:
  tid=one(tok,n);r=slot(model,tok,c)
  if tid is not None and r is not None and int(r[0].argmax(-1))==tid:good.append((c,n,tid))
 if not good:raise RuntimeError('no eligible anchors')
 rng=random.Random(269);rows=[]
 for d in DEPTHS:
  reps=4 if d<=16 else 2
  for k in range(reps):
   city,country,tid=good[(d+k)%len(good)];s,e=mk(rng,d,d*100+k);kns=execp(s,e);sl,sns,stok,_=slot(model,tok,city)
   for mode in ('text_rag','graph_rag'):
    txt=rag_context(s,e,city,compact=(mode=='graph_rag'));b=tok(txt,return_tensors='pt',add_special_tokens=False,truncation=True,max_length=16384);lg,rns=fwd(model,b)
    rows.append({'depth':d,'mode':mode,'country_correct':int(lg.argmax(-1))==tid,'vm_correct':int(sl.argmax(-1))==tid,'rag_ns':rns,'vm_ns':sns+kns,'rag_tokens':int(b.input_ids.numel()),'vm_knowledge_tokens':0})
 by={}
 for d in DEPTHS:
  by[str(d)]={}
  for mode in ('text_rag','graph_rag'):
   x=[r for r in rows if r['depth']==d and r['mode']==mode];v=statistics.median(r['vm_ns'] for r in x);q=statistics.median(r['rag_ns'] for r in x)
   by[str(d)][mode]={'n':len(x),'vm_accuracy':sum(r['vm_correct'] for r in x)/len(x),'rag_accuracy':sum(r['country_correct'] for r in x)/len(x),'vm_median_ms':v/1e6,'rag_median_ms':q/1e6,'speedup_rag_over_vm':q/v,'rag_median_tokens':statistics.median(r['rag_tokens'] for r in x),'vm_knowledge_text_tokens':0}
 report={'stage':'R269-REAL-QWEN-ORACLE-RAG-BENCHMARK','model_id':MODEL_ID,'revision':REV,'parameter_count':sum(p.numel() for p in model.parameters()),'trainable_parameter_count':0,'by_depth':by,'oracle_text_rag_has_perfect_relevant_context':True,'oracle_graph_rag_has_perfect_external_path_execution':True,'retrieval_search_cost_included':False,'rerank_cost_included':False,'scientific_scope':'same frozen Qwen. VM executes exact KCIR path then one tokenless typed city view. Text-RAG gets every relevant edge as text. Graph-RAG is stronger: it may externally execute the path and only verbalize the final city as text.','dod_status':'NOT_DOD; fair real-model latency boundary audit'}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
 if any(by[str(d)]['text_rag']['vm_accuracy']<.95 for d in DEPTHS):return 2
 return 0
if __name__=='__main__':raise SystemExit(main())