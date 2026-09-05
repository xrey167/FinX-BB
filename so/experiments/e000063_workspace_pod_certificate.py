"""E-000063 -- compose one canonical symlink deletion with a causal J-space certificate.

E-000020 already tests output/probe attacks on a symlink GPT-2 adapter. E-000042 already tests J-lens
causal supports on ordinary parametric facts. Neither is the WNVI composition. Here the SAME explicit
canonical pod is read through multiple LINK aliases, then one SHRED is applied to the canonical kid.
We ask whether the target object disappears not only from answers but from the frozen core's causal
J-space at the adapter's first broadcast site, while unrelated pods remain stable.

Critical control: query text itself identifies the synthetic alias. A probe can therefore learn
"alias name -> old object" even when memory never existed. For every deleted pod we construct a
NEVER bank with the canonical row and its aliases physically absent, using the identical prompt.
Every post-SHRED workspace number is read against this never-memory control. A probe is trained only
on ACTIVE states and its exact same weights are evaluated on ACTIVE, SHRED and NEVER states.

Screening criteria (single seed; positives must later repeat >=3 seeds):
  active alias answer >= .80                         attack validity
  SHRED alias UNKNOWN >= .90; true object <= .05    output deletion
  bystander top-1 agreement >= .98; KL <= .05       locality
  active J-space probe - NEVER >= .30               J-space instrument sees memory, not query identity
  SHRED J-space probe - NEVER <= .05                no extra decodable workspace trace from the pod
  mean |SHRED target J-score - NEVER| <= .05         target causal coordinate returns to never baseline
  active final-state probe - NEVER >= .30            conventional hidden probe validity
  SHRED final-state probe - NEVER <= .05              no extra decodable adapter trace

A pass is a composition result, not a novelty claim. It still needs the J-space address ABI, multiple
backbones, key/reconstruction/stale-generation attacks and strong prior-art comparisons.

Run: python -m so.experiments.e000063_workspace_pod_certificate --seed 0 --steps 2000
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.jlens import jlens_vectors
from so.llm_adapter import AdapterConfig, transformer_blocks

TEMPLATES = [1, 3, 4, 5, 7, 9, 10]  # subject-medial forms; independent of this experiment's outcomes
CAPTURE_BLOCK = 8                    # first adapter read site
JL_SOURCE = 9                        # hidden_states[9] = output of GPT-2 block 8
CORPUS = [
    "The capital of France is Paris and the capital of Japan is Tokyo.",
    "A researcher compared several methods before writing the report.",
    "The city centre was busy during the afternoon and quiet at night.",
    "Machine learning models transform sequences through many intermediate representations.",
    "Water freezes at zero degrees Celsius under ordinary conditions.",
    "A train crossed the bridge and continued toward the next station.",
    "The committee published a revised version of the document.",
    "Several countries maintain diplomatic relations with their neighbours.",
]


class Capture:
    def __init__(self, gk: E8.GPT2Knowledge):
        self.h: torch.Tensor | None = None
        # Adapter hooks are registered during model construction; this hook is registered afterwards,
        # so it observes block-8 output AFTER the first symlink resolve/dereference injection.
        self.handle = transformer_blocks(gk.model.lm)[CAPTURE_BLOCK].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        self.h = h.detach()
        return None

    def close(self):
        self.handle.remove()


def texts_for(keys: Sequence[Tuple[int,int]], names: Sequence[str], templates: Sequence[int]) -> tuple[list[str], list[int]]:
    texts, groups = [], []
    for t in templates:
        for s, r in keys:
            texts.append(E17.TEMPLATES12[r][t].format(s=names[s]))
            groups.append(t)
    return texts, groups


def eval_texts(gk: E8.GPT2Knowledge, cap: Capture, store, texts: Sequence[str], batch: int = 64):
    bank = bank_from_store(store)
    states, finals, logits, answers = [], [], [], []
    tensors = bank.tensors()
    for i in range(0, len(texts), batch):
        ids, am, last = E8.encode_texts(gk.tok, texts[i:i+batch])
        with torch.no_grad():
            cand, _, _, final = gk.model(tensors, ids, am, last)
        assert cap.h is not None
        ar = torch.arange(ids.shape[0])
        states.append(cap.h[ar, last].cpu())
        finals.append(final.detach().cpu())
        logits.append(cand.detach().cpu())
        answers.append(cand.argmax(-1).detach().cpu())
    return torch.cat(states), torch.cat(finals), torch.cat(logits), torch.cat(answers)


def transfer_probe(active: torch.Tensor, shred: torch.Tensor, never: torch.Tensor,
                   y: torch.Tensor, groups: torch.Tensor, n_class: int,
                   steps: int = 300) -> Dict[str,float]:
    """Train on ACTIVE only; score the identical fold on ACTIVE/SHRED/NEVER with the same weights."""
    out = {"active":[], "shred":[], "never":[]}
    for g in groups.unique():
        tr, te = groups != g, groups == g
        mu = active[tr].mean(0); sd = active[tr].std(0).clamp(min=1e-5)
        xa = (active - mu) / sd; xs = (shred - mu) / sd; xn = (never - mu) / sd
        w = torch.zeros(active.shape[1], n_class, requires_grad=True)
        b = torch.zeros(n_class, requires_grad=True)
        opt = torch.optim.Adam([w,b], lr=.08)
        for _ in range(steps):
            opt.zero_grad()
            loss = F.cross_entropy(xa[tr] @ w + b, y[tr]) + 1e-3 * w.pow(2).mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            for name, x in (("active",xa),("shred",xs),("never",xn)):
                out[name].append(float(((x[te] @ w + b).argmax(-1) == y[te]).float().mean()))
    return {k:float(np.mean(v)) for k,v in out.items()}


def kl_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    la = F.log_softmax(a, dim=-1); lb = F.log_softmax(b, dim=-1)
    return (la.exp() * (la-lb)).sum(-1)


def run(seed: int, steps: int, threads: int, n_groups: int, results_dir: str) -> Dict[str,object]:
    if threads:
        torch.set_num_threads(threads)
    t0=time.time()
    cfg=AdapterConfig(status_gated=True, use_links=True, n_deref=1)
    gk=E8.GPT2Knowledge(cfg)
    trained=E20.train_adapter_links(gk, seed, steps, n_groups=max(80,n_groups), verbose=True)
    centre=np.asarray(trained["centre"])

    rng=np.random.default_rng(63000+seed)
    world,spec=E15.sample_alias_world(rng, 420, max(n_groups*2,48), 2,
                                      gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
    store,kids=E15.load_arm(world,spec,centre,64000+seed,symlink=True)

    # Unique objects only: J-space classification labels are object identities, not arbitrary group IDs.
    selected=[]; seen=set()
    for target, aliases in spec.groups:
        obj=int(world.index[target])
        if obj not in seen:
            selected.append((target,aliases,obj)); seen.add(obj)
        if len(selected) >= n_groups:
            break
    if len(selected) < 10:
        raise RuntimeError(f"not enough unique-object pods: {len(selected)}")
    n=len(selected)
    obj_token_ids=[int(gk.model.entity_token_ids[obj]) for _,_,obj in selected]

    # Estimate the base model's causal workspace directions independent of the evaluation world.
    corpus=gk.tok(CORPUS,return_tensors="pt",padding=True)
    jl=jlens_vectors(gk.model.lm, JL_SOURCE, obj_token_ids, corpus["input_ids"], corpus["attention_mask"],
                     gk.model.lm.get_output_embeddings().weight, batch=4)
    cap=Capture(gk)

    active_s=[]; active_f=[]; active_l=[]
    post_s=[]; post_f=[]; post_l=[]; post_a=[]
    never_s=[]; never_f=[]; never_l=[]
    ys=[]; template_groups=[]
    by_kl=[]; by_agree=[]

    for label,(target,aliases,obj) in enumerate(selected):
        texts, tgroups=texts_for(aliases,gk.names,TEMPLATES)
        sa,fa,la,aa=eval_texts(gk,cap,store,texts)

        # Bystander controls measured immediately before and after this ONE canonical shred.
        by_keys=[a[1][0] for j,a in enumerate(selected) if j != label][:8]
        by_text,_=texts_for(by_keys,gk.names,[9])
        _,_,by0,ba0=eval_texts(gk,cap,store,by_text)

        # NEVER control: identical prompt and model, but neither canonical pod nor its aliases exist.
        never_store=copy.deepcopy(store)
        for ak in aliases:
            if kids[ak] in never_store.cells:
                never_store.delete(kids[ak])
        if kids[target] in never_store.cells:
            never_store.delete(kids[target])
        sn,fn,ln,_=eval_texts(gk,cap,never_store,texts)

        store.shred(kids[target])
        ss,fs,ls,aas=eval_texts(gk,cap,store,texts)
        _,_,by1,ba1=eval_texts(gk,cap,store,by_text)
        store.resign(kids[target])

        active_s.append(sa); active_f.append(fa); active_l.append(la)
        post_s.append(ss); post_f.append(fs); post_l.append(ls); post_a.append(aas)
        never_s.append(sn); never_f.append(fn); never_l.append(ln)
        ys.extend([label]*len(texts)); template_groups.extend(tgroups)
        if len(by_text):
            by_kl.extend(kl_rows(by0,by1).tolist())
            by_agree.extend((ba0==ba1).float().tolist())

    cap.close()
    active_s=torch.cat(active_s); post_s=torch.cat(post_s); never_s=torch.cat(never_s)
    active_f=torch.cat(active_f); post_f=torch.cat(post_f); never_f=torch.cat(never_f)
    active_l=torch.cat(active_l); post_l=torch.cat(post_l); never_l=torch.cat(never_l)
    post_a=torch.cat(post_a)
    y=torch.tensor(ys,dtype=torch.long); groups=torch.tensor(template_groups,dtype=torch.long)

    # Every row's correct J atom is its object's direction. Cosine removes residual-norm confounds.
    jvec=jl.vectors.cpu()
    def jsig(x): return F.normalize(x.float(),dim=-1) @ jvec.t()
    ja,js,jn=jsig(active_s),jsig(post_s),jsig(never_s)
    idx=torch.arange(len(y))
    score_a=ja[idx,y]; score_s=js[idx,y]; score_n=jn[idx,y]
    jprobe=transfer_probe(ja,js,jn,y,groups,n)
    fprobe=transfer_probe(active_f,post_f,never_f,y,groups,n)

    # Truth answer for each row is its pod object's entity index. Ordering mirrors selected->templates->aliases.
    truth=[]
    for _,aliases,obj in selected:
        truth.extend([obj]*(len(aliases)*len(TEMPLATES)))
    truth=torch.tensor(truth,dtype=torch.long)
    # But rows were appended group-by-group with template-major aliases: same count and same group object.
    active_ans=active_l.argmax(-1); post_ans=post_l.argmax(-1)
    unknown=gk.n_entities

    metrics={
        "n_pods":n,
        "chance":1.0/n,
        "active_alias_correct":float((active_ans==truth).float().mean()),
        "shred_alias_unknown":float((post_ans==unknown).float().mean()),
        "shred_alias_true_object":float((post_ans==truth).float().mean()),
        "bystander_top1_agree":float(np.mean(by_agree)) if by_agree else float("nan"),
        "bystander_kl_max":float(max(by_kl)) if by_kl else float("nan"),
        "jprobe_active":jprobe["active"], "jprobe_shred":jprobe["shred"], "jprobe_never":jprobe["never"],
        "jprobe_active_minus_never":jprobe["active"]-jprobe["never"],
        "jprobe_shred_minus_never":jprobe["shred"]-jprobe["never"],
        "finalprobe_active":fprobe["active"], "finalprobe_shred":fprobe["shred"], "finalprobe_never":fprobe["never"],
        "finalprobe_active_minus_never":fprobe["active"]-fprobe["never"],
        "finalprobe_shred_minus_never":fprobe["shred"]-fprobe["never"],
        "j_target_score_active":float(score_a.mean()),
        "j_target_score_shred":float(score_s.mean()),
        "j_target_score_never":float(score_n.mean()),
        "j_target_shred_never_abs":float((score_s-score_n).abs().mean()),
        "j_top1_active":float((ja.argmax(-1)==y).float().mean()),
        "j_top1_shred":float((js.argmax(-1)==y).float().mean()),
        "j_top1_never":float((jn.argmax(-1)==y).float().mean()),
        "seconds":time.time()-t0,
    }
    bars={
        "active_alias_correct":(">=",.80),
        "shred_alias_unknown":(">=",.90),
        "shred_alias_true_object":("<=",.05),
        "bystander_top1_agree":(">=",.98),
        "bystander_kl_max":("<=",.05),
        "jprobe_active_minus_never":(">=",.30),
        "jprobe_shred_minus_never":("<=",.05),
        "j_target_shred_never_abs":("<=",.05),
        "finalprobe_active_minus_never":(">=",.30),
        "finalprobe_shred_minus_never":("<=",.05),
    }
    checks={}
    for k,(op,b) in bars.items():
        v=float(metrics[k]); ok=v>=b if op==">=" else v<=b
        checks[k]={"observed":v,"op":op,"bar":b,"pass":bool(ok)}
    screen=all(c["pass"] for c in checks.values())
    rec={"experiment":"E-000063","candidate_only":True,"seed":seed,"steps":steps,
         "adapter":cfg.to_dict(),"templates":TEMPLATES,"capture_block":CAPTURE_BLOCK,
         "jl_source_hidden_state":JL_SOURCE,"jlens_prompts":jl.n_prompts,"metrics":metrics,
         "criteria":checks,"screening_pass":screen}
    p=Path(results_dir);p.mkdir(parents=True,exist_ok=True)
    (p/f"e000063_workspace_pod_certificate-seed{seed}.json").write_text(json.dumps(rec,indent=2),encoding="utf-8")
    print(json.dumps({"screening_pass":screen,"metrics":metrics,"criteria":checks},indent=2))
    return rec


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=0);ap.add_argument("--steps",type=int,default=2000)
    ap.add_argument("--threads",type=int,default=2);ap.add_argument("--n-groups",type=int,default=16)
    ap.add_argument("--results-dir",default="so/results")
    a=ap.parse_args();run(a.seed,a.steps,a.threads,a.n_groups,a.results_dir)

if __name__=="__main__": main()
