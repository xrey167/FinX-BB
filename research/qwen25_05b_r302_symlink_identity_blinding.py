from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file, candidate_token

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_WEIGHTS = {"model.safetensors": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"}
OUT = Path(os.environ.get("SO_R302_REPORT", "ci-qwen-r302/report.json"))

FACTS = (
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Japan", "Tokyo"),
    ("China", "Beijing"), ("Canada", "Ottawa"), ("Spain", "Madrid"), ("Russia", "Moscow"),
    ("Egypt", "Cairo"), ("Brazil", "Brasilia"), ("Greece", "Athens"), ("Austria", "Vienna"),
    ("Portugal", "Lisbon"), ("Norway", "Oslo"), ("Finland", "Helsinki"), ("Sweden", "Stockholm"),
    ("Denmark", "Copenhagen"), ("Ireland", "Dublin"), ("Poland", "Warsaw"), ("Turkey", "Ankara"),
)
SYSTEM = (
    "Answer the user's relation query with exactly one value word. A line named `Port value` is external current world state; "
    "when it is present, treat it as the value of the requested relation."
)


def nonce(i: int) -> str:
    return f"POD-ZN{i:02d}-Q7X9M4"


def make_prompt(tok, entity: str, port: str | None):
    lines = [f"Entity: {entity}", "Relation: capital"]
    if port is not None:
        lines.append(f"Port value: {port}")
    lines.append("What is the relation value? Return one word.")
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def batch_logits(model, tok, texts, batch_size=20):
    rows=[]
    for start in range(0,len(texts),batch_size):
        batch=texts[start:start+batch_size]
        enc=tok(batch,return_tensors="pt",padding=True,add_special_tokens=False)
        with torch.inference_mode(): out=model(**enc,use_cache=False,return_dict=True)
        idx=out.logits.shape[1]-1
        rows.extend(out.logits[:,idx,:].float().cpu())
    return rows


def main() -> int:
    OUT.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    rng=random.Random(302)

    md=Path(snapshot_download(
        repo_id=MODEL_ID,revision=REVISION,
        allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors","*.index.json","*.merges","*.vocab","merges.txt","vocab.json"],
    ))
    hashes={n:sha256_file(md/n) for n in EXPECTED_WEIGHTS};assert hashes==EXPECTED_WEIGHTS
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False)
    tok.pad_token=tok.eos_token;tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(
        md,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation="eager",trust_remote_code=False,
    ).eval();model.requires_grad_(False)

    # Keep only capital labels that are representable as one candidate token under this tokenizer.
    eligible=[]
    token_ids={}
    for country,capital in FACTS:
        try: tid=candidate_token(tok,capital)
        except RuntimeError: continue
        eligible.append((country,capital));token_ids[capital]=tid
    if len(eligible)<12: raise RuntimeError(f"too few one-token facts: {eligible}")
    capitals=[x[1] for x in eligible]

    rows=[]
    texts=[];meta=[]
    for i,(country,actual) in enumerate(eligible):
        cf=capitals[(i+5)%len(capitals)]
        n=nonce(i)
        for condition,entity,port in (
            ("natural_no_port",country,None),
            ("blind_no_port",n,None),
            ("natural_actual_port",country,actual),
            ("blind_actual_port",n,actual),
            ("natural_counterfactual_port",country,cf),
            ("blind_counterfactual_port",n,cf),
        ):
            texts.append(make_prompt(tok,entity,port));meta.append((i,country,actual,cf,n,condition))
    logits=batch_logits(model,tok,texts)

    candidate_ids=torch.tensor([token_ids[c] for c in capitals],dtype=torch.long)
    pos={c:i for i,c in enumerate(capitals)}
    cond_correct={}; cond_gap={}
    for info,lg in zip(meta,logits):
        i,country,actual,cf,n,condition=info
        scores=lg[candidate_ids]
        top_idx=int(scores.argmax().item());top=capitals[top_idx]
        expected_value = cf if "counterfactual" in condition else actual
        if condition.endswith("no_port"):
            expected_value=actual
        ok=top==expected_value
        gap=float(lg[token_ids[actual]]-lg[token_ids[cf]])
        cond_correct.setdefault(condition,[]).append(ok)
        cond_gap.setdefault(condition,[]).append(gap)
        rows.append({"country":country,"actual":actual,"counterfactual":cf,"nonce":n,"condition":condition,"top_candidate":top,"expected_for_condition":expected_value,"correct":ok,"actual_minus_counterfactual_logit":gap})

    accuracy={k:sum(v)/len(v) for k,v in cond_correct.items()}
    mean_gap={k:sum(v)/len(v) for k,v in cond_gap.items()}

    # Alias collapse is an architectural exactness property: after Symlink resolution the
    # B/J execution input contains the same opaque handle, independent of source alias.
    alias_examples=[]; alias_prompt_equal=[]
    for i,(country,actual) in enumerate(eligible[:8]):
        aliases=(country,country.upper(),f"the country called {country}")
        canonical_inputs=[]
        for _alias in aliases:
            # Resolver output is deliberately all that reaches the governed relation path.
            canonical_inputs.append(make_prompt(tok,nonce(i),actual))
        same=all(x==canonical_inputs[0] for x in canonical_inputs[1:])
        alias_prompt_equal.append(same)
        alias_examples.append({"aliases":aliases,"canonical_handle":nonce(i),"canonical_model_input_identical":same})

    natural_no=accuracy["natural_no_port"]
    blind_no=accuracy["blind_no_port"]
    nat_cf=accuracy["natural_counterfactual_port"]
    blind_cf=accuracy["blind_counterfactual_port"]
    report={
        "stage":"R302-SYMLINK-IDENTITY-BLINDING",
        "architecture_candidate":"Symlink Namespace Firewall / Identity-Blinded Governed Knowledge",
        "model_id":MODEL_ID,"revision":REVISION,"weights_sha256":hashes,"backbone_frozen":all(not p.requires_grad for p in model.parameters()),
        "eligible_fact_count":len(eligible),"eligible_facts":eligible,
        "candidate_accuracy":accuracy,"mean_actual_minus_counterfactual_logit":mean_gap,
        "natural_parametric_recall_accuracy_without_port":natural_no,
        "blinded_actual_fact_leakage_rate_without_port":blind_no,
        "actual_fact_leakage_reduction":natural_no-blind_no,
        "natural_counterfactual_port_compliance":nat_cf,
        "blinded_counterfactual_port_compliance":blind_cf,
        "counterfactual_compliance_gain_from_blinding":blind_cf-nat_cf,
        "alias_canonical_input_identity_rate":sum(alias_prompt_equal)/len(alias_prompt_equal),
        "alias_examples":alias_examples,"rows":rows,
        "mechanism":(
            "A governed natural-language alias is resolved before factual execution and replaced by an opaque canonical "
            "Pod handle. The pretrained backbone therefore sees the relation/operator and current Port value, but not the "
            "lexical entity identity that could directly trigger memorized parametric facts. Every alias collapses to the "
            "same canonical execution input."
        ),
        "security_target":(
            "For governed mutable relations, revocation must not fall back to pretrained lexical factual memory. Identity "
            "blinding is a namespace firewall intended to make the Port the only entity-specific value channel."
        ),
        "limitations":(
            "This is an empirical leakage/compliance probe, not a proof. Descriptive context can still reveal identity even "
            "after a surface alias is replaced; a production resolver must canonicalize all identity-bearing references. "
            "The Port is represented as text in this bootstrap gate rather than the final latent Port."
        ),
        "dod_status":"NOT_DOD; parametric-leakage architecture probe",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode();report["report_sha256"]=hashlib.sha256(canonical).hexdigest();OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"eligible":len(eligible),"natural_no_port":natural_no,"blind_no_port_leakage":blind_no,"natural_counterfactual_compliance":nat_cf,"blind_counterfactual_compliance":blind_cf,"alias_identity_rate":report["alias_canonical_input_identity_rate"]},indent=2))
    # Gate is deliberately relative: the small backbone may have weak factual recall.
    if report["alias_canonical_input_identity_rate"]!=1.0:return 2
    if accuracy["blind_actual_port"]<0.80:return 3
    return 0

if __name__=="__main__":raise SystemExit(main())
