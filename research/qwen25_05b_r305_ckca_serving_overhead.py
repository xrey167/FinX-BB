from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file

MODEL_ID="Qwen/Qwen2.5-0.5B"
REVISION="060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_WEIGHTS={"model.safetensors":"88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"}
OUT=Path(os.environ.get("SO_R305_REPORT","ci-qwen-r305/report.json"))
KEY=b"r305-benchmark-key"


def median_ns(fn,repeats:int):
    xs=[]
    for _ in range(repeats):
        t=time.perf_counter_ns();fn();xs.append(time.perf_counter_ns()-t)
    return statistics.median(xs),xs


def main()->int:
    OUT.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)

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

    text=tok.apply_chat_template([
        {"role":"system","content":"Answer briefly."},
        {"role":"user","content":"Give one synonym for fast."},
    ],tokenize=False,add_generation_prompt=True)
    enc=tok(text,return_tensors="pt",add_special_tokens=False)

    # Minimal runtime structures: ordinary request is not governed, so CKCA returns directly
    # to the exact B-plane call without Port/J allocation.
    governed_routes={"governed:demo":17}
    factor_valid={17:True}
    authority={17:(9,"deadbeef",REVISION)}
    cap_payload=f"17|9|deadbeef|{REVISION}".encode()
    cap_mac=hmac.new(KEY,cap_payload,hashlib.sha256).digest()

    def direct():
        with torch.inference_mode():
            return model(**enc,use_cache=False,return_dict=True).logits

    def ckca_no_port():
        route=governed_routes.get("ordinary:request")
        if route is None:
            with torch.inference_mode():
                return model(**enc,use_cache=False,return_dict=True).logits
        raise AssertionError("unexpected governed route")

    def control_only_no_port():
        route=governed_routes.get("ordinary:request")
        return route is None

    def active_control_only():
        pod=governed_routes.get("governed:demo")
        if pod is None:return False
        gen,digest,rev=authority[pod]
        payload=f"{pod}|{gen}|{digest}|{rev}".encode()
        good=hmac.compare_digest(hmac.new(KEY,payload,hashlib.sha256).digest(),cap_mac)
        return good and rev==REVISION and factor_valid.get(pod,False)

    # Exact functional bypass check.
    dlog=direct();wlog=ckca_no_port();max_delta=float(torch.max(torch.abs(dlog-wlog)).item())

    for _ in range(3): direct();ckca_no_port()
    direct_times=[];wrapper_times=[]
    # Interleave to reduce drift bias.
    for i in range(12):
        funcs=(direct,ckca_no_port) if i%2==0 else (ckca_no_port,direct)
        for fn in funcs:
            t=time.perf_counter_ns();fn();dt=time.perf_counter_ns()-t
            (direct_times if fn is direct else wrapper_times).append(dt)
    direct_med=statistics.median(direct_times);wrapper_med=statistics.median(wrapper_times)
    observed=(wrapper_med/direct_med)-1.0

    no_port_control_med,no_port_control_times=median_ns(control_only_no_port,100000)
    active_control_med,active_control_times=median_ns(active_control_only,50000)
    modeled_no_port_fraction=no_port_control_med/direct_med
    modeled_active_control_fraction=active_control_med/direct_med

    report={
        "stage":"R305-CKCA-SERVING-OVERHEAD",
        "architecture_candidate":"CKCA zero-touch B-plane bypass",
        "model_id":MODEL_ID,"revision":REVISION,"weights_sha256":hashes,"backbone_frozen":all(not p.requires_grad for p in model.parameters()),
        "direct_forward_median_ms":direct_med/1e6,"ckca_no_port_forward_median_ms":wrapper_med/1e6,
        "observed_no_port_wrapper_overhead_fraction":observed,
        "no_port_control_only_median_ns":no_port_control_med,
        "active_authority_plus_factor_control_median_ns":active_control_med,
        "modeled_no_port_control_fraction_of_direct_forward":modeled_no_port_fraction,
        "modeled_active_control_fraction_of_direct_forward":modeled_active_control_fraction,
        "no_port_full_vocab_max_logit_delta":max_delta,
        "no_port_path_allocates_port_or_jspace":False,
        "no_port_path":"one route-table miss then the exact unchanged frozen-model forward",
        "active_control_path":"route lookup + authority generation/revision/MAC verification + O(1) lifetime-factor bit",
        "mechanism":(
            "CKCA is conditional rather than always-on. An ordinary non-governed request performs a route miss and then "
            "executes the identical B-plane model call. Governed requests pay authority/lifetime checks before Port/J use."
        ),
        "benchmark_boundary":(
            "CPU/Python microbenchmark on Qwen2.5-0.5B. The control-only fraction estimates architectural overhead but is "
            "not a GPU production serving benchmark. Observed wrapper timing is noisy and must not be generalized."
        ),
        "dod_status":"NOT_DOD; local serving-overhead mechanism gate",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode();report["report_sha256"]=hashlib.sha256(canonical).hexdigest();OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
    if max_delta!=0.0:return 2
    if modeled_no_port_fraction>=0.05:return 3
    if not active_control_only():return 4
    return 0

if __name__=="__main__":raise SystemExit(main())
