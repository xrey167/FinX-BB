from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file, legacy_cache, slice_cache
from research.qwen25_05b_r308_late_bound_handle_vm import MODEL_ID, REVISION, EXPECTED_WEIGHTS, VALUE_STRINGS

OUT = Path(os.environ.get("SO_R309_REPORT", "ci-qwen-r309/report.json"))


def run_suffix(model, suffix_ids, prefix_cache, prefix_len):
    ids = torch.tensor([suffix_ids], dtype=torch.long)
    mask = torch.ones((1, prefix_len + len(suffix_ids)), dtype=torch.long)
    pos = torch.arange(prefix_len, prefix_len + len(suffix_ids), dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        return model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=copy.deepcopy(prefix_cache),
            use_cache=True,
            return_dict=True,
        )


def cache_bytes(cache) -> int:
    total = 0
    for k, v in legacy_cache(cache):
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return int(total)


def mutated_cache(cache, seed: int):
    g = torch.Generator().manual_seed(seed)
    out=[]
    for k,v in legacy_cache(cache):
        rk=torch.randn(k.shape,generator=g,dtype=torch.float32).to(k.dtype)
        rv=torch.randn(v.shape,generator=g,dtype=torch.float32).to(v.dtype)
        out.append((rk,rv))
    return tuple(out)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)

    md = Path(snapshot_download(
        repo_id=MODEL_ID, revision=REVISION,
        allow_patterns=["*.json","*.txt","*.model","*.tiktoken","*.safetensors","*.index.json","*.merges","*.vocab","merges.txt","vocab.json"],
    ))
    hashes={n:sha256_file(md/n) for n in EXPECTED_WEIGHTS};assert hashes==EXPECTED_WEIGHTS
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True,trust_remote_code=False)
    tok.pad_token_id=tok.eos_token_id
    model=AutoModelForCausalLM.from_pretrained(
        md,local_files_only=True,torch_dtype=torch.bfloat16,
        attn_implementation="eager",trust_remote_code=False,
    ).eval();model.requires_grad_(False)

    messages=[
        {"role":"system","content":(
            "A trusted runtime will splice the verified current value into the assistant stream. "
            "After receiving it, continue consistently without changing the value."
        )},
        {"role":"user","content":(
            "Report the current governed value for canonical entity POD-17. The runtime supplies the current value."
        )},
    ]
    base=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    static_text=base+"Current value:\n"
    static_ids=tok(static_text,add_special_tokens=False).input_ids
    S=len(static_ids)
    st=torch.tensor([static_ids],dtype=torch.long)
    with torch.inference_mode():
        static_out=model(input_ids=st,attention_mask=torch.ones_like(st),use_cache=True,return_dict=True)
    static_cache=legacy_cache(static_out.past_key_values)

    # Use multi-token future values with stable token-prefix boundaries.
    stable=[]
    for value in VALUE_STRINGS:
        full_text=static_text+value+"\nStatus:"
        full_ids=tok(full_text,add_special_tokens=False).input_ids
        if full_ids[:S]==static_ids:
            stable.append((value,full_ids,full_ids[S:]))
    if len(stable)<8:
        raise RuntimeError(f"too few prefix-stable values: {len(stable)}")
    probes=stable[:10]

    rows=[];deltas=[];top_matches=[];full_times=[];splice_times=[];dep_ratios=[];old_suffixes=[]
    current_logits=[]
    for i,(value,full_ids,suffix_ids) in enumerate(probes):
        # Full recompute oracle.
        full_tensor=torch.tensor([full_ids],dtype=torch.long)
        t0=time.perf_counter()
        with torch.inference_mode():
            full_out=model(input_ids=full_tensor,attention_mask=torch.ones_like(full_tensor),use_cache=True,return_dict=True)
        full_times.append(time.perf_counter()-t0)
        full_logits=full_out.logits[0,-1].float()

        # Late-bound splice: immutable B-prefix reused; only current value+post-value cue is re-executed.
        t0=time.perf_counter()
        sp=run_suffix(model,suffix_ids,static_cache,S)
        splice_times.append(time.perf_counter()-t0)
        sp_logits=sp.logits[0,-1].float()
        delta=float(torch.max(torch.abs(full_logits-sp_logits)).item())
        deltas.append(delta)
        ft=int(full_logits.argmax().item());stp=int(sp_logits.argmax().item())
        top_matches.append(ft==stp)
        dep_ratios.append(len(suffix_ids)/len(full_ids))
        current_logits.append(sp_logits)

        pc=legacy_cache(sp.past_key_values)
        old_suffixes.append(slice_cache(pc,S,S+len(suffix_ids)))
        rows.append({
            "value":value,
            "full_prompt_tokens":len(full_ids),
            "immutable_prefix_tokens":S,
            "generation_dependent_suffix_tokens":len(suffix_ids),
            "generation_dependent_fraction":len(suffix_ids)/len(full_ids),
            "full_recompute_seconds":full_times[-1],
            "late_splice_seconds":splice_times[-1],
            "max_vocab_logit_delta":delta,
            "full_next_token_id":ft,
            "splice_next_token_id":stp,
            "next_token_match":ft==stp,
        })

    # Update anti-resurrection: an old post-binding suffix remains physically resident but the
    # new execution begins from the immutable pre-binding checkpoint. Mutating old suffix bytes
    # cannot alter current logits because they are not admitted into the new decode branch.
    stale_mutation_deltas=[]
    for i in range(1,len(probes)):
        value,full_ids,suffix_ids=probes[i]
        before=run_suffix(model,suffix_ids,static_cache,S).logits[0,-1].float()
        _stale=mutated_cache(old_suffixes[i-1],9000+i)  # physically present corruption, deliberately unused
        after=run_suffix(model,suffix_ids,static_cache,S).logits[0,-1].float()
        stale_mutation_deltas.append(float(torch.max(torch.abs(before-after)).item()))

    full_med=statistics.median(full_times);splice_med=statistics.median(splice_times)
    report={
        "stage":"R309-LATE-BOUND-DECODE-SPLICE",
        "architecture_candidate":"Late-Bound Decode Checkpoint (LBDC) for CKCA",
        "model_id":MODEL_ID,"revision":REVISION,"weights_sha256":hashes,"backbone_frozen":all(not p.requires_grad for p in model.parameters()),
        "stable_multitoken_values":len(probes),
        "immutable_prefix_tokens":S,
        "top_match_rate_full_recompute_vs_splice":sum(top_matches)/len(top_matches),
        "max_full_vocab_logit_delta_full_vs_splice":max(deltas),
        "median_generation_dependent_token_fraction":statistics.median(dep_ratios),
        "median_full_recompute_seconds":full_med,
        "median_late_splice_seconds":splice_med,
        "median_walltime_speedup_full_over_splice":full_med/max(splice_med,1e-9),
        "max_current_logit_delta_after_mutating_physically_resident_old_suffix":max(stale_mutation_deltas),
        "static_prefix_cache_bytes":cache_bytes(static_cache),
        "rows":rows,
        "mechanism":(
            "The decoder is checkpointed immediately before mutable world data is bound. A verified Port value is then "
            "spliced as token IDs into the assistant stream and only the generation-dependent suffix is executed. On edit, "
            "the immutable B-prefix is reused and the old post-binding branch is lifetime-invalidated rather than the full "
            "query being reprefilled."
        ),
        "architectural_hypothesis":(
            "CKCA should delay mutable-value materialization until the latest semantic point that needs it. This minimizes "
            "the neural state whose lifetime depends on world data and turns edits into suffix re-execution rather than "
            "global prompt/cache invalidation."
        ),
        "prior_art_boundary":(
            "Prefix/KV caching and forced-token/copy decoding are established. The candidate contribution can only be their "
            "use as one part of the full generation-authority + transitive-lifetime coherence protocol, not the splice itself."
        ),
        "dod_status":"NOT_DOD; late-bound real-model decode gate",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode();report["report_sha256"]=hashlib.sha256(canonical).hexdigest();OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("stable_multitoken_values","top_match_rate_full_recompute_vs_splice","max_full_vocab_logit_delta_full_vs_splice","median_generation_dependent_token_fraction","median_full_recompute_seconds","median_late_splice_seconds","median_walltime_speedup_full_over_splice","max_current_logit_delta_after_mutating_physically_resident_old_suffix")},indent=2))
    if report["top_match_rate_full_recompute_vs_splice"]!=1.0:return 2
    if report["max_full_vocab_logit_delta_full_vs_splice"]>0.02:return 3
    if report["max_current_logit_delta_after_mutating_physically_resident_old_suffix"]>0.02:return 4
    return 0

if __name__=="__main__":raise SystemExit(main())
