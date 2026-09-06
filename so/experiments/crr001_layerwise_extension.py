"""Post-registered stronger CRR001-L: independent K and V bases per layer."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from so.experiments.crr001_finite_response_rank import (
    PROMPTS, identical, oracle_screen, legacy_cache,
)

REVISIONS = {
    "distilbert/distilgpt2": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
    "EleutherAI/pythia-70m": "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
}


def cache_arrays(output):
    return [[x.detach().cpu().numpy().ravel().copy() for x in pair]
            for pair in legacy_cache(output.past_key_values)]


def flatten(cache):
    return np.concatenate([x for pair in cache for x in pair])


def run(name, output_dir):
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    revision = REVISIONS[name]
    tok = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
    model = AutoModel.from_pretrained(name, revision=revision, torch_dtype=torch.float64,
        attn_implementation="eager", trust_remote_code=False, use_safetensors=True).eval()
    model.requires_grad_(False)
    assert next(model.parameters()).dtype == torch.float64
    blocks = model.h if hasattr(model, "h") else model.layers
    amplitudes = np.linspace(-1., 3., 33)
    record = dict(experiment="CRR-001-L", model=name, revision=revision,
        torch=torch.__version__, transformers=transformers.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cells=[])
    for prompt_index, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            absent = model(ids, use_cache=True, output_hidden_states=True, return_dict=True)
        absent_cache = cache_arrays(absent)
        source_site = absent.hidden_states[1][0, 3].detach().clone()
        scale = float(torch.sqrt(torch.mean(source_site**2)))*.5
        del absent
        for seed in range(3):
            gen = torch.Generator().manual_seed(seed)
            direction = torch.randn(source_site.shape, generator=gen, dtype=torch.float64)
            direction *= scale/torch.sqrt(torch.mean(direction**2))
            current = [1.]
            def inject(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                new = h.clone()
                new[:, 3, :] += current[0]*direction
                return (new, *output[1:]) if isinstance(output, tuple) else new
            handle = blocks[0].register_forward_hook(inject)
            states = []
            try:
                with torch.no_grad():
                    for amplitude in amplitudes:
                        current[0] = float(amplitude)
                        out = model(ids, use_cache=True, output_hidden_states=True, return_dict=True)
                        states.append(cache_arrays(out))
                    current[0] = 1.
                    repeat = cache_arrays(model(ids, use_cache=True, output_hidden_states=True, return_dict=True))
            finally:
                handle.remove()
            repeat_exact = identical(flatten(repeat), flatten(states[16]))
            absent_exact = identical(flatten(absent_cache), flatten(states[8]))
            untouched = all(identical(row[0][k], absent_cache[0][k]) for row in states for k in (0, 1))
            assert repeat_exact and absent_exact and untouched
            cell = dict(seed=seed, prompt_index=prompt_index, amplitudes=amplitudes.tolist(),
                repeat_forward_exact=repeat_exact, zero_source_equals_absent_exact=absent_exact,
                before_injection_kv_unchanged=untouched,
                aggregate_kv=oracle_screen([flatten(x) for x in states],16,8), tensors=[])
            for layer in range(len(blocks)):
                for kind, index in [("key",0),("value",1)]:
                    values = [row[layer][index] for row in states]
                    stats = oracle_screen(values,16,8)
                    stats.update(layer=layer,kind=kind)
                    cell["tensors"].append(stats)
            record["cells"].append(cell)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir/(name.replace("/","--")+"-layerwise.json")).write_text(json.dumps(record,indent=2)+"\n")
            print(json.dumps(dict(model=name,seed=seed,prompt=prompt_index,
                deepest=[dict(kind=t["kind"], rank16=t["oracle"][-1],roundoff=t["full_rank_reconstruction_roundoff_maxabs"])
                         for t in cell["tensors"] if t["layer"]==len(blocks)-1])),flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=list(REVISIONS),required=True)
    parser.add_argument("--output-dir",type=Path,default=Path("crr001-layerwise-results"))
    args=parser.parse_args()
    run(args.model,args.output_dir)

if __name__=="__main__":
    main()
