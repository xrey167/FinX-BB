"""NIC001: oracle nonlinear chart composition, not a new repair algorithm."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import platform
import numpy as np

REVISIONS = {
    "distilbert/distilgpt2": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
    "EleutherAI/pythia-70m": "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
}
PROMPTS = [
    "The engineer checked the updated reference before making a decision.",
    "A researcher compared the measurements with the previous laboratory notes.",
]
CHUNKS = [" Then the assistant reviewed the record.",
          " Next it compared the evidence again.",
          " Finally it prepared a short conclusion."]
POSITIONS = (3, 6, 9)


def identical(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def mobius(values):
    """Classical anchored subset transform. Also supports exact object arrays."""
    values = np.asarray(values)
    size = len(values)
    if size < 1 or size & (size-1):
        raise ValueError("Need a power-of-two number of worlds")
    out = values.copy()
    for bit in range(size.bit_length()-1):
        for mask in range(size):
            if mask & (1 << bit):
                out[mask] = out[mask] - out[mask ^ (1 << bit)]
    return out


def reconstruct(coefficients, mask, order):
    out = coefficients[0].copy()
    for subset in range(1, len(coefficients)):
        if subset & mask == subset and subset.bit_count() <= order:
            out = out + coefficients[subset]
    return out


def exact_screen():
    rows = []
    for family in ("separable", "pairwise", "pure_triple", "compact_joint"):
        values = []
        for mask in range(8):
            a, b, c = [Fraction(int(bool(mask & (1 << i)))) for i in range(3)]
            base = Fraction(2)+a+2*b+3*c
            if family == "separable":
                value = base + a/(1+a) + b/(2+b) + c/(3+c)
            elif family == "pairwise":
                value = base + a*b/3 + b*c/5 + a*c/7
            elif family == "pure_triple":
                value = base + 11*a*b*c
            else:
                value = Fraction(1)/(1+a+2*b+3*c)
            values.append([value])
        values = np.array(values, dtype=object)
        co = mobius(values)
        exact_by_order = {str(k): [bool(np.array_equal(reconstruct(co, m, k), values[m]))
                                  for m in range(8)] for k in (1,2,3)}
        final_by_order = {str(k): str(reconstruct(co, 7, k)[0]-values[7,0]) for k in (1,2,3)}
        order_results = []
        for permutation in itertools.permutations(range(3)):
            x = values[0,0]
            for i in permutation:
                x += values[1 << i,0]-values[0,0]
            order_results.append(x)
        if family == "pure_triple":
            assert all(exact_by_order['2'][:7]) and not exact_by_order['2'][7]
            assert len(set(order_results)) == 1 and order_results[0] != values[7,0]
        assert all(exact_by_order['3'])
        rows.append(dict(family=family, values=[str(x[0]) for x in values],
                         exact_by_order=exact_by_order, final_error_by_order=final_by_order,
                         three_way_interaction=str(co[7,0]),
                         all_increment_orders_agree=len(set(order_results)) == 1,
                         increment_orders_correct=all(x == values[7,0] for x in order_results)))
    return rows


def residual(predicted, fresh):
    diff = predicted-fresh
    return dict(byte_identical=identical(predicted,fresh),
                maxabs=float(np.max(np.abs(diff), initial=0)),
                l2=float(np.linalg.norm(diff.ravel())),
                unequal_coordinates=int(np.count_nonzero(predicted != fresh)),
                coordinates=int(fresh.size))


def tensor_screen(worlds):
    worlds = np.stack(worlds)
    coefficients = mobius(worlds)
    all_order = residual(reconstruct(coefficients,7,3),worlds[7])
    floor = max(1e-10, 100*all_order['maxabs'])
    rows = {}
    for k in (1,2):
        rows[str(k)] = residual(reconstruct(coefficients,7,k),worlds[7])
        rows[str(k)]['material'] = rows[str(k)]['maxabs'] > floor
    pairs = {str(mask): float(np.max(np.abs(coefficients[mask]),initial=0)) for mask in (3,5,6)}
    return dict(orders=rows, full_order_roundoff=all_order, descriptive_material_floor=floor,
                pair_chart_drift_maxabs=pairs,
                triple_interaction_maxabs=float(np.max(np.abs(coefficients[7]),initial=0)),
                old_vs_all_removed=residual(worlds[0],worlds[7]),
                pair_removal_independent_errors={str(m):residual(reconstruct(coefficients,m,1),worlds[m]) for m in (3,5,6)})


def cache_arrays(out):
    cache = out.past_key_values
    if hasattr(cache, 'to_legacy_cache'):
        cache = cache.to_legacy_cache()
    return [[t.detach().cpu().numpy().copy() for t in pair[:2]] for pair in cache]


def snapshots_identical(left, right):
    if len(left) != len(right):
        return False
    for xs,ys in zip(left,right):
        if len(xs) != len(ys):
            return False
        for xp,yp in zip(xs,ys):
            if len(xp) != len(yp) or not all(identical(a,b) for a,b in zip(xp,yp)):
                return False
    return True


def frozen_screen(name, destination):
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    revision = REVISIONS[name]
    tokenizer = AutoTokenizer.from_pretrained(name,revision=revision,trust_remote_code=False)
    model = AutoModel.from_pretrained(name,revision=revision,torch_dtype=torch.float64,
                attn_implementation='eager',trust_remote_code=False,use_safetensors=True).eval()
    model.requires_grad_(False)
    assert next(model.parameters()).dtype == torch.float64
    blocks = model.h if hasattr(model,'h') else model.layers
    chunk_ids = [tokenizer(s,return_tensors='pt')['input_ids'] for s in CHUNKS]
    record = dict(experiment='NIC-001',model=name,revision=revision,
                  python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                  transformers=transformers.__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  positions=list(POSITIONS),chunk_ids=[x.tolist() for x in chunk_ids],cells=[])
    for pi,text in enumerate(PROMPTS):
        ids = tokenizer(text,return_tensors='pt')['input_ids']
        assert ids.shape[1] > max(POSITIONS)
        with torch.no_grad():
            base = model(ids,use_cache=True,output_hidden_states=True,return_dict=True)
        source_sites = [base.hidden_states[1][0,pos].detach().clone() for pos in POSITIONS]
        del base
        for seed in range(3):
            directions = []
            for i,site in enumerate(source_sites):
                gen = torch.Generator().manual_seed(1000*seed+i)
                d = torch.randn(site.shape,generator=gen,dtype=torch.float64)
                d *= .5*torch.sqrt(torch.mean(site**2))/torch.sqrt(torch.mean(d**2))
                directions.append(d)

            def run_world(mask, with_hook=True):
                def inject(module,args,out):
                    h = out[0] if isinstance(out,tuple) else out
                    changed = h.clone()
                    for i,pos in enumerate(POSITIONS):
                        if not mask & (1 << i):
                            changed[:,pos,:] += directions[i]
                    return (changed,*out[1:]) if isinstance(out,tuple) else changed
                handle = blocks[0].register_forward_hook(inject) if with_hook else None
                try:
                    with torch.no_grad():
                        out = model(ids,use_cache=True,return_dict=True)
                finally:
                    if handle is not None:
                        handle.remove()
                snapshots = [cache_arrays(out)]
                cache = out.past_key_values
                for chunk in chunk_ids:
                    with torch.no_grad():
                        out = model(chunk,past_key_values=cache,use_cache=True,return_dict=True)
                    current = cache_arrays(out)
                    for previous_pair,new_pair in zip(snapshots[-1],current):
                        for old,new in zip(previous_pair,new_pair):
                            assert identical(old,new[...,:old.shape[-2],:]), 'Prior K/V overwritten'
                    snapshots.append(current)
                    cache = out.past_key_values
                return snapshots

            worlds = [run_world(mask) for mask in range(8)]
            repeat0, repeat7 = run_world(0), run_world(7)
            no_hook = run_world(7,False)
            assert snapshots_identical(worlds[0],repeat0)
            assert snapshots_identical(worlds[7],repeat7)
            assert snapshots_identical(worlds[7],no_hook)
            for mask in range(8):
                for stage in range(4):
                    for kind in range(2):
                        assert identical(worlds[mask][stage][0][kind],no_hook[stage][0][kind]), 'Pre-injection cache depends on source'
            cell = dict(seed=seed,prompt_index=pi,input_ids=ids.tolist(),
                        repeat_old_exact=True,repeat_absent_exact=True,zero_hook_vs_no_hook_exact=True,
                        prior_cache_prefixes_unchanged=True,block0_independent=True,stages=[])
            for stage in range(4):
                entry = dict(stage=stage,tensors=[])
                for layer in range(len(blocks)):
                    for kind,index in [('key',0),('value',1)]:
                        arrays = [worlds[mask][stage][layer][index] for mask in range(8)]
                        start = 0 if stage == 0 else worlds[0][stage-1][layer][index].shape[-2]
                        full_stats = tensor_screen(arrays)
                        new_stats = tensor_screen([a[...,start:,:] for a in arrays])
                        entry['tensors'].append(dict(layer=layer,kind=kind,total_tokens=arrays[0].shape[-2],
                            new_tokens=arrays[0].shape[-2]-start,all_slots=full_stats,new_slots=new_stats))
                cell['stages'].append(entry)
            record['cells'].append(cell)
            destination.mkdir(parents=True,exist_ok=True)
            (destination/(name.replace('/','--')+'.json')).write_text(json.dumps(record,indent=2)+'\n')
            last = cell['stages'][-1]['tensors'][-1]
            print(json.dumps(dict(model=name,prompt=pi,seed=seed,
                 final_value_pairwise_new=last['new_slots']['orders']['2'],
                 final_value_allorder_floor=last['new_slots']['full_order_roundoff']['maxabs'])),flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=list(REVISIONS))
    parser.add_argument('--output-dir',type=Path,default=Path('nic001-results'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.model:
        frozen_screen(args.model,args.output_dir)
    else:
        record = dict(experiment='NIC-001',exact=exact_screen(),
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (args.output_dir/'exact.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2))

if __name__ == '__main__':
    main()
