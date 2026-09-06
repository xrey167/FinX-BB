"""QCR001: native-precision repair, with exact scalar/rounding-box controls.

No oracle result is a deployable certificate or a major-invention claim.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
from scipy.linalg import qr
from scipy.optimize import linprog

REVISIONS = {
    "distilbert/distilgpt2": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
    "EleutherAI/pythia-70m": "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
}
PROMPTS = ["The engineer checked the updated reference before making a decision.",
           "A researcher compared the measurements with the previous laboratory notes."]
CONTINUATION = " The recorded result was checked again."
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
RANKS = (1, 2, 4, 8, 16, 32)


def array_bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def same(a, b):
    return a.shape == b.shape and np.array_equal(array_bits(a), array_bits(b))


def round_native(a, dtype):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dtype).to(torch.float64).numpy()


def compare(a, b):
    if a.shape != b.shape:
        raise ValueError("Comparison requires identical tensor shapes")
    eq = array_bits(a) == array_bits(b)
    return dict(coordinates=int(eq.size), matches=int(eq.sum()),
                match_fraction=float(eq.mean()), byte_identical=bool(eq.all()),
                maxabs=float(np.max(np.abs(a-b), initial=0)))


def rn_grid(x, step=F(1, 128)):
    """Exact rational uniform-grid round-to-nearest-ties-even."""
    x, step = F(x), F(step)
    if step <= 0:
        raise ValueError("step must be positive")
    v = x/step
    low = v.numerator//v.denominator
    tail = v-low
    n = low + int(tail > F(1, 2) or (tail == F(1, 2) and low % 2 != 0))
    return n*step


def certify_grid(center, radius, step=F(1, 128)):
    center, radius = F(center), F(radius)
    if radius < 0:
        raise ValueError("negative error radius")
    a, b = rn_grid(center-radius, step), rn_grid(center+radius, step)
    return a if a == b else None


def scalar_screen():
    q = F(1,128)
    centers = [F(5,4)+j*q for j in range(64)]
    accepted = 0
    for center in centers:
        pred, truth, bound = center+q/16, center-q/32, q/8
        assert abs(pred-truth) <= bound
        y = certify_grid(pred, bound, q)
        assert y == rn_grid(truth, q) and pred != truth
        accepted += 1
    crossings=[]
    for power in (10, 20, 40):
        mid=F(5,4)+q/2
        eps=F(1, 2**power)*q
        assert rn_grid(mid-eps,q) != rn_grid(mid+eps,q)
        assert certify_grid(mid,eps,q) is None
        crossings.append(dict(error=str(2*eps), same_quantized_output=False))
    x=F(1)+q
    native=x
    for _ in range(2):
        native=rn_grid((native+1)/2,q)
    ideal=rn_grid(((x+1)/2+1)/2,q)
    # This particular two-stage example can agree; a cancellation exposes the
    # information discarded by the FIRST rounding, independently of smallness.
    rounded_mid=rn_grid((x+1)/2,q)
    staged=rn_grid(2*rounded_mid-1,q)
    unrounded=rn_grid(2*((x+1)/2)-1,q)
    assert staged != unrounded
    xt=torch.tensor(float(x),dtype=torch.bfloat16)
    pt=2*((xt+1)/2)-1
    assert float(pt)==float(staged)
    return dict(nonzero_error_exact_rounding_controls=accepted,
                midpoint_crossings=crossings, staged_result=str(staged),
                final_only_rounding_result=str(unrounded),
                staged_torch_bfloat16_agrees=True,
                theorem_or_certificate_novelty=False)


def rounding_box(y, dtype):
    """Closed superset of RN cells for finite nonzero normal target values.

    fp32/bf16 adjacent values and midpoints are exactly representable in fp64.
    Zero/overflow endpoints are excluded by the witness caller.
    """
    t=torch.from_numpy(np.asarray(y,dtype=np.float64)).to(dtype)
    prev=torch.nextafter(t,torch.full_like(t,-float('inf'))).double().numpy()
    nxt=torch.nextafter(t,torch.full_like(t,float('inf'))).double().numpy()
    return (prev+y)/2, (nxt+y)/2


def fraction_solve(a,b):
    """Gauss-Jordan over exact rationals; singular candidates fail closed."""
    n=len(b)
    z=[[F(v) for v in row]+[F(rhs)] for row,rhs in zip(a,b)]
    for c in range(n):
        pivot=next((i for i in range(c,n) if z[i][c]),None)
        if pivot is None:
            raise ValueError("singular rational anchor")
        z[c],z[pivot]=z[pivot],z[c]
        p=z[c][c]
        z[c]=[v/p for v in z[c]]
        for i in range(n):
            if i!=c and z[i][c]:
                p=z[i][c]
                z[i]=[v-p*w for v,w in zip(z[i],z[c])]
    return [row[-1] for row in z]


def make_witness(basis, low, high, indices, old=None):
    """Construct exact row relation; only return a proven disjoint box."""
    r=basis.shape[1]
    old=np.zeros(len(low)) if old is None else old
    if len(indices)<r+1:
        return None
    _,_,p=qr(basis[indices].T,pivoting=True,mode='economic')
    anchor=np.asarray(indices)[p[:r]]
    remaining=np.asarray(indices)[p[r:]]
    A=[[F(float(x)) for x in row] for row in basis[anchor]]
    for extra in remaining[:6]:
        row=[F(float(x)) for x in basis[extra]]
        try:
            coeff=fraction_solve(list(map(list,zip(*A))),row)
        except ValueError:
            continue
        al=[F(float(low[i]))-F(float(old[i])) for i in anchor]
        ah=[F(float(high[i]))-F(float(old[i])) for i in anchor]
        bound_low=sum(c*(l if c>=0 else h) for c,l,h in zip(coeff,al,ah))
        bound_high=sum(c*(h if c>=0 else l) for c,l,h in zip(coeff,al,ah))
        el=F(float(low[extra]))-F(float(old[extra]))
        eh=F(float(high[extra]))-F(float(old[extra]))
        if bound_low>eh or bound_high<el:
            w=dict(rank=r,anchor_indices=anchor.tolist(),extra_index=int(extra),
                   anchor_rows=[[str(x) for x in line] for line in A],
                   extra_row=list(map(str,row)),coefficients=list(map(str,coeff)),
                   anchor_low=list(map(str,al)),anchor_high=list(map(str,ah)),
                   extra_low=str(el),extra_high=str(eh),
                   span_low=str(bound_low),span_high=str(bound_high),
                   original_old=[float(old[i]).hex() for i in list(anchor)+[extra]],
                   absolute_low=[float(low[i]).hex() for i in list(anchor)+[extra]],
                   absolute_high=[float(high[i]).hex() for i in list(anchor)+[extra]],
                   scope="fixed represented-real affine basis then single RN cast")
            assert verify_witness(w)
            return w
    return None


def verify_witness(w):
    A=[[F(x) for x in row] for row in w['anchor_rows']]
    b=[F(x) for x in w['extra_row']]
    c=[F(x) for x in w['coefficients']]
    r=w['rank']
    if len(A)!=r or len(c)!=r or len(b)!=r or any(len(row)!=r for row in A):
        return False
    if any(sum(c[i]*A[i][j] for i in range(r))!=b[j] for j in range(r)):
        return False
    lo=[F(x) for x in w['anchor_low']]; hi=[F(x) for x in w['anchor_high']]
    if 'original_old' in w:
        old=[F(float.fromhex(v)) for v in w['original_old']]
        absolute_low=[F(float.fromhex(v)) for v in w['absolute_low']]
        absolute_high=[F(float.fromhex(v)) for v in w['absolute_high']]
        if lo+[F(w['extra_low'])] != [a-b for a,b in zip(absolute_low,old)]:
            return False
        if hi+[F(w['extra_high'])] != [a-b for a,b in zip(absolute_high,old)]:
            return False
    if len(lo)!=r or len(hi)!=r or any(a>b for a,b in zip(lo,hi)):
        return False
    lower=sum(v*(a if v>=0 else b) for v,a,b in zip(c,lo,hi))
    upper=sum(v*(b if v>=0 else a) for v,a,b in zip(c,lo,hi))
    el,eh=F(w['extra_low']),F(w['extra_high'])
    return el<=eh and (lower>eh or upper<el)


def separation_search(basis, old, target, dtype):
    """Numerical row/LP proposals followed by exact rational checking.

    Unsuccessful search is inconclusive. No LP status alone counts as proof.
    """
    absolute_low,absolute_high=rounding_box(target,dtype)
    low,high=absolute_low-old,absolute_high-old
    valid=np.isfinite(low)&np.isfinite(high)&(target!=0)&(np.abs(target)>2*torch.finfo(dtype).tiny)
    valid&=np.linalg.norm(basis,axis=1)>1e-14
    indices=np.flatnonzero(valid)
    r=basis.shape[1]
    result=dict(witness=None,status='INCONCLUSIVE')
    if len(indices)<r+1:
        return result
    _,_,pivot=qr(basis[indices].T,pivoting=True,mode='economic')
    anchors=indices[pivot[:r]]
    center=(low+high)/2
    half=(high-low)/2
    fit=basis @ (basis.T @ (target-old))
    score=np.abs(fit-center)/np.maximum(half,np.finfo(float).tiny)
    selected=np.unique(np.concatenate((anchors,indices[np.argsort(score[indices])[-768:]])))
    # Minimize maximum normalized box violation, a standard phase-I LP.
    A=basis[selected]/half[selected,None]
    lower=low[selected]/half[selected]; upper=high[selected]/half[selected]
    Aub=np.vstack((np.column_stack((A,-np.ones(len(A)))),np.column_stack((-A,-np.ones(len(A))))))
    bvec=np.concatenate((upper,-lower))
    objective=np.zeros(r+1); objective[-1]=1
    sol=linprog(objective,A_ub=Aub,b_ub=bvec,bounds=[(None,None)]*r+[(0,None)],
                method='highs',options={'time_limit':20.})
    result.update(lp_status=int(sol.status),lp_slack=float(sol.fun) if sol.fun is not None else None,
                  selected_constraints=int(len(selected)))
    if sol.success and sol.fun>1e-7:
        dual=np.asarray(sol.ineqlin.marginals)
        support=np.unique(np.flatnonzero(np.abs(dual)>1e-9)%len(selected))
        w=make_witness(basis,absolute_low,absolute_high,selected[support],old)
        if w:
            result.update(witness=w,status='EXACT_SEPARATION')
    return result


def oracle(states,dtype,new_start,witness=False):
    states=np.asarray(states,dtype=np.float64)
    old=states[16]; delta=states-old
    u,s,vt=np.linalg.svd(delta,full_matrices=False)
    invariant=np.all(delta==0,axis=0)
    results=[]
    for r in RANKS:
        k=min(r,len(s))
        predicted=old+(u[:,:k]*s[:k])@vt[:k]
        predicted[:,invariant]=old[invariant]
        predicted[16]=old
        quantized=round_native(predicted,dtype)
        all_rows=[compare(a,b) for a,b in zip(quantized,states)]
        new_rows=[compare(a[new_start:],b[new_start:]) for a,b in zip(quantized,states)]
        results.append(dict(rank=r,per_revision=all_rows,new_per_revision=new_rows,
                            all_nontrivial_exact=sum(x['byte_identical'] for i,x in enumerate(all_rows) if i!=16),
                            new_nontrivial_exact=sum(x['byte_identical'] for i,x in enumerate(new_rows) if i!=16),
                            pre_cast_maxabs=float(np.max(np.abs(predicted-states)))))
    row=dict(coordinates=int(states.shape[1]),new_coordinates=int(states.shape[1]-new_start),
             oracle_invariant_coordinates=int(invariant.sum()),singular_values=s.tolist(),ranks=results)
    if witness:
        row['rounding_box_separation']=separation_search(vt[:16].T,old,states[8],dtype)
    return row


def cache_list(out):
    c=out.past_key_values
    if hasattr(c,'to_legacy_cache'):
        c=c.to_legacy_cache()
    return [(p[0],p[1]) for p in c]


def flatten_cache(out,prefix_length=None):
    arrays=[]; splits=[]
    for layer,pair in enumerate(cache_list(out)):
        for kind,t in zip(('key','value'),pair):
            # Token-major flatten makes the old/new split contiguous.
            x=t.detach().permute(0,2,1,3).contiguous().double().cpu().numpy()
            arrays.append(x.reshape(-1).copy())
            splits.append(int(prefix_length*x.shape[2]*x.shape[3]) if prefix_length is not None else 0)
    return arrays,splits


def parameters_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode()); h.update(p.detach().double().contiguous().numpy().tobytes())
    return h.hexdigest()


def run_world(model,blocks,ids,tail,direction,amplitude):
    handle=None
    if amplitude is not None:
        def hook(module,args,out):
            x=out[0] if isinstance(out,tuple) else out
            y=x.clone(); y[:,3,:]+=float(amplitude)*direction
            return (y,*out[1:]) if isinstance(out,tuple) else y
        handle=blocks[0].register_forward_hook(hook)
    try:
        with torch.no_grad():
            pre=model(ids,use_cache=True,output_hidden_states=True,return_dict=True)
    finally:
        if handle is not None:
            handle.remove()
    pre_arrays,_=flatten_cache(pre)
    with torch.no_grad():
        post=model(tail,past_key_values=pre.past_key_values,use_cache=True,return_dict=True)
    post_arrays,splits=flatten_cache(post,ids.shape[1])
    assert all(same(a,b[:n]) for a,b,n in zip(pre_arrays,post_arrays,splits))
    return pre_arrays,post_arrays,splits


def run_model(name,dtype_name,directory):
    from transformers import AutoModel,AutoTokenizer
    import transformers
    dtype=DTYPES[dtype_name]
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    revision=REVISIONS[name]
    tok=AutoTokenizer.from_pretrained(name,revision=revision,trust_remote_code=False)
    model=AutoModel.from_pretrained(name,revision=revision,torch_dtype=dtype,
                attn_implementation='eager',trust_remote_code=False,use_safetensors=True).eval()
    model.requires_grad_(False)
    assert next(model.parameters()).dtype==dtype
    blocks=model.h if hasattr(model,'h') else model.layers
    parameters_hash=parameters_digest(model)
    record=dict(experiment='QCR-001',model=name,revision=revision,dtype=dtype_name,
                torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,
                parameters_float64_hash=parameters_hash,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cells=[])
    directory.mkdir(parents=True,exist_ok=True)
    stored=[]
    for pi,prompt in enumerate(PROMPTS):
        ids=tok(prompt,return_tensors='pt')['input_ids']; tail=tok(CONTINUATION,return_tensors='pt')['input_ids']
        with torch.no_grad():
            absent=model(ids,use_cache=True,output_hidden_states=True,return_dict=True)
        site=absent.hidden_states[1][0,3].double()
        rms=float(torch.sqrt(torch.mean(site*site)))*.5
        del absent
        for seed in range(3):
            gen=torch.Generator().manual_seed(seed)
            d=torch.randn(site.shape,generator=gen,dtype=torch.float64)
            d=(d*rms/torch.sqrt(torch.mean(d*d))).to(dtype)
            worlds=[run_world(model,blocks,ids,tail,d,a) for a in np.linspace(-1,3,33)]
            repeat=run_world(model,blocks,ids,tail,d,1.)
            no_hook=run_world(model,blocks,ids,tail,d,None)
            assert all(same(a,b) for a,b in zip(worlds[16][1],repeat[1]))
            assert all(same(a,b) for a,b in zip(worlds[8][1],no_hook[1]))
            assert all(same(world[0][k],no_hook[0][k]) for world in worlds for k in (0,1))
            cell=dict(prompt_index=pi,seed=seed,input_ids=ids.tolist(),tail_ids=tail.tolist(),
                      repeat_exact=True,absent_hook_equals_zero=True,old_prefix_invariant=True,
                      source_direction_rms_requested=rms,source_direction_double=d.double().tolist(),tensors=[])
            for index in range(2*len(blocks)):
                # Full persistent write trajectory: prefill followed by complete continuation cache.
                samples=[np.concatenate((w[0][index],w[1][index])) for w in worlds]
                new_start=len(worlds[0][0][index])+worlds[0][2][index]
                stats=oracle(samples,dtype,new_start,witness=(index//2==len(blocks)-1))
                stats.update(layer=index//2,kind='key' if index%2==0 else 'value')
                cell['tensors'].append(stats)
            cell['joint_by_rank']=[]
            for ri,r in enumerate(RANKS):
                exact=[];new_exact=[]
                for j in range(33):
                    exact.append(all(t['ranks'][ri]['per_revision'][j]['byte_identical'] for t in cell['tensors']))
                    new_exact.append(all(t['ranks'][ri]['new_per_revision'][j]['byte_identical'] for t in cell['tensors']))
                cell['joint_by_rank'].append(dict(rank=r,per_revision=exact,new_per_revision=new_exact,
                    nontrivial_exact=sum(v for j,v in enumerate(exact) if j!=16),
                    new_nontrivial_exact=sum(v for j,v in enumerate(new_exact) if j!=16)))
            stored.append((pi,seed,ids,tail,d.double(),worlds[16],worlds[8]))
            record['cells'].append(cell)
            (directory/(name.replace('/','--')+'-'+dtype_name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps(dict(model=name,dtype=dtype_name,prompt=pi,seed=seed,joint=cell['joint_by_rank'][4])),flush=True)
            del worlds
    # Same already-quantized parameters, now lifted losslessly for arithmetic-target control.
    model.to(torch.float64)
    assert parameters_digest(model)==parameters_hash
    for cell,(_,_,ids,tail,d,old,never) in zip(record['cells'],stored):
        controls=[]
        for amplitude,native in [(1.,old),(0.,never)]:
            lifted=run_world(model,blocks,ids,tail,d,amplitude)
            rows=[]
            for i in range(2*len(blocks)):
                a=np.concatenate((native[0][i],native[1][i]))
                b=round_native(np.concatenate((lifted[0][i],lifted[1][i])),dtype)
                rows.append(compare(a,b))
            controls.append(dict(amplitude=amplitude,all_tensors_exact=all(x['byte_identical'] for x in rows),tensors=rows))
        cell['native_vs_lifted_then_cast']=controls
    record['lifted_parameter_hash_exact']=True
    (directory/(name.replace('/','--')+'-'+dtype_name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=list(REVISIONS));p.add_argument('--dtype',choices=list(DTYPES),default='bfloat16')
    p.add_argument('--output-dir',type=Path,default=Path('qcr001-results'))
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.model:
        run_model(args.model,args.dtype,args.output_dir)
    else:
        result=dict(experiment='QCR-001',python=platform.python_version(),scalar=scalar_screen())
        (args.output_dir/'scalar.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
