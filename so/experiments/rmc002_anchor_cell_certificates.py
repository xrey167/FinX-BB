"""RMC002: source-anchored native-cell reuse inside a dense dependency cone.

Exact integer/ties-to-even operator assay. The operator dimensions, edit path,
and primary sparse-delta baseline are inherited unchanged from RMC001.
This is not trained-model evidence or a novelty claim.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from so.experiments import rmc001_rounding_margin_certificates as R

# Conservative cost: subtract + abs + accumulate per source coordinate.
ANCHOR_COORD_COST = 3


@dataclass
class AnchorBlock:
    start: int
    end: int
    max_abs_active_weight: int
    safe_radius: int


@dataclass
class AnchorLayer:
    source_anchor: np.ndarray
    blocks: list[AnchorBlock]
    trusted_digest: str
    refreshes: int = 0


@dataclass
class AnchorState:
    source: np.ndarray
    h: list[np.ndarray]
    active_z: list[np.ndarray]
    layers: list[AnchorLayer]


def _digest_layer(layer: AnchorLayer) -> str:
    payload = dict(anchor=layer.source_anchor.tolist(), blocks=[
        dict(start=b.start,end=b.end,maxw=b.max_abs_active_weight,radius=b.safe_radius)
        for b in layer.blocks])
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _make_layer(model: R.Model, layer: int, source_anchor: np.ndarray,
                protected_z: np.ndarray, protected_h: np.ndarray) -> AnchorLayer:
    blocks=[]
    for start in range(R.A,R.D,R.BLOCK):
        end=min(R.D,start+R.BLOCK)
        radius=min(R.safe_radius(int(z),int(h)) for z,h in
                   zip(protected_z[start-R.A:end-R.A],protected_h[start-R.A:end-R.A]))
        maxw=int(np.max(np.abs(model.weights[layer][start:end,:R.A])))
        blocks.append(AnchorBlock(start,end,maxw,radius))
    item=AnchorLayer(source_anchor.copy(),blocks,'')
    item.trusted_digest=_digest_layer(item)
    return item


def init_anchor_state(model: R.Model) -> AnchorState:
    layers=[]
    zero=model.old_source.copy()
    for layer in range(R.L):
        layers.append(_make_layer(model,layer,zero,
                                  model.old_z[layer][R.A:],model.old_h[layer][R.A:]))
    state=AnchorState(zero.copy(),[x.copy() for x in model.old_h],
                      [z[:R.A].copy() for z in model.old_z],layers)
    assert verify_all_layers(model,state)
    return state


def verify_layer(layer: AnchorLayer) -> bool:
    return layer.trusted_digest == _digest_layer(layer)


def verify_all_layers(model: R.Model, state: AnchorState) -> bool:
    """Integrity plus exact initial-certificate check where source anchor is old source.

    Runtime refreshed certificates are integrity-bound by trusted_digest. The
    initialized certificates are additionally recomputed from the model arrays.
    This is an assay verifier, not production authentication.
    """
    for i,layer in enumerate(state.layers):
        if not verify_layer(layer): return False
        if np.array_equal(layer.source_anchor,model.old_source):
            expected=_make_layer(model,i,model.old_source,
                                 model.old_z[i][R.A:],model.old_h[i][R.A:])
            if _digest_layer(expected)!=layer.trusted_digest: return False
    return True


def _displacement_cost_and_l1(current: np.ndarray, anchor: np.ndarray):
    delta=current-anchor
    return ANCHOR_COORD_COST*len(delta), int(np.sum(np.abs(delta),dtype=np.int64))


def anchor_edit(model: R.Model,state: AnchorState,new_source: np.ndarray):
    prev_old=model.input_background.copy(); prev_old[:R.A]=state.source
    prev_new=model.input_background.copy(); prev_new[:R.A]=new_source
    hs=[]; active_zs=[]
    ops=0; refresh_ops=0; refreshed_layers=0; refreshed_blocks=0
    parent_protected_changed=False
    layer_rows=[]
    for layer_idx in range(R.L):
        old_h=state.h[layer_idx]
        h=old_h.copy()
        changed=np.flatnonzero(prev_new!=prev_old)
        delta=prev_new[changed]-prev_old[changed]
        az=state.active_z[layer_idx].copy()
        if len(changed):
            az += model.weights[layer_idx][:R.A,changed] @ delta
            ops += R.A*len(changed)
        h[:R.A]=R.qround(az)

        cert=state.layers[layer_idx]
        if not verify_layer(cert):
            raise ValueError('certificate integrity failure')
        displacement_cost,l1=_displacement_cost_and_l1(h[:R.A],cert.source_anchor)
        ops += displacement_cost

        failed=parent_protected_changed
        if not failed:
            for block in cert.blocks:
                ops += 1
                bound=block.max_abs_active_weight*l1
                if bound > block.safe_radius:
                    failed=True
                    break
        # If one block cannot be certified under the shared layer anchor, refresh
        # the entire protected layer and create a single new anchor epoch.
        if failed:
            zprot=model.weights[layer_idx][R.A:] @ prev_new + model.bias[layer_idx][R.A:]
            fresh=R.qround(zprot)
            h[R.A:]=fresh
            rows=R.D-R.A
            refresh_ops += rows*R.D
            refreshed_layers += 1
            refreshed_blocks += len(cert.blocks)
            new_cert=_make_layer(model,layer_idx,h[:R.A],zprot,fresh)
            new_cert.refreshes=cert.refreshes+1
            state.layers[layer_idx]=new_cert
        protected_changed=not R.exact_equal(h[R.A:],old_h[R.A:])
        layer_rows.append(dict(layer=layer_idx,failed_shared_anchor=failed,
                               protected_changed=protected_changed,l1_from_anchor=l1,
                               refreshed=failed))
        parent_protected_changed=protected_changed
        hs.append(h.copy()); active_zs.append(az.copy())
        prev_old,prev_new=old_h,h
    state.source=new_source.copy(); state.h=hs; state.active_z=active_zs
    return dict(ops=ops+refresh_ops,refresh_ops=refresh_ops,
                refreshed_layers=refreshed_layers,refreshed_blocks=refreshed_blocks,
                layers=layer_rows)


def main_path(seed:int):
    model=R.build_model(seed)
    sparse=R.init_sparse(model); candidate=init_anchor_state(model)
    sparse_ops=candidate_ops=dense_ops=0
    rows=[]
    for index,target in enumerate(R.edit_path(seed)):
        fresh,_=R.full_rebuild(model,target)
        sop,_=R.sparse_edit(model,sparse,target)
        c=anchor_edit(model,candidate,target)
        assert all(R.exact_equal(a,b) for a,b in zip(sparse.h,fresh))
        assert all(R.exact_equal(a,b) for a,b in zip(candidate.h,fresh))
        sparse_ops+=sop; candidate_ops+=c['ops']; dense_ops+=R.L*R.D*R.D
        rows.append(dict(edit=index,source=target.tolist(),sparse_ops=sop,
                         candidate_ops=c['ops'],refreshed_layers=c['refreshed_layers'],
                         every_write_exact=True))
    ratio=sparse_ops/candidate_ops
    return dict(seed=seed,sparse_ops=sparse_ops,candidate_ops=candidate_ops,dense_ops=dense_ops,
                sparse_over_candidate=ratio,dense_over_candidate=dense_ops/candidate_ops,
                total_refreshed_layers=sum(x['refreshed_layers'] for x in rows),edits=rows,
                candidate_aux_scalars=R.L*R.A + R.L*((R.D-R.A+R.BLOCK-1)//R.BLOCK)*2 + R.L*R.A,
                sparse_exact_preactivation_scalars=R.L*R.D)


def large_refresh_control(seed:int):
    model=R.build_model(seed); state=init_anchor_state(model)
    large=np.full(R.A,30,dtype=np.int64)
    fresh,_=R.full_rebuild(model,large)
    first=anchor_edit(model,state,large)
    assert first['refreshed_layers']>=1
    assert all(R.exact_equal(a,b) for a,b in zip(state.h,fresh))
    # One small follow-up after the new anchor must remain exact.
    follow=large.copy(); follow[0]+=1
    fresh2,_=R.full_rebuild(model,follow)
    second=anchor_edit(model,state,follow)
    assert all(R.exact_equal(a,b) for a,b in zip(state.h,fresh2))
    return dict(first_refreshed_layers=first['refreshed_layers'],
                first_refresh_ops=first['refresh_ops'],follow_refreshed_layers=second['refreshed_layers'],
                exact=True)


def leaky_control(seed:int):
    model=R.build_model(seed,leaky=True); state=init_anchor_state(model)
    target=np.ones(R.A,dtype=np.int64)
    fresh,_=R.full_rebuild(model,target)
    cell=anchor_edit(model,state,target)
    assert all(R.exact_equal(a,b) for a,b in zip(state.h,fresh))
    return dict(refreshed_layers=cell['refreshed_layers'],refresh_ops=cell['refresh_ops'],
                exact=True,candidate_ops=cell['ops'],dense_ops=R.L*R.D*R.D,
                candidate_over_dense=cell['ops']/(R.L*R.D*R.D))


def edit_revert_control(seed:int):
    model=R.build_model(seed); state=init_anchor_state(model)
    one=np.zeros(R.A,dtype=np.int64); one[0]=1
    anchor=np.zeros(R.A,dtype=np.int64)
    c1=anchor_edit(model,state,one); c2=anchor_edit(model,state,anchor)
    assert c1['refreshed_layers']==0 and c2['refreshed_layers']==0
    assert all(x['l1_from_anchor']==0 for x in c2['layers'])
    fresh,_=R.full_rebuild(model,anchor)
    assert all(R.exact_equal(a,b) for a,b in zip(state.h,fresh))
    return dict(return_displacement_zero=True,refreshes=0,every_write_exact=True)


def tamper_control(seed:int):
    model=R.build_model(seed); state=init_anchor_state(model)
    assert verify_all_layers(model,state)
    state.layers[0].blocks[0].safe_radius += 1
    radius_detected=not verify_all_layers(model,state)
    state=init_anchor_state(model); state.layers[0].blocks[0].max_abs_active_weight -= 1
    sensitivity_detected=not verify_all_layers(model,state)
    state=init_anchor_state(model); state.layers[0].source_anchor[0]+=1
    anchor_detected=not verify_all_layers(model,state)
    assert radius_detected and sensitivity_detected and anchor_detected
    return dict(radius_detected=radius_detected,sensitivity_detected=sensitivity_detected,
                anchor_detected=anchor_detected)


def benchmark(seed:int,rounds=7):
    model=R.build_model(seed); path=R.edit_path(seed)[:8]
    def candidate():
        s=init_anchor_state(model)
        for p in path: anchor_edit(model,s,p)
    def sparse():
        s=R.init_sparse(model)
        for p in path: R.sparse_edit(model,s,p)
    out={}
    for name,fn in [('candidate',candidate),('sparse',sparse)]:
        fn(); samples=[]
        for _ in range(rounds):
            t=time.perf_counter_ns(); fn(); samples.append(time.perf_counter_ns()-t)
        out[name+'_ns']=float(np.median(samples))
    out['sparse_over_candidate_wall']=out['sparse_ns']/out['candidate_ns']
    out['wall_not_application_gate']=True
    return out


def run_seed(seed:int):
    main=main_path(seed)
    assert main['sparse_over_candidate']>=10.0,main['sparse_over_candidate']
    assert main['total_refreshed_layers']==0
    return dict(seed=seed,main=main,large=large_refresh_control(seed),
                leaky=leaky_control(seed),revert=edit_revert_control(seed),
                tamper=tamper_control(seed),benchmark=benchmark(seed))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--output',type=Path,default=Path('rmc002-results.json'))
    a=p.parse_args()
    result=dict(experiment='RMC-002',status='candidate_operator_screen_not_invention',
                preregistered_gate=10.0,cost_anchor_coordinate_factor=ANCHOR_COORD_COST,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                python=platform.python_version(),numpy=np.__version__,
                dimensions=dict(width=R.D,depth=R.L,active=R.A,q=R.Q,block=R.BLOCK,edits=R.EDITS),
                seeds=[run_seed(s) for s in a.seeds],trained_backbones=0,
                full_system_gates='NOT_EVALUATED')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(seed=r['seed'],ratio=r['main']['sparse_over_candidate'],
                               candidate_ops=r['main']['candidate_ops'],sparse_ops=r['main']['sparse_ops'],
                               large_refresh=r['large']['first_refreshed_layers'],
                               leaky_refresh=r['leaky']['refreshed_layers'],wall=r['benchmark']['sparse_over_candidate_wall'])
                      for r in result['seeds']],indent=2))

if __name__=='__main__': main()
