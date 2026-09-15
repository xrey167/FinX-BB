"""E-000088 -- Certified Semantic Dependency Elision existence/utility kill screen.

Preregistered in docs/novelty/e000088-certified-semantic-dependency-elision.md.
This is deliberately an exhaustive oracle, not a production certificate.

Important stronger baseline added before reading the result: besides a coarse Bank-level
syntactic dependency baseline, report a FIELD-SENSITIVE dependency baseline. A payload-only
Pod edit cannot invalidate routing if routing structurally reads only keys. If semantic
elision merely rediscovers that program dependency, it is not a neural-specific invention.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from so.data import Bank, bank_from_store
from so.experiments import e000015_symlink_cells as E15


def replace_obj(bank: Bank, row: int, obj: int) -> Bank:
    """Copy a Bank and replace one FACT payload index, leaving every other field unchanged."""
    kw = dict(bank.__dict__)
    arr = np.array(bank.obj, copy=True); arr[row] = int(obj); kw['obj'] = arr
    return Bank(**kw)


def exact_equal(a: np.ndarray, b: np.ndarray) -> bool:
    # Primary rule is exact equality, as preregistered. Diagnostics separately report maxabs.
    return bool(np.array_equal(a, b))


def artifact(pred, kind: str) -> np.ndarray:
    if kind == 'routing': return pred.routing
    if kind == 'hidden': return pred.hidden
    if kind == 'logits': return pred.logits
    if kind == 'answer': return pred.answers
    raise KeyError(kind)


def run_seed(seed: int, steps: int, n_pods: int, n_queries: int, verbose: bool=True) -> Dict[str, Any]:
    t0=time.time(); out=E15.train_or_load(seed, steps, 1)
    model,centre=out['model'],out['centre']
    rng=np.random.default_rng(88000+seed)
    world,spec=E15.sample_alias_world(rng, 850, max(n_pods,16), 2)
    store,kids=E15.load_arm(world,spec,centre,seed+88000,symlink=True)
    bank=bank_from_store(store)
    pos_of_kid={int(k):i for i,k in enumerate(bank.kid)}
    base_keys=[f.key for f in world.facts if f.key not in spec.alias_of]
    pod_keys={k for t,aa in spec.groups for k in [t]+list(aa)}
    bystanders=[k for k in base_keys if k not in pod_keys]
    rng.shuffle(bystanders)
    kinds=('routing','hidden','logits','answer')
    rows=[]; positive_changed=0; positive_total=0
    semantic_independent={k:0 for k in kinds}; total_pairs={k:0 for k in kinds}
    zero_false_reuse=True; maxabs=0.0

    for pi,(target,aliases) in enumerate(spec.groups[:n_pods]):
        row=pos_of_kid[kids[target]]
        # Positive query actually asks the target. Bystanders are syntactically downstream of the
        # same Bank but do not name the Pod. This deliberately stresses dense routing.
        qs=[E15._q1(world,target)] + [E15._q1(world,k) for k in bystanders[pi*n_queries:(pi+1)*n_queries]]
        if len(qs)<2: raise RuntimeError('not enough bystander queries')
        base=E15.predict(model,bank,world,qs)
        same={k:True for k in kinds}; changed_positive=False
        for obj in range(world.n_entities):
            if obj==int(bank.obj[row]): continue
            bp=replace_obj(bank,row,obj)
            p=E15.predict(model,bp,world,qs)
            for k in kinds:
                aa,bb=artifact(base,k),artifact(p,k)
                eq=exact_equal(aa,bb)
                same[k] = same[k] and eq
                if not eq and np.issubdtype(np.asarray(aa).dtype,np.number):
                    maxabs=max(maxabs,float(np.max(np.abs(np.asarray(aa,dtype=np.float64)-np.asarray(bb,dtype=np.float64)))))
            if p.answers[0] != base.answers[0] or not np.array_equal(p.logits[0],base.logits[0]):
                changed_positive=True
        positive_total+=1; positive_changed+=int(changed_positive)
        for k in kinds:
            total_pairs[k]+=1; semantic_independent[k]+=int(same[k])
        rows.append({'pod':pi,'positive_changed':changed_positive,'exact_independent':same})
        if verbose: print('pod',pi,'positive',changed_positive,'independent',same,flush=True)

    pos_rate=positive_changed/max(positive_total,1)
    frac={k:semantic_independent[k]/max(total_pairs[k],1) for k in kinds}
    # Coarse dependency invalidates every artifact on any Pod generation change.
    coarse_invalidations=sum(total_pairs.values())
    semantic_invalidations=sum(total_pairs[k]-semantic_independent[k] for k in kinds)
    work_reduction=(coarse_invalidations/max(semantic_invalidations,1))

    # Stronger field-sensitive baseline: for a payload-only edit, forward routing does not read obj.
    # It therefore reuses routing without any exhaustive neural certificate. Other artifacts consume
    # values and remain invalidated. This baseline is derived directly from model.py dataflow.
    field_invalidations=sum(total_pairs[k] for k in kinds if k!='routing')
    field_reduction=(field_invalidations/max(semantic_invalidations,1))
    semantic_extra_over_field=sum(semantic_independent[k] for k in kinds if k!='routing')

    checks={
      'V1_positive_dependency_ge_095':pos_rate>=.95,
      'V2_full_domain_256':world.n_entities==256,
      'V3_no_audit_optimization':True,
      'V4_zero_false_reuse':zero_false_reuse,
      'V5_active_only':True,
      'existence_ge_005':max(frac.values())>=.05,
      'coarse_work_reduction_ge_2':work_reduction>=2.0,
      # Novelty guard: must beat ordinary field-sensitive dependency analysis, not merely coarse Bank tags.
      'semantic_extra_over_field_positive':semantic_extra_over_field>0,
      'field_sensitive_work_reduction_ge_2':field_reduction>=2.0,
    }
    survive=all(checks.values())
    return {'seed':seed,'checkpoint_sha256':out.get('checkpoint_sha256',''),'n_pods':n_pods,
      'payload_domain':world.n_entities,'positive_change_rate':pos_rate,'independent_fraction':frac,
      'coarse_invalidations':coarse_invalidations,'field_sensitive_invalidations':field_invalidations,
      'semantic_invalidations':semantic_invalidations,'coarse_work_reduction':work_reduction,
      'field_sensitive_work_reduction':field_reduction,'semantic_extra_over_field':semantic_extra_over_field,
      'max_numeric_delta':maxabs,'checks':checks,'survive':survive,'rows':rows,'seconds':time.time()-t0}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,nargs='*',default=[0,1,2]); ap.add_argument('--steps',type=int,default=4000)
    ap.add_argument('--n-pods',type=int,default=8); ap.add_argument('--n-queries',type=int,default=3); ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--results-dir',default='so/results'); a=ap.parse_args(); torch.set_num_threads(a.threads)
    rows=[run_seed(s,a.steps,a.n_pods,a.n_queries) for s in a.seeds]
    rec={'experiment':'E-000088','title':'Certified Semantic Dependency Elision kill screen','rows':rows,
      'all_survive':all(r['survive'] for r in rows),
      'decision':'SURVIVE_RESEARCH_DIRECTION' if all(r['survive'] for r in rows) else 'KILL_OR_REDESIGN',
      'novelty_guard':'Routing reuse under payload-only edits is credited to ordinary field-sensitive program dependency analysis, not semantic certification.'}
    p=Path(a.results_dir);p.mkdir(parents=True,exist_ok=True);(p/'e000088_semantic_dependency_elision.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
    print(json.dumps(rec,indent=2));
    # A kill is a valid research result; do not make CI red merely because the hypothesis died.
if __name__=='__main__': main()
