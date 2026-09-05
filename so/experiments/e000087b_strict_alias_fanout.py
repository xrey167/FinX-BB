"""E-000087B — alias fan-out utility rerun under the strict E87 marker contract.

No novelty claim. This exists only to determine whether E-000086's semantic failures were caused
by the train/eval marker-contract bug. Canonical indirection itself is prior art.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Any
import numpy as np
from so.data import bank_from_store
from so.experiments import e000015_symlink_cells as E15
from so.experiments.e000087_marker_validity_contract import install_strict_contract
from so.train import make_centre


def run(seed:int, fanout:int, rounds:int=11)->dict[str,Any]:
    install_strict_contract()
    rng=np.random.default_rng(870087+seed*1009+fanout)
    centre=make_centre(seed,32)
    world,spec=E15.sample_alias_world(rng,256,8,fanout,2048,6,2)
    ss,sk=E15.load_arm(world,spec,centre,seed+1000,True)
    ds,dk=E15.load_arm(world,spec,centre,seed+1000,False)
    target,aliases=spec.groups[0]; old=int(world.index[target]); new=(old+17)%world.n_entities
    ss.update(sk[target],new); sb=bank_from_store(ss)
    symlink_update=sb.index_view.get(target)==new and all(sb.index_view.get(a)==new for a in aliases)
    ds.update(dk[target],new); db1=bank_from_store(ds)
    duplicate_one_does_not_propagate=all(db1.index_view.get(a)==old for a in aliases)
    for a in aliases: ds.update(dk[a],new)
    db=bank_from_store(ds)
    duplicate_full=all(db.index_view.get(k)==new for k in [target,*aliases])
    ss.rollback(sk[target],1)
    for k in [target,*aliases]: ds.rollback(dk[k],1)
    rb1=bank_from_store(ss); rb2=bank_from_store(ds)
    rollback=all(rb1.index_view.get(k)==old for k in [target,*aliases]) and all(rb2.index_view.get(k)==old for k in [target,*aliases])

    # Independent fresh stores for SHRED.
    s2,k2=E15.load_arm(world,spec,centre,seed+2000,True)
    d2,j2=E15.load_arm(world,spec,centre,seed+2000,False)
    s2.shred(k2[target]); sh=bank_from_store(s2)
    symlink_shred=target not in sh.index_view and all(a not in sh.index_view for a in aliases)
    d2.shred(j2[target]); dh1=bank_from_store(d2)
    duplicate_one_shred_does_not_propagate=all(dh1.index_view.get(a)==old for a in aliases)
    for a in aliases: d2.shred(j2[a])
    dh=bank_from_store(d2)
    duplicate_full_shred=all(k not in dh.index_view for k in [target,*aliases])

    us,ud,ssn,sdn=[],[],[],[]
    for r in range(rounds):
        rr=np.random.default_rng(880087+seed*10007+fanout*97+r)
        w,sp=E15.sample_alias_world(rr,256,8,fanout,2048,6,2); c=make_centre(seed+r,32)
        a,ak=E15.load_arm(w,sp,c,seed+r+3000,True); b,bk=E15.load_arm(w,sp,c,seed+r+3000,False)
        t,aa=sp.groups[0]; nv=(int(w.index[t])+19)%w.n_entities
        x=time.perf_counter_ns(); a.update(ak[t],nv); us.append(time.perf_counter_ns()-x)
        x=time.perf_counter_ns(); b.update(bk[t],nv); [b.update(bk[q],nv) for q in aa]; ud.append(time.perf_counter_ns()-x)
        a,ak=E15.load_arm(w,sp,c,seed+r+4000,True); b,bk=E15.load_arm(w,sp,c,seed+r+4000,False)
        x=time.perf_counter_ns(); a.shred(ak[t]); ssn.append(time.perf_counter_ns()-x)
        x=time.perf_counter_ns(); b.shred(bk[t]); [b.shred(bk[q]) for q in aa]; sdn.append(time.perf_counter_ns()-x)
    med=lambda z:float(np.median(z))
    checks={"symlink_update":symlink_update,"duplicate_single_update_stays_local":duplicate_one_does_not_propagate,
            "duplicate_full_update":duplicate_full,"rollback":rollback,"symlink_shred":symlink_shred,
            "duplicate_single_shred_stays_local":duplicate_one_shred_does_not_propagate,"duplicate_full_shred":duplicate_full_shred}
    return {"seed":seed,"fanout":fanout,"checks":checks,"pass":all(checks.values()),
            "operation_counts":{"symlink":1,"duplicate":1+fanout},
            "median_ns":{"symlink_update":med(us),"duplicate_update":med(ud),"symlink_shred":med(ssn),"duplicate_shred":med(sdn)},
            "ratios":{"update_duplicate_over_symlink":med(ud)/max(med(us),1),"shred_duplicate_over_symlink":med(sdn)/max(med(ssn),1)}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='*',default=[0,1,2,3,4]);p.add_argument('--fanouts',type=int,nargs='*',default=[1,2,4,8,16,32,64]);p.add_argument('--rounds',type=int,default=11);p.add_argument('--results-dir',default='so/results');a=p.parse_args()
    rows=[run(s,f,a.rounds) for f in a.fanouts for s in a.seeds]
    rec={"experiment":"E-000087B","result":"utility_semantics_pass" if all(x['pass'] for x in rows) else "still_falsified","rows":rows,
         "breakthrough":False,"novelty_claim":False,"marker_radius":0.35,
         "interpretation":"Store-level utility only. No neural, CAVI, J-space, locality or novelty inference."}
    d=Path(a.results_dir);d.mkdir(parents=True,exist_ok=True);(d/'e000087b_strict_alias_fanout.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
    if not all(x['pass'] for x in rows): raise SystemExit(2)
if __name__=='__main__':main()
