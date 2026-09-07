"""FIR001: structurally finite influence horizon versus persistent recall.

This is an exact integer/Fraction architecture-boundary assay, not a trained model.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from fractions import Fraction
from pathlib import Path
import numpy as np

K=4; B=128; ZW=32; STEPS=80


def zero_preserving(x):
    return x*x*x + 3*x


def bg_step(b, seed, t):
    # Source-independent, bounded integer recurrence. Permute + affine exogenous drive.
    shift=(seed*7+t*3)%B
    out=np.roll(b,shift)
    drive=((np.arange(B,dtype=object)+1)*(seed+1)*(t+1))%17-8
    return out+drive


def source_step(z, injected=None):
    out=[np.zeros(ZW,dtype=object) for _ in range(K)]
    if injected is not None:
        out[0]=np.array(injected,dtype=object)
    for i in range(1,K):
        out[i]=np.array([zero_preserving(v) for v in z[i-1]],dtype=object)
    return out


def readout(b,z):
    s=sum(sum(int(v) for v in stage) for stage in z)
    # nonlinear source-dependent observable while z is alive
    return int(sum(int(v) for v in b[:8])) + s + s*s


def run_world(seed,payload,inject_at=0,query_reread_at=None,leaky=False):
    b=np.zeros(B,dtype=object); z=[np.zeros(ZW,dtype=object) for _ in range(K)]
    states=[]; reads=[]
    payload=np.array(payload,dtype=object)
    for t in range(STEPS):
        inject = payload if t==inject_at or (query_reread_at is not None and t==query_reread_at) else None
        z=source_step(z,inject)
        b=bg_step(b,seed,t)
        if leaky:
            leak=sum(int(v) for v in z[-1])
            b=b.copy(); b[0]+=leak
        states.append((b.copy(),[x.copy() for x in z]))
        reads.append(readout(b,z))
    return states,reads


def equal_state(a,b):
    return np.array_equal(a[0],b[0]) and all(np.array_equal(x,y) for x,y in zip(a[1],b[1]))


def payload(seed,which):
    rng=np.random.default_rng(1000+seed*11+which)
    return rng.integers(-3,4,size=ZW,dtype=np.int64).astype(object)


def benchmark(fn,rounds=101):
    vals=[]
    for _ in range(7): fn()
    for _ in range(rounds):
        t=time.perf_counter_ns(); fn(); vals.append(time.perf_counter_ns()-t)
    return float(np.median(vals))


def fraction_control(seed,p):
    # Separate exact scalar recurrence proving stage support vanishes after K shifts.
    stages=[Fraction(0) for _ in range(K)]
    history=[]
    for t in range(K+2):
        new=[Fraction(0) for _ in range(K)]
        if t==0: new[0]=Fraction(int(p[0]))
        for i in range(1,K): new[i]=stages[i-1]**3+3*stages[i-1]
        stages=new; history.append(tuple(stages))
    assert any(any(v for v in h) for h in history[:K])
    assert all(v==0 for v in history[K])
    return [list(map(str,h)) for h in history]


def materialize_current_source(background,p):
    # Candidate and strongest reread baseline are deliberately given exactly
    # the same canonical payload access and source-channel operator budget.
    z=[np.zeros(ZW,dtype=object) for _ in range(K)]
    z=source_step(z,p)
    return readout(background,z)


def seed_case(seed):
    pa,pb=payload(seed,0),payload(seed,1)
    a,ra=run_world(seed,pa)
    b,rb=run_world(seed,pb)
    first_equal=next(i for i,(x,y) in enumerate(zip(a,b)) if equal_state(x,y))
    assert first_equal==K
    assert all(equal_state(a[i],b[i]) for i in range(K,STEPS))
    assert any(ra[i]!=rb[i] for i in range(K))
    assert ra[64]==rb[64]

    # canonical reread gives current payload-dependent late recall
    ar,rar=run_world(seed,pa,query_reread_at=64)
    br,rbr=run_world(seed,pb,query_reread_at=64)
    assert rar[64]!=rbr[64]
    updated,ru=run_world(seed,pb,query_reread_at=64)
    assert rbr==ru

    # leaky control retains long-term source influence and therefore fails finite horizon
    al,_=run_world(seed,pa,leaky=True)
    bl,_=run_world(seed,pb,leaky=True)
    leaky_equal=[i for i,(x,y) in enumerate(zip(al,bl)) if equal_state(x,y)]
    assert not leaky_equal

    # Strongest conventional baseline gets identical current-source access and computation.
    def candidate_query():
        return materialize_current_source(a[64][0],pa)
    def baseline_query():
        return materialize_current_source(a[64][0],pa)
    assert candidate_query()==baseline_query()==rar[64]

    t_full=benchmark(lambda: run_world(seed,pb)[0],31)
    t_reread=benchmark(baseline_query,201)
    t_candidate=benchmark(candidate_query,201)
    return dict(seed=seed,first_complete_coalescence_write=first_equal,
        complete_state_equal_through_end=all(equal_state(a[i],b[i]) for i in range(K,STEPS)),
        late_no_reread_source_distinguishable=ra[64]!=rb[64],
        late_reread_source_distinguishable=rar[64]!=rbr[64],
        leaky_control_ever_coalesces=bool(leaky_equal),
        finite_horizon_state_words=B+K*ZW,
        canonical_payload_words=ZW,
        finite_window_event_words=K*ZW,
        candidate_query_ns=t_candidate, conventional_reread_query_ns=t_reread,
        full_80_step_replay_ns=t_full,
        conventional_over_candidate=t_reread/t_candidate,
        full_replay_over_candidate_query=t_full/t_candidate,
        fraction_support_history=fraction_control(seed,pa))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=Path('fir001-results.json'))
    a=ap.parse_args(); rows=[seed_case(s) for s in range(5)]
    result=dict(experiment='FIR-001',status='finite_horizon_boundary_not_invention',K=K,
        dimensions=dict(background=B,source_stage=ZW,stages=K,steps=STEPS),rows=rows,
        interpretation='Exact bounded source extinction is compatible with long-term source recall only if source-dependent information is retained/reinjected elsewhere. Canonical reread then collapses to late binding and matches the conventional baseline.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),trained_backbones=0,
        full_system_gates='NOT_EVALUATED')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('seed','first_complete_coalescence_write','late_no_reread_source_distinguishable','late_reread_source_distinguishable','conventional_over_candidate')} for r in rows],indent=2))
if __name__=='__main__': main()
