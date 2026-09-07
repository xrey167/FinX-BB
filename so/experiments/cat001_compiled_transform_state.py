"""CAT001: exact editable sufficient state in an associative transform algebra.

Constructive baseline screen, NOT a trained neural-memory or novelty claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

P = 2_147_483_647  # prime
I = (1, 0, 0, 1)


def mm(a, b):
    """2x2 product a @ b over the prime field."""
    a0,a1,a2,a3=a; b0,b1,b2,b3=b
    return ((a0*b0+a1*b2)%P, (a0*b1+a1*b3)%P,
            (a2*b0+a3*b2)%P, (a2*b1+a3*b3)%P)


def combine(left, right):
    """Canonical source order: left sources act first, then right sources."""
    return mm(right, left)


def det(m):
    return (m[0]*m[3]-m[1]*m[2])%P


def inv(m):
    z=det(m)
    if not z:
        raise ValueError("singular")
    q=pow(z,P-2,P)
    return (m[3]*q%P, -m[1]*q%P, -m[2]*q%P, m[0]*q%P)


def mix64(x):
    x &= (1<<64)-1
    x ^= x >> 30; x = (x * 0xbf58476d1ce4e5b9) & ((1<<64)-1)
    x ^= x >> 27; x = (x * 0x94d049bb133111eb) & ((1<<64)-1)
    x ^= x >> 31
    return x


def encode(source_id, payload):
    """Controlled algebraic stand-in for a learned Pod->transform encoder."""
    x=mix64((source_id+1)*0x9e3779b97f4a7c15 ^ (payload+17)*0xd6e8feb86659fd93)
    vals=[]
    for j in range(4):
        x=mix64(x+j+0x9e3779b97f4a7c15)
        vals.append(1 + x % (P-1))
    m=tuple(vals)
    if det(m)==0:
        m=(m[0],m[1],m[2],(m[3]+1)%P or 1)
    assert det(m)
    return m


def project(m, x):
    den=(m[2]*x+m[3])%P
    if not den:
        raise ZeroDivisionError
    return ((m[0]*x+m[1])%P)*pow(den,P-2,P)%P


def fold(leaves):
    count=0
    out=I
    for m in leaves:
        out=mm(m,out)  # apply next source after the accumulated prefix
        count+=1
    # Do not charge identity initialization as a transform composition.
    return out, max(0,count-1)


class ProductTree:
    def __init__(self, leaves):
        n=len(leaves)
        if n<1 or n&(n-1):
            raise ValueError("power-of-two leaves required")
        self.n=n
        self.t=[I]*(2*n)
        self.t[n:]=list(leaves)
        self.build_calls=0
        for i in range(n-1,0,-1):
            self.t[i]=combine(self.t[2*i],self.t[2*i+1])
            self.build_calls+=1

    @property
    def root(self): return self.t[1]

    def update(self, pos, value):
        i=self.n+pos; self.t[i]=value; calls=0
        i//=2
        while i:
            self.t[i]=combine(self.t[2*i],self.t[2*i+1]); calls+=1; i//=2
        return calls

    def clone(self):
        other=object.__new__(ProductTree); other.n=self.n; other.t=self.t.copy(); other.build_calls=self.build_calls
        return other


class CompiledMemory:
    def __init__(self, payloads):
        self.payloads=list(payloads); self.generation=[0]*len(payloads); self.epoch=0
        self.tree=ProductTree([encode(i,p) for i,p in enumerate(payloads)])

    def snapshot(self):
        return self.tree.root, self.epoch

    def consume(self, snap, queries):
        root,epoch=snap
        if epoch!=self.epoch:
            raise RuntimeError("stale generation epoch")
        return query_outputs(root,queries)

    def edit(self,pos,payload=None,revoke=False):
        old=self.payloads[pos]
        self.generation[pos]+=1; self.epoch+=1
        if revoke:
            value=I
        else:
            if payload is None: raise ValueError
            self.payloads[pos]=payload; value=encode(pos,payload)
        return old,self.tree.update(pos,value)


def query_outputs(root, xs):
    out=[]
    for x in xs:
        try: out.append(project(root,x))
        except ZeroDivisionError: out.append(None)
    return out


def safe_queries(roots, count=64):
    xs=[]; x=2
    while len(xs)<count:
        if all((m[2]*x+m[3])%P for m in roots): xs.append(x)
        x+=1
    return xs


def interaction_control(leaves, i):
    j=i+1
    base,_=fold(leaves)
    li=leaves.copy(); li[i]=encode(i,999_001)
    lj=leaves.copy(); lj[j]=encode(j,999_002)
    lij=li.copy(); lij[j]=lj[j]
    ri,_=fold(li); rj,_=fold(lj); rij,_=fold(lij)
    xs=safe_queries([base,ri,rj,rij],16)
    interactions=[]
    for x in xs:
        y0,yi,yj,yij=[project(r,x) for r in (base,ri,rj,rij)]
        interactions.append((yij-yi-yj+y0)%P)
    swapped=leaves.copy(); swapped[i],swapped[j]=swapped[j],swapped[i]
    rs,_=fold(swapped)
    sx=safe_queries([base,rs],16)
    return dict(nonzero_projective_interactions=sum(v!=0 for v in interactions),
                adjacent_swap_changes_root=rs!=base,
                adjacent_swap_changes_queries=any(project(base,x)!=project(rs,x) for x in sx))


def naive_patch_control(leaves,pos,new_value):
    oldroot,_=fold(leaves); fresh=leaves.copy(); old=leaves[pos]; fresh[pos]=new_value; target,_=fold(fresh)
    # Deliberately context-free O(1) removal/replacement. Correct only in special positions/commuting cases.
    guess=mm(mm(oldroot,inv(old)),new_value)
    return dict(naive_equals_fresh=guess==target, oldroot=oldroot, guess=guess, fresh=target)


def median_ns(fn, rounds=101):
    for _ in range(5): fn()
    xs=[]
    for _ in range(rounds):
        t=time.perf_counter_ns(); fn(); xs.append(time.perf_counter_ns()-t)
    return int(statistics.median(xs))


def seed_size(seed,n,updates=32):
    rng=random.Random(100_000+seed*10_000+n)
    payloads=[rng.randrange(1,10_000_000) for _ in range(n)]
    leaves=[encode(i,p) for i,p in enumerate(payloads)]
    fullroot,fullcalls=fold(leaves)
    candidate=ProductTree(leaves); baseline=ProductTree(leaves)
    assert candidate.root==baseline.root==fullroot
    rows=[]
    ratios=[]
    first_stale=None
    for u in range(updates):
        pos=rng.randrange(1,n-1); newpayload=rng.randrange(10_000_001,20_000_000); new=encode(pos,newpayload)
        old_snapshot=(candidate.root,u)
        cc=candidate.update(pos,new); bc=baseline.update(pos,new)
        leaves[pos]=new
        fresh,fc=fold(leaves)
        assert candidate.root==baseline.root==fresh
        qs=safe_queries([fresh],64)
        assert query_outputs(candidate.root,qs)==query_outputs(fresh,qs)
        assert cc==bc==int(math.log2(n)) and fc==n-1
        ratios.append(fc/cc)
        if u==0:
            first_stale=dict(root_same_after_update=old_snapshot[0]==candidate.root,
                             stale_epoch=old_snapshot[1],current_epoch=u+1)
        rows.append(dict(update=u,pos=pos,candidate_calls=cc,conventional_tree_calls=bc,
                         full_rebuild_calls=fc,root_exact=True,queries_exact=True))
    # Separate generation/ABA check with wrapper, where equal payload can return equal numerical root but stale epoch remains invalid.
    cm=CompiledMemory(payloads)
    snap=cm.snapshot(); pos=n//2; original=cm.payloads[pos]
    cm.edit(pos,payload=original+1234567); cm.edit(pos,payload=original)
    aba_numerical_root_returns=(cm.tree.root==snap[0])
    stale_rejected=False
    try: cm.consume(snap,safe_queries([snap[0]],8))
    except RuntimeError: stale_rejected=True
    assert stale_rejected

    ic=interaction_control([encode(i,p) for i,p in enumerate(payloads)],n//3)
    assert ic['nonzero_projective_interactions'] and ic['adjacent_swap_changes_root'] and ic['adjacent_swap_changes_queries']
    npc=naive_patch_control([encode(i,p) for i,p in enumerate(payloads)],n//2,encode(n//2,88_888_888))
    assert not npc['naive_equals_fresh']

    # Timing uses immutable copies and one fixed edit; operation counts are authoritative.
    base_leaves=[encode(i,p) for i,p in enumerate(payloads)]
    pos=n//2; nv=encode(pos,77_777_777)
    def full_timed():
        x=base_leaves.copy(); x[pos]=nv; return fold(x)[0]
    base_tree=ProductTree(base_leaves)
    def tree_timed():
        t=base_tree.clone(); t.update(pos,nv); return t.root
    tf=median_ns(full_timed,31); tt=median_ns(tree_timed,101)
    return dict(seed=seed,n=n,build_calls=candidate.build_calls,stored_matrices=len(candidate.t),
                update_rows=rows,operation_full_over_candidate=min(ratios),
                operation_conventional_over_candidate=1.0,
                full_rebuild_median_ns=tf,tree_update_median_ns=tt,wall_full_over_tree=tf/tt,
                interactions=ic,naive_global_inverse_patch=npc,
                aba_numerical_root_returns=aba_numerical_root_returns,stale_epoch_rejected=stale_rejected,
                query_reads_raw_payload=False,query_root_matrix_words=4,
                candidate_and_conventional_same_algorithm=True,
                first_stale_snapshot=first_stale)


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--output',type=Path,default=Path('cat001-results.json'))
    args=ap.parse_args()
    rows=[seed_size(s,n) for s in range(5) for n in (64,256,1024,4096)]
    rec=dict(experiment='CAT-001',status='constructive_baseline_not_invention',prime=P,
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             python=platform.python_version(),rows=rows,
             min_full_over_candidate_operations=min(r['operation_full_over_candidate'] for r in rows),
             max_conventional_over_candidate_operations=max(r['operation_conventional_over_candidate'] for r in rows),
             trained_backbones=0,full_system_gates='NOT_EVALUATED')
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),min_full_over_candidate_operations=rec['min_full_over_candidate_operations'],
                          strongest_baseline_ratio=rec['max_conventional_over_candidate_operations']),indent=2))

if __name__=='__main__': main()
