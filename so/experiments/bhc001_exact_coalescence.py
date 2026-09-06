"""BHC001: exact native coalescence is not implied by real contraction.
Designed countermodels and ordinary change propagation, not an invention.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import torch

FORMATS = {"bfloat16": (torch.bfloat16, 7), "float16": (torch.float16, 10),
           "float32": (torch.float32, 23), "float64": (torch.float64, 52)}


def raw(t):
    return t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and raw(a) == raw(b)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rn_binade(x, bits):
    """Exact RN-even on [1,2], sufficient for the scalar witnesses."""
    x = F(x)
    require(F(1) <= x <= F(2), "Outside registered binade")
    scaled = (x-1)*(1 << bits)
    q, r = divmod(scaled.numerator, scaled.denominator)
    if 2*r > scaled.denominator or (2*r == scaled.denominator and q % 2):
        q += 1
    return F(1) + F(q, 1 << bits)


def scalar_screen(name):
    dtype, bits = FORMATS[name]
    unit = F(1, 1 << bits)
    b, a = 1+unit, 1+2*unit
    bx, ax = torch.tensor(float(b), dtype=dtype), torch.tensor(float(a), dtype=dtype)
    step = lambda x: bx + .5*(x-bx)
    require(same(step(bx), bx) and same(step(ax), ax), "Native fixed-pair control failed")
    require(rn_binade(b+(a-b)/2, bits) == a, "Exact tie-even check failed")
    require(rn_binade(b, bits) == b, "Exact baseline check failed")
    even = torch.tensor(1., dtype=dtype)
    neighbor = torch.tensor(float(1+unit), dtype=dtype)
    require(same(even + .5*(neighbor-even), even), "Even-bias convergence failed")
    return dict(format=name, mantissa_bits=bits, low=str(b), high=str(a),
                half_step=str(b+(a-b)/2), native_low_fixed=True,
                native_high_fixed=True, rational_rn_witness=True,
                gap=float(unit), real_contraction=0.5,
                even_bias_one_step_coalesces=True,
                real_gap_bound_after_64=str(unit/F(2)**64))


@dataclass
class Machine:
    weights: torch.Tensor
    bias: torch.Tensor
    unit: torch.Tensor

    def initial(self, present):
        state = self.bias.expand(self.weights.shape[0], self.weights.shape[1]).clone()
        if present:
            state[0] += self.unit
        return state

    def step(self, state, reset_first=False):
        require(state.shape == self.weights.shape[:2], "State shape mismatch")
        parent = torch.cat((state[:1], state[:-1]), dim=0)
        delta = parent - self.bias
        projected = torch.bmm(self.weights, delta.unsqueeze(-1)).squeeze(-1)
        result = self.bias + .5*torch.tanh(projected)
        if reset_first:
            result[0] = self.bias
        return result


def machine(seed, name, odd=True, layers=4, width=16):
    dtype, bits = FORMATS[name]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(32, np.full(width, 1/width), size=(layers,width))
    require(np.all(counts.sum(axis=-1) == 32), "Bad stochastic matrix")
    w = torch.tensor(counts/32, dtype=dtype)
    require(torch.equal(w.double().sum(-1), torch.ones((layers,width),dtype=torch.float64)),
            "Weights changed row sums under casting")
    unit = torch.tensor(2.**(-bits), dtype=dtype)
    b = torch.tensor(1.+(2.**(-bits) if odd else 0.), dtype=dtype)
    return Machine(w,b,unit)


def rollout(m, present, steps, reset_at=None):
    require(steps > 0 and (reset_at is None or 1 <= reset_at <= steps), "Invalid schedule")
    state = m.initial(present)
    writes = [state.clone()]
    for t in range(1, steps+1):
        state = m.step(state, reset_first=t == reset_at)
        writes.append(state.clone())
    return torch.stack(writes)


def replay_to_join(m, old, reset_at, first_layer_only=False):
    """Ordinary exact change propagation with an unsafe partial-state control.
    Every prefix write is recomputed. No fresh future oracle states are used.
    """
    current = m.initial(False)
    repaired = old.clone()
    repaired[0] = current
    calls = 0
    for t in range(1, len(old)):
        current = m.step(current, reset_first=t == reset_at)
        calls += 1
        repaired[t] = current
        a, b = (current[0], old[t,0]) if first_layer_only else (current, old[t])
        if same(a,b):
            return repaired, calls, t
    return repaired, calls, None


def first_match(a,b):
    return next((t for t,(x,y) in enumerate(zip(a,b)) if same(x,y)), None)


def neural_screen(seed,name):
    m = machine(seed,name)
    old = rollout(m,True,256)
    never = rollout(m,False,256)
    require(same(old,rollout(m,True,256)), "Repeat-forward failed")
    require(first_match(old,never) is None, "Odd-bias hypothesis falsified")
    stable = next((t for t in range(1,len(old)) if same(old[t],old[t-1]) and
                   same(never[t],never[t-1])),None)
    require(stable is not None, "No stable native pair reached")
    require(same(m.step(old[stable]),old[stable]) and
            same(m.step(never[stable]),never[stable]), "Fixed-pair witness failed")
    gap = (old[-1].double()-never[-1].double()).abs()
    require(bool(torch.all(gap == float(m.unit))), "Gap not one ULP everywhere")
    # Exposes a surviving bit; not learned semantic recall or a utility claim.
    read_old = (old[-1].double()-float(m.bias))/float(m.unit)
    read_never = (never[-1].double()-float(m.bias))/float(m.unit)
    even = machine(seed,name,odd=False)
    even_old,even_never=rollout(even,True,8),rollout(even,False,8)
    even_join=first_match(even_old,even_never)
    # Identical exogenous reset appears in BOTH reference executions.
    old_reset=rollout(m,True,512,32)
    fresh_reset=rollout(m,False,512,32)
    fixed,calls,join=replay_to_join(m,old_reset,32)
    ordinary,ordinary_calls,ordinary_join=replay_to_join(m,old_reset,32)
    unsafe,unsafe_calls,unsafe_join=replay_to_join(m,old_reset,32,True)
    require(same(fixed,fresh_reset), "Complete-state repair disagrees with rebuild")
    require(same(fixed,ordinary) and calls==ordinary_calls, "Baseline mismatch")
    require(not same(unsafe,fresh_reset), "Incomplete-state control unexpectedly exact")
    return dict(seed=seed,format=name,layers=4,width=16,source_dimension=1,
        real_maxnorm_lipschitz_upper_bound=0.5,native_steps=256,
        first_complete_join=None,first_adjacent_fixed_pair_time=stable,
        fixed_pair_next_transition_verified=True,first_layer_only_source_injection=True,
        different_final_coordinates=int(torch.count_nonzero(gap)),final_gap_max=float(gap.max()),
        final_gap_by_layer=[float(x.max()) for x in gap],
        unit_rescaled_old_read_values=sorted(set(read_old.flatten().tolist())),
        unit_rescaled_never_read_values=sorted(set(read_never.flatten().tolist())),
        even_bias_first_complete_join=even_join,
        reset_control=dict(reset_first_layer_at=32,trajectory_steps=512,
            complete_join_at=join,exact_all_rebuilt_writes=True,
            actual_transition_calls=calls,ordinary_transition_calls=ordinary_calls,
            conventional_baseline_same_result=True,
            unsafe_first_layer_join_at=unsafe_join,unsafe_transition_calls=unsafe_calls,
            unsafe_all_writes_exact=False,
            unsafe_mismatched_coordinates=int(torch.count_nonzero(unsafe != fresh_reset))),
        pretrained_model=False,full_system_utility='NOT_EVALUATED')


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return dict(experiment='BHC-001',status='contract_falsification_not_invention',
        python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scalar=[scalar_screen(n) for n in FORMATS],
        neural=[neural_screen(s,n) for n in FORMATS for s in range(5)],
        trained_backbones=0,application_gates='NOT_EVALUATED')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('bhc001-results.json'))
    args=p.parse_args()
    result=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(experiment=result['experiment'],native_cells=len(result['neural']),
        fixed_pairs=sum(r['fixed_pair_next_transition_verified'] for r in result['neural']),
        complete_joins=sorted(set(r['reset_control']['complete_join_at'] for r in result['neural'])),
        scalar=result['scalar']),indent=2))

if __name__=='__main__':
    main()
