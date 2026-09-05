"""Post-failure recorder for preregistered RMC001.

The original executable aborts on the fixed >=10x sparse-delta operation gate.
This recorder reruns the SAME scientific functions and emits all failed cells.
It does not change the edit sequence, dimensions, cost model, certificate rule,
or threshold.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from so.experiments import rmc001_rounding_margin_certificates as R


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--output',type=Path,default=Path('rmc001-failure.json'))
    a=p.parse_args()
    rows=[]
    for seed in a.seeds:
        model=R.build_model(seed)
        main=R.exact_costs(model,R.edit_path(seed))
        assert all(x['every_write_exact'] for x in main['edits'])
        rows.append(dict(seed=seed,main=main,leaky=R.leaky_control(seed),
                         fixed_gate_ge_10=main['sparse_over_candidate']>=10.0))
    out=dict(experiment='RMC-001',status='PREREGISTERED_OPERATION_GATE_FAILED',
             original_source_sha256=hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest(),
             recorder_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             threshold=10.0,threshold_changed=False,protocol_changed=False,
             seeds=rows,all_seed_gate=all(r['fixed_gate_ge_10'] for r in rows))
    assert not out['all_seed_gate'], 'This recorder is only valid for the observed failed screen.'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps([dict(seed=r['seed'],ratio=r['main']['sparse_over_candidate'],
                           refreshes=r['main']['total_refreshes'],
                           candidate_ops=r['main']['candidate_ops'],
                           sparse_ops=r['main']['sparse_ops'],
                           leaky_ratio=r['leaky']['sparse_over_candidate']) for r in rows],indent=2))

if __name__=='__main__': main()
