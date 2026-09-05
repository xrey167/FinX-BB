"""E-000088 — E77 reproduction under E87's strict marker-validity contract.

No CAVI/novelty interpretation is permitted unless every requested training seed reaches
fresh real-symlink correctness >=0.95. Historical E77 remains 0.99/0.95/0.94; this run does not
rewrite it. The only setup correction is that samples labelled mechanically valid must satisfy
the existing radius=0.35 predicate in both training and MVCC evaluation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from so.experiments.e000087_marker_validity_contract import install_strict_contract


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds',type=int,nargs='*',default=[0]);p.add_argument('--steps',type=int,default=3000);p.add_argument('--groups',type=int,default=100);p.add_argument('--template',type=int,default=9);p.add_argument('--threads',type=int,default=2);p.add_argument('--results-dir',default='so/results');a=p.parse_args()
    install_strict_contract()
    # Import after installation so every runtime path observes the aligned marker contract.
    from so.experiments import e000070_cavi_live_symlink_boundary as E70
    torch.set_num_threads(a.threads)
    rows=[]
    for seed in a.seeds:
        r=E70.run(seed,a.steps,a.groups,a.template)
        strict_gate=float(r['fresh_alias_read_rate'])>=0.95
        structural=all(bool(v) for k,v in r['checks'].items() if k!='fresh_real_symlink_read')
        rows.append({**r,'strict_fresh_gate_ge_095':strict_gate,'structural_checks_without_old_060_screen':structural,
                     'qualified_for_attack_interpretation':strict_gate and structural})
    rec={'experiment':'E-000088','title':'strict-marker E77 reproduction','rows':rows,
         'all_three_seed_gate':len(rows)>=3 and all(r['qualified_for_attack_interpretation'] for r in rows),
         'breakthrough':False,'novelty_claim':False,
         'historical_e77_preserved':'0.99/0.95/0.94 remains failed evidence',
         'boundary':'This candidate-set fresh-read check is only a prerequisite. Full-vocabulary/heldout/lifecycle/locality/J-lens gates remain separate and unresolved.'}
    d=Path(a.results_dir);d.mkdir(parents=True,exist_ok=True);(d/'e000088_strict_marker_reader_gate.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
    if not rec['all_three_seed_gate']: raise SystemExit(3)
if __name__=='__main__':main()
