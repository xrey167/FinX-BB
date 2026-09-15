"""E-000054 — hard decision frontier over the two remaining locality channels.

Train the E-000052 full architecture with soft gates, then harden the same learned decisions only at
inference: (1) absolute key-match sigmoid -> steep threshold, and (2) query-relevance sigmoid -> steep
threshold around a chosen learned probability. This directly tests whether the remaining generic KL and
missing-key UNKNOWN failures are soft leakage rather than a representational limitation.

Candidate only: any positive cell must be repeated across seeds and attacked.
"""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
import torch
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.llm_adapter import AdapterConfig

BARS = {
    "train/active_correct": (">=", .95),
    "heldout/active_correct": (">=", .95),
    "revoke_train_min": (">=", .95),
    "revoke_heldout_min": (">=", .95),
    "shred_heldout_min": (">=", .95),
    "heldout/revoked_deleted_object": ("<=", .02),
    "broken1_unknown": (">=", .90),
    "generic/kl_to_base": ("<=", .05),
}

def harden(model, match_tau: float, rel_prob_tau: float, gain: float) -> None:
    """Make already-learned soft decisions approximately binary without retraining.

    Match gate: m=sigmoid((cos-tau)*temp). Raising temp preserves its threshold.
    Relevance: rel=sigmoid(z). Replace z with gain*(z-logit(p_tau)), so the new 0.5 boundary is the
    chosen probability under the original learned head. This preserves ranking and only sharpens the decision.
    """
    with torch.no_grad():
        model.match_tau.fill_(float(match_tau))
        model.match_temp.fill_(float(gain))
        logit_tau = math.log(rel_prob_tau / (1.0 - rel_prob_tau))
        for head in model.query_relevance.values():
            last = head[-1]
            last.weight.mul_(gain)
            last.bias.copy_(gain * (last.bias - logit_tau))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=0); ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--generic-share',type=float,default=.25)
    ap.add_argument('--match-tau',type=float,default=.5); ap.add_argument('--rel-prob-tau',type=float,default=.5)
    ap.add_argument('--gain',type=float,default=80.0)
    ap.add_argument('--threads',type=int,default=2); ap.add_argument('--results-dir',default='ci-e54')
    a=ap.parse_args(); torch.set_num_threads(a.threads); os.environ['SO_BOS']='1'
    cfg=AdapterConfig(status_gated=True, match_gate=True, two_channel_null=True)
    gk=E8.GPT2Knowledge(cfg); t0=time.time()
    out=E18.train_arm(gk,a.seed,a.steps,generic_share=a.generic_share)
    harden(gk.model,a.match_tau,a.rel_prob_tau,a.gain)
    m=E17.evaluate_templates(gk, 6400+a.seed, out['centre'], E18.N_TRAIN_TEMPLATES)
    metrics={k:float(v) for k,v in m.items() if isinstance(v,(int,float,bool))}
    checks={}
    for k,(op,b) in BARS.items():
        v=metrics.get(k,float('nan')); ok=(v>=b) if op=='>=' else (v<=b)
        checks[k]={'value':v,'op':op,'bar':b,'pass':bool(ok)}
    rec={'experiment':'E-000054','candidate_only':True,'seed':a.seed,'steps':a.steps,
         'generic_share':a.generic_share,'match_tau':a.match_tau,'rel_prob_tau':a.rel_prob_tau,'gain':a.gain,
         'metrics':metrics,'criteria':checks,'screening_pass':all(x['pass'] for x in checks.values()),
         'seconds':time.time()-t0}
    p=Path(a.results_dir); p.mkdir(parents=True,exist_ok=True)
    fn=f"e54-m{a.match_tau}-r{a.rel_prob_tau}-s{a.seed}.json"; (p/fn).write_text(json.dumps(rec,indent=2))
    print(json.dumps({k:rec[k] for k in ('match_tau','rel_prob_tau','seed','screening_pass')},indent=2))
    for k,c in checks.items(): print(k, round(c['value'],4), c['op'], c['bar'], 'PASS' if c['pass'] else 'FAIL')
if __name__=='__main__': main()
