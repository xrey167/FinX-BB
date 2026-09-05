"""Reference-head rerun; the failed optimized-head gradient check stays negative.

The first Pythia-2 optimization preflight in Actions run 33963051896 failed
(atol=.0002, observed .0197546482). This runner does NOT relax that check.
It removes the optimization entirely and executes KnowledgeAdapterLM.forward
for training and evaluation. Reader validity, budget, worlds and CAVI semantics
are unchanged. This is implementation validation, not a scientific breakthrough.
"""
from __future__ import annotations
import sys
from so.llm_adapter import KnowledgeAdapterLM
from so.experiments import cavi_capability_continuation as C

class ReferenceAdapter(KnowledgeAdapterLM):
    """Original full-LM-head implementation, with unchanged inherited forward."""
    pass

original_preflight = C.preflight

def reference_preflight(gk, centre):
    record = original_preflight(gk, centre)
    record['execution'] = 'original KnowledgeAdapterLM full-vocabulary full-sequence head'
    record['optimized_head_accepted'] = False
    record['rejected_optimization_run'] = 33963051896
    print('REFERENCE_HEAD_PREFLIGHT', record, flush=True)
    return record

if __name__ == '__main__':
    if '--backbone' not in sys.argv or sys.argv[sys.argv.index('--backbone')+1] != 'pythia':
        raise SystemExit('This recorded rerun is Pythia-only')
    C.LastTokenAdapter = ReferenceAdapter
    C.preflight = reference_preflight
    C.main()
