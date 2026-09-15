"""Fixed-budget capability continuation. Exploratory, never a breakthrough certificate.

Existing E79 checkpoints are immutable parents. GPT-2 receives 1500 fresh-world
steps at 2e-4; Pythia-70m receives the original 3000-step recipe. No held-out
phrasing enters either training run. All three seeds are retained. CAVI's
freshness semantics and the >=.95 validity threshold are unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from so.llm_adapter import AdapterConfig, KnowledgeAdapterLM
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000079_joint_contract as E79
from so.experiments.e000070_cavi_live_symlink_boundary import _authority_and_manifest, _text
from so.data import bank_from_store

class LastTokenAdapter(KnowledgeAdapterLM):
    """Head-only optimization, numerically checked against the full LM below."""
    def forward(self, bank, input_ids, attention_mask, last_idx, cell_mask=None):
        if self.lm.config.model_type not in ('gpt2', 'gpt_neox'):
            raise ValueError('unvalidated backbone for head optimization')
        with self._memory_request(bank, last_idx, cell_mask) as ctx:
            out = self.lm.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                    use_cache=False, return_dict=True)
            ar = torch.arange(input_ids.shape[0], device=input_ids.device)
            hidden = out.last_hidden_state[ar, last_idx]
            full = self.lm.get_output_embeddings()(hidden)
            routing = torch.stack(ctx['routing'], 1) if ctx is not None and ctx['routing'] else None
            self.last_query = torch.stack(ctx['query'], 1) if ctx is not None and ctx.get('query') else None
            return full[:, self.candidate_ids], full, routing, hidden

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(1048576), b''): h.update(data)
    return h.hexdigest()

def preflight(gk, centre):
    world, spec = E15.sample_alias_world(np.random.default_rng(989001), 180, 8, 2,
                                        gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
    store, _ = E15.load_arm(world, spec, centre, 989002, symlink=True)
    bank = bank_from_store(store).tensors()
    ids, am, last = E8.encode_texts(gk.tok, [_text(gk, k, 0) for k in spec.alias_keys[:4]])
    model = gk.model
    model.eval()
    fast = model(bank, ids, am, last)
    full = KnowledgeAdapterLM.forward(model, bank, ids, am, last)
    out = {}
    for i, label in ((0,'candidate'),(1,'full_vocab'),(2,'routing'),(3,'hidden')):
        torch.testing.assert_close(fast[i], full[i], rtol=1e-4, atol=1e-4)
        out[label+'_maxabs'] = float((fast[i]-full[i]).abs().max())
    params = model.adapter_parameters()
    gf = torch.autograd.grad(fast[0].square().mean(), params, allow_unused=True)
    gr = torch.autograd.grad(full[0].square().mean(), params, allow_unused=True)
    differences = []
    for a, b in zip(gf, gr):
        if a is None or b is None:
            if a is not b: raise AssertionError('gradient participation mismatch')
        else:
            torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-4)
            differences.append(float((a-b).abs().max()))
    out['adapter_gradient_maxabs'] = max(differences, default=0.)
    return out

def fresh_audit(gk, centre, seed):
    rows = []
    # Preserve the E79 world; add two independent worlds. No best-world selection.
    for world_seed in (70000+seed, 91000+seed, 92000+seed):
        world, spec = E15.sample_alias_world(np.random.default_rng(world_seed), 180, 100, 2,
                                            gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
        store, _ = E15.load_arm(world, spec, centre, world_seed+1000, symlink=True)
        bank = bank_from_store(store)
        auth, manifest = _authority_and_manifest(store, bank)
        truth = np.asarray([world.index[k] for k in spec.alias_keys])
        token_truth = np.asarray(gk.entity_ids)[truth]
        for template in (8,9,10,11):
            pred, full = E79.infer(gk, bank, [_text(gk,k,template) for k in spec.alias_keys], auth, manifest)
            rows.append({'world_seed':world_seed,'template':template,'n':len(truth),
                         'candidate_correct':int((pred==truth).sum()),
                         'candidate_accuracy':float((pred==truth).mean()),
                         'full_vocab_accuracy':float((full.argmax(-1).numpy()==token_truth).mean())})
    return {'rows':rows, 'minimum_candidate_accuracy':min(r['candidate_accuracy'] for r in rows),
            'minimum_full_vocab_accuracy':min(r['full_vocab_accuracy'] for r in rows),
            'valid_reader':all(r['candidate_accuracy'] >= .95 for r in rows),
            'scope':'candidate read gate; full-vocabulary performance reported separately'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, choices=(0,1,2), required=True)
    ap.add_argument('--backbone', choices=('gpt2','pythia'), required=True)
    ap.add_argument('--parent', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(2); torch.manual_seed(a.seed); os.environ['SO_BOS']='1'
    a.output.mkdir(parents=True, exist_ok=False)
    model_name = 'openai-community/gpt2' if a.backbone=='gpt2' else 'EleutherAI/pythia-70m'
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=1,
                        read_layers=(8,10) if a.backbone=='gpt2' else (3,5))
    factory = E8.KnowledgeAdapterLM
    E8.KnowledgeAdapterLM = LastTokenAdapter
    try: gk = E8.GPT2Knowledge(cfg, model_name=model_name)
    finally: E8.KnowledgeAdapterLM = factory
    started = time.time()
    record = {'experiment':'CAVI-CAPABILITY-CONTINUATION','seed':a.seed,'model_name':model_name,
              'model_revision':getattr(gk.model.lm.config,'_commit_hash',None),
              'config':cfg.to_dict(), 'source_commit':os.getenv('GITHUB_SHA','unknown'),
              'python':platform.python_version(),'torch':str(torch.__version__),
              'breakthrough':False,'threshold':.95}
    if a.backbone=='gpt2':
        if a.parent is None: raise ValueError('GPT-2 must reuse an E79 checkpoint')
        record['parent_checkpoint_sha256'] = digest(a.parent)
        ck = torch.load(a.parent, map_location='cpu', weights_only=True)
        if ck['seed'] != a.seed or ck['config'] != cfg.to_dict() or ck['model_name'] != model_name:
            raise ValueError('parent checkpoint identity mismatch')
        current = E8.adapter_state(gk.model)
        if set(current) != set(ck['adapter']): raise ValueError('adapter key mismatch')
        for name in ('candidate_ids','entity_token_ids'):
            if not torch.equal(current[name], ck['adapter'][name]): raise ValueError('token identity mismatch')
        missing = gk.model.load_state_dict(ck['adapter'],strict=False)
        if missing.unexpected_keys or any(not k.startswith('lm.') for k in missing.missing_keys):
            raise ValueError('non-backbone state omitted')
        centre = np.asarray(ck['centre'])
        record['parent_training_source_commit'] = ck['source_commit']
        record['preflight'] = preflight(gk, centre)
        record['before_continuation'] = fresh_audit(gk, centre, a.seed)
        (a.output/'before.json').write_text(json.dumps(record,indent=2))
        original_centre = E20.make_centre
        E20.make_centre = lambda seed, dim: centre.copy()
        try:
            trained = E20.train_adapter_links(gk,81000+a.seed,1500,lr=2e-4,route_only_steps=0,n_groups=100)
        finally: E20.make_centre = original_centre
        record['training'] = {'parent_steps':ck['steps'],'additional_steps':1500,
                              'additional_rng_seed':81000+a.seed,'lr':2e-4,'route_only_steps':0,
                              'optimizer':'new AdamW; not an exact optimizer-state resume'}
    else:
        centre = E20.make_centre(a.seed,cfg.marker_dim)
        record['preflight'] = preflight(gk, centre)
        trained = E20.train_adapter_links(gk,a.seed,3000,n_groups=100)
        centre = np.asarray(trained['centre'])
        record['training'] = {'steps':3000,'rng_seed':a.seed,'lr':2e-3,'route_only_steps':400}
    checkpoint = a.output/'reader.pt'
    torch.save({'adapter':E8.adapter_state(gk.model),'centre':centre.tolist(),'seed':a.seed,
                'config':cfg.to_dict(),'model_name':model_name,'provenance':record,
                'history':trained['history']},checkpoint)
    record['checkpoint_sha256'] = digest(checkpoint)
    gk.model.eval()
    record['fresh'] = fresh_audit(gk,centre,a.seed)
    (a.output/'result.json').write_text(json.dumps(record,indent=2))
    # Run joint lifecycle screening only after the unchanged real-reader gate.
    if record['fresh']['valid_reader']:
        record['joint_screen'] = E79.evaluate(gk,centre,a.seed,100)
    else:
        record['joint_screen'] = {'status':'SKIPPED_INVALID_READER'}
    record['seconds'] = time.time()-started
    record['not_established'] = ['full adversarial battery','independent J-space audit',
                                 'universal linguistic coverage','novelty','latency bound']
    (a.output/'result.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)
    if not record['fresh']['valid_reader']: raise SystemExit(2)
    if not record['joint_screen'].get('screening_pass',False): raise SystemExit(3)

if __name__ == '__main__': main()
