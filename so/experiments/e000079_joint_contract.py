"""E-000079: retained-checkpoint joint audit; no novelty claim.

Train the E-000077 BOS+dense symlink recipe once. Optionally use the separately
numerically tested GPT-2 last-token head. Keep every seed, checkpoint and failing
metric. Audit actual current reads and rejection behavior with the same weights;
raw memory-enabled prose is compared against real no-memory execution.
This is a deliberately stronger screening contract, NOT a proof that every
linguistic path or every possible neural cache is safe.
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
import torch.nn.functional as F

from so.cavi_snapshot import ForwardSnapshotConsumptionGuard
from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments.e000070_cavi_live_symlink_boundary import _authority_and_manifest, _text
from so.experiments.e000071_cavi_read_hook_race import _manifest_from_live
from so.last_token_adapter import LastTokenGPT2Adapter
from so.llm_adapter import AdapterConfig

GENERIC = (
    "The ocean was calm under the moon.",
    "A train crossed the bridge and continued toward the next station.",
    "The committee published a revised version of the document.",
    "A researcher compared several methods before writing the report.",
    "Water freezes at zero degrees Celsius under ordinary conditions.",
    "The city centre was busy during the afternoon and quiet at night.",
    "Please explain why the sky appears blue during the day.",
    "The first chapter introduces the central argument of the book.",
    "A careful programmer checks the inputs before calling a function.",
    "The recipe calls for flour, water, salt and a little patience.",
    "Several musicians gathered to rehearse before the concert.",
    "A compass needle points approximately toward magnetic north.",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def infer(gk, bank, texts, auth=None, manifest=None, batch=24):
    preds, logits = [], []
    tensors = None if bank is None else bank.tensors()
    guard = None
    if auth is not None:
        guard = ForwardSnapshotConsumptionGuard(gk.model, lambda: auth.row_mask(manifest), auth.lock)
    try:
        for start in range(0, len(texts), batch):
            ids, am, last = E8.encode_texts(gk.tok, texts[start:start + batch])
            with torch.no_grad():
                cand, full, _, _ = gk.model(tensors, ids, am, last)
            preds.extend(cand.argmax(-1).tolist())
            logits.append(full.detach().cpu())
    finally:
        if guard is not None: guard.close()
    return np.asarray(preds), torch.cat(logits)


def mean_kl(p, q):
    lp, lq = F.log_softmax(p, -1), F.log_softmax(q, -1)
    return float((lp.exp() * (lp - lq)).sum(-1).mean())


def evaluate(gk, centre, seed, groups):
    world, spec = E15.sample_alias_world(np.random.default_rng(70000 + seed),
        180, groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
    store, kids = E15.load_arm(world, spec, centre, 71000 + seed, symlink=True)
    bank = bank_from_store(store)
    auth, manifest = _authority_and_manifest(store, bank)
    keys = spec.alias_keys
    truth = np.asarray([world.index[k] for k in keys])
    metrics, details = {}, {}
    for template in (8, 9, 10, 11):
        pred, _ = infer(gk, bank, [_text(gk, k, template) for k in keys], auth, manifest)
        metrics[f'fresh_template_{template}'] = float((pred == truth).mean())
        details[f'fresh_template_{template}'] = {'correct': int((pred == truth).sum()),
                                               'total': len(keys), 'predictions': pred.tolist()}
    metrics['fresh_heldout_min'] = min(metrics[f'fresh_template_{t}'] for t in (8,9,10,11))
    _, base_generic = infer(gk, None, list(GENERIC))
    _, active_generic = infer(gk, bank, list(GENERIC), auth, manifest)
    metrics['generic_memory_enabled_kl_to_base'] = mean_kl(base_generic, active_generic)
    metrics['generic_memory_enabled_maxabs'] = float((active_generic-base_generic).abs().max())
    # True explicit no-memory arm is measured separately, not sold as semantic scope.
    _, explicit_bypass = infer(gk, None, list(GENERIC))
    metrics['explicit_bypass_repeat_maxabs'] = float((base_generic-explicit_bypass).abs().max())

    # A deterministic half of groups is edited; the other half is untouched.
    changed = spec.groups[:max(1, groups//2)]
    untouched_keys = [k for _, aliases in spec.groups[max(1, groups//2):] for k in aliases]
    by_text = [_text(gk, k, 9) for k in untouched_keys]
    by_before, by_logits_before = infer(gk, bank, by_text, auth, manifest)
    with auth.lock:
        for target, _ in changed:
            store.update(kids[target], (int(world.index[target])+17) % gk.n_entities)
            auth.update_pod(kids[target])
    updated = bank_from_store(store)
    updated_manifest = _manifest_from_live(auth, store, updated)
    changed_keys = [k for _, aliases in changed for k in aliases]
    changed_text = [_text(gk, k, 9) for k in changed_keys]
    want = np.asarray([updated.index_view[k] for k in changed_keys])
    pred, _ = infer(gk, updated, changed_text, auth, updated_manifest)
    metrics['updated_alias_correct'] = float((pred==want).mean())
    stale_pred, _ = infer(gk, bank, changed_text, auth, manifest)
    metrics['stale_updated_bank_unknown'] = float((stale_pred==gk.n_entities).mean())
    metrics['stale_updated_bank_old_answer'] = float((stale_pred==np.asarray([world.index[k] for k in changed_keys])).mean())
    by_after, by_logits_after = infer(gk, updated, by_text, auth, updated_manifest)
    metrics['update_bystander_agreement'] = float((by_before==by_after).mean())
    metrics['update_bystander_kl'] = mean_kl(by_logits_before, by_logits_after)

    # Relink the first alias of each changed group, keeping its old pod live.
    relink_keys, relink_want, old_witnesses = [], [], []
    with auth.lock:
        for i, (_, aliases) in enumerate(changed):
            key = aliases[0]
            new_target = spec.groups[(i+1) % groups][0]
            aid = kids[key]
            old_witnesses.append(auth.witness(aid))
            store.relink(aid, kids[new_target]); auth.relink_alias(aid, kids[new_target])
            relink_keys.append(key)
    current = bank_from_store(store)
    current_manifest = _manifest_from_live(auth, store, current)
    relink_text = [_text(gk, k, 9) for k in relink_keys]
    pred, _ = infer(gk, current, relink_text, auth, current_manifest)
    metrics['relinked_alias_correct'] = float((pred==np.asarray([current.index_view[k] for k in relink_keys])).mean())
    stale_pred, _ = infer(gk, updated, relink_text, auth, updated_manifest)
    metrics['stale_relinked_bank_unknown'] = float((stale_pred==gk.n_entities).mean())
    metrics['relink_old_witnesses_rejected'] = float(np.mean([not auth.validate_witness(w) for w in old_witnesses]))
    metrics['relink_pod_only_accepts_old'] = float(np.mean([auth.validate_pod_only(w) for w in old_witnesses]))

    # Shred all canonical targets once, and test every alias, not just a row mask.
    with auth.lock:
        for target, _ in spec.groups:
            store.shred(kids[target]); auth.shred_pod(kids[target])
    shredded = bank_from_store(store)
    shredded_manifest = _manifest_from_live(auth, store, shredded)
    pred, _ = infer(gk, shredded, [_text(gk, k, 9) for k in keys], auth, shredded_manifest)
    metrics['shred_alias_unknown'] = float((pred==gk.n_entities).mean())
    criteria = {
        'fresh_template_9': metrics['fresh_template_9'] >= .95,
        'fresh_all_heldout': metrics['fresh_heldout_min'] >= .95,
        'updated_alias': metrics['updated_alias_correct'] >= .95,
        'relinked_alias': metrics['relinked_alias_correct'] >= .95,
        'stale_updated_unknown': metrics['stale_updated_bank_unknown'] >= .95,
        'stale_relinked_unknown': metrics['stale_relinked_bank_unknown'] >= .95,
        'shred_unknown': metrics['shred_alias_unknown'] >= .95,
        'generic_locality': metrics['generic_memory_enabled_kl_to_base'] <= .05,
        'bystander_agreement': metrics['update_bystander_agreement'] >= .98,
        'bystander_kl': metrics['update_bystander_kl'] <= .05,
        'alias_lineage_rejection': metrics['relink_old_witnesses_rejected'] == 1.,
        'pod_only_differentiation': metrics['relink_pod_only_accepts_old'] == 1.,
    }
    return {'metrics': metrics, 'criteria': criteria, 'screening_pass': all(criteria.values()),
            'fresh_observations': details,
            'not_established': ['universal linguistic coverage', 'independent J-space audit',
                               'all derived-cache dependencies', 'multiple backbones', 'novelty']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--groups', type=int, default=100)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--model-name', default='gpt2')
    ap.add_argument('--fast-head', action='store_true')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--results-dir', default='ci-e79')
    a = ap.parse_args()
    if a.groups < 4: ap.error('--groups must be >= 4')
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); os.environ['SO_BOS']='1'
    folder = Path(a.results_dir); folder.mkdir(parents=True, exist_ok=True)
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=1)
    old_factory = E8.KnowledgeAdapterLM
    if a.fast_head: E8.KnowledgeAdapterLM = LastTokenGPT2Adapter
    try: gk = E8.GPT2Knowledge(cfg, model_name=a.model_name)
    finally: E8.KnowledgeAdapterLM = old_factory
    t0 = time.time()
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
        if ck['seed'] != a.seed or ck['config'] != cfg.to_dict():
            raise ValueError('checkpoint seed/config mismatch')
        gk.model.load_state_dict(ck['adapter'], strict=False)
        centre = np.asarray(ck['centre'])
        checkpoint = Path(a.checkpoint)
    else:
        trained = E20.train_adapter_links(gk, a.seed, a.steps, n_groups=a.groups)
        centre = np.asarray(trained['centre'])
        checkpoint = folder / f'e000079-s{a.seed}.pt'
        ck = {'adapter': E8.adapter_state(gk.model), 'centre': centre.tolist(),
              'seed': a.seed, 'steps': a.steps, 'config': cfg.to_dict(),
              'history': trained['history'], 'model_name': a.model_name,
              'fast_head': a.fast_head, 'source_commit': os.getenv('GITHUB_SHA', 'local')}
        tmp = checkpoint.with_suffix('.tmp'); torch.save(ck, tmp); tmp.replace(checkpoint)
    gk.model.eval()
    record = {'experiment':'E-000079', 'seed': a.seed, 'steps': ck['steps'],
              'fast_head': a.fast_head, 'groups': a.groups, 'checkpoint_sha256':sha256(checkpoint),
              'training_source_commit': ck['source_commit'], 'evaluation_source_commit':os.getenv('GITHUB_SHA','local'),
              'python': platform.python_version(), 'torch':str(torch.__version__)}
    # Write provenance before evaluation, so a failed audit never loses the trained weights.
    out = folder / f'e000079-s{a.seed}.json'
    out.write_text(json.dumps(record, indent=2))
    record.update(evaluate(gk, centre, a.seed, a.groups)); record['seconds']=time.time()-t0
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!='fresh_observations'}, indent=2), flush=True)
    if not record['screening_pass']: raise SystemExit(2)

if __name__ == '__main__': main()
