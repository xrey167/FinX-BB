"""E-000073 -- stale DERIVED neural state replay across the CAVI consumption boundary.

E-000070/071 validate serialized Bank rows at the real symlink read hook.  That is still not the
strongest boundary: a caller can cache state *after* an authorized neural-memory read (router result,
resolved payload, residual/read activation) and replay that derived state later without presenting the
Bank at all.  If such material remains self-authorizing, row freshness has merely moved the replay
problem downstream.

This experiment captures the actual last-token hidden state immediately AFTER the adapter's final
memory-consuming transformer block during a correct, live symlink read.  The tensor is serialized and
reloaded, then the alias is relinked while the old canonical pod remains live.  The stale tensor is
replayed with bank=None at the same transformer block.  This is deliberately downstream of routing:
there is no row left for a Bank guard to reject.

Baselines:
  none       trust the cached derived tensor;
  commit     trust the authorization boolean captured when the tensor was produced;
  pod_only   recheck only the canonical pod incarnation;
  cavi       recheck the full alias-binding + canonical-pod witness at the exact replay/injection hook.

The alias-relink case differentiates CAVI from a simple referent-version check: P remains current/live,
so pod_only still authorizes the old A->P-derived activation.  Full CAVI must reject it and produce the
exact no-memory BYPASS result.  A second case tests pod SHRED->RESTORE ABA.  A third case mutates the
alias inside forward, after commit-time authorization but before the replay hook, to test the derived
state TOCTOU boundary.

No J-space signal participates in routing, training, authorization or gating.  J-space/J-lens remains
an independent audit line only.  Versions, capabilities, locks, commit-time authorization and freshness
are baselines/prior art, not novelty claims.  This is a structural falsification screen for the composed
CAVI execution property.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from so.cavi import CAVIAuthority, NeuralConsumptionGuard, ResolveWitness
from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments.e000070_cavi_live_symlink_boundary import _authority_and_manifest, _text
from so.experiments.e000071_cavi_read_hook_race import _manifest_from_live
from so.llm_adapter import AdapterConfig, transformer_blocks


def _encoded(gk, text: str):
    return E8.encode_texts(gk.tok, [text])


def _capture_post_read_hidden(gk, bank, text: str, layer: int):
    """Capture the real post-adapter block state at the final memory-consumption layer."""
    ids, am, last = _encoded(gk, text)
    block = transformer_blocks(gk.model.lm)[layer]
    box = {}

    def capture(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        box["hidden"] = h[0, int(last[0])].detach().cpu().clone()
        return None

    # Adapter hooks were registered at model construction, so this later hook observes their result.
    handle = block.register_forward_hook(capture)
    try:
        with torch.no_grad():
            cand, full, _, _ = gk.model(bank.tensors(), ids, am, last)
    finally:
        handle.remove()
    if "hidden" not in box:
        raise RuntimeError("post-read hidden state was not captured")
    return int(cand.argmax(-1)[0]), full.detach().cpu(), box["hidden"]


def _bypass(gk, text: str):
    ids, am, last = _encoded(gk, text)
    with torch.no_grad():
        cand, full, _, _ = gk.model(None, ids, am, last)
    return int(cand.argmax(-1)[0]), full.detach().cpu()


def _replay(gk, text: str, layer: int, cached: torch.Tensor, *,
            allow_fn=None, lock=None, mutate_pre=None):
    """Replay a cached post-memory hidden state with NO Bank supplied to the model.

    `allow_fn` is evaluated at the actual injection hook.  If `lock` is supplied, validation and
    replacement occur under the same live-authority lock.  `mutate_pre` is registered first and can
    invalidate authority inside forward immediately before the replay hook executes.
    """
    ids, am, last = _encoded(gk, text)
    block = transformer_blocks(gk.model.lm)[layer]
    state = {"injected": False, "mutated": False}
    handles = []

    if mutate_pre is not None:
        def mutator(module, inputs):
            if not state["mutated"]:
                mutate_pre()
                state["mutated"] = True
        handles.append(block.register_forward_pre_hook(mutator))

    def replay_hook(module, inputs, output):
        def apply():
            allowed = True if allow_fn is None else bool(allow_fn())
            if not allowed:
                return None
            h = output[0] if isinstance(output, tuple) else output
            h2 = h.clone()
            h2[0, int(last[0])] = cached.to(device=h.device, dtype=h.dtype)
            state["injected"] = True
            return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

        if lock is None:
            return apply()
        with lock:
            return apply()

    handles.append(block.register_forward_hook(replay_hook))
    try:
        with torch.no_grad():
            cand, full, _, _ = gk.model(None, ids, am, last)
    finally:
        for h in handles:
            h.remove()
    return int(cand.argmax(-1)[0]), full.detach().cpu(), state


def _serialize_roundtrip(x: torch.Tensor) -> torch.Tensor:
    b = io.BytesIO()
    torch.save(x, b)
    b.seek(0)
    return torch.load(b, map_location="cpu", weights_only=True)


def _case(gk, centre, seed: int, groups: int, template: int, offset: int):
    rng = np.random.default_rng(offset + seed)
    world, spec = E15.sample_alias_world(rng, 180, groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
    store, kids = E15.load_arm(world, spec, centre, offset + 1000 + seed, symlink=True)
    bank = bank_from_store(store)
    auth, manifest = _authority_and_manifest(store, bank)
    target, aliases = spec.groups[0]
    target2, _ = spec.groups[1]
    aid = kids[aliases[0]]
    p1 = kids[target]
    p2 = kids[target2]
    text = _text(gk, aliases[0], template)
    return world, spec, store, kids, bank, auth, manifest, aid, p1, p2, text


def run(seed: int, steps: int, groups: int, template: int) -> Dict[str, object]:
    torch.manual_seed(seed)
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
    gk = E8.GPT2Knowledge(cfg)
    trained = E20.train_adapter_links(gk, seed, steps, n_groups=max(groups, 24), verbose=True)
    gk.model.eval()
    centre = np.asarray(trained["centre"])
    layer = int(gk.model.cfg.read_layers[-1])

    # ---------------------------------------------------------------- alias relink: differentiates full CAVI from pod-only
    world, spec, store, kids, bank, auth, manifest, aid, p1, p2, text = _case(
        gk, centre, seed, groups, template, 91000
    )
    old_truth = int(world.index[spec.groups[0][0]])
    new_truth = int(world.index[spec.groups[1][0]])
    fresh_pred, fresh_logits, cached_hidden = _capture_post_read_hidden(gk, bank, text, layer)
    cached_hidden = _serialize_roundtrip(cached_hidden)  # explicit serialized-tensor attack
    witness = auth.witness(aid)
    commit_authorized = auth.validate_witness(witness)
    bypass_pred, bypass_logits = _bypass(gk, text)

    # Change only the reference binding.  Old P remains live and at the same incarnation.
    store.relink(aid, p2)
    auth.relink_alias(aid, p2)
    unguarded_pred, unguarded_logits, unguarded_state = _replay(gk, text, layer, cached_hidden)
    commit_pred, commit_logits, commit_state = _replay(
        gk, text, layer, cached_hidden, allow_fn=lambda: commit_authorized
    )
    pod_pred, pod_logits, pod_state = _replay(
        gk, text, layer, cached_hidden, allow_fn=lambda: auth.validate_pod_only(witness), lock=auth.lock
    )
    cavi_pred, cavi_logits, cavi_state = _replay(
        gk, text, layer, cached_hidden, allow_fn=lambda: auth.validate_witness(witness), lock=auth.lock
    )

    fresh_bank = bank_from_store(store)
    fresh_manifest = _manifest_from_live(auth, store, fresh_bank)
    with NeuralConsumptionGuard(gk.model, lambda: auth.row_mask(fresh_manifest, full=True), lock=auth.lock):
        ids, am, last = _encoded(gk, text)
        with torch.no_grad():
            fresh_after_cand, fresh_after_logits_t, _, _ = gk.model(fresh_bank.tensors(), ids, am, last)
    fresh_after_pred = int(fresh_after_cand.argmax(-1)[0])
    fresh_after_logits = fresh_after_logits_t.detach().cpu()

    relink = {
        "fresh_before_pred": fresh_pred,
        "old_truth": old_truth,
        "new_truth": new_truth,
        "fresh_before_correct": fresh_pred == old_truth,
        "old_full_witness_rejected": not auth.validate_witness(witness),
        "old_pod_witness_still_valid": auth.validate_pod_only(witness),
        "serialized_replay_injected": bool(unguarded_state["injected"]),
        "serialized_replay_pred": unguarded_pred,
        "serialized_replay_vs_fresh_maxabs": float((unguarded_logits - fresh_logits).abs().max()),
        "serialized_replay_vs_bypass_maxabs": float((unguarded_logits - bypass_logits).abs().max()),
        "commit_time_replay_injected": bool(commit_state["injected"]),
        "commit_time_vs_unguarded_maxabs": float((commit_logits - unguarded_logits).abs().max()),
        "pod_only_replay_injected": bool(pod_state["injected"]),
        "pod_only_vs_unguarded_maxabs": float((pod_logits - unguarded_logits).abs().max()),
        "cavi_replay_injected": bool(cavi_state["injected"]),
        "cavi_vs_bypass_maxabs": float((cavi_logits - bypass_logits).abs().max()),
        "cavi_vs_unguarded_maxabs": float((cavi_logits - unguarded_logits).abs().max()),
        "fresh_after_pred": fresh_after_pred,
        "fresh_after_current_correct": fresh_after_pred == new_truth,
        "fresh_after_vs_stale_replay_maxabs": float((fresh_after_logits - unguarded_logits).abs().max()),
    }

    # ---------------------------------------------------------------- pod ABA: old derived state must stay dead after same-id restore
    w2, sp2, st2, k2, b2, a2, m2, aid2, pp1, pp2, txt2 = _case(
        gk, centre, seed, 6, template, 93000
    )
    pred2, logits2, hidden2 = _capture_post_read_hidden(gk, b2, txt2, layer)
    hidden2 = _serialize_roundtrip(hidden2)
    wit2 = a2.witness(aid2)
    a2.shred_pod(pp1)
    a2.restore_pod(pp1)  # same logical id, strictly newer incarnation
    bp2, bl2 = _bypass(gk, txt2)
    aba_pred, aba_logits, aba_state = _replay(
        gk, txt2, layer, hidden2, allow_fn=lambda: a2.validate_witness(wit2), lock=a2.lock
    )
    aba = {
        "old_witness_rejected": not a2.validate_witness(wit2),
        "newer_incarnation": a2.pod_incarnation(pp1) > wit2.pod_incarnation,
        "replay_not_injected": not bool(aba_state["injected"]),
        "rejected_replay_equals_bypass_maxabs": float((aba_logits - bl2).abs().max()),
    }

    # ---------------------------------------------------------------- in-forward replay race
    # Two independent cases: commit-time auth is cached before mutation; CAVI validates at injection.
    wr, spr, sr, kr, br, ar, mr, raid, rp1, rp2, rtxt = _case(
        gk, centre, seed, 6, template, 95000
    )
    _, _, rh = _capture_post_read_hidden(gk, br, rtxt, layer)
    rh = _serialize_roundtrip(rh)
    rw = ar.witness(raid)
    rcommit = ar.validate_witness(rw)

    def mutate_commit():
        ar.relink_alias(raid, rp2)

    rcommit_pred, rcommit_logits, rcommit_state = _replay(
        gk, rtxt, layer, rh, allow_fn=lambda: rcommit, mutate_pre=mutate_commit
    )

    wc, spc, sc, kc, bc, ac, mc, caid, cp1, cp2, ctxt = _case(
        gk, centre, seed, 6, template, 97000
    )
    _, _, ch = _capture_post_read_hidden(gk, bc, ctxt, layer)
    ch = _serialize_roundtrip(ch)
    cw = ac.witness(caid)
    cbp, cbl = _bypass(gk, ctxt)

    def mutate_cavi():
        ac.relink_alias(caid, cp2)

    rcavi_pred, rcavi_logits, rcavi_state = _replay(
        gk, ctxt, layer, ch, allow_fn=lambda: ac.validate_witness(cw), lock=ac.lock, mutate_pre=mutate_cavi
    )
    race = {
        "commit_mutation_happened": bool(rcommit_state["mutated"]),
        "commit_replay_still_injected": bool(rcommit_state["injected"]),
        "cavi_mutation_happened": bool(rcavi_state["mutated"]),
        "cavi_replay_rejected": not bool(rcavi_state["injected"]),
        "cavi_rejected_equals_bypass_maxabs": float((rcavi_logits - cbl).abs().max()),
    }

    checks = {
        "real_symlink_capability_before": relink["fresh_before_correct"],
        "alias_relink_invalidates_full_not_pod_only": relink["old_full_witness_rejected"] and relink["old_pod_witness_still_valid"],
        "serialized_post_read_state_is_replayable": relink["serialized_replay_injected"] and relink["serialized_replay_vs_fresh_maxabs"] <= 1e-6,
        "serialized_replay_is_not_bypass": relink["serialized_replay_vs_bypass_maxabs"] > 1e-6,
        "commit_time_authorization_is_insufficient": relink["commit_time_replay_injected"] and relink["commit_time_vs_unguarded_maxabs"] <= 1e-7,
        "pod_only_version_check_is_insufficient": relink["pod_only_replay_injected"] and relink["pod_only_vs_unguarded_maxabs"] <= 1e-7,
        "cavi_rejects_stale_derived_state": (not relink["cavi_replay_injected"]) and relink["cavi_vs_bypass_maxabs"] <= 1e-7,
        "cavi_rejection_materially_changes_stale_replay": relink["cavi_vs_unguarded_maxabs"] > 1e-6,
        "fresh_current_incarnation_retains_capability": relink["fresh_after_current_correct"],
        "aba_old_derived_state_rejected": aba["old_witness_rejected"] and aba["newer_incarnation"] and aba["replay_not_injected"] and aba["rejected_replay_equals_bypass_maxabs"] <= 1e-7,
        "commit_time_loses_in_forward_race": race["commit_mutation_happened"] and race["commit_replay_still_injected"],
        "cavi_closes_in_forward_derived_replay_race": race["cavi_mutation_happened"] and race["cavi_replay_rejected"] and race["cavi_rejected_equals_bypass_maxabs"] <= 1e-7,
    }

    return {
        "seed": seed,
        "steps": steps,
        "layer": layer,
        "candidate_only": True,
        "screening_pass": all(checks.values()),
        "checks": checks,
        "alias_relink": relink,
        "aba": aba,
        "race": race,
        "not_claimed": "versions, epochs, HMAC, freshness, locks, pointers, capabilities or commit-time authorization individually; J-space is not used",
        "scope": "post-read hidden-state replay. Cached router-score and resolved-payload replay remain separate required attacks.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--template", type=int, default=9)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rows = [run(s, a.steps, a.groups, a.template) for s in a.seeds]
    rec = {
        "experiment": "E-000073",
        "title": "CAVI derived neural-state replay boundary",
        "all_screening_pass": all(r["screening_pass"] for r in rows),
        "rows": rows,
    }
    p = Path(a.results_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "e000073_cavi_derived_state_replay.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not rec["all_screening_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
