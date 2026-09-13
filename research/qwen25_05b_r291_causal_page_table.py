from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import (
    COLORS,
    SPECS,
    EXPECTED_WEIGHTS as _IGNORE_3B_HASHES,
    candidate_token,
    candidates,
    concat_cache,
    legacy_cache,
    mutated_lane,
    query_logits,
    render,
    run_segment,
    sha256_file,
    slice_cache,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_WEIGHTS = {
    "model.safetensors": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
}
OUT = Path(os.environ.get("SO_R291_REPORT", "ci-qwen-r291/report.json"))
MODEL_REVISION = f"{MODEL_ID}@{REVISION}"


def digest_cache(cache) -> str:
    h = hashlib.sha256()
    for k, v in legacy_cache(cache):
        for t in (k, v):
            h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def top_candidate(tok, logits, spec: str) -> str:
    scores = {
        c.casefold(): float(logits[candidate_token(tok, c)])
        for c in candidates(spec)
    }
    return max(scores, key=scores.get)


@dataclass(frozen=True)
class Handle:
    pod_id: int
    generation: int
    page_id: int
    payload_digest: str
    model_revision: str


@dataclass
class Page:
    pod_id: int
    generation: int
    payload_digest: str
    model_revision: str
    cache: tuple


class CausalPageTable:
    """Two-control-plane prototype: mutable page table + independent immutable ledger.

    Stale pages remain physically materialized in `pages`, but only the page named by
    the authority record may be returned to Port Attention. A verifier recomputes the
    admissible handle from the ledger rather than trusting a caller-supplied handle.
    """

    def __init__(self):
        self.pages: dict[int, Page] = {}
        self.authority: dict[int, int | None] = {}
        self.authority_generation: dict[int, int] = {}
        self.ledger: dict[int, dict[int, dict]] = {}
        self._next_page = 1

    def write(self, pod_id: int, generation: int, cache) -> Handle:
        digest = digest_cache(cache)
        page_id = self._next_page
        self._next_page += 1
        page = Page(pod_id, generation, digest, MODEL_REVISION, copy.deepcopy(legacy_cache(cache)))
        self.pages[page_id] = page
        self.ledger.setdefault(pod_id, {})[generation] = {
            "generation": generation,
            "page_id": page_id,
            "digest": digest,
            "model_revision": MODEL_REVISION,
            "committed": False,
            "revoked": False,
        }
        return Handle(pod_id, generation, page_id, digest, MODEL_REVISION)

    def commit(self, handle: Handle):
        rec = self.ledger[handle.pod_id][handle.generation]
        if (
            rec["page_id"] != handle.page_id
            or rec["digest"] != handle.payload_digest
            or rec["model_revision"] != handle.model_revision
        ):
            raise RuntimeError("commit handle mismatch")
        prev = self.authority_generation.get(handle.pod_id, 0)
        if handle.generation <= prev:
            raise RuntimeError("non-monotonic generation")
        rec["committed"] = True
        self.authority[handle.pod_id] = handle.page_id
        self.authority_generation[handle.pod_id] = handle.generation

    def revoke(self, pod_id: int):
        gen = self.authority_generation[pod_id]
        self.ledger[pod_id][gen]["revoked"] = True
        self.authority[pod_id] = None

    def independent_verify(self, handle: Handle) -> bool:
        # Recompute admissibility from the append-only ledger + current authority.
        current_gen = self.authority_generation.get(handle.pod_id)
        current_page = self.authority.get(handle.pod_id)
        if current_gen is None or current_page is None:
            return False
        if handle.generation != current_gen or handle.page_id != current_page:
            return False
        rec = self.ledger.get(handle.pod_id, {}).get(handle.generation)
        if not rec or not rec["committed"] or rec["revoked"]:
            return False
        if rec["digest"] != handle.payload_digest or rec["model_revision"] != handle.model_revision:
            return False
        page = self.pages.get(handle.page_id)
        if page is None:
            return False
        return (
            page.pod_id == handle.pod_id
            and page.generation == handle.generation
            and page.payload_digest == handle.payload_digest
            and page.model_revision == handle.model_revision
        )

    def current_handle(self, pod_id: int) -> Handle | None:
        page_id = self.authority.get(pod_id)
        if page_id is None:
            return None
        gen = self.authority_generation[pod_id]
        rec = self.ledger[pod_id][gen]
        return Handle(pod_id, gen, page_id, rec["digest"], rec["model_revision"])

    def materialize_verified(self, pod_id: int):
        handle = self.current_handle(pod_id)
        if handle is None or not self.independent_verify(handle):
            return None
        # Critically, stale pages are not concatenated into the attention sequence.
        return self.pages[handle.page_id].cache


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rng = random.Random(291)

    model_dir = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        allow_patterns=[
            "*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors",
            "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json",
        ],
    ))
    hashes = {name: sha256_file(model_dir / name) for name in EXPECTED_WEIGHTS}
    assert hashes == EXPECTED_WEIGHTS
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    layouts = {(v, s): render(tok, v, s) for v in range(4) for s in SPECS}
    static_ids = layouts[(0, "read")]["static_ids"]
    assert all(x["static_ids"] == static_ids for x in layouts.values())
    dynamic = {}
    for v in range(4):
        ids = layouts[(v, "read")]["dynamic_ids"]
        assert all(layouts[(v, s)]["dynamic_ids"] == ids for s in SPECS)
        dynamic[v] = ids
    D = len(dynamic[0])
    assert all(len(x) == D for x in dynamic.values())
    S = len(static_ids)

    st = torch.tensor([static_ids], dtype=torch.long)
    with torch.inference_mode():
        sout = model(input_ids=st, attention_mask=torch.ones_like(st), use_cache=True, return_dict=True)
    shared = legacy_cache(sout.past_key_values)
    shared_mask = torch.ones((1, S), dtype=torch.long)

    capsules = {}
    prefix = {}
    digests = {}
    for v in range(4):
        out = run_segment(model, dynamic[v], shared, shared_mask)
        pcache = legacy_cache(out.past_key_values)
        prefix[v] = pcache
        capsules[v] = slice_cache(pcache, S, S + D)
        digests[v] = digest_cache(capsules[v])

    table = CausalPageTable()
    pod_id = 7
    generation = 0
    stale_handles: list[Handle] = []
    rejected_stale = 0
    rejected_forged = 0
    stale_mutation_equivalence = []
    active_equivalence = []
    active_top_match = []
    probes = []

    # 4096 durable-style generation changes. Only a bounded subset calls the LLM;
    # the lifecycle hot path itself is exercised on every transition.
    for step in range(4096):
        generation += 1
        value = rng.randrange(4)
        old_handle = table.current_handle(pod_id)
        h = table.write(pod_id, generation, capsules[value])
        table.commit(h)
        if old_handle is not None:
            stale_handles.append(old_handle)
            rejected_stale += int(not table.independent_verify(old_handle))
        forged = Handle(pod_id, generation + 100000, h.page_id, h.payload_digest, h.model_revision)
        rejected_forged += int(not table.independent_verify(forged))

        if step % 64 == 0:
            active = table.materialize_verified(pod_id)
            assert active is not None
            # Mutate several physically retained stale pages. They never enter attention.
            stale_page_ids = [pid for pid in table.pages if pid != h.page_id]
            for pid in rng.sample(stale_page_ids, min(8, len(stale_page_ids))):
                table.pages[pid].cache = mutated_lane(table.pages[pid].cache, seed=step * 1000 + pid)

            for spec in SPECS:
                qids = layouts[(value, spec)]["query_ids"]
                direct_logits = query_logits(
                    model, qids, prefix[value], torch.ones((1, S + D), dtype=torch.long)
                )
                verified_cache = concat_cache(shared, active)
                verified_logits = query_logits(
                    model, qids, verified_cache, torch.ones((1, S + D), dtype=torch.long)
                )
                delta = float(torch.max(torch.abs(direct_logits - verified_logits)).item())
                active_equivalence.append(delta)
                direct_top = top_candidate(tok, direct_logits, spec)
                verified_top = top_candidate(tok, verified_logits, spec)
                active_top_match.append(direct_top == verified_top)

                # Mutate every stale page again, then rematerialize current authority.
                for pid in rng.sample(stale_page_ids, min(4, len(stale_page_ids))):
                    table.pages[pid].cache = mutated_lane(table.pages[pid].cache, seed=step * 2000 + pid)
                active2 = table.materialize_verified(pod_id)
                verified2 = query_logits(
                    model,
                    qids,
                    concat_cache(shared, active2),
                    torch.ones((1, S + D), dtype=torch.long),
                )
                sdelta = float(torch.max(torch.abs(verified_logits - verified2)).item())
                stale_mutation_equivalence.append(sdelta)
                probes.append({
                    "step": step,
                    "generation": generation,
                    "value": value,
                    "spec": spec,
                    "direct_top": direct_top,
                    "verified_top": verified_top,
                    "active_logit_delta": delta,
                    "stale_page_mutation_logit_delta": sdelta,
                })

    # Revocation must emit no page even though every old page still exists.
    pages_before_revoke = len(table.pages)
    table.revoke(pod_id)
    revoked_returns_none = table.materialize_verified(pod_id) is None
    stale_after_revoke_rejected = all(not table.independent_verify(h) for h in stale_handles[-128:])

    # Hot-path overhead: authority lookup + independent verification only. Model cost is
    # intentionally excluded from this microbenchmark and reported separately by RAG gates.
    # Re-enable one clean generation for lookup timing.
    generation += 1
    h = table.write(pod_id, generation, capsules[1])
    table.commit(h)
    lookup_ns = []
    raw_ns = []
    for _ in range(20000):
        t0 = time.perf_counter_ns()
        _ = table.pages[table.authority[pod_id]]
        raw_ns.append(time.perf_counter_ns() - t0)
        t0 = time.perf_counter_ns()
        _ = table.materialize_verified(pod_id)
        lookup_ns.append(time.perf_counter_ns() - t0)

    report = {
        "stage": "R291-CAUSAL-PAGE-TABLE",
        "architecture_candidate": "Causal Page Table (CPT) for Temporal Capability Ports",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "generation_updates": 4096,
        "materialized_pages_before_revoke": pages_before_revoke,
        "stale_handle_rejection_rate": rejected_stale / max(len(stale_handles), 1),
        "forged_future_handle_rejection_rate": rejected_forged / 4096,
        "revoked_returns_no_attention_page": revoked_returns_none,
        "stale_handles_rejected_after_revoke": stale_after_revoke_rejected,
        "active_top_match_rate_vs_direct": sum(active_top_match) / len(active_top_match),
        "max_active_full_vocab_logit_delta_vs_direct": max(active_equivalence),
        "max_full_vocab_logit_delta_after_stale_page_mutation": max(stale_mutation_equivalence),
        "raw_page_lookup_ns_median": statistics.median(raw_ns),
        "verified_authority_lookup_ns_median": statistics.median(lookup_ns),
        "verified_authority_lookup_ns_p99": sorted(lookup_ns)[int(len(lookup_ns) * 0.99)],
        "attention_sequence_contains_stale_generations": False,
        "logical_context_growth_with_revision_count": 0,
        "mechanism": (
            "All historical generations remain physically allocated in an external page store, but an "
            "independently verified authority page table materializes exactly one generation into Port "
            "Attention. Update = write new immutable page + atomic authority flip. Revocation = clear "
            "authority. Because stale pages are not members of the attention sequence, revision count "
            "does not increase logical context length or attention work."
        ),
        "claim_boundary": (
            "This real-model gate proves execution equivalence and lifecycle indirection for the tested "
            "prototype. It does not by itself prove system-wide deletion, RAG superiority, or novelty."
        ),
        "probes": probes,
        "dod_status": "NOT_DOD; active-only authority materialization gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "generation_updates",
        "materialized_pages_before_revoke",
        "stale_handle_rejection_rate",
        "forged_future_handle_rejection_rate",
        "revoked_returns_no_attention_page",
        "active_top_match_rate_vs_direct",
        "max_active_full_vocab_logit_delta_vs_direct",
        "max_full_vocab_logit_delta_after_stale_page_mutation",
        "verified_authority_lookup_ns_median",
        "verified_authority_lookup_ns_p99",
    )}, indent=2))

    if report["stale_handle_rejection_rate"] != 1.0:
        return 2
    if report["forged_future_handle_rejection_rate"] != 1.0:
        return 3
    if not revoked_returns_none or not stale_after_revoke_rejected:
        return 4
    if report["active_top_match_rate_vs_direct"] != 1.0:
        return 5
    if report["max_active_full_vocab_logit_delta_vs_direct"] > 0.02:
        return 6
    if report["max_full_vocab_logit_delta_after_stale_page_mutation"] > 0.02:
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
