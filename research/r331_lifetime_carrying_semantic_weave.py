from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

STAGE = "R331-LIFETIME-CARRYING-SEMANTIC-WEAVE"
PODS = int(os.environ.get("SO_R331_PODS", "10000"))
DOCUMENTS = int(os.environ.get("SO_R331_DOCUMENTS", "20000"))
CLAUSES = int(os.environ.get("SO_R331_CLAUSES", "8"))
UPDATES = int(os.environ.get("SO_R331_UPDATES", "12000"))
REPORT_PATH = Path(os.environ.get("SO_R331_REPORT", "r331_report.json"))
SEED = 3310914


@dataclass
class Cell:
    generation: int
    value: int


@dataclass
class Clause:
    direct_pods: tuple[int, ...]
    parents: tuple[int, ...]
    template: int
    lifetime: frozenset[tuple[int, int]] = frozenset()
    text: str = ""


@dataclass
class Document:
    clauses: list[Clause]


def render_clause(c: Clause, world: list[Cell], parent_texts: list[str]) -> str:
    vals = [world[p].value for p in c.direct_pods]
    # Deterministic stand-in for free-form clause generation: the text depends on
    # current world inputs and optional parent conclusions. What matters here is
    # causal segmentation/lifetime, not language quality.
    h = 1469598103934665603
    for v in vals:
        h ^= v + 0x9E3779B97F4A7C15
        h = (h * 1099511628211) & ((1 << 64) - 1)
    for t in parent_texts:
        for ch in t.encode()[:24]:
            h ^= ch
            h = (h * 1099511628211) & ((1 << 64) - 1)
    if c.template == 0:
        return f"current assessment {h % 997}"
    if c.template == 1:
        return f"derived comparison {h % 991}"
    if c.template == 2:
        return f"therefore state {h % 983}"
    return f"explanation marker {h % 977}"


def compute_clause(doc: Document, idx: int, world: list[Cell]) -> tuple[str, frozenset[tuple[int, int]]]:
    c = doc.clauses[idx]
    parent_texts = [doc.clauses[p].text for p in c.parents]
    deps = {(pid, world[pid].generation) for pid in c.direct_pods}
    for p in c.parents:
        deps.update(doc.clauses[p].lifetime)
    return render_clause(c, world, parent_texts), frozenset(deps)


def fresh_document(doc: Document, world: list[Cell]) -> list[str]:
    out = []
    tmp = Document([Clause(c.direct_pods, c.parents, c.template) for c in doc.clauses])
    for i in range(len(tmp.clauses)):
        txt, life = compute_clause(tmp, i, world)
        tmp.clauses[i].text = txt
        tmp.clauses[i].lifetime = life
        out.append(txt)
    return out


def build_docs(rng: random.Random, world: list[Cell]):
    docs: list[Document] = []
    pod_to_nodes: dict[int, list[tuple[int, int]]] = defaultdict(list)
    parent_to_children: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for d in range(DOCUMENTS):
        clauses = []
        for i in range(CLAUSES):
            # ~20% static clauses; others read one or two current Pods.
            if rng.random() < 0.20:
                pods = ()
            else:
                pods = tuple(rng.sample(range(PODS), 1 if rng.random() < 0.75 else 2))
            # Sparse semantic DAG; parents always earlier so one forward pass works.
            parents = ()
            if i > 0 and rng.random() < 0.32:
                count = 1 if i == 1 or rng.random() < 0.80 else 2
                parents = tuple(sorted(rng.sample(range(i), min(count, i))))
            clauses.append(Clause(pods, parents, rng.randrange(4)))
        doc = Document(clauses)
        for i in range(CLAUSES):
            txt, life = compute_clause(doc, i, world)
            doc.clauses[i].text = txt
            doc.clauses[i].lifetime = life
            for pid in doc.clauses[i].direct_pods:
                pod_to_nodes[pid].append((d, i))
            for p in doc.clauses[i].parents:
                parent_to_children[(d, p)].append((d, i))
        docs.append(doc)
    return docs, pod_to_nodes, parent_to_children


def descendants(start_nodes: list[tuple[int, int]], parent_to_children) -> set[tuple[int, int]]:
    out = set(start_nodes)
    q = deque(start_nodes)
    while q:
        n = q.popleft()
        for c in parent_to_children.get(n, ()):
            if c not in out:
                out.add(c); q.append(c)
    return out


def main() -> None:
    rng = random.Random(SEED)
    world = [Cell(1, rng.randrange(1_000_000)) for _ in range(PODS)]
    t0 = time.perf_counter()
    docs, pod_to_nodes, parent_to_children = build_docs(rng, world)
    build_seconds = time.perf_counter() - t0

    total_clauses = DOCUMENTS * CLAUSES
    weave_invalidated = 0
    monolithic_suffix_invalidated = 0
    full_document_invalidated = 0
    same_value_updates = 0
    exact_audit_mismatches = 0
    stale_lifetime_survivors = 0
    touched_counts = []
    mono_counts = []
    update_ns = []

    for u in range(UPDATES):
        pid = rng.randrange(PODS)
        cell = world[pid]
        old_gen = cell.generation
        same = rng.random() < 0.40
        old_value = cell.value
        cell.generation += 1
        cell.value = old_value if same else rng.randrange(1_000_000)
        same_value_updates += int(same)

        ts = time.perf_counter_ns()
        direct = list(pod_to_nodes.get(pid, ()))
        affected = descendants(direct, parent_to_children)
        touched_counts.append(len(affected))
        weave_invalidated += len(affected)

        # Conservative monolithic autoregressive control: once a world-dependent
        # clause appears, every later clause in that document is downstream through
        # hidden state and must be regenerated.
        mono_nodes = set()
        by_doc: dict[int, int] = {}
        for d, i in direct:
            by_doc[d] = min(by_doc.get(d, CLAUSES), i)
        for d, start in by_doc.items():
            for i in range(start, CLAUSES):
                mono_nodes.add((d, i))
        mono_counts.append(len(mono_nodes))
        monolithic_suffix_invalidated += len(mono_nodes)
        full_document_invalidated += len(by_doc) * CLAUSES

        # Regenerate only affected semantic nodes in topological order.
        for d, i in sorted(affected):
            txt, life = compute_clause(docs[d], i, world)
            docs[d].clauses[i].text = txt
            docs[d].clauses[i].lifetime = life
        update_ns.append(time.perf_counter_ns() - ts)

        # Any clause still claiming the old generation is an escaped stale lifetime.
        old_dep = (pid, old_gen)
        for d, i in affected:
            if old_dep in docs[d].clauses[i].lifetime:
                stale_lifetime_survivors += 1

        # Periodic full-current oracle over sampled documents.
        if u % 120 == 0:
            for d in rng.sample(range(DOCUMENTS), 40):
                fresh = fresh_document(docs[d], world)
                exact_audit_mismatches += sum(
                    a != b for a, b in zip(fresh, [c.text for c in docs[d].clauses])
                )

    mean_weave = statistics.mean(touched_counts)
    mean_mono = statistics.mean(mono_counts)
    report = {
        "stage": STAGE,
        "architecture_candidate": "Lifetime-Carrying Semantic Weave (LCSW) / segment-isolated generation DAG",
        "pods": PODS,
        "documents": DOCUMENTS,
        "clauses_per_document": CLAUSES,
        "total_semantic_clauses": total_clauses,
        "updates": UPDATES,
        "same_value_generation_updates": same_value_updates,
        "full_current_oracle_mismatches": exact_audit_mismatches,
        "stale_generation_lifetime_survivors": stale_lifetime_survivors,
        "semantic_weave_invalidated_clauses": weave_invalidated,
        "monolithic_autoregressive_suffix_invalidated_clauses": monolithic_suffix_invalidated,
        "whole_document_invalidated_clauses": full_document_invalidated,
        "mean_weave_invalidated_clauses_per_update": mean_weave,
        "mean_monolithic_suffix_invalidated_clauses_per_update": mean_mono,
        "invalidation_reduction_vs_monolithic_suffix": 1.0 - weave_invalidated / max(1, monolithic_suffix_invalidated),
        "invalidation_reduction_vs_whole_document": 1.0 - weave_invalidated / max(1, full_document_invalidated),
        "p99_weave_invalidated_clauses_per_update": sorted(touched_counts)[int(0.99 * (len(touched_counts) - 1))],
        "median_update_repair_ns_python": statistics.median(update_ns),
        "build_seconds": build_seconds,
        "contract_pass": exact_audit_mismatches == 0 and stale_lifetime_survivors == 0,
        "mechanism": (
            "Free-form response structure is compiled into independently regenerable semantic clauses. Each clause is rendered from immutable "
            "language/plan context plus explicit current world inputs and optional parent semantic clauses; its lifetime is the transitive union "
            "of those dependencies. A world update regenerates only the affected semantic DAG descendants instead of the entire later token suffix."
        ),
        "architectural_hypothesis": (
            "Autoregressive hidden state makes every later token a conservative descendant of an early mutable read. Segment-isolated semantic "
            "generation inserts explicit causal reset boundaries between clauses, allowing natural-language output to remain lifecycle-aware "
            "without forcing all subsequent prose to inherit every earlier world lifetime."
        ),
        "dod_status": "NOT_DOD; semantic-segmentation lifetime mechanism gate",
        "claim_boundary": (
            "Incremental computation, DAG recomputation, structured generation and modular decoding are established. R331 tests their use as "
            "explicit neural/semantic lifetime boundaries for governed free-form output; standalone novelty is not claimed."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("Lifetime-Carrying Semantic Weave contract failed")


if __name__ == "__main__":
    main()
