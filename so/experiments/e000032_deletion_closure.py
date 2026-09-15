"""Experiment E-000032 — the deletion closure of a store, and the certificate that composes with it.

E-000030 produced this programme's first certified deletions: for every value a deleted payload could
hold, the model's computation is bit-identical, so no attack that exists or will be invented can tell
the values apart. That is a statement about a RECORD. A data subject does not ask about a record. They
ask whether the FACT is gone, and this repository already holds the case where the two come apart
completely: `dependency/derivable_recovery_after_revoke_K3 = 1.0` in every seed of E-000019 -- a
certified record deletion under which every derivable fact survives, totally.

THE COMPOSITION THIS EXPERIMENT MEASURES.

    record-level certificate over R  +  R covers the fact closure of the fact
    ------------------------------------------------------------------------
                        fact-level certificate

The second premise is a property of the STORE, not of a checkpoint: `so.closure.fact_closure` computes
it with the mechanical resolver and no model at all, and reports a certified lower bound beside the
greedy answer so "optimal" is verified rather than assumed. The guarantee therefore factorises into
one expensive model-side proof and one cheap store-side search -- and the search is exactly what
canonicalisation makes trivial.

WHY THE SYMLINK IS THE POINT AND NOT AN ORNAMENT. Two stores are built from the SAME world with the
SAME ground truth (E-000015's `sample_alias_world` and `load_arm`), so they present an IDENTICAL
interface: every key resolves to the same object in both. In the canonical arm the group's k access
keys are LINK cells sharing one object; in the duplicated arm each key carries its own copy. At the
record level the two stores are indistinguishable -- per-key closure is one in both -- and their
erasure cost differs by a factor of k. That gap is what a record-level certificate cannot see and what
this experiment puts a number on.

The honest framing, stated here so the record carries it: the gap is redundancy, and one operation
reaching only one of its places -- Codd's MODIFICATION anomaly applied to a delete, not his DELETION
anomaly, which is the opposite failure of losing information nobody asked to lose. Normalization as
the remedy is 1971. What is not Codd is that normalization is free in a database,
where a join is exact, and is NOT free in a neural memory, where the reader must LEARN to dereference
and can refuse or fail. E-000025 priced that on a frozen GPT-2: sharing costs 0.0954 and having
trained on links at all costs 0.0688, worst of three seeds over all twelve phrasings. This experiment
adds the other half -- what the price BUYS, in units of certificate.

WHAT COULD FALSIFY IT. Four controls, each able to void the comparison:
  * the two arms must present the same interface, or nothing below is a like-for-like comparison;
  * per-key closure must be one in BOTH arms, or the stores were already distinguishable at the
    record level and the fact level is not doing the work;
  * the model must READ the fact before any deletion, or "the fact is unreachable afterwards" is
    vacuous;
  * the payload must still BE in the store afterwards, because EVICT is retention-preserving by
    design: the honest verdict is two-part -- unreachable to the reader, retained in the store -- and
    if retention failed, EVICT would have become a DELETE and RESTORE would be broken;
  * the removed payload must not reach any SURVIVING bank row, swept over its whole domain in the
    store -- a membership test cannot see that MVCCStore.bank() builds an alias row's link key from the
    target cell, so a surviving row is computed from the removed one;
  * the payload must REACH the outputs while its row is in the bank, or its absence afterwards is not
    evidence that anything was deleted -- this is the control that keeps the membership claim honest,
    and its lack is what made the first version of E-000030's DELETE arm certify by not testing;
  * removing the whole closure must certify in BOTH arms, or the instrument reports "canonical" where
    it should report "enough records removed".

WHAT A REVIEW CORRECTED, AND THIS VERSION DOES DIFFERENTLY (ledger §31.33). The first version
reported `model_evaluations_per_deletion = 0.0` as a literal, timed the reachability control outside
the per-deletion cost although it runs once per fact, inherited E-000030's mediation check from a
configuration without links, and described the reader as the frozen GPT-2 when it is the E-000015
`MutableKnowledgeTransformer` trained from scratch. Now: what the model is asked to do inside the
certification window is COUNTED by wrapping `forward` and `encode_bank`, the per-fact cost is
reported with the control inside it, `check_mediation` is run on THIS configuration
(`use_links=True, n_deref=1`) in every arm as a pre-registered control, and the prose names the
reader. The `(closure - 1) / keys_per_group` comparison is kept and re-described: on a star it is
the store's arithmetic, on a chain it is wrong (`test_closure_minus_one_over_keys_is_star_arithmetic
_and_not_a_store_law`), and what the 0.0000 measures is the reader's fidelity to the resolver.

Trains nothing when the E-000015 checkpoints are present.

Run:  python -m so.experiments.e000032_deletion_closure [--seeds 0 1 2] [--n-groups 25]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from so import ledger
from so.audit import (certify_encoding, certify_fact, certify_store_absence,
                      certify_structural, check_absence, check_mediation, check_retention)
from so.closure import closure_profile, fact_closure
from so.data import bank_from_store
from so.experiments.e000015_symlink_cells import (EVAL, _q1, encode_slots, load_arm, predict,
                                                  sample_alias_world, train_or_load)
from so.experiments.common import position_of_kid
from so.experiments.e000001b_mini_transformer import _sha256, CHECKPOINTS, CKPT_SUFFIX
from so.reference import ReferenceResolver
from so.world import UNKNOWN, World

ARMS = ("canonical", "mixed", "duplicated")
N_GROUPS = 25            # alias groups certified per seed; each costs one encoding sweep per arm


def load_mixed(world, spec, centre, seed: int, n_linked: int):
    """A store where only SOME of each group's aliases are links; the rest are copies.

    Partial normalization is the realistic state of any store that grew, and it is what makes the
    closure a scale rather than a label. ``n_linked`` aliases per group become LINK cells and the
    remainder are written out as their own facts, so the group's closure is
    ``1 + (n_alias - n_linked)`` while its interface is unchanged.
    """
    from so.mvcc import MVCCStore
    store = MVCCStore(marker_dim=centre.shape[0], seed=seed, marker_centre=centre)
    kids: Dict[Tuple[int, int], int] = {}
    linked: Dict[Tuple[int, int], bool] = {}
    for t, aliases in spec.groups:
        for i, k in enumerate(aliases):
            linked[k] = i < n_linked
    for f in world.facts:
        if f.key not in spec.alias_of:
            kids[f.key] = store.write(f.subject, f.relation, f.obj, provenance="fact")
    for f in world.facts:
        if f.key in spec.alias_of:
            if linked.get(f.key, False):
                kids[f.key] = store.link(f.subject, f.relation, kids[spec.alias_of[f.key]],
                                         provenance="alias")
            else:
                kids[f.key] = store.write(f.subject, f.relation, f.obj, provenance="copy")
    return store, kids


def group_keys(spec, target_key) -> Tuple[Tuple[int, int], ...]:
    """The group's access keys: the target's own plus its aliases.

    Taken from the SPEC rather than from either store, so the two arms are measured over exactly the
    same key set and the only thing that differs is how the store holds them.
    """
    for t, aliases in spec.groups:
        if t == target_key:
            return tuple([t] + list(aliases))
    raise KeyError(target_key)


def read_keys(model, store, world, keys: Sequence[Tuple[int, int]]) -> np.ndarray:
    return predict(model, bank_from_store(store), world, [_q1(world, k) for k in keys]).answers


class ModelCalls:
    """Count what the model is asked to do inside a window, instead of asserting it.

    Wraps the instance's ``forward`` and ``encode_bank``. An ``encode_bank`` reached from inside a
    ``forward`` is that forward's own work and is not counted twice; ``encodes`` counts the standalone
    calls (``certify_encoding`` makes them). ``evaluations`` is forwards plus standalone encodes -- a
    standalone encode is a partial evaluation and is counted as one, which over-counts rather than
    under-counts. The wrappers are removed on exit so the model is as it was.
    """

    def __init__(self, model):
        self.model, self.forwards, self.encodes, self._depth = model, 0, 0, 0

    def __enter__(self):
        m = self.model
        self._forward, self._encode = m.forward, m.encode_bank

        def forward(*a, **k):
            self.forwards += 1
            self._depth += 1
            try:
                return self._forward(*a, **k)
            finally:
                self._depth -= 1

        def encode_bank(*a, **k):
            if self._depth == 0:
                self.encodes += 1
            return self._encode(*a, **k)

        m.forward, m.encode_bank = forward, encode_bank
        return self

    def __exit__(self, *exc):
        del self.model.forward
        del self.model.encode_bank
        return False

    @property
    def evaluations(self) -> int:
        return self.forwards + self.encodes


def reachability_control(model, store, world, kids: Sequence[int], batch_queries) -> Any:
    """Does the payload reach the outputs WHILE its rows are still in the bank?

    This is the positive control, and without it the whole arm is empty: "the row is not in the bank
    afterwards" is evidence of a deletion only if the row was there and mattered. It is run before any
    eviction, at the bank positions the cells occupy.
    """
    bank = bank_from_store(store)
    rows = [position_of_kid(store, k) for k in kids]
    batch = encode_slots(list(batch_queries), bank, world, model.cfg.max_hops, model.cfg.n_deref)
    return certify_structural(model, bank.tensors(), rows,
                              lambda b: model(b, batch.mode, batch.start, batch.rels, batch.hop_valid),
                              model.cfg.d_model, outputs_of=lambda o: o[0])


def certify_removal(model, store, world, bank_before, removed_kids: Sequence[int], control) -> Tuple[Any, Any]:
    """The record-level half, on the bank as it stands after the removal.

    EVICT takes the row out of the bank, so nothing is left to perturb -- and BOTH sweep instruments
    then answer vacuously. `certify_encoding` over an empty row set certifies with one evaluation, and
    `certify_structural` over an empty row set answers "no path", the strongest label in the ladder,
    on any bank at all including one whose rows are live. Neither is evidence.

    What carries the claim after an eviction is membership: the model reads the store only through the
    bank (`so/model.py:246`), so a payload with no bank row is not an input, over any domain and for
    every query. `check_absence` states that and requires the reachability CONTROL above, so the
    absence is tied to a payload that demonstrably mattered while it was there.
    """
    bank_after = bank_from_store(store)
    record = certify_encoding(model, bank_after.tensors(), [], world.n_entities,
                              interface_keys=("k_f", "v_f", "k_r", "v_r", "active"))
    absence = check_absence(bank_before, bank_after, list(removed_kids), control=control)
    # Stronger than membership, and needed here: MVCCStore.bank() builds a LINK row's
    # link_subject/link_relation from the TARGET cell, so a surviving alias row is computed from the
    # removed cell and no membership test can see it. Sweeping the removed payload over its whole
    # domain and comparing the surviving bank settles it exhaustively. store.bank() is the right
    # object to compare: every tensor Bank.tensors() hands the model is a function of its arrays.
    payload_gone = certify_store_absence(store, list(removed_kids), lambda st: st.bank(),
                                         world.n_entities, fields=("obj",))
    return record, absence, payload_gone


def run_seed(seed: int, n_groups: int, steps: int, verbose: bool = True) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000015_deref1{CKPT_SUFFIX}_seed{seed}.pt"
    out = train_or_load(seed, steps, n_deref=1)
    model, centre = out["model"], out["centre"]

    rng = np.random.default_rng(seed)
    world, spec = sample_alias_world(rng, EVAL["n_base"], EVAL["n_groups"], EVAL["n_alias_per_group"])
    n_alias = EVAL["n_alias_per_group"]
    stores = {"canonical": load_arm(world, spec, centre, seed, symlink=True),
              "mixed": load_mixed(world, spec, centre, seed, n_linked=n_alias // 2),
              "duplicated": load_arm(world, spec, centre, seed, symlink=False)}

    m: Dict[str, Any] = {"seed": seed, "n_entities": world.n_entities,
                         "checkpoint_sha256": _sha256(path) if path.exists() else "",
                         "n_alias_per_group": EVAL["n_alias_per_group"],
                         "keys_per_group": EVAL["n_alias_per_group"] + 1}
    t0 = time.time()

    # ---- control 1: the two arms present the same interface, or nothing below compares anything
    views = {a: ReferenceResolver(st).view() for a, (st, _) in stores.items()}
    answers = {a: {k: int(o) for k, (o, _) in v.items()} for a, v in views.items()}
    m["control/interface_identical"] = float(all(answers[a] == answers["canonical"] for a in ARMS))
    m["control/n_keys"] = len(answers["canonical"])

    # ---- control 2: per-key closure is one in BOTH arms -- indistinguishable at the record level
    chosen = [spec.groups[int(i)][0] for i in rng.permutation(len(spec.groups))[:n_groups]]
    all_keys = [k for t in chosen for k in group_keys(spec, t)]
    for arm, (store, _) in stores.items():
        prof = closure_profile(store, all_keys)
        m[f"{arm}/per_key_closure_max"] = float(prof.max)
        m[f"{arm}/per_key_closure_mean"] = float(prof.mean)
        m[f"{arm}/per_key_n"] = int(prof.n)

    # ---- control 3: the model reads the fact before anything is deleted
    truth = np.array([answers["canonical"][k] for k in all_keys])
    for arm, (store, _) in stores.items():
        m[f"{arm}/read_before_deletion"] = float((read_keys(model, store, world, all_keys) == truth).mean())
    m["control/read_before_deletion"] = min(m[f"{a}/read_before_deletion"] for a in ARMS)

    # ---- the closure itself, and the composition
    probe_queries = [_q1(world, k) for k in all_keys[: min(64, len(all_keys))]]

    # ---- control 4: the premise of the interface certificate, falsification-tested ON THIS
    # configuration (use_links=True, n_deref=1). E-000030 tested it on the bare configuration only,
    # and the first version of this experiment inherited it without re-checking (ledger §31.33). The
    # rows are the OBJECT cells of the first groups -- FACT cells in every arm, dereferenced by their
    # aliases in the canonical and mixed arms -- so the forward that is probed runs through the link
    # path the certificate is later used on. The check samples: it can refute, not establish.
    for arm, (store, kids) in stores.items():
        bank = bank_from_store(store)
        rows = [position_of_kid(store, kids[t]) for t in chosen[:4]]
        batch = encode_slots(list(probe_queries), bank, world, model.cfg.max_hops, model.cfg.n_deref)

        def run(b, _batch=batch):
            with torch.no_grad():
                return model(b, _batch.mode, _batch.start, _batch.rels, _batch.hop_valid)

        med = check_mediation(model, bank.tensors(), rows, world.n_entities, run,
                              outputs_of=lambda o: o[0], n_probes=8, seed=seed)
        m[f"{arm}/mediation_consistent"] = float(med.consistent)
        m[f"{arm}/mediation_encoding_invariant"] = float(med.encoding_invariant)
        m[f"{arm}/mediation_output_invariant"] = float(med.output_invariant)
        if verbose:
            print(f"  seed {seed} {arm:<11} mediation on use_links/n_deref=1: "
                  f"{'consistent' if med.consistent else 'VOID'} -- {med.note}", flush=True)

    per_arm: Dict[str, Dict[str, List[float]]] = {a: {} for a in ARMS}
    for arm in ARMS:
        store, kids = stores[arm]
        sizes, optimal, one_valid, all_valid = [], [], [], []
        one_reads, all_reads, one_absent, control_ok = [], [], [], []
        one_store, one_addr = [], []
        search_seconds, guarantee_seconds, control_seconds = [], [], []
        one_retained = []
        one_forwards, one_encodes, all_forwards, all_encodes, control_forwards = [], [], [], [], []
        for t_key in chosen:
            keys = group_keys(spec, t_key)
            obj = answers[arm][t_key]
            t_search = time.time()
            fc = fact_closure(store, keys, obj=obj)
            search_seconds.append(time.time() - t_search)
            sizes.append(fc.size)
            optimal.append(float(fc.optimal))

            # the positive control, BEFORE anything is removed: the payload must reach the outputs,
            # or its absence afterwards is not evidence that anything was deleted. This is a validity
            # check on the INSTRUMENT rather than a per-deletion cost -- it establishes that the
            # measurement can fail -- so it is timed separately from the guarantee itself.
            whole = sorted(set(list(fc.records) + [kids[t_key]]))
            t_ctl = time.time()
            with ModelCalls(model) as ctl_calls:
                control = reachability_control(model, store, world, whole, probe_queries)
            control_seconds.append(time.time() - t_ctl)
            control_forwards.append(float(ctl_calls.forwards))
            control_ok.append(float(control.reachable))
            bank_before = bank_from_store(store)

            # ARM "one record": remove exactly what a record-level certificate would cover -- the
            # object itself. In a pod that IS the closure; under duplication it is one of k.
            # the deletion and everything that certifies it. What the model is asked to do inside
            # this window is COUNTED by the wrapper, not asserted: the first version wrote 0.0 here as
            # a literal while certify_encoding's reference fingerprint runs encode_bank once.
            t_guarantee = time.time()
            with ModelCalls(model) as calls:
                store.evict(kids[t_key])
                rec, absence, payload_gone = certify_removal(model, store, world, bank_before,
                                                             [kids[t_key]], control)
                retention = check_retention(store, [kids[t_key]])
                cert_one = certify_fact(rec, fc, [kids[t_key]], store_after=store, keys=keys,
                                        absence=absence, store_absence=payload_gone, retention=retention,
                                        residual_note="says nothing about what the core knew before the store existed")
            guarantee_seconds.append(time.time() - t_guarantee + search_seconds[-1])
            one_forwards.append(float(calls.forwards))
            one_encodes.append(float(calls.encodes))
            one_valid.append(float(cert_one.valid))
            one_retained.append(float(retention.retained))
            one_absent.append(float(absence.certified_absent))
            one_store.append(float(payload_gone.certified))
            # the address is a separate claim and it does NOT hold: an alias row keeps pointing at the
            # evicted cell, so link_subject/link_relation move when its key moves. Recorded, not hidden.
            addr = certify_store_absence(store, [kids[t_key]], lambda st: st.bank(),
                                         world.n_entities, fields=("subject",))
            one_addr.append(float(addr.certified))
            one_reads.append(float((read_keys(model, store, world, keys) == obj).mean()))

            # ARM "whole closure": remove every record the store's own semantics needs
            with ModelCalls(model) as calls_all:
                for kid in [k for k in fc.records if k != kids[t_key]]:
                    store.evict(kid)
                rec_all, absence_all, payload_all = certify_removal(model, store, world, bank_before,
                                                                    whole, control)
                cert_all = certify_fact(rec_all, fc, whole, store_after=store, keys=keys,
                                        absence=absence_all, store_absence=payload_all)
            all_forwards.append(float(calls_all.forwards))
            all_encodes.append(float(calls_all.encodes))
            all_valid.append(float(cert_all.valid))
            all_reads.append(float((read_keys(model, store, world, keys) == obj).mean()))

            for kid in whole:
                store.restore(kid)
        per_arm[arm] = dict(sizes=sizes, optimal=optimal, one_valid=one_valid, all_valid=all_valid,
                            one_reads=one_reads, all_reads=all_reads, one_absent=one_absent,
                            control_ok=control_ok)
        m[f"{arm}/fact_closure_mean"] = float(np.mean(sizes))
        m[f"{arm}/fact_closure_min"] = float(np.min(sizes))
        m[f"{arm}/fact_closure_max"] = float(np.max(sizes))
        m[f"{arm}/fact_closure_optimal_rate"] = float(np.mean(optimal))
        m[f"{arm}/one_record_fact_certified"] = float(np.mean(one_valid))
        m[f"{arm}/one_record_payload_absent"] = float(np.mean(one_absent))
        m[f"{arm}/one_record_payload_store_absent"] = float(np.mean(one_store))
        m[f"{arm}/one_record_address_store_absent"] = float(np.mean(one_addr))
        # EVICT keeps the payload in the store on purpose, so the verdict is two-part:
        # unreachable to the reader, retained in the store. Recorded, never elided.
        m[f"{arm}/one_record_retained_in_store"] = float(np.mean(one_retained))
        # what a certified fact deletion COSTS, as measured and not as asserted. The reachability
        # control runs once PER FACT in this design (it is the positive control for that fact's
        # payload), so it is inside the per-fact cost; the first version timed it separately and
        # described it as once per instrument. The evaluation counts come from the wrapper.
        m[f"{arm}/closure_search_seconds"] = float(np.mean(search_seconds))
        m[f"{arm}/certified_deletion_seconds"] = float(np.mean(guarantee_seconds))
        m[f"{arm}/instrument_control_seconds"] = float(np.mean(control_seconds))
        m[f"{arm}/per_fact_seconds"] = float(np.mean(np.array(guarantee_seconds) + np.array(control_seconds)))
        m[f"{arm}/model_forwards_per_deletion"] = float(np.mean(one_forwards))
        m[f"{arm}/model_encodes_per_deletion"] = float(np.mean(one_encodes))
        m[f"{arm}/model_evaluations_per_deletion"] = float(np.mean(np.array(one_forwards) + np.array(one_encodes)))
        m[f"{arm}/model_evaluations_whole_closure"] = float(np.mean(np.array(all_forwards) + np.array(all_encodes)))
        m[f"{arm}/control_model_forwards"] = float(np.mean(control_forwards))
        m[f"{arm}/control_reachable_before"] = float(np.mean(control_ok))
        m[f"{arm}/one_record_still_readable"] = float(np.mean(one_reads))
        m[f"{arm}/whole_closure_fact_certified"] = float(np.mean(all_valid))
        m[f"{arm}/whole_closure_still_readable"] = float(np.mean(all_reads))
        # The closure is a store statistic; this is what makes it more than bookkeeping. After
        # removing ONLY the object, the records that survive are exactly the closure's other members,
        # one per access path that is a copy rather than a pointer, so the reader should still answer
        # for (closure - 1) of the group's keys and for no others. Predicted per group from the store
        # alone, then compared -- a store-side number put at risk against a neural measurement rather
        # than reported beside one.
        predicted = [(c - 1) / m["keys_per_group"] for c in sizes]
        err = np.abs(np.array(predicted) - np.array(one_reads))
        m[f"{arm}/predicted_still_readable"] = float(np.mean(predicted))
        m[f"{arm}/prediction_error"] = float(np.mean(err))
        m[f"{arm}/prediction_error_max"] = float(np.max(err))
        if verbose:
            print(f"  seed {seed} {arm:<11} closure {m[f'{arm}/fact_closure_mean']:.2f} "
                  f"(optimal {m[f'{arm}/fact_closure_optimal_rate']:.2f})  one-record certified "
                  f"{m[f'{arm}/one_record_fact_certified']:.2f}  still readable "
                  f"{m[f'{arm}/one_record_still_readable']:.2f}  whole-closure certified "
                  f"{m[f'{arm}/whole_closure_fact_certified']:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    m["gap/certificates_per_record"] = (m["canonical/fact_closure_mean"] and
                                        m["duplicated/fact_closure_mean"] / m["canonical/fact_closure_mean"])
    m["n_groups"] = len(chosen)
    m["seconds"] = time.time() - t0
    return m


KEYS = (["control/interface_identical", "control/read_before_deletion", "control/n_keys",
         "gap/certificates_per_record", "n_groups"] +
        [f"{a}/{k}" for a in ARMS
         for k in ("per_key_closure_max", "per_key_closure_mean", "read_before_deletion",
                   "fact_closure_mean", "fact_closure_min", "fact_closure_max",
                   "fact_closure_optimal_rate", "one_record_fact_certified",
                   "one_record_payload_absent", "one_record_payload_store_absent",
                   "one_record_address_store_absent", "one_record_retained_in_store",
                   "control_reachable_before",
                   "one_record_still_readable",
                   "whole_closure_fact_certified", "whole_closure_still_readable",
                   "predicted_still_readable", "prediction_error", "prediction_error_max",
                   "closure_search_seconds", "certified_deletion_seconds",
                   "instrument_control_seconds", "per_fact_seconds",
                   "model_forwards_per_deletion", "model_encodes_per_deletion",
                   "model_evaluations_per_deletion", "model_evaluations_whole_closure",
                   "control_model_forwards", "mediation_consistent",
                   "mediation_encoding_invariant", "mediation_output_invariant")])

CRITERIA = {
    # controls: any of these failing voids the comparison rather than weakening it
    "control/interface_identical": (">=", 1.0),
    "control/read_before_deletion": (">=", 0.90),
    "canonical/per_key_closure_max": ("<=", 1.0),
    "duplicated/per_key_closure_max": ("<=", 1.0),
    # the claim
    "canonical/fact_closure_max": ("<=", 1.0),
    "duplicated/fact_closure_min": (">=", 3.0),
    "canonical/fact_closure_optimal_rate": (">=", 1.0),
    "duplicated/fact_closure_optimal_rate": (">=", 1.0),
    "canonical/control_reachable_before": (">=", 1.0),
    # the interface certificate's premise, falsification-tested on the configuration in use; a
    # VOID here voids every certificate below it (added for the re-run, ledger §31.33)
    "canonical/mediation_consistent": (">=", 1.0),
    "mixed/mediation_consistent": (">=", 1.0),
    "duplicated/mediation_consistent": (">=", 1.0),
    # EVICT is retention-preserving by design, so this must be 1.0 -- if it were not, EVICT would
    # have quietly become a DELETE and RESTORE would be broken
    "canonical/one_record_retained_in_store": (">=", 1.0),
    # the exhaustive store counterfactual, which is what actually carries "the payload is unreachable"
    "canonical/one_record_payload_store_absent": (">=", 1.0),
    "duplicated/one_record_payload_store_absent": (">=", 1.0),
    "duplicated/control_reachable_before": (">=", 1.0),
    "canonical/one_record_fact_certified": (">=", 1.0),
    "duplicated/one_record_fact_certified": ("<=", 0.0),
    # the model confirms the verdict rather than the verdict standing alone
    "canonical/one_record_still_readable": ("<=", 0.10),
    "duplicated/one_record_still_readable": (">=", 0.60),
    # and removing the whole closure certifies in BOTH arms: the instrument is not arm-shaped
    "canonical/whole_closure_fact_certified": (">=", 1.0),
    "duplicated/whole_closure_fact_certified": (">=", 1.0),
    "duplicated/whole_closure_still_readable": ("<=", 0.10),
    # partial normalization buys exactly the part it normalized, and no more
    "mixed/fact_closure_mean": (">=", 2.0),
    "mixed/fact_closure_max": ("<=", 2.0),
    "mixed/one_record_fact_certified": ("<=", 0.0),
    # the store statistic predicts the reader, in every arm, or it is only bookkeeping
    "canonical/prediction_error": ("<=", 0.05),
    "mixed/prediction_error": ("<=", 0.05),
    "duplicated/prediction_error": ("<=", 0.05),
}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-groups", type=int, default=N_GROUPS)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    per_seed = [run_seed(s, args.n_groups, args.steps) for s in args.seeds]
    numeric = [{k: float(v) for k, v in s.items() if isinstance(v, (bool, int, float))} for s in per_seed]
    keys = [k for k in KEYS if all(k in s for s in numeric)]
    agg = ledger.aggregate(numeric, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    rows = [[arm,
             f"{agg[f'{arm}/per_key_closure_mean']['mean']:.2f}",
             f"{agg[f'{arm}/fact_closure_mean']['mean']:.2f}",
             f"{agg[f'{arm}/fact_closure_optimal_rate']['min']:.2f}",
             f"{agg[f'{arm}/one_record_fact_certified']['min']:.4f}",
             f"{agg[f'{arm}/one_record_still_readable']['mean']:.4f}",
             f"{agg[f'{arm}/predicted_still_readable']['mean']:.4f}",
             f"{agg[f'{arm}/whole_closure_fact_certified']['min']:.4f}"]
            for arm in ARMS]
    tbl = ledger.table(["store", "closure per KEY", "closure per FACT", "proved optimal",
                        "one record: fact certified", "one record: still readable",
                        "predicted from the closure", "whole closure: fact certified"], rows)

    record = {"experiment": "E-000032",
              "title": "the deletion closure of a store, and the certificate that composes with it",
              "trains_nothing": True, "seeds": args.seeds, "n_groups": args.n_groups,
              "arms": list(ARMS), "per_seed": per_seed, "aggregate": agg, "criteria": check}
    md = [f"# E-000032 — {record['title']}", "",
          f"Seeds {args.seeds}, {args.n_groups} alias groups per seed, the recorded E-000015 one-slot",
          "checkpoints, no training. The reader is the E-000015 `MutableKnowledgeTransformer` -- trained",
          "from scratch on a 256-entity world with an explicit UNKNOWN head, worlds resampled every",
          "training step so no evaluation fact is in its weights -- and NOT the frozen GPT-2 adapter; an",
          "earlier version of this report said GPT-2 and was wrong (ledger §31.33). Both arms are built",
          "from the SAME world with the same ground truth, so they present an identical interface: every",
          "key resolves to the same object in both.", "",
          "## The gap a record-level certificate cannot see", "", tbl, "",
          "`closure per KEY` is how many records must go before THAT KEY stops answering. It is one in",
          "both arms, which is the point: at the record level the two stores are indistinguishable.",
          "`closure per FACT` is how many must go before NO key in the group yields the object, and it",
          "is where they separate. `proved optimal` is the fraction where the greedy search MET a",
          "certified lower bound (every live derivation is a must-hit set, so a pairwise-disjoint",
          "subfamily bounds the optimum from below) rather than merely being assumed exact.", "",
          "`one record: fact certified` removes exactly the object -- what a record-level certificate",
          "covers today -- and asks whether that licenses a fact-level statement. `still readable` is",
          "the model's own answer afterwards, so the verdict is confirmed by behaviour and not only by",
          "bookkeeping. `predicted from the closure` is `(closure - 1) / keys_per_group`, computed from",
          "the store before the model is run at all: removing only the object leaves exactly the copies",
          "that are separate records.", "",
          "What that agreement IS, stated after a review rather than before it (ledger §31.33): on the",
          "star topologies these arms are built from, every non-target closure member backs exactly one",
          "key, so the formula is that invariant restated and the 0.0000 measures the reader's FIDELITY",
          "to the store's own resolver, which E-000015 had already recorded at 1.0000 on these",
          "checkpoints. It is not a forecast. On a chain -- an alias pointing at a copy rather than at",
          "the object -- the formula is wrong by a full grid step against the mechanical resolver with no",
          "model in the loop (`test_closure_minus_one_over_keys_is_star_arithmetic_and_not_a_store_law`);",
          "the quantity that IS a function of the store is the post-deletion resolver count, and",
          "`certify_fact` already checks that one through `store_after`.", "",
          "## What a certified fact deletion costs, counted", "",
          "The model calls inside the certification window are COUNTED by wrapping `forward` and",
          "`encode_bank` -- the first version of this report wrote `0` as a literal, while",
          "`certify_encoding`'s reference fingerprint runs `encode_bank` once even over an empty row set.",
          "The reachability control is the positive control for each fact's payload and runs once PER",
          "FACT, so it belongs inside the per-fact cost; the first version timed it apart and called it",
          "once per instrument. `all in` is search + certification + control.", "",
          ledger.table(["store", "closure search (s)", "certification (s)", "forwards inside",
                        "standalone encodes inside", "control (s)", "control forwards",
                        "per fact, all in (s)"],
                       [[arm,
                         f"{agg[f'{arm}/closure_search_seconds']['mean']:.4f}",
                         f"{agg[f'{arm}/certified_deletion_seconds']['mean']:.4f}",
                         f"{agg[f'{arm}/model_forwards_per_deletion']['mean']:.1f}",
                         f"{agg[f'{arm}/model_encodes_per_deletion']['mean']:.1f}",
                         f"{agg[f'{arm}/instrument_control_seconds']['mean']:.2f}",
                         f"{agg[f'{arm}/control_model_forwards']['mean']:.1f}",
                         f"{agg[f'{arm}/per_fact_seconds']['mean']:.2f}"] for arm in ARMS]),
          "",
          "The store-side parts -- the closure search and the exhaustive store counterfactual -- run",
          "no forward pass; the certification window as a whole is not model-free, and the number in",
          "the table is what it costs.", "",
          "E-000024 is the comparison: deleting 50 facts from a LoRA took 129 s by gradient ascent and",
          "335 s by relabelling, changed 2,359,296 parameters, moved perplexity on ordinary prose from",
          "42.9 to 6.19e+09 and 6.39e+06, and admits no certificate at all -- there is no finite payload",
          "domain to sweep and no interface the data passes through.", "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), "",
          "## What this is and is not", "",
          "The gap between the two arms is Codd's MODIFICATION anomaly applied to a delete -- his",
          "DELETION anomaly is the opposite failure, unintended loss -- and normalization is its 1971",
          "remedy; this experiment does not claim otherwise. What it adds is that the anomaly decides whether a",
          "DELETION CERTIFICATE for a neural memory means anything, that the store-side half of the",
          "guarantee (closure search and store counterfactual) is computable without the model, and that in a neural memory the normalization is",
          "not free -- E-000025 prices it at 0.0954 for sharing and 0.0688 for link training on a frozen",
          "GPT-2, worst of three seeds across all twelve phrasings.", ""]
    path = ledger.save("e000032_deletion_closure", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(tbl)
    print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
