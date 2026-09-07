"""Experiment E-000051 -- the residue against the reader.

After a pod's object is evicted, ``ON DELETE SET NULL`` (BLANK) leaves two live self-referencing LINK
rows and ``NO ACTION`` (a dangling link) leaves two live LINK rows exporting the tombstone key. E-000046
counted them at the store level (exported-level history independence 0.0000 for both; 1.0000 for the
CASCADE that evicts every row). This experiment reads them through the two frozen readers and asks the
question the store cannot answer: do those rows change the reader's answers on queries that are NOT
about the pod?

THE ADVERSARY. Query-only, one system, no pre-deletion snapshot. For each pod p and each bank, five
features per probe query from the returned logits (the hypothesised object's logit, the UNKNOWN logit,
the best other-entity logit, the log-sum-exp, the softmax entropy), concatenated over a query class,
standardised on the training fold, a linear probe (so.attacks.LinearProbe) fitted on 80 pods and scored
on 20, five folds, AUC = the Mann-Whitney rank statistic over the held-out scores. Null band at n = 200
about 0.42-0.58; thresholds 0.60 / 0.75 with the grey zone (0.60, 0.75) pre-registered as inconclusive.

BANKS PER POD, all built from clones of one store. LIVE; CASCADE(p) = evict target and both aliases;
BLANK(p) = evict target, blank both aliases; DANGLE(p) = evict target only; NEVER(p) = a fresh store
written by the same code over the world with p's three facts removed (every row written after p's
target draws its marker one generator position earlier); MATCHED(p) = CASCADE(p), so BLANK and DANGLE
differ from it in exactly their two residue rows. Calibration, PER POD (the completeness critic's fix:
one bank per seed would give the probe two point masses): PERM(p), PERM2(p) = LIVE with two distinct
row permutations (the float-summation floor of a permutation-invariant reader); ADD2(p) = LIVE plus two
fresh live LINK rows at free keys pointing at a live base fact held by no pod (the row-count floor).

QUERY CLASSES. (i) the deleted keys: target and both aliases (synthetic: both surface forms plus the
reverse query for p's object, E-000028's channel; GPT-2: templates 3 and 10); (ii) bystanders: 16
fixed base keys held by no pod, the alias keys of two fixed other pods, and the 4 base rows written
immediately after p's target; (iii) generic: GPT-2, E-000017's five generic templates with p's subject
plus three with a bystander's, scored on the full vocabulary; synthetic, 8 fixed free keys.

ARMS (per reader, per seed, n pods). present: LIVE vs NEVER on class (i) -- adversary validity.
cascade_soft: CASCADE vs NEVER on (i)(ii)(iii) -- the seeded-marker channel alone. blank / dangle vs
MATCHED on (i)(ii)(iii) -- THE MEASUREMENT on (ii)(iii); (i) is a validity row (a live row against none
at the asked key). blank / dangle vs NEVER -- table only. perm: PERM vs PERM2 on (ii)(iii); add2: ADD2
vs PERM on (ii)(iii). Beside every AUC: max KL(post || reference) and top-1 agreement per class, and the
interface residual (max |delta| over the consumed encoding on rows common to both banks, aligned by
key). Store-level columns from so.audit.check_history_independence on the same stores.

WHAT IS OWNED AND CITED, NOT CLAIMED. Weak history independence: Naor and Teague 2001. Store + certified
reader: Garg, Goldwasser and Vasudevan 2020; Godin and Vasudevan 2022; Cohen, Smith, Swanberg and
Vasudevan 2023 -- 'HI of bank() is not HI of the reader' is their two-clause structure. Single-system
output-only unlearning detection on forget-irrelevant inputs: Chen et al., arXiv:2506.14003. Soft-delete
residue moving other queries: Ghost Vectors (2606.18497). Exact record deletion on a frozen LM's memory
against a never-stored reference, LiRA at chance: Ramesh, arXiv:2607.27539 -- the CASCADE-vs-NEVER cell
is his and is a pipeline check here. SET NULL / NO ACTION / CASCADE: SQL-92. What has not been read
through any reader: a SET NULL or NO ACTION pointer residue against a never-held reference on queries
off the pod, calibrated against two fresh live rows; and the UNKNOWN rate on a blanked alias, a row
shape (a self-referencing link) neither reader's training distribution contained.

IDENTITIES, DECLARED. ``dangle/deleted_key_unknown`` is E-000026's ``delete_target/alias_unknown``
re-measured: ``bank()`` exports the same key for an evicted target as for a deleted one. It is reported
as a reproduction and is not a criterion.

WHAT FALSIFIES WHAT (the decision rule, fixed before the run, is DECISION_RULE below). V1 present/auc_i
< 0.95 voids the reader. V3 perm AUC > 0.60 voids every AUC row (the adversary reads summation order).
V4 add2 AUC > 0.60 means the reader's off-pod outputs carry the number of rows: the ROW-COUNT reading,
under which no residue AUC is a history channel and only KL / top-1 carry. V5 blank/dangle vs MATCHED
on class (i) < 0.75 is the INVISIBLE reading. With V1-V5 passing: M1 (cascade vs NEVER <= 0.60 on every
class) is the marker channel at chance; M2 (blank/dangle vs MATCHED <= 0.60 off the pod) is CONFINED --
the negative sentence; any M2 row >= 0.75 with its add2 row <= 0.60 is the positive sentence, naming the
arm and the class, with the KL beside it (M3: KL <= 0.05 nats and top-1 agreement >= 0.98 are the
programme's own locality bars). L1, read independently: blank/deleted_key_unknown >= 0.90 and
blank/deleted_key_wrong_entity <= 0.05. Any AUC in (0.60, 0.75) is recorded as inconclusive at this n.

Trains nothing. Evidence level E5.

Run:  python -m so.experiments.e000051_residue_reader [--readers syn gpt2] [--seeds 0 1 2] [--n-pods 100]
      python -m so.experiments.e000051_residue_reader --quick --readers syn --seeds 0 --n-pods 6 --threads 1   (smoke)
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from so import ledger
from so.attacks import LinearProbe
from so.audit import check_history_independence
from so.data import Bank, bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000026_lifecycle_at_a_readable_template as E26
from so.world import Query, UNKNOWN, World, free_keys

READERS = ("syn", "gpt2")
CLASSES = ("i", "ii", "iii")
GPT2_TEMPLATES = (3, 10)             # E-000026's strong trained template and strong held-out template
GPT2_BYSTANDER_TEMPLATE = 3
N_BYSTANDER_BASE = 16
N_AFTER = 4
N_GENERIC_FREE = 8


# ------------------------------------------------------------------------------------- statistics
def auc_mann_whitney(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC as the Mann-Whitney U statistic over ranks, ties counted half. No sklearn on this box."""
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i: j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def cv_auc(x_pos: np.ndarray, x_neg: np.ndarray, seed: int, folds: int = 5, epochs: int = 300) -> float:
    """Five-fold cross-validated AUC of a linear probe: positives and negatives are paired by pod, so a
    pod's two banks are always in the same fold. Standardisation is fitted on the training fold only."""
    n = x_pos.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    pos_scores, neg_scores = [], []
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        xt = np.concatenate([x_pos[train], x_neg[train]])
        yt = np.concatenate([np.ones(len(train), dtype=np.int64), np.zeros(len(train), dtype=np.int64)])
        mu, sd = xt.mean(0), xt.std(0) + 1e-6
        probe = LinearProbe(xt.shape[1], 2, seed=seed + k)
        probe.fit((xt - mu) / sd, yt, epochs=epochs)
        with torch.no_grad():
            def score(x):
                z = probe.w(torch.as_tensor((x - mu) / sd, dtype=torch.float32)).numpy()
                return z[:, 1] - z[:, 0]
            pos_scores.append(score(x_pos[test])); neg_scores.append(score(x_neg[test]))
    return auc_mann_whitney(np.concatenate(pos_scores), np.concatenate(neg_scores))


def features(logits: np.ndarray, obj: np.ndarray, unknown_col: int, entity_cols: Optional[np.ndarray] = None
             ) -> np.ndarray:
    """Five features per row. ``logits`` (Q, K); ``obj`` (Q,) the hypothesised object's column;
    ``entity_cols`` restricts 'other entity' and the softmax to the entity columns (full-vocab GPT-2)."""
    lg = logits.astype(np.float64)
    q = np.arange(lg.shape[0])
    if entity_cols is None:
        ent = lg.copy()
        ent[:, unknown_col] = -np.inf
    else:
        ent = lg[:, entity_cols]
    obj_logit = lg[q, obj]
    unk_logit = lg[:, unknown_col]
    masked = ent.copy()
    if entity_cols is None:
        masked[q, obj] = -np.inf
    else:
        pos = {int(c): i for i, c in enumerate(entity_cols)}
        for r in q:
            if int(obj[r]) in pos:
                masked[r, pos[int(obj[r])]] = -np.inf
    other = masked.max(1)
    m = lg.max(1, keepdims=True)
    lse = (m[:, 0] + np.log(np.exp(lg - m).sum(1)))
    p = np.exp(lg - lse[:, None])
    ent_h = -(p * np.log(p + 1e-30)).sum(1)
    return np.stack([obj_logit, unk_logit, other, lse, ent_h], 1)


def kl_top1(post: np.ndarray, ref: np.ndarray) -> Tuple[float, float, float]:
    """max KL(post || ref) over rows, top-1 agreement, max |delta logit|."""
    lp = torch.log_softmax(torch.as_tensor(post, dtype=torch.float64), -1)
    lr = torch.log_softmax(torch.as_tensor(ref, dtype=torch.float64), -1)
    kl = (lp.exp() * (lp - lr)).sum(-1)
    agree = float((post.argmax(1) == ref.argmax(1)).mean())
    return float(kl.max()), agree, float(np.abs(post - ref).max())


# ------------------------------------------------------------------------------------------ banks
def permute_bank(b: Bank, perm: np.ndarray) -> Bank:
    inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    arr = lambda x: None if x is None else x[perm]
    remap = lambda d: None if d is None else {k: int(inv[v]) for k, v in d.items()}
    return Bank(subject=b.subject[perm], relation=b.relation[perm], obj=b.obj[perm], marker=b.marker[perm],
                active=b.active[perm], usable=b.usable[perm], kid=b.kid[perm], index_view=b.index_view,
                kid_of_key=remap(b.kid_of_key), active_pos=remap(b.active_pos), marker_valid=arr(b.marker_valid),
                routable=arr(b.routable), routable_pos=remap(b.routable_pos), is_link=arr(b.is_link),
                link_subject=arr(b.link_subject), link_relation=arr(b.link_relation),
                trace_of_key=None if b.trace_of_key is None else
                {k: tuple(int(inv[p]) for p in v) for k, v in b.trace_of_key.items()})


class Pod:
    def __init__(self, target, aliases, obj):
        self.target, self.aliases, self.obj = target, list(aliases), int(obj)

    @property
    def keys(self):
        return [self.target] + self.aliases


class Setting:
    """One reader, one seed: the world, the store, the fixed query sets, and the bank builders."""

    def __init__(self, reader: str, seed: int, n_pods: int, threads: int, content_markers: bool = False):
        self.reader, self.seed = reader, seed
        self.content_markers = content_markers          # E-000053: history-independent markers, default off
        rng = np.random.default_rng(2000 + seed)
        if reader == "syn":
            out = E15.train_or_load(seed, 4000, 1)
            self.model, self.centre = out["model"], out["centre"]
            self.sha = out.get("checkpoint_sha256", "")
            self.world, self.spec = E15.sample_alias_world(rng, 850, 100, 2)
            self.n_ent = self.world.n_entities
        else:
            self.gk, self.centre, self.sha = E26.load_link_adapter(seed)
            self.model = self.gk.model
            self.world, self.spec = E15.sample_alias_world(rng, 700, 100, 2, self.gk.n_entities, 4,
                                                           E20.N_TRAIN_TEMPLATES)
            self.n_ent = self.gk.n_entities
        self.store, self.kids = E15.load_arm(self.world, self.spec, self.centre, seed, symlink=True,
                                             content_markers=content_markers)
        self.live = bank_from_store(self.store)
        groups = self.spec.groups
        self.pods = [Pod(t, ks, self.world.index[t]) for t, ks in groups[:n_pods]]
        pod_keys = {k for t, ks in groups for k in [t] + list(ks)}
        self.base_keys = [f.key for f in self.world.facts if f.key not in self.spec.alias_of]
        self.base_pos = {k: i for i, k in enumerate(self.base_keys)}
        free_base = [k for k in self.base_keys if k not in pod_keys]
        r2 = np.random.default_rng(7000 + seed)
        self.bystander_base = [free_base[int(i)] for i in r2.choice(len(free_base), N_BYSTANDER_BASE, replace=False)]
        self.other_pod_aliases = [k for t, ks in groups[-2:] for k in ks]          # two fixed pods at the end
        self.add2_targets = [k for k in free_base if k not in self.bystander_base]
        self.free = free_keys(self.world)
        self.generic_free = [self.free[int(i)] for i in r2.choice(len(self.free), N_GENERIC_FREE, replace=False)]
        # ADD2's two fresh rows must not land on a generic-class key: the first synthetic run let them,
        # and a generic free key that had just become a live link read the link's target (KL 16.8 on
        # class iii for add2) -- an instrument leak, not a row-count effect
        self.add2_free = [k for k in self.free if k not in set(self.generic_free)]
        self.rng_pod = np.random.default_rng(9000 + seed)

    # ---- banks
    def _clone(self):
        return self.store.clone_by_replay()

    def bank_cascade(self, p: Pod):
        s = self._clone()
        for k in p.keys:
            s.evict(self.kids[k])
        return bank_from_store(s), s

    def bank_blank(self, p: Pod):
        s = self._clone(); s.evict(self.kids[p.target])
        for a in p.aliases:
            s.blank(self.kids[a])
        return bank_from_store(s), s

    def bank_dangle(self, p: Pod):
        s = self._clone(); s.evict(self.kids[p.target])
        return bank_from_store(s), s

    def bank_never(self, p: Pod):
        drop = set(p.keys)
        w = World(self.world.n_entities, self.world.n_relations, self.world.n_synonyms,
                  [f for f in self.world.facts if f.key not in drop])
        spec = E15.AliasSpec({k: v for k, v in self.spec.alias_of.items() if k not in drop},
                             [g for g in self.spec.groups if g[0] != p.target])
        s, _ = E15.load_arm(w, spec, self.centre, self.seed, symlink=True, content_markers=self.content_markers)
        return bank_from_store(s), s

    def bank_perm(self, p: Pod, which: int):
        r = np.random.default_rng(11000 + self.seed * 1000 + 2 * self.pods.index(p) + which)
        return permute_bank(self.live, r.permutation(self.live.size))

    def bank_add2(self, p: Pod):
        r = np.random.default_rng(13000 + self.seed * 1000 + self.pods.index(p))
        s = self._clone()
        fk = [self.add2_free[int(i)] for i in r.choice(len(self.add2_free), 2, replace=False)]
        tgt = self.add2_targets[int(r.integers(len(self.add2_targets)))]
        for k in fk:
            s.link(k[0], k[1], self.kids[tgt], provenance="add2")
        return bank_from_store(s), s

    # ---- queries
    def queries(self, p: Pod, cls: str):
        w = self.world
        if cls == "i":
            keys = p.keys
            if self.reader == "syn":
                qs = [Query("fwd", k[0], (k[1],), (w.surface_of(k[1], s),)) for k in keys for s in (0, 1)]
                qs.append(Query("rev", p.obj, (p.target[1],), (w.surface_of(p.target[1], 0),)))
                objs = [p.obj] * len(qs)
                return qs, np.array(objs)
            return [(k, t) for k in keys for t in GPT2_TEMPLATES], np.array([p.obj] * (len(keys) * len(GPT2_TEMPLATES)))
        if cls == "ii":
            i0 = self.base_pos[p.target]
            # the N_AFTER base rows written immediately after p's target (their markers shift in NEVER(p));
            # cyclic so every pod has exactly the same number of class-(ii) queries -- a pod whose target
            # is among the last base facts otherwise had fewer, and np.stack refused the feature matrix
            after = [self.base_keys[(i0 + 1 + j) % len(self.base_keys)] for j in range(N_AFTER)]
            keys = list(self.bystander_base) + list(self.other_pod_aliases) + after
            objs = np.array([w.index[k] for k in keys])
            if self.reader == "syn":
                return [E15._q1(w, k) for k in keys], objs
            return [(k, GPT2_BYSTANDER_TEMPLATE) for k in keys], objs
        # generic
        if self.reader == "syn":
            keys = self.generic_free
            return [E15._q1(w, k) for k in keys], np.array([p.obj] * len(keys))
        names = self.gk.names
        texts = [t.format(s=names[p.target[0]]) for t in E17.GENERIC] + \
                [t.format(s=names[self.bystander_base[0][0]]) for t in E17.GENERIC[:3]]
        return texts, np.array([p.obj] * len(texts))

    # ---- reads
    @torch.no_grad()
    def read(self, bank: Bank, qs, cls: str) -> np.ndarray:
        """Logits for the class: (Q, n_ent + 1) for syn and GPT-2 classes i/ii; (Q, V) for GPT-2 generic."""
        if self.reader == "syn":
            return E15.predict(self.model, bank, self.world, list(qs)).logits
        tensors = bank.tensors()
        if cls == "iii":
            ids, am, last = E8.encode_texts(self.gk.tok, list(qs))
            _, full, _, _ = self.model(tensors, ids, am, last)
            return full.numpy()
        texts = [E17.TEMPLATES12[k[1]][t].format(s=self.gk.names[k[0]]) for k, t in qs]
        out = []
        for i in range(0, len(texts), 64):
            ids, am, last = E8.encode_texts(self.gk.tok, texts[i: i + 64])
            cand, _, _, _ = self.model(tensors, ids, am, last)
            out.append(cand.numpy())
        return np.concatenate(out)

    def feats(self, logits: np.ndarray, objs: np.ndarray, cls: str) -> np.ndarray:
        if self.reader == "gpt2" and cls == "iii":
            ent = np.asarray(self.gk.entity_ids)
            return features(logits, ent[objs], self.gk.unknown_id, entity_cols=ent).reshape(-1)
        return features(logits, objs, self.n_ent).reshape(-1)

    @torch.no_grad()
    def encoding(self, bank: Bank) -> Dict[str, np.ndarray]:
        enc = self.model.encode_bank(bank.tensors())
        keys = ("k_f", "v_f", "k_r", "v_r", "active") if self.reader == "syn" else ("keys", "values", "active")
        return {k: enc[k].float().numpy() if torch.is_tensor(enc[k]) else np.asarray(enc[k]) for k in keys}

    def enc_residual(self, a: Bank, b: Bank) -> float:
        ea, eb = self.encoding(a), self.encoding(b)
        ka = {(int(s), int(r)): i for i, (s, r) in enumerate(zip(a.subject, a.relation))}
        kb = {(int(s), int(r)): i for i, (s, r) in enumerate(zip(b.subject, b.relation))}
        common = [k for k in ka if k in kb]
        ia = np.array([ka[k] for k in common]); ib = np.array([kb[k] for k in common])
        return max(float(np.abs(ea[n][ia] - eb[n][ib]).max()) for n in ea)

    def unknown_rate(self, bank: Bank, p: Pod) -> Tuple[float, float]:
        """Fraction of the pod's ALIAS keys answered UNKNOWN / answered with some entity."""
        if self.reader == "syn":
            ans = E15.predict(self.model, bank, self.world, [E15._q1(self.world, a) for a in p.aliases]).answers
        else:
            ans, _, _ = E20._answers(self.gk, bank, p.aliases, self.gk.names, template=GPT2_BYSTANDER_TEMPLATE)
        ans = np.asarray(ans)
        return float((ans == UNKNOWN).mean()), float((ans != UNKNOWN).mean())


# ------------------------------------------------------------------------------------------ run
ARMS = {                        # name -> (positive bank, reference bank, classes)
    "present": ("live", "never", ("i",)),
    "cascade_soft": ("cascade", "never", CLASSES),
    "blank_matched": ("blank", "cascade", CLASSES),
    "dangle_matched": ("dangle", "cascade", CLASSES),
    "blank_never": ("blank", "never", ("ii", "iii")),
    "dangle_never": ("dangle", "never", ("ii", "iii")),
    "perm": ("perm", "perm2", ("ii", "iii")),
    "add2": ("add2", "perm", ("ii", "iii")),
}


def run_reader_seed(reader: str, seed: int, n_pods: int, threads: int, n_hardgate: int, verbose: bool = True,
                    content_markers: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    S = Setting(reader, seed, n_pods, threads, content_markers=content_markers)
    m: Dict[str, Any] = {"reader": reader, "seed": seed, "n_pods": len(S.pods), "checkpoint_sha256": S.sha,
                         "content_markers": content_markers}
    feats: Dict[Tuple[str, str], List[np.ndarray]] = {}
    logits: Dict[Tuple[str, str], List[np.ndarray]] = {}
    enc_res = {"cascade_never": [], "blank_cascade": [], "dangle_cascade": [], "add2_perm": []}
    hi = {"cascade": [], "blank": [], "dangle": []}
    unk = {"blank": [], "dangle": []}
    forwards = 0
    for pi, p in enumerate(S.pods):
        banks: Dict[str, Bank] = {"live": S.live}
        stores = {}
        banks["cascade"], stores["cascade"] = S.bank_cascade(p)
        banks["blank"], stores["blank"] = S.bank_blank(p)
        banks["dangle"], stores["dangle"] = S.bank_dangle(p)
        banks["never"], _ = S.bank_never(p)
        banks["perm"] = S.bank_perm(p, 0); banks["perm2"] = S.bank_perm(p, 1)
        banks["add2"], _ = S.bank_add2(p)
        for name in ("cascade", "blank", "dangle"):
            h = check_history_independence(stores[name])
            hi[name].append((float(h.exported_hi), float(h.residue_rows), float(h.markers_equal)))
        for cls in CLASSES:
            qs, objs = S.queries(p, cls)
            for bname, bank in banks.items():
                if cls == "i" and bname in ("perm", "perm2", "add2"):
                    continue
                lg = S.read(bank, qs, cls); forwards += len(qs)
                feats.setdefault((bname, cls), []).append(S.feats(lg, objs, cls))
                logits.setdefault((bname, cls), []).append(lg)
        enc_res["cascade_never"].append(S.enc_residual(banks["cascade"], banks["never"]))
        enc_res["blank_cascade"].append(S.enc_residual(banks["blank"], banks["cascade"]))
        enc_res["dangle_cascade"].append(S.enc_residual(banks["dangle"], banks["cascade"]))
        enc_res["add2_perm"].append(S.enc_residual(banks["add2"], banks["perm"]))
        unk["blank"].append(S.unknown_rate(banks["blank"], p))
        unk["dangle"].append(S.unknown_rate(banks["dangle"], p))
        if verbose and (pi + 1) % 10 == 0:
            print(f"  {reader} seed {seed}: {pi + 1}/{len(S.pods)} pods ({time.time() - t0:.0f}s)", flush=True)

    # ---- the pipeline check: hard gate, cascade vs never -> the marker channel closes at the interface
    hg = []
    S.model.cfg.hard_gate = True
    for p in S.pods[:n_hardgate]:
        bc, _ = S.bank_cascade(p); bn, _ = S.bank_never(p)
        qs, _ = S.queries(p, "ii")
        hg.append((S.enc_residual(bc, bn), float(np.abs(S.read(bc, qs, "ii") - S.read(bn, qs, "ii")).max())))
    S.model.cfg.hard_gate = False
    m["hardgate/enc_maxabs"] = float(max(h[0] for h in hg)) if hg else float("nan")
    m["hardgate/logit_maxabs"] = float(max(h[1] for h in hg)) if hg else float("nan")

    # ---- arms
    for arm, (pos, ref, classes) in ARMS.items():
        for cls in classes:
            X = np.stack(feats[(pos, cls)]); Y = np.stack(feats[(ref, cls)])
            m[f"{arm}/auc_{cls}"] = cv_auc(X, Y, seed) if len(S.pods) >= 5 else float("nan")
            kls, agrees, dl = zip(*[kl_top1(a, b) for a, b in zip(logits[(pos, cls)], logits[(ref, cls)])])
            m[f"{arm}/kl_max_{cls}"] = float(max(kls)); m[f"{arm}/top1_agree_{cls}"] = float(np.mean(agrees))
            m[f"{arm}/dlogit_max_{cls}"] = float(max(dl))
    for k, v in enc_res.items():
        m[f"enc/{k}_maxabs"] = float(max(v))
    for name, rows in hi.items():
        a = np.array(rows)
        m[f"store/{name}/exported_hi"] = float(a[:, 0].mean()); m[f"store/{name}/residue_rows"] = float(a[:, 1].mean())
        m[f"store/{name}/markers_equal"] = float(a[:, 2].mean())
    for name, rows in unk.items():
        a = np.array(rows)
        m[f"{name}/deleted_key_unknown"] = float(a[:, 0].mean()); m[f"{name}/deleted_key_wrong_entity"] = float(a[:, 1].mean())
    m["forwards"] = forwards
    m["seconds"] = time.time() - t0
    if verbose:
        print(f"  {reader} seed {seed}: present {m['present/auc_i']:.3f} | cascade ii/iii {m['cascade_soft/auc_ii']:.3f}/"
              f"{m['cascade_soft/auc_iii']:.3f} | blank i/ii/iii {m['blank_matched/auc_i']:.3f}/{m['blank_matched/auc_ii']:.3f}/"
              f"{m['blank_matched/auc_iii']:.3f} | dangle ii/iii {m['dangle_matched/auc_ii']:.3f}/{m['dangle_matched/auc_iii']:.3f} | "
              f"perm ii/iii {m['perm/auc_ii']:.3f}/{m['perm/auc_iii']:.3f} | add2 ii/iii {m['add2/auc_ii']:.3f}/{m['add2/auc_iii']:.3f} | "
              f"blank unk {m['blank/deleted_key_unknown']:.2f} | KL ii blank {m['blank_matched/kl_max_ii']:.4f}  ({m['seconds']:.0f}s)",
              flush=True)
    return m


def criteria_for(reader: str) -> Dict[str, Tuple[str, float]]:
    R = reader
    c = {
        f"{R}/present/auc_i": (">=", 0.95),                                   # V1
        f"{R}/enc/cascade_never_maxabs": (">=", 1e-6),                        # V2 the marker channel exists
        f"{R}/perm/auc_ii": ("<=", 0.60), f"{R}/perm/auc_iii": ("<=", 0.60),  # V3 float floor
        f"{R}/add2/auc_ii": ("<=", 0.60), f"{R}/add2/auc_iii": ("<=", 0.60),  # V4 row-count floor
        f"{R}/blank_matched/auc_i": (">=", 0.75), f"{R}/dangle_matched/auc_i": (">=", 0.75),   # V5
        f"{R}/cascade_soft/auc_i": ("<=", 0.60), f"{R}/cascade_soft/auc_ii": ("<=", 0.60),
        f"{R}/cascade_soft/auc_iii": ("<=", 0.60),                            # M1
        f"{R}/blank_matched/auc_ii": ("<=", 0.60), f"{R}/blank_matched/auc_iii": ("<=", 0.60),
        f"{R}/dangle_matched/auc_ii": ("<=", 0.60), f"{R}/dangle_matched/auc_iii": ("<=", 0.60),   # M2
        f"{R}/blank_matched/kl_max_ii": ("<=", 0.05), f"{R}/dangle_matched/kl_max_ii": ("<=", 0.05),
        f"{R}/blank_matched/top1_agree_ii": (">=", 0.98), f"{R}/dangle_matched/top1_agree_ii": (">=", 0.98),  # M3
        f"{R}/blank/deleted_key_unknown": (">=", 0.90), f"{R}/blank/deleted_key_wrong_entity": ("<=", 0.05),  # L1
    }
    if R == "gpt2":
        c[f"{R}/blank_matched/kl_max_iii"] = ("<=", 0.05); c[f"{R}/dangle_matched/kl_max_iii"] = ("<=", 0.05)
    return c


DECISION_RULE = (
    "Per reader. VOID if present/auc_i < 0.95 (the adversary cannot see a live row against none) or if "
    "the hard-gate pipeline check leaves an interface residual. If perm AUC > 0.60 every AUC row is void "
    "(the adversary reads summation order) and only KL / top-1 rows are read. If add2 AUC > 0.60 the "
    "ROW-COUNT reading is recorded: the reader's off-pod outputs carry the number of active rows, a "
    "property of dense routing and not of history, and the residue AUCs are reported against add2's as a "
    "calibrated null. If blank/dangle vs matched on class (i) < 0.75 the INVISIBLE reading is recorded. "
    "With V1-V5 passing: NEGATIVE (confined) if cascade, blank and dangle AUCs off the pod are all <= 0.60 and "
    "KL <= 0.05 nats with top-1 agreement >= 0.98; POSITIVE if any blank/dangle off-pod AUC >= 0.75 with its "
    "add2 row <= 0.60, naming the arm and the class, with the KL beside it; MARKER if any cascade-vs-never "
    "AUC >= 0.75 (the seeded generator's position is a history channel the reader exposes under the soft "
    "gate). Any AUC in (0.60, 0.75) is inconclusive at this n and licenses neither sentence. L1 is read "
    "independently: a blanked alias must read UNKNOWN in >= 0.90 of cases and as some entity in <= 0.05; "
    "dangle/deleted_key_unknown is E-000026's delete_target/alias_unknown under a new name and is reported, "
    "not decided on. Fixed before the run.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", nargs="*", default=list(READERS), choices=READERS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-pods", type=int, default=100)
    ap.add_argument("--n-hardgate", type=int, default=20)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--quick", action="store_true", help="reduced sizes: written with a -smoke suffix, not a record")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.quick:
        args.n_hardgate = min(args.n_hardgate, 3)

    per: List[Dict[str, Any]] = []
    for reader in args.readers:
        for seed in args.seeds:
            print(f"=== {reader} seed {seed}: {args.n_pods} pods ===", flush=True)
            per.append(run_reader_seed(reader, seed, args.n_pods, args.threads, args.n_hardgate))
            if args.results_dir:                       # partial results survive a later crash
                import json as _json
                os.makedirs(args.results_dir, exist_ok=True)
                with open(os.path.join(args.results_dir, f"e000051_{reader}_seed{seed}.json"), "w") as f:
                    _json.dump(per[-1], f, indent=1, default=float)
    # aggregate per reader over seeds, keyed reader/measure
    rows_by_reader: Dict[str, List[Dict[str, float]]] = {}
    for m in per:
        rows_by_reader.setdefault(m["reader"], []).append(
            {k: float(v) for k, v in m.items() if isinstance(v, (int, float)) and k not in ("seed",)})
    agg: Dict[str, Dict[str, float]] = {}
    for reader, rows in rows_by_reader.items():
        keys = sorted(set.intersection(*[set(r) for r in rows]))
        a = ledger.aggregate(rows, keys)
        agg.update({f"{reader}/{k}": v for k, v in a.items()})
    crit = {}
    for reader in rows_by_reader:
        crit.update({k: v for k, v in criteria_for(reader).items() if k in agg})
    check = ledger.check_criteria(agg, crit)

    def cell(reader, k, f="mean"):
        return f"{agg[f'{reader}/{k}'][f]:.3f}" if f"{reader}/{k}" in agg else "-"

    md = [f"# E-000051 — the residue against the reader", "",
          f"Readers {args.readers}, seeds {args.seeds}, {args.n_pods} pods per seed, trains nothing"
          + ("; REDUCED SIZES (--quick): not a record" if args.quick else "") + ". AUCs are five-fold cross-validated "
          "Mann-Whitney statistics over held-out pods (worst seed = max for a <= bar, min for a >= bar); the null band at "
          "n = 200 is about 0.42-0.58.", ""]
    for reader in rows_by_reader:
        md += [f"## {reader}", "",
               ledger.table(["arm (positive vs reference)", "AUC (i) deleted keys", "AUC (ii) bystanders", "AUC (iii) generic",
                             "max KL (ii)", "top-1 agree (ii)", "max KL (iii)"],
                            [[f"{arm}: {pos} vs {ref}", cell(reader, f"{arm}/auc_i"), cell(reader, f"{arm}/auc_ii"),
                              cell(reader, f"{arm}/auc_iii"), cell(reader, f"{arm}/kl_max_ii", "max"),
                              cell(reader, f"{arm}/top1_agree_ii", "min"), cell(reader, f"{arm}/kl_max_iii", "max")]
                             for arm, (pos, ref, _) in ARMS.items()]), "",
               ledger.table(["store-level (mean over pods)", "exported HI", "residue rows", "markers equal"],
                            [[name, cell(reader, f"store/{name}/exported_hi"), cell(reader, f"store/{name}/residue_rows"),
                              cell(reader, f"store/{name}/markers_equal")] for name in ("cascade", "blank", "dangle")]), "",
               ledger.table(["lifecycle row", "alias answers UNKNOWN", "alias answers some entity"],
                            [["BLANK (SET NULL by hand)", cell(reader, "blank/deleted_key_unknown", "min"),
                              cell(reader, "blank/deleted_key_wrong_entity", "max")],
                             ["DANGLE (evict object; = E-000026 delete_target/alias_unknown, reproduction)",
                              cell(reader, "dangle/deleted_key_unknown", "min"), cell(reader, "dangle/deleted_key_wrong_entity", "max")]]),
               "", f"Interface residuals (max |delta| on rows common to both banks, aligned by key): cascade vs never "
               f"{cell(reader, 'enc/cascade_never_maxabs', 'max')}, blank vs cascade {cell(reader, 'enc/blank_cascade_maxabs', 'max')}, "
               f"add2 vs perm {cell(reader, 'enc/add2_perm_maxabs', 'max')}; hard-gate pipeline check: encoding "
               f"{cell(reader, 'hardgate/enc_maxabs', 'max')}, logits {cell(reader, 'hardgate/logit_maxabs', 'max')}.", ""]
    md += ["## The rule, fixed before the run", "", DECISION_RULE, "", "## Pre-registered criteria", "",
           ledger.criteria_table(check), ""]
    record = {"experiment": "E-000051", "title": "the residue against the reader", "evidence_level": "E5",
              "trains_nothing": True, "readers": args.readers, "seeds": args.seeds, "n_pods": args.n_pods,
              "quick": args.quick, "decision_rule": DECISION_RULE, "per_reader_seed": per, "aggregate": agg,
              "criteria": check}
    name = ("e000051_residue_reader" + ("" if list(args.readers) == list(READERS) else "-" + "-".join(args.readers))
            + ("-smoke" if args.quick else ""))
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        import json
        with open(os.path.join(args.results_dir, name + ".json"), "w") as f:
            json.dump(record, f, indent=1, default=float)
        with open(os.path.join(args.results_dir, name + ".md"), "w") as f:
            f.write("\n".join(md))
        path = os.path.join(args.results_dir, name + ".md")
    else:
        path = ledger.save(name, record, "\n".join(md))
    print("\n".join(md[1:])); print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
