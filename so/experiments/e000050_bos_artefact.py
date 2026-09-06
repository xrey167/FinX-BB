"""Experiment E-000050-A -- the paraphrase gap is the position-0 token: a BOS at inference, no training.

THE HYPOTHESIS. Every GPT-2 adapter in this repository was trained and evaluated on prompts tokenised
WITHOUT a BOS: GPT-2's tokenizer has ``add_bos_token`` False, so ``encode_texts`` (E-000008) never
prepends ``<|endoftext|>`` and a subject-initial template such as "{s} lives in" puts the SUBJECT at
token position 0.  Position 0 in GPT-2 is anomalous -- it is the attention sink (Xiao et al. 2023), the
site of the massive activations (Sun et al. 2024), and its residual is dominated by a fixed direction
that does not depend on which token sits there.  The adapter's routing query is a projection of the
residual at the LAST token, but what that residual knows about the subject was read through attention
from the subject's position; if the subject is the sink, the query is built from the sink's
representation and addressing fails.  The prediction is then exact and cheap: any token in front of the
subject moves it to position 1 and restores addressing; a BARE BOS -- a token the adapter never saw --
should do it as well as the text prefix of E-000039-A; subject-MEDIAL templates should not move; and
the things that were measured as "held-out failures" downstream of addressing -- deletion not reaching
held-out phrasings (E-000017's kill criterion 5), E-000025's bimodality by template, E-000026's need to
select a template, E-000039-B's negative, E-000013's override_heldout_min 0.0000 and revoke
heldout_kl_max 4.47, E-000020's direct 0.5633 at template 0 -- should follow the same switch.

WHAT WOULD FALSIFY IT. (i) A BOS does not help where the text prefix does: then the effect is what the
prefix SAYS, not where the subject SITS, and E-000039-A's prefix finding stands as a semantic one.
(ii) A newline or a neutral word does not match the BOS: then it is a BOS-specific effect rather than a
position-0 one.  (iii) Subject-medial templates change under a BOS: then the prefix is not a pure
position-0 fix -- the adapter also learned features of whichever token sat at position 0 -- and the gain
is not free of a retrain.  (iv) Trained subject-medial templates fall under a BOS: same reading, and the
"no training" part of the claim goes.  (v) Deletion does not propagate to held-out phrasings once the
reading does: then the paraphrase gap of DELETION is not the paraphrase gap of READING and §31.21's
re-scoping of kill criterion 5 was wrong.  Every one of these is a registered row below.

WHAT IS MEASURED, WITH NOTHING TRAINED.  Three recorded checkpoint families, three seeds each,
under five prompt variants -- none / BOS "<|endoftext|>" / the text prefix of E-000039-A / a newline /
a neutral single word ("Also") -- applied at inference through E-000008's ``encode_texts``:

  E-000017-B  (e000017_t8_c0)  per template (12): reading, route_hit (addressing), oracle_read
              (transport under ``cell_mask``), first-read null; E-000017's battery: held-out reading,
              SHRED / REVOKE propagation per template, broken-key UNKNOWN, generic-text KL.
  E-000013    (e000013_gpt2)   override on the trained and the two held-out phrasings while the
              counterfactual cell is ACTIVE, and the KL to the base model per phrasing after REVOKE.
  E-000020    (e000020_gpt2)   E-000020's lifecycle battery (direct, alias_direct, dup_direct, update
              reach, SHRED / REVOKE / DELETE through the aliases) at template 0 and the held-out ones.

CONTROLS THAT CAN FAIL.  Subject-medial templates must not change under a BOS (bar: no medial template
moves by more than 0.05); no trained template may fall (bar: -0.05); the text prefix must reproduce
E-000039-A's ceiling (0.97 read / 0.98 route on the held-out subject-initial forms); the subject-token
index is read off the tokenizer for every variant and recorded, so the split is never a recorded accuracy.

PRIOR ART, AND WHAT IS AND IS NOT CLAIMED.  The mechanisms are owned: attention sinks (Xiao et al.
2023), massive activations (Sun et al. 2024), first-token / BOS handling in TransformerLens
(``prepend_bos``) and mechanistic-interpretability practice.  The closest sentence is owned too:
Yang et al. 2024 ("The Fall of ROME") find that subject-first prompts break ROME's subject key on
GPT-2-family models because of the special distribution of position 0, that Llama's ``<s>`` prevents
it, and that a prefix at inference cures the collapse -- with random-text prefixes, and with paraphrase
generalisation on those cases staying low after the cure.  The ledger must cite that as the parent of
both the diagnosis and the remedy.  Two cautions travel with it: Yang et al. A.4 find a
position-embedding swap does NOT remove the first-token anomaly (the first token's self-only attention
is the other cause), so this is a POSITION-0 artefact and not a positional-EMBEDDING one; and sink
formation is data-dependent (Barbero et al., Gu et al.), so a BOS gain on GPT-2 is a property of this
model family and not a general law.  What may be unowned is the measured sentence only: that an
EXTERNAL addressable-memory adapter on a FROZEN GPT-2 fails held-out paraphrases exactly on
subject-initial templates because its routing reads the position-0 representation; that a bare
``<|endoftext|>`` -- not random text -- at inference, with no training, lifts held-out reading and
addressing to the trained ceiling; that DELETION propagation, broken-key UNKNOWN, override and revoke
KL on held-out phrasings follow the same switch; and the controls.  Also worth carrying: the released
CounterFact paraphrase prompts always carry a generated prefix, so the field's paraphrase numbers were
measured with the subject never at position 0 and §31.36's "ROME reports ~96% paraphrase" is not a
like-for-like comparison with this repository's unprefixed subject-initial held-out templates.  Both
readings -- artefact, and semantic -- are written into DECISION_RULE below before the run.

E-000050-B (``e000050_bos_trained``) is the trained arm: the same adapter re-trained WITH a BOS.  This
file trains nothing and evaluates the recorded checkpoints as they stand.

Run:  python -m so.experiments.e000050_bos_artefact [--seeds 0 1 2] [--families e17 e13 e20]
      python -m so.experiments.e000050_bos_artefact --quick --seeds 0 --threads 1 --n-targets 30 \
          --families e17 --variants none bos text --results-dir /path/to/scratch     (a smoke run)
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from so import ledger
from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000013_prior_conflict as E13
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000039_address_tying as E39
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256
from so.llm_adapter import AdapterConfig
from so.mvcc import MVCCStore
from so.reference import load_world
from so.world import Fact, Query, UNKNOWN, World, fill_random

N_TRAIN, N_T = E39.N_TRAIN, E39.N_T
TEMPLATES12 = E17.TEMPLATES12
BOS = "<|endoftext|>"

# The five prompt variants. Entity names carry a leading space (they are "Ġ..." tokens), so the BOS and
# the newline are concatenated as they are; the text prefix is E-000039-A's string VERBATIM, double
# space included, so that its numbers reproduce; the neutral word gets a space only when the text does
# not already begin with one ("Also Anna lives in", "Also The home of Anna is").
VARIANTS: "OrderedDict[str, str]" = OrderedDict([
    ("none", ""),
    ("bos", BOS),
    ("text", E39.PREFIX),
    ("newline", "\n"),
    ("word", "Also"),
    # THE DOUBLE-SPACE CONFOUND (the landing's completeness critic). Entity names are decoded from
    # 'G[A-Z]...' tokens and carry a leading space, so every subject-MEDIAL template tokenises with a lone
    # 'G' (id 220) immediately before the subject, and "It is known that " reproduces that trained bigram
    # while a bare BOS does not. Three arms separate the marker from the position: "space" is that lone
    # token alone at position 0 (subject at 1); "bos_sp" is the BOS followed by the marker (subject at 2);
    # "text_nosp" is the text prefix without the marker (subject at 4). bos_sp ~ text > bos means the
    # residue is the trained marker; text_nosp ~ text > bos means the words; all equal means position.
    ("space", " "),
    ("bos_sp", BOS + " "),
    ("text_nosp", E39.PREFIX.rstrip()),
])
FAMILIES = ("e17", "e13", "e20")
E20_TEMPLATES = (0, 8, 9, 10, 11)      # the recorded template and the four held-out ones
E13_HELDOUT = E13.HELDOUT_TEMPLATES     # (2, 3)


# ------------------------------------------------------------------------------ the prefix, applied
_ORIGINAL_ENCODE = E8.encode_texts


def with_prefix(prefix: str, texts: Sequence[str]) -> List[str]:
    if not prefix:
        return list(texts)
    out = []
    for t in texts:
        sep = "" if (prefix[-1].isspace() or prefix.endswith("|>") or t[:1].isspace()) else " "
        out.append(prefix + sep + t)
    return out


@contextlib.contextmanager
def prefixed(prefix: str) -> Iterator[None]:
    """Every prompt that goes through E-000008's ``encode_texts`` carries ``prefix`` for the duration.

    E-000013, E-000017, E-000020 and E-000039 all call it through the module attribute (``E8.encode_texts``),
    and E-000008's own ``predict`` reads the module global, so rebinding the attribute reaches every read
    path in the repository without touching any of those files.  ``SO_BOS`` (E-000050-B's switch inside
    ``encode_texts``) is forced off so a prefix is never applied twice.
    """
    if os.environ.get("SO_BOS") == "1":
        raise RuntimeError("SO_BOS=1 would prepend a second BOS inside encode_texts; unset it for this experiment")

    def wrapped(tok, texts: List[str]):
        return _ORIGINAL_ENCODE(tok, with_prefix(prefix, texts))

    E8.encode_texts = wrapped
    try:
        yield
    finally:
        E8.encode_texts = _ORIGINAL_ENCODE


def subject_index(tok, text: str, name: str) -> int:
    """Token index of the subject name in ``text`` -- read off the tokenizer, never off an accuracy."""
    toks = tok.convert_ids_to_tokens(tok(text)["input_ids"])
    key = name.strip()
    return next(i for i, s in enumerate(toks) if key in s)


def subject_positions(tok, templates: Dict[int, List[str]], name: str = " Anna") -> Dict[str, List[int]]:
    """Subject-token index per template (relation 0) under every variant, plus a check that the BOS is
    one token and that no variant changes the subject's own token."""
    base_tok = tok.convert_ids_to_tokens(tok(templates[0][0].format(s=name))["input_ids"])
    subj_tok = next(s for s in base_tok if name.strip() in s)
    out: Dict[str, List[int]] = {}
    for v, p in VARIANTS.items():
        idx = []
        for t in range(len(templates[0])):
            text = with_prefix(p, [templates[0][t].format(s=name)])[0]
            toks = tok.convert_ids_to_tokens(tok(text)["input_ids"])
            assert subj_tok in toks, (v, t, toks)
            idx.append(toks.index(subj_tok))
        out[v] = idx
    assert tok(BOS)["input_ids"] == [tok.eos_token_id], "the BOS must be exactly GPT-2's <|endoftext|> token"
    return out


# --------------------------------------------------------- E-000039's decomposition, with a prefix
@torch.no_grad()
def decompose_prefixed(gk: E8.GPT2Knowledge, seed: int, centre: np.ndarray, n_targets: int, prefix: str,
                       oracle: bool = True) -> Dict[str, Any]:
    """E-000039's ``decompose`` per-template loop, on prompts carrying ``prefix``, over all twelve forms.

      read(t)   top-1 = object, everything addressable
      hit(t)    argmax of the LAST read slot's routing = the cell that holds the fact   (addressing)
      orc(t)    top-1 = object when the ONLY addressable cell is that one (``cell_mask``) (transport)
      null0(t)  the EARLIER read slot routes to the null column, as a 1-hop question should

    The world, the targets and the seed are E-000039-A's, so the ``none`` variant must reproduce its
    record and the ``text`` variant its prefixed ceiling.  Copied rather than called so that e000039 is
    left exactly as recorded.
    """
    rng = np.random.default_rng(seed)
    world = fill_random(rng, World(gk.n_entities, 4, N_TRAIN, []), E17.EVAL["n_cells"])
    store = MVCCStore(marker_dim=centre.shape[0], seed=seed, marker_centre=centre)
    load_world(store, world)
    bank = bank_from_store(store)
    tensors = bank.tensors()
    idx = rng.choice(len(world.facts), size=min(n_targets, len(world.facts)), replace=False)
    targets = [world.facts[int(i)] for i in idx]
    truth = np.array([f.obj for f in targets])
    cstar = np.array([bank.kid_of_key[f.key] for f in targets])
    gk.model.eval()
    m: Dict[str, Any] = {"n_targets": len(targets)}

    read = np.zeros((N_T, len(targets)), dtype=np.int64)
    hit = np.zeros((N_T, len(targets)), dtype=bool)
    cell = np.zeros((N_T, len(targets)), dtype=np.int64)
    null0 = np.zeros((N_T, len(targets)), dtype=bool)
    qs = []
    for t in range(N_T):
        texts = with_prefix(prefix, [TEMPLATES12[f.relation][t].format(s=gk.names[f.subject]) for f in targets])
        qt = []
        for i in range(0, len(texts), 32):
            ids, am, last = _ORIGINAL_ENCODE(gk.tok, texts[i:i + 32])
            cand, _, routing, _ = gk.model(tensors, ids, am, last)
            a = cand.argmax(-1).numpy()
            sl = slice(i, i + ids.shape[0])
            read[t, sl] = np.where(a == gk.n_entities, UNKNOWN, a)
            cell[t, sl] = routing[:, -1].numpy().argmax(-1)
            hit[t, sl] = cell[t, sl] == cstar[sl]
            null0[t, sl] = routing[:, 0].numpy().argmax(-1) == routing.shape[-1] - 1
            qt.append(gk.model.last_query.numpy())
        qs.append(np.concatenate(qt))
    Q = np.stack(qs)                                                   # (T, N, R, d_key)
    U = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-9)
    for r in range(U.shape[2]):
        M = U[:, :, r]
        within = float(np.mean([(M[i] * M[j]).sum(-1).mean() for i in range(N_T) for j in range(i + 1, N_T)]))
        G = M.reshape(-1, M.shape[-1]) @ M.reshape(-1, M.shape[-1]).T
        lab = np.repeat(np.arange(len(targets))[None], N_T, 0).reshape(-1)
        m[f"query_cos_within_fact/read{r}"] = within
        m[f"query_cos_between_fact/read{r}"] = float(G[lab[:, None] != lab[None, :]].mean())
    m["address_collision"] = float(np.mean([1.0 - len(set(cell[t].tolist())) / len(targets)
                                            for t in range(N_TRAIN, N_T)]))

    orc = np.zeros((N_T, len(targets)), dtype=np.int64)
    for j, f in enumerate(targets if oracle else []):
        mask = torch.zeros(bank.size, dtype=torch.bool)
        mask[cstar[j]] = True
        texts = with_prefix(prefix, [TEMPLATES12[f.relation][t].format(s=gk.names[f.subject]) for t in range(N_T)])
        ids, am, last = _ORIGINAL_ENCODE(gk.tok, texts)
        cand, _, _, _ = gk.model(tensors, ids, am, last, cell_mask=mask)
        a = cand.argmax(-1).numpy()
        orc[:, j] = np.where(a == gk.n_entities, UNKNOWN, a)

    r = {t: float((read[t] == truth).mean()) for t in range(N_T)}
    o = {t: (float((orc[t] == truth).mean()) if oracle else float("nan")) for t in range(N_T)}
    for t in range(N_T):
        kind = "train" if t < N_TRAIN else "heldout"
        m[f"t{t}/{kind}/read"] = r[t]
        m[f"t{t}/{kind}/route_hit"] = float(hit[t].mean())
        m[f"t{t}/{kind}/first_read_null"] = float(null0[t].mean())
        m[f"t{t}/{kind}/oracle_read"] = o[t]
        m[f"t{t}/{kind}/oracle_unknown"] = float((orc[t] == UNKNOWN).mean()) if oracle else float("nan")
    best = max(range(N_T), key=lambda t: r[t])
    gaps = [max(r[best] - r[t], 0.0) for t in range(N_TRAIN, N_T)]
    resid = [max(o[best] - o[t], 0.0) for t in range(N_TRAIN, N_T)]
    g, s = float(sum(gaps)), float(sum(resid))
    m["best_template"] = best
    m["heldout/gap"] = g
    m["heldout/residual_gap"] = s
    m["heldout/routing_share"] = (float(np.clip(1.0 - s / g, 0.0, 1.0)) if (oracle and g > 1e-9)
                                  else float("nan"))
    return m


# ------------------------------------------------------------------------- family 1: E-000017-B
def _kind(t: int) -> str:
    return "train" if t < N_TRAIN else "heldout"


def e17_variant(gk: E8.GPT2Knowledge, seed: int, centre: np.ndarray, prefix: str, n_targets: int,
                oracle: bool, battery: bool) -> Dict[str, float]:
    """One prompt variant on one E-000017-B checkpoint: the decomposition and E-000017's battery."""
    d = decompose_prefixed(gk, 1700 + seed, centre, n_targets, prefix, oracle)
    out: Dict[str, float] = {k: float(v) for k, v in d.items() if isinstance(v, (int, float, bool))}
    if battery:
        with prefixed(prefix):
            b = E17.evaluate_templates(gk, 1700 + seed, centre, N_TRAIN)
        for k, v in b.items():
            if isinstance(v, (int, float, bool)) and k not in ("seed", "n_targets"):
                out[k] = float(v)
    initial, medial = E39.subject_initial_templates(gk.tok)
    held = range(N_TRAIN, N_T)
    hi = [t for t in held if t in initial]
    hm = [t for t in held if t in medial]
    ti = [t for t in range(N_TRAIN) if t in initial]
    tm = [t for t in range(N_TRAIN) if t in medial]
    out["heldout/read_min"] = min(d[f"t{t}/heldout/read"] for t in held)
    out["heldout/route_hit_min"] = min(d[f"t{t}/heldout/route_hit"] for t in held)
    out["heldout_initial/read_min"] = min(d[f"t{t}/heldout/read"] for t in hi)
    out["heldout_initial/route_hit_min"] = min(d[f"t{t}/heldout/route_hit"] for t in hi)
    out["heldout_initial/oracle_read_min"] = min(d[f"t{t}/heldout/oracle_read"] for t in hi)
    out["heldout_medial/read_min"] = min(d[f"t{t}/heldout/read"] for t in hm)
    out["heldout_medial/route_hit_min"] = min(d[f"t{t}/heldout/route_hit"] for t in hm)
    out["train_initial/read_min"] = min(d[f"t{t}/train/read"] for t in ti)
    out["train_initial/route_hit_min"] = min(d[f"t{t}/train/route_hit"] for t in ti)
    out["train_medial/read_min"] = min(d[f"t{t}/train/read"] for t in tm)
    out["train/read_min"] = min(d[f"t{t}/train/read"] for t in range(N_TRAIN))
    if battery:
        out["shred_heldout_initial_min"] = min(b[f"shred_hard/template{t}_unknown"] for t in hi)
        out["revoke_heldout_initial_min"] = min(b[f"revoke/template{t}_unknown"] for t in hi)
        out["shred_heldout_medial_min"] = min(b[f"shred_hard/template{t}_unknown"] for t in hm)
    return out


def e17_changes(none: Dict[str, float], var: Dict[str, float], tok) -> Dict[str, float]:
    """The controls: what a variant does to the templates the hypothesis says it must NOT touch."""
    initial, medial = E39.subject_initial_templates(tok)
    dr = {t: var[f"t{t}/{_kind(t)}/read"] - none[f"t{t}/{_kind(t)}/read"] for t in range(N_T)}
    dh = {t: var[f"t{t}/{_kind(t)}/route_hit"] - none[f"t{t}/{_kind(t)}/route_hit"] for t in range(N_T)}
    out = {
        "medial_abs_change_max": max(abs(dr[t]) for t in medial),
        "medial_route_abs_change_max": max(abs(dh[t]) for t in medial),
        "train_medial_abs_change_max": max(abs(dr[t]) for t in medial if t < N_TRAIN),
        "train_read_change_min": min(dr[t] for t in range(N_TRAIN)),
        "initial_read_gain_mean": float(np.mean([dr[t] for t in initial])),
        "initial_route_gain_mean": float(np.mean([dh[t] for t in initial])),
        "heldout_initial_read_gain_min": min(dr[t] for t in initial if t >= N_TRAIN),
        "medial_read_change_min": min(dr[t] for t in medial),
    }
    for t in range(N_T):
        out[f"t{t}/read_change"] = dr[t]
        out[f"t{t}/route_hit_change"] = dh[t]
    return out


def run_e17(seed: int, variants: Sequence[str], n_targets: int, oracle: bool, battery: bool,
            verbose: bool = True) -> Tuple[Dict[str, float], Dict[str, Any]]:
    path = CHECKPOINTS / f"e000017_t8_c0{CKPT_SUFFIX}_seed{seed}.pt"
    gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))
    ck = torch.load(path, weights_only=False)
    gk.model.load_state_dict(ck["adapter"], strict=False)
    gk.model.eval()
    centre = np.asarray(ck["centre"])
    meta: Dict[str, Any] = {"checkpoint_sha256": _sha256(path), "subject_positions": subject_positions(gk.tok, TEMPLATES12)}
    initial, _ = E39.subject_initial_templates(gk.tok)
    per: Dict[str, Dict[str, float]] = {}
    out: Dict[str, float] = {}
    for v in variants:
        t0 = time.time()
        per[v] = e17_variant(gk, seed, centre, VARIANTS[v], n_targets, oracle, battery)
        if v != "none" and "none" in per:
            per[v].update(e17_changes(per["none"], per[v], gk.tok))
        out.update({f"e17/{v}/{k}": val for k, val in per[v].items()})
        meta[f"seconds/{v}"] = time.time() - t0
        if verbose:
            r = per[v]
            line = (f"  seed {seed} e17 {v:8s} held-out initial read/route {r['heldout_initial/read_min']:.2f}/"
                    f"{r['heldout_initial/route_hit_min']:.2f}  medial {r['heldout_medial/read_min']:.2f}/"
                    f"{r['heldout_medial/route_hit_min']:.2f}  trained read_min {r['train/read_min']:.2f}")
            if battery:
                line += (f"  heldout_read {r['heldout/active_correct']:.4f} train_read {r['train/active_correct']:.4f} "
                         f"shred_heldout {r['shred_heldout_min']:.4f} revoke_heldout {r['revoke_heldout_min']:.4f} "
                         f"broken_unk {r['broken1_unknown']:.4f} generic_kl {r['generic/kl_to_base']:.3f}")
            print(line + f"  ({time.time() - t0:.0f}s)", flush=True)
    if verbose:
        head = "  t  kind     pos      " + "  ".join(f"{v:>11s}" for v in variants) + "   " + \
               "  ".join(f"{'orc_' + v:>11s}" for v in variants if v in ("none", "bos"))
        print(head)
        for t in range(N_T):
            cells = [f"{per[v][f't{t}/{_kind(t)}/read']:.2f}/{per[v][f't{t}/{_kind(t)}/route_hit']:.2f}" for v in variants]
            orcs = [f"{per[v][f't{t}/{_kind(t)}/oracle_read']:.2f}" for v in variants if v in ("none", "bos")]
            print(f"  {t:2d} {_kind(t):8s} {'initial' if t in initial else 'medial':8s} "
                  + "  ".join(f"{c:>11s}" for c in cells) + "   " + "  ".join(f"{c:>11s}" for c in orcs), flush=True)
    del gk
    return out, meta


# --------------------------------------------------------------------------- family 2: E-000013
@torch.no_grad()
def e13_override_revert(gk: E13.GPT2KnowledgePrior, seed: int, centre: np.ndarray) -> Dict[str, float]:
    """E-000013's override / revoke measurement, on E-000013's own world and derangement (same rng
    sequence as its ``evaluate``), restricted to the rows the hypothesis is about: override per template
    while the counterfactual cell is ACTIVE, KL to the base model per template after REVOKE, generic KL.
    The prompt variant is whatever ``prefixed`` has installed."""
    rng = np.random.default_rng(seed)
    n_ent, n_c = gk.n_entities, len(E13.PAIRS)
    country, capital = np.array(gk.country_idx), np.array(gk.capital_idx)
    while True:
        perm = rng.permutation(n_c)
        if not np.any(perm == np.arange(n_c)):
            break
    cf_facts = [Fact(int(country[i]), 0, int(capital[perm[i]])) for i in range(n_c)]
    world = fill_random(rng, World(n_ent, 4, gk.n_synonyms, cf_facts), E13.EVAL["n_cells"])
    store = MVCCStore(marker_dim=centre.shape[0], seed=seed, marker_centre=centre)
    kids = load_world(store, world)
    q_c = [Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),)) for f in cf_facts]
    cf_obj = np.array([f.obj for f in cf_facts])
    cap_tok = np.array(gk.entity_ids)[capital]
    m: Dict[str, float] = {}

    def bank():
        return bank_from_store(store)

    base = {t: gk.predict_full(None, q_c, template=t) for t in range(4)}
    for t in range(4):
        sub = base[t]["full"][:, cap_tok]
        m[f"prior/template{t}_restricted_top1"] = float((sub.argmax(-1) == np.arange(n_c)).mean())
    pos_c = [int(np.where(store.bank()["kid"] == kids[f.key])[0][0]) for f in cf_facts]
    for t in range(4):
        po = gk.predict_full(bank(), q_c, template=t)
        m[f"override/template{t}_direct"] = float((po["answers"] == cf_obj).mean())
        m[f"override/template{t}_full_vocab_top1"] = float((po["full"].argmax(-1) == np.array(gk.entity_ids)[cf_obj]).mean())
        m[f"override/template{t}_route_hit"] = float(np.mean([int(po["routing"][i, -1].argmax()) == pos_c[i]
                                                              for i in range(n_c)]))
    m["override/direct"] = m["override/template0_direct"]
    m["override_heldout_min"] = float(min(m[f"override/template{t}_direct"] for t in E13_HELDOUT))
    m["override_heldout_route_hit_min"] = float(min(m[f"override/template{t}_route_hit"] for t in E13_HELDOUT))
    for f in cf_facts:
        store.revoke(kids[f.key])
    kls = []
    for t in range(4):
        pa = gk.predict_full(bank(), q_c, template=t)
        kl = E13.kl_rows(base[t]["full"], pa["full"])
        kls.append(float(kl.mean()))
        m[f"revoke/template{t}_kl_to_base"] = float(kl.mean())
        m[f"revoke/template{t}_top1_matches_base"] = float((pa["full"].argmax(-1) == base[t]["full"].argmax(-1)).mean())
        m[f"revoke/template{t}_counterfactual_top1"] = float((pa["answers"] == cf_obj).mean())
    for f in cf_facts:
        store.restore(kids[f.key])
    m["revoke/kl_to_base"] = kls[0]
    m["revoke/heldout_kl_max"] = float(max(kls[t] for t in E13_HELDOUT))
    m["revoke/kl_to_base_pooled"] = float(np.mean(kls))
    m["revoke/top1_matches_base_pooled"] = float(np.mean([m[f"revoke/template{t}_top1_matches_base"] for t in range(4)]))
    m["revoke/counterfactual_top1_pooled"] = float(np.mean([m[f"revoke/template{t}_counterfactual_top1"] for t in range(4)]))
    gen = [E13.GENERIC[int(rng.integers(0, len(E13.GENERIC)))].format(s=gk.names[int(rng.integers(0, gk.n_names))])
           for _ in range(E13.EVAL["n_generic"])]
    pg, bg = gk.predict_full(bank(), [], texts=gen), gk.predict_full(None, [], texts=gen)
    m["generic/kl_to_base"] = float(E13.kl_rows(bg["full"], pg["full"]).mean())
    return m


def run_e13(seed: int, variants: Sequence[str], verbose: bool = True) -> Tuple[Dict[str, float], Dict[str, Any]]:
    path = CHECKPOINTS / f"e000013_gpt2{CKPT_SUFFIX}_seed{seed}.pt"
    gk = E13.GPT2KnowledgePrior(AdapterConfig(status_gated=True, fallback="prior"))
    ck = torch.load(path, weights_only=False)
    gk.model.load_state_dict(ck["adapter"], strict=False)
    gk.model.eval()
    centre = np.asarray(ck["centre"])
    meta: Dict[str, Any] = {"checkpoint_sha256": _sha256(path),
                            "subject_positions": subject_positions(gk.tok, E13.TEMPLATES_PC, " France")}
    out: Dict[str, float] = {}
    for v in variants:
        t0 = time.time()
        with prefixed(VARIANTS[v]):
            m = e13_override_revert(gk, 1300 + seed, centre)
        out.update({f"e13/{v}/{k}": val for k, val in m.items()})
        meta[f"seconds/{v}"] = time.time() - t0
        if verbose:
            print(f"  seed {seed} e13 {v:8s} override t0..3 "
                  + " ".join(f"{m[f'override/template{t}_direct']:.2f}" for t in range(4))
                  + f"  heldout_min {m['override_heldout_min']:.2f}  revoke kl t0..3 "
                  + " ".join(f"{m[f'revoke/template{t}_kl_to_base']:.3f}" for t in range(4))
                  + f"  heldout_kl_max {m['revoke/heldout_kl_max']:.3f}  generic_kl {m['generic/kl_to_base']:.3f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    del gk
    return out, meta


# --------------------------------------------------------------------------- family 3: E-000020
E20_KEYS = ["direct", "alias_direct", "dup_direct", "alias_heldout_min", "shared_update/alias_new_object",
            "duplicate_update/alias_new_object", "rollback/alias_direct", "shred_target/alias_unknown",
            "shred_target/alias_true_object", "dup_shred/copy_direct_acc", "resign_target/alias_direct",
            "revoke_alias/alias_unknown", "revoke_alias/sibling_readable", "revoke_alias/target_readable",
            "delete_target/alias_unknown", "delete_target/alias_true_object", "shred_target/alias_probe_top1",
            "active/alias_probe_top1"]


def run_e20(seed: int, variants: Sequence[str], templates: Sequence[int], verbose: bool = True
            ) -> Tuple[Dict[str, float], Dict[str, Any]]:
    path = CHECKPOINTS / f"e000020_gpt2{CKPT_SUFFIX}_seed{seed}.pt"
    ck = torch.load(path, weights_only=False)
    cfg = AdapterConfig(**ck["adapter_config"]) if "adapter_config" in ck else \
        AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
    gk = E8.GPT2Knowledge(cfg)
    gk.model.load_state_dict(ck["adapter"], strict=False)
    gk.model.eval()
    centre = np.asarray(ck["centre"])
    meta: Dict[str, Any] = {"checkpoint_sha256": _sha256(path), "templates": list(templates)}
    out: Dict[str, float] = {}
    for v in variants:
        for t in templates:
            t0 = time.time()
            with prefixed(VARIANTS[v]):
                m = E20.evaluate(gk, 2000 + seed, centre, template=t)
            for k in E20_KEYS:
                out[f"e20/{v}/t{t}/{k}"] = float(m[k])
            meta[f"seconds/{v}/t{t}"] = time.time() - t0
            if verbose:
                print(f"  seed {seed} e20 {v:8s} t{t:<2d} direct {m['direct']:.4f} alias {m['alias_direct']:.4f} "
                      f"dup {m['dup_direct']:.4f} update {m['shared_update/alias_new_object']:.4f} "
                      f"shred_unk {m['shred_target/alias_unknown']:.4f} delete_unk {m['delete_target/alias_unknown']:.4f} "
                      f"revoke_unk {m['revoke_alias/alias_unknown']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        held = [t for t in templates if t >= E20.N_TRAIN_TEMPLATES]
        if held:
            for k in ("direct", "alias_direct", "shared_update/alias_new_object", "shred_target/alias_unknown"):
                out[f"e20/{v}/heldout/{k}_min"] = min(out[f"e20/{v}/t{t}/{k}"] for t in held)
    del gk
    return out, meta


# ------------------------------------------------------------------------------- pre-registered
CRITERIA: Dict[str, Dict[str, Tuple[str, float]]] = {
    # the record reproduced in-process: without a prefix the held-out reading is where E-000017-B and
    # E-000039-A left it (0.7288 worst seed; t11 0.39 / 0.54)
    "record_reproduced": {"e17/none/heldout/active_correct": ("<=", 0.80),
                          "e17/none/heldout_initial/read_min": ("<=", 0.70),
                          "e17/none/heldout_initial/route_hit_min": ("<=", 0.80)},
    # H1: a bare BOS -- a token the adapter never saw -- lifts the held-out subject-initial forms to the
    # prefixed ceiling with no weight changed
    "bos_restores_heldout_addressing": {"e17/bos/heldout_initial/read_min": (">=", 0.90),
                                        "e17/bos/heldout_initial/route_hit_min": (">=", 0.90),
                                        "e17/bos/heldout/active_correct": (">=", 0.90)},
    # H2: the effect is where the subject sits, not what precedes it -- any single token does it, and
    # E-000039-A's text prefix reproduces its own ceiling
    "any_token_does_it": {"e17/newline/heldout_initial/read_min": (">=", 0.90),
                          "e17/newline/heldout_initial/route_hit_min": (">=", 0.90),
                          "e17/word/heldout_initial/read_min": (">=", 0.90),
                          "e17/word/heldout_initial/route_hit_min": (">=", 0.90),
                          "e17/text/heldout_initial/read_min": (">=", 0.90),
                          "e17/text/heldout_initial/route_hit_min": (">=", 0.90)},
    # H3, the controls that can fail: subject-medial templates must not move under a BOS, no trained
    # template may fall, and the trained reading must hold
    "controls_hold": {"e17/bos/medial_abs_change_max": ("<=", 0.05),
                      "e17/bos/train_read_change_min": (">=", -0.05),
                      "e17/bos/train/active_correct": (">=", 0.95)},
    # H4: deletion follows the same switch (E-000039-B's bars, which E-000017-B failed at 0.8650)
    "deletion_follows": {"e17/bos/shred_heldout_min": (">=", 0.95),
                         "e17/bos/revoke_heldout_min": (">=", 0.95),
                         "e17/bos/heldout/revoked_deleted_object": ("<=", 0.02)},
    # H5: no new collateral (bars set at the control's own recorded values, as in E-000039-B)
    "no_new_collateral": {"e17/bos/broken1_unknown": (">=", 0.63),
                          "e17/bos/generic/kl_to_base": ("<=", 3.65)},
    # E-000013's "behaves like own knowledge" rows at E-000013's own bars (recorded: 0.0000 and 4.47)
    "e13_override_and_revert_on_heldout": {"e13/bos/override_heldout_min": (">=", 0.70),
                                           "e13/bos/override/direct": (">=", 0.90),
                                           "e13/bos/revoke/heldout_kl_max": ("<=", 0.10)},
    # E-000020's template-0 battery at E-000020's own bars (recorded at template 0: 0.5633 / 0.5000 / 0.8850)
    "e20_lifecycle_at_template0": {"e20/bos/t0/direct": (">=", 0.85),
                                   "e20/bos/t0/alias_direct": (">=", 0.80),
                                   "e20/bos/t0/shared_update/alias_new_object": (">=", 0.90)},
}

DECISION_RULE = (
    "Worst seed on every row. READING 1 (artefact): record_reproduced, bos_restores_heldout_addressing, "
    "any_token_does_it and deletion_follows all pass -> the held-out paraphrase gap of this addressable "
    "memory on a frozen GPT-2 is the missing-BOS position-0 artefact; the honest held-out numbers for the "
    "memory are the BOS-prefixed ones, measured here with no training; kill criterion 5 (E-000017), the "
    "bimodality of E-000025, the template selection of E-000026, E-000039-B's negative and the 'behaves like "
    "own knowledge: no' row of section 31.36 are re-scoped as measured with the subject at position 0. If "
    "controls_hold also passes the fix is free; if controls_hold fails because subject-medial or trained "
    "templates MOVE under a BOS, the artefact reading survives (the subject-initial rows are its evidence) "
    "but the fix is not free: the adapter learned features of whichever token sat at position 0, and the "
    "BOS has to be applied consistently at training time (E-000050-B) before any number is quoted as a "
    "ceiling. READING 2 (semantic): the BOS fails its rows where the text prefix passes its own -> the gain "
    "is in what the prefix says and not where the subject sits; E-000039-A's prefix finding stands as it "
    "is, the position-0 diagnosis is withdrawn, and nothing is re-scoped. If the BOS passes and the newline "
    "or the word fails, the effect is BOS-specific (sink-token-specific) rather than position-0-specific "
    "and is recorded as such. The E-000013 and E-000020 groups are reported, not decided on: they say how "
    "far the switch reaches into the records the target was scored on (section 31.36). Prior art: the "
    "diagnosis and the remedy are Yang et al. 2024 (Fall of ROME) for ROME on GPT-2-family models; the "
    "mechanism is attention sinks / massive activations; only the memory-adapter measurement, the "
    "BOS-specific test, the deletion / override outcome and the controls are this record's. Fixed before "
    "the run.")

NOT_CLAIMED = ("that prepending a BOS is a general law (sink formation is data-dependent; this is GPT-2 124M); "
               "that the positional EMBEDDING is the cause (Yang et al. A.4: the first token's self-only "
               "attention is the other cause -- this record says position 0, not position embedding); "
               "anything about a trained-with-BOS adapter (E-000050-B); LLM scale; multi-token entities.")


# --------------------------------------------------------------------------------------- record
def _fmt(x: Optional[float]) -> str:
    return "-" if x is None or x != x else f"{x:.4f}"


def per_template_table(agg: Dict[str, Dict[str, float]], variants: Sequence[str], tok) -> str:
    initial, _ = E39.subject_initial_templates(tok)
    rows = []
    for t in range(N_T):
        k = _kind(t)
        cells = [f"{agg[f'e17/{v}/t{t}/{k}/read']['min']:.2f} / {agg[f'e17/{v}/t{t}/{k}/route_hit']['min']:.2f}"
                 for v in variants]
        orc = [f"{agg[f'e17/{v}/t{t}/{k}/oracle_read']['min']:.2f}" for v in variants if v in ("none", "bos")]
        rows.append([t, k, "initial" if t in initial else "medial", *cells, *orc])
    return ledger.table(["t", "kind", "subject", *[f"{v}: read / route" for v in variants],
                         *[f"oracle_read ({v})" for v in variants if v in ("none", "bos")]], rows)


def battery_table(agg: Dict[str, Dict[str, float]], variants: Sequence[str], family: str,
                  keys: Sequence[Tuple[str, bool]]) -> str:
    rows = []
    for k, lower in keys:
        vals = []
        for v in variants:
            a = agg.get(f"{family}/{v}/{k}")
            vals.append("-" if a is None else _fmt(ledger.worst(a, lower)))
        if any(x != "-" for x in vals):
            rows.append([k, *vals])
    return ledger.table(["measure (worst seed)", *variants], rows)


E17_ROWS = [("heldout/active_correct", False), ("train/active_correct", False), ("heldout/read_min", False),
            ("heldout/route_hit_min", False), ("heldout_initial/read_min", False),
            ("heldout_initial/route_hit_min", False), ("heldout_initial/oracle_read_min", False),
            ("heldout_medial/read_min", False), ("heldout_medial/route_hit_min", False),
            ("train_initial/read_min", False), ("train_medial/read_min", False),
            ("medial_abs_change_max", True), ("train_medial_abs_change_max", True), ("train_read_change_min", False),
            ("initial_read_gain_mean", False), ("heldout/routing_share", False),
            ("shred_heldout_min", False), ("revoke_heldout_min", False), ("shred_heldout_initial_min", False),
            ("shred_heldout_medial_min", False), ("shred_train_min", False), ("revoke_train_min", False),
            ("heldout/revoked_deleted_object", True), ("broken1_unknown", False), ("generic/kl_to_base", True),
            ("query_cos_between_fact/read1", True), ("address_collision", True)]
E13_ROWS = [(f"override/template{t}_direct", False) for t in range(4)] + [("override_heldout_min", False)] + \
           [(f"override/template{t}_route_hit", False) for t in range(4)] + \
           [(f"revoke/template{t}_kl_to_base", True) for t in range(4)] + \
           [("revoke/heldout_kl_max", True), ("revoke/top1_matches_base_pooled", False),
            ("revoke/counterfactual_top1_pooled", True), ("generic/kl_to_base", True)]


def decide_reading(met: Dict[str, Optional[bool]]) -> str:
    """DECISION_RULE applied to the claim groups; ``None`` marks a group that was not measured."""
    bos, tok, ctl, dele, rec = (met.get(k) for k in ("bos_restores_heldout_addressing", "any_token_does_it",
                                                       "controls_hold", "deletion_follows", "record_reproduced"))
    if bos is None:
        return "not decided (E-000017-B family not run)"
    if not bos:
        return ("semantic: the prefix finding stands, the position-0 diagnosis does not" if tok
                else "neither the BOS nor another token restores the held-out addressing: the hypothesis is false")
    if tok is False:
        return "BOS-specific (sink-token-specific), not position-0-specific"
    head = "artefact" + ("" if rec else " (record NOT reproduced in-process: check sizes before re-scoping anything)")
    if dele is False:
        return head + ", for reading but not for deletion"
    if ctl:
        return head + ", free"
    return head + ", not free (subject-medial or trained templates moved under the BOS)"


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--families", nargs="*", default=list(FAMILIES), choices=FAMILIES)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS),
                    help="prompt variants; 'none' is always run first (the changes are measured against it)")
    ap.add_argument("--n-targets", type=int, default=100, help="targets for the per-template decomposition")
    ap.add_argument("--no-oracle", action="store_true", help="skip the cell_mask arm (transport)")
    ap.add_argument("--no-battery", action="store_true", help="skip E-000017's battery (deletion, broken key, generic KL)")
    ap.add_argument("--e13-variants", nargs="*", default=["none", "bos", "text"], choices=list(VARIANTS))
    ap.add_argument("--e20-variants", nargs="*", default=["none", "bos"], choices=list(VARIANTS))
    ap.add_argument("--e20-templates", type=int, nargs="*", default=list(E20_TEMPLATES))
    ap.add_argument("--quick", action="store_true",
                    help="reduced sizes for a smoke run; the result is written with suffix -smoke and is not a record")
    ap.add_argument("--results-dir", default=None, help="write the result files here instead of so/results")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    os.environ.pop("SO_BOS", None)
    if args.results_dir:
        ledger.RESULTS_DIR = Path(args.results_dir)
    if args.quick:
        os.environ.setdefault("SO_RESULT_SUFFIX", "-smoke")
        # the bank stays at 1000 cells (cheap, and the addressing difficulty depends on it: the smoke
        # run at 300 cells read the held-out subject-initial forms 0.16 above the record); only the
        # target counts shrink
        E17.EVAL.update(n_targets=max(args.n_targets, 8), n_broken=30, n_generic=30)
        E13.EVAL.update(n_generic=30)
        E20.EVAL.update(n_base=120, n_groups=20, n_direct=40, n_targets=20)
    variants = ["none"] + [v for v in args.variants if v != "none"]
    e13_variants = ["none"] + [v for v in args.e13_variants if v != "none"]
    e20_variants = ["none"] + [v for v in args.e20_variants if v != "none"]

    per_seed: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    tok = None
    t_start = time.time()
    for seed in args.seeds:
        m: Dict[str, Any] = {"seed": seed}
        if "e17" in args.families:
            print(f"=== seed {seed}: E-000017-B checkpoint, variants {variants} ===", flush=True)
            out, md = run_e17(seed, variants, args.n_targets, not args.no_oracle, not args.no_battery)
            m.update(out); meta[f"e17/seed{seed}"] = md
        if "e13" in args.families:
            print(f"=== seed {seed}: E-000013 checkpoint, variants {e13_variants} ===", flush=True)
            out, md = run_e13(seed, e13_variants)
            m.update(out); meta[f"e13/seed{seed}"] = md
        if "e20" in args.families:
            print(f"=== seed {seed}: E-000020 checkpoint, variants {e20_variants}, templates {args.e20_templates} ===",
                  flush=True)
            out, md = run_e20(seed, e20_variants, args.e20_templates)
            m.update(out); meta[f"e20/seed{seed}"] = md
        per_seed.append(m)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")

    keys = [k for k in per_seed[0] if k != "seed" and isinstance(per_seed[0][k], (int, float))]
    agg = ledger.aggregate(per_seed, keys)
    all_criteria = {k: v for g in CRITERIA.values() for k, v in g.items()}
    measured = {k: v for k, v in all_criteria.items() if k in agg}
    not_measured = sorted(k for k in all_criteria if k not in agg)
    check = ledger.check_criteria(agg, measured)
    met = {g: (all(check["criteria"][k]["pass"] for k in ks if k in measured) if any(k in measured for k in ks)
               else None) for g, ks in CRITERIA.items()}
    reading = decide_reading(met)

    record = {
        "experiment": "E-000050-A",
        "title": "the paraphrase gap is the position-0 token: a BOS at inference, no training",
        "evidence_level": "E5", "deletion_level": None,
        "trains_nothing": "E-000017-B, E-000013 and E-000020's recorded checkpoints are evaluated as recorded; "
                          "the only change is a prefix on the prompt at inference.",
        "variants": {v: p for v, p in VARIANTS.items() if v in set(variants) | set(e13_variants) | set(e20_variants)},
        "families": args.families, "seeds": args.seeds, "n_targets": args.n_targets, "quick": args.quick,
        "oracle": not args.no_oracle, "battery": not args.no_battery, "e20_templates": args.e20_templates,
        "decision_rule": DECISION_RULE, "not_claimed": NOT_CLAIMED, "reading": reading,
        "prior_art": ["Yang et al. 2024, The Fall of ROME: subject-first prompts break the subject key on GPT-2-family "
                      "models because of the special distribution of position 0; any prefix or Llama's <s> repairs it; "
                      "generalisation on the repaired cases stays low (16.88% paraphrase on GPT-2-XL collapse cases)",
                      "Xiao et al. 2023 (attention sinks); Sun et al. 2024 (massive activations); TransformerLens "
                      "prepend_bos and the mechanistic-interpretability folklore of prepending a BOS to GPT-2",
                      "Yang et al. A.4: a position-embedding swap does not remove the first-token anomaly, so this is a "
                      "position-0 artefact, not a positional-embedding one; Barbero et al. / Gu et al.: sink formation "
                      "is data-dependent, so the BOS gain is a property of this model family",
                      "CounterFact's released paraphrase prompts always carry a generated prefix: the field's paraphrase "
                      "numbers were measured with the subject never at position 0"],
        "claim_groups_met": met, "criteria": check["criteria"], "criteria_not_measured": not_measured,
        "claim_supported": check["claim_supported"] and not not_measured,
        "meta": meta, "per_seed": per_seed, "aggregate": agg,
        "seconds_total": time.time() - t_start,
    }
    md = [f"# E-000050-A — {record['title']}", "",
          record["trains_nothing"], "",
          f"Seeds {args.seeds}; families {args.families}; {args.n_targets} targets per seed for the decomposition"
          + ("; REDUCED SIZES (--quick): not a record" if args.quick else "") + ". Worst seed everywhere.", "",
          f"**Reading: {reading}.**", "",
          ledger.table(["claim group", "supported"], [(g, "not measured" if v is None else ("yes" if v else "**no**"))
                                                      for g, v in met.items()]), ""]
    if "e17" in args.families:
        md += ["## E-000017-B, per template (read / route_hit, worst seed over seeds)", "",
               "The subject column is the token index of the subject name read off the tokenizer: "
               + "; ".join(f"{v}: {meta[f'e17/seed{args.seeds[0]}']['subject_positions'][v]}"
                           for v in variants), "",
               per_template_table(agg, variants, tok), "",
               "## E-000017-B, the battery", "", battery_table(agg, variants, "e17", E17_ROWS), ""]
    if "e13" in args.families:
        md += ["## E-000013 (fallback to the prior): override while ACTIVE and KL to the base model after REVOKE", "",
               "Templates: " + "; ".join(f"t{t} `{E13.TEMPLATES_PC[0][t]}`" for t in range(4))
               + f"; held-out {E13_HELDOUT}; subject positions "
               + "; ".join(f"{v}: {meta[f'e13/seed{args.seeds[0]}']['subject_positions'][v]}" for v in e13_variants), "",
               battery_table(agg, e13_variants, "e13", E13_ROWS), ""]
    if "e20" in args.families:
        rows = [(f"t{t}/{k}", False) for t in args.e20_templates for k in E20_KEYS[:8] + ["delete_target/alias_unknown"]]
        md += ["## E-000020 (link cells): the lifecycle battery at template 0 and the held-out templates", "",
               battery_table(agg, e20_variants, "e20", rows), ""]
    md += ["## Pre-registered criteria (worst seed)", "", ledger.criteria_table(check), ""]
    if not_measured:
        md += ["Not measured in this run (families or variants skipped): " + ", ".join(not_measured), ""]
    md += ["## The rule, fixed before the run", "", DECISION_RULE, "", "Not claimed: " + NOT_CLAIMED, "",
           "Prior art: " + " | ".join(record["prior_art"])]
    path = ledger.save("e000050a_bos_artefact", record, "\n".join(md))
    print("\n".join(md)); print(f"\nsaved {path}  ({record['seconds_total']:.0f}s)")
    return record


if __name__ == "__main__":
    main()
