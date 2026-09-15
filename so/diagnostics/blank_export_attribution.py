"""BLANK / SET NULL attribution on the retrained E-000020 seed-0 symlink adapter (judge rank 1, critic items 2-4).
Store state: every pod's target EVICTED and both aliases BLANKed (E-000051's BLANK), 100 pods, one bank.
Arms differ only in the EXPORT of the blanked rows (no weight changed), plus controls:
  a  self-reference export (recorded: link -> own key)
  b  link -> (own subject, absent relation)
  c  link -> (absent subject, own relation)      [a subject with no rows in the bank]
  d  link -> the evicted target's key            [DANGLE's export on BLANK's state]
  e  DANGLE: target evicted, aliases untouched   [tombstone key export]
  f  bare frozen LM (bank=None): the copy prior
  g  alias rows masked out of routing (cell_mask)
Per read: answer class {UNKNOWN, self-subject, sibling-object, true-object, other}, and the paired logit gap
unknown - self_subject.  Templates 0 (+lone space), 3, 8.  200 reads per arm per template; McNemar a-vs-b, a-vs-d."""
import copy, json, numpy as np, torch
from math import comb
torch.set_num_threads(1)
from so.llm_adapter import AdapterConfig
from so.data import bank_from_store
from so.world import UNKNOWN
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000052_symlink_bos_battery as E52
cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
gk = E8.GPT2Knowledge(cfg); out = E20.train_or_load(gk, 0, 3000); centre = out["centre"]; model = gk.model; model.eval()
world, spec, st, kids = E52.world_and_stores(gk, 2000, centre)
pods = [(t, ks) for t, ks in spec.groups]
aliases = [a for _, ks in pods for a in ks]; tgt_of = {a: t for t, ks in pods for a in ks}
sib_of = {ks[0]: ks[1] for _, ks in pods}; sib_of.update({ks[1]: ks[0] for _, ks in pods})
true_obj = np.array([world.index[tgt_of[a]] for a in aliases]); subj = np.array([a[0] for a in aliases])
sibl_obj = true_obj.copy()   # sibling alias resolves to the same object -> "sibling object" == true object here; keep subject-of-sibling instead
sib_subj = np.array([sib_of[a][0] for a in aliases])
n_rel = 4
rng_c = np.random.default_rng(1)
def other_subject_free(a):
    """a subject != own with (subject, own relation) absent from the world: an absent key under another subject"""
    while True:
        s = int(rng_c.integers(gk.n_entities))
        if s != a[0] and (s, a[1]) not in world.index: return s
# --- state B: evict targets, blank aliases
stB = copy.deepcopy(st)
for t, ks in pods:
    stB.evict(kids[t])
    for a in ks: stB.blank(kids[a])
bankB = bank_from_store(stB); rowB = {int(k): i for i, k in enumerate(bankB.kid)}
arows = np.array([rowB[kids[a]] for a in aliases])
def export(bank, ls, lr):
    b = copy.copy(bank); b.link_subject = bank.link_subject.copy(); b.link_relation = bank.link_relation.copy()
    b.link_subject[arows] = ls; b.link_relation[arows] = lr; return b
free_rel = np.array([next((r for r in range(n_rel) if (int(a[0]), r) not in world.index), -1) for a in aliases])
has_free = free_rel >= 0
print('aliases with a free relation under their own subject:', int(has_free.sum()), 'of', len(aliases), flush=True)
arms = {
    "a_self": bankB,
    "b_own_subject_absent_rel": export(bankB, subj, np.where(has_free, free_rel, np.array([a[1] for a in aliases]))),
    "c_other_subject_absent_key": export(bankB, np.array([other_subject_free(a) for a in aliases]), np.array([a[1] for a in aliases])),
    "d_target_key": export(bankB, np.array([tgt_of[a][0] for a in aliases]), np.array([tgt_of[a][1] for a in aliases])),
}
stE = copy.deepcopy(st)
for t, _ in pods: stE.evict(kids[t])
bankE = bank_from_store(stE); arms["e_dangle"] = bankE
mask = np.ones(bankB.size, dtype=bool); mask[arows] = False
res = {}
def read(bank, texts, cell_mask=None, bare=False):
    L = []
    for i in range(0, len(texts), 50):
        ids, am, last = E8.encode_texts(gk.tok, texts[i:i+50])
        with torch.no_grad():
            cand, _, _, _ = model(None if bare else bank.tensors(), ids, am, last, cell_mask=None if cell_mask is None else torch.as_tensor(cell_mask[:], dtype=torch.bool))
        L.append(cand.numpy())
    return np.concatenate(L)
def classify(lg):
    a = lg.argmax(-1); n = gk.n_entities
    cls = np.where(a == n, "unknown", np.where(a == subj, "self_subject", np.where(a == true_obj, "true_object", np.where(a == sib_subj, "sibling_subject", "other"))))
    gap = lg[:, n] - lg[np.arange(len(a)), subj]
    return cls, gap
def mcnemar(x, y):
    b = int((x & ~y).sum()); c = int((~x & y).sum()); n = b + c
    p = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n * 2 if n else 1.0
    return {"b": b, "c": c, "p_two_sided": min(1.0, p)}
for template, prefix in ((0, " "), (0, ""), (3, ""), (8, "")):
    key = f"t{template}{'_space' if prefix else ''}"
    texts = [prefix + E17.TEMPLATES12[r][template].format(s=gk.names[s]) for s, r in aliases]
    R = {}
    for name, bank in arms.items():
        cls, gap = classify(read(bank, texts)); R[name] = {"cls": cls, "gap": gap}
    cls, gap = classify(read(bankB, texts, cell_mask=mask)); R["g_alias_masked"] = {"cls": cls, "gap": gap}
    cls, gap = classify(read(None, texts, bare=True)); R["f_bare_lm"] = {"cls": cls, "gap": gap}
    summary = {}
    for name, d in R.items():
        c = d["cls"]; summary[name] = {k: float((c == k).mean()) for k in ("unknown", "self_subject", "true_object", "sibling_subject", "other")}
        summary[name]["wrong_entity"] = float((c != "unknown").mean()); summary[name]["gap_mean"] = float(d["gap"].mean())
    for x, y in (("a_self", "b_own_subject_absent_rel"), ("a_self", "d_target_key"), ("a_self", "e_dangle"), ("d_target_key", "e_dangle")):
        sel = has_free if "b_own" in (x + y) else np.ones(len(aliases), dtype=bool)
        summary[f"mcnemar_wrong_{x}_vs_{y}"] = mcnemar((R[x]["cls"] != "unknown")[sel], (R[y]["cls"] != "unknown")[sel])
        summary[f"gap_diff_{x}_vs_{y}"] = float((R[y]["gap"] - R[x]["gap"])[sel].mean())
        summary[f"wrong_{x}_vs_{y}_on_paired_set"] = [float((R[x]["cls"] != "unknown")[sel].mean()), float((R[y]["cls"] != "unknown")[sel].mean()), int(sel.sum())]
    res[key] = summary
    print(key); [print(f"   {n:28s} unk {s['unknown']:.3f} wrong {s['wrong_entity']:.3f} self {s['self_subject']:.3f} true {s['true_object']:.3f} sib {s['sibling_subject']:.3f} other {s['other']:.3f} gap {s['gap_mean']:+.2f}") for n, s in summary.items() if isinstance(s, dict) and 'unknown' in s]
    [print("   ", k, v) for k, v in summary.items() if k.startswith("mcnemar") or k.startswith("gap_diff") or k.startswith("wrong_")]
json.dump(res, open(os.environ.get("SO_OUT", ".") + "/blank_attrib_seed0.json", "w"), indent=1)
print("BLANK DONE")
