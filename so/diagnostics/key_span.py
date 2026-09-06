"""Is 31.48's R4 -- the held-out routing failure -- MEMORISATION or UNDERDETERMINATION?

k_proj is nn.Linear(768, 256, bias=False) (so/llm_adapter.py:115). Its training inputs are
ln_key(W_in[s] + rel_emb[r]) over the 256 TRAINED subjects and 4 relations -- at most 260 directions
of a 768-dimensional input space before the LayerNorm. The routing loss therefore says nothing about
the orthogonal complement except through weight decay. If a held-out subject's key input has a large
component OFF the trained span, its routing failure is a statement about where the map was fitted,
not about per-entity memorisation.

Zero training. Reads so/results/checkpoints/e000020_gpt2_seed0.pt.

Run:  python -m so.diagnostics.key_span
Per held-out subject: (a) the off-span energy of its key input, (b) whether the adapter routes it
top-1 on a 600-fact bank, at the templates R4 used. Then Spearman(off-span, routing success).
Controls: the same two quantities for the TRAINED subjects (whose off-span energy is ~0 by
construction -- that is the control that says the measure is doing what it claims), and a
norm control, since ||k|| enters the score linearly and a norm difference alone would explain a
threshold effect without any span story.
"""
import json, numpy as np, torch
torch.set_num_threads(1)
from so.llm_adapter import AdapterConfig
from so.data import bank_from_store
from so.mvcc import MVCCStore
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20

cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
gk = E8.GPT2Knowledge(cfg); out = E20.train_or_load(gk, 0, 3000); model = gk.model; model.eval()
centre = out["centre"]; tok = gk.tok
ids512 = E8.select_entities(tok, 512)
trained_ids = list(ids512[:256])
inter = list(ids512[1::2])                      # frequency-matched split, as in the R4 pilot
trained_set = set(trained_ids)

# --- the trained span: the actual inputs k_proj saw, post-ln_key
with torch.no_grad():
    w_in = model.w_in
    rel = model.rel_emb.weight                                   # (n_relations, d)
    X = []
    for r in range(rel.shape[0]):
        X.append(model.ln_key(w_in[torch.as_tensor(trained_ids)] + rel[r][None]))
    X = torch.cat(X, 0)                                          # (256*R, 768)
    Xc = X - X.mean(0, keepdim=True)
    U, Sv, Vh = torch.linalg.svd(Xc, full_matrices=False)
    energy = (Sv ** 2).cumsum(0) / (Sv ** 2).sum()
    rank95 = int((energy < 0.95).sum().item()) + 1
    rank99 = int((energy < 0.99).sum().item()) + 1
    V = Vh[:rank99]                                              # the trained span, 99% energy
    mu = X.mean(0, keepdim=True)

def off_span(ids, r):
    with torch.no_grad():
        x = model.ln_key(w_in[torch.as_tensor(ids)] + rel[r][None])
        xc = x - mu
        proj = (xc @ V.t()) @ V
        return (xc - proj).norm(dim=-1) / xc.norm(dim=-1).clamp_min(1e-9), x.norm(dim=-1)

res = {"rank95": rank95, "rank99": rank99, "n_train_vectors": int(X.shape[0]), "d_in": int(X.shape[1])}
print(f"trained span: rank95={rank95} rank99={rank99} of {X.shape[0]} vectors in R^{X.shape[1]}", flush=True)
for name, ids in (("trained", trained_ids), ("interleaved_held", [i for i in inter if i not in trained_set]),
                  ("next256", list(ids512[256:512]))):
    o, n = off_span(ids, 0)
    res[name] = {"off_span_mean": float(o.mean()), "off_span_p10": float(o.quantile(0.1)), "off_span_p90": float(o.quantile(0.9)),
                 "key_norm_mean": float(n.mean()), "n": len(ids)}
    print(f"{name:20s} off-span {o.mean():.4f} [{o.quantile(0.1):.4f},{o.quantile(0.9):.4f}]  ||x|| {n.mean():.2f}  n={len(ids)}", flush=True)

# --- per-entity routing success on the interleaved table (the R4 pilot's own setting)
rng = np.random.default_rng(0)
table = inter; held_mask = [i not in trained_set for i in table]
saved = (model.entity_token_ids.clone(), model.candidate_ids.clone())
model.entity_token_ids.copy_(torch.as_tensor(table)); model.candidate_ids.copy_(torch.as_tensor(list(table) + [gk.unknown_id]))
names = [tok.decode([i]) for i in table]
store = MVCCStore(marker_dim=centre.shape[0], seed=7, marker_centre=centre)
keys, facts = set(), []
while len(facts) < 600:
    s, r, o = int(rng.integers(256)), int(rng.integers(4)), int(rng.integers(256))
    if (s, r) in keys: continue
    keys.add((s, r)); facts.append((s, r, o)); store.write(s, r, o)
bank = bank_from_store(store); T = bank.tensors()
rows_of = {(int(s), int(r)): i for i, (s, r) in enumerate(zip(bank.subject, bank.relation))}
hit_by_subject = {}
for template in (1, 9):
    texts = [E17.TEMPLATES12[r][template].format(s=names[s]) for s, r, o in facts]
    hits = []
    for i in range(0, len(texts), 50):
        ids_t, am, last = E8.encode_texts(tok, texts[i:i+50])
        with torch.no_grad(): _, _, routing, _ = model(T, ids_t, am, last)
        top = routing[:, 2].argmax(-1).numpy()
        hits.extend([top[j] == rows_of[(facts[i+j][0], facts[i+j][1])] for j in range(len(top))])
    for (s, r, o), h in zip(facts, hits):
        hit_by_subject.setdefault((template, s), []).append(bool(h))
model.entity_token_ids.copy_(saved[0]); model.candidate_ids.copy_(saved[1])

def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

for template in (1, 9):
    subs = sorted({s for (t, s) in hit_by_subject if t == template})
    ids_of = [table[s] for s in subs]
    o, n = off_span(ids_of, 0)
    succ = np.array([np.mean(hit_by_subject[(template, s)]) for s in subs])
    held = np.array([held_mask[s] for s in subs])
    row = {"n_subjects": len(subs),
           "spearman_offspan_vs_routing": spearman(o.numpy(), succ),
           "spearman_keynorm_vs_routing": spearman(n.numpy(), succ),
           "routing_trained": float(succ[~held].mean()), "routing_held": float(succ[held].mean()),
           "offspan_trained": float(o.numpy()[~held].mean()), "offspan_held": float(o.numpy()[held].mean()),
           "keynorm_trained": float(n.numpy()[~held].mean()), "keynorm_held": float(n.numpy()[held].mean())}
    # within the held-out subjects only: does off-span still order routing success?
    if held.sum() > 10:
        row["spearman_offspan_within_held"] = spearman(o.numpy()[held], succ[held])
        row["spearman_keynorm_within_held"] = spearman(n.numpy()[held], succ[held])
    res[f"t{template}"] = row
    print(f"t{template}: rho(offspan,route)={row['spearman_offspan_vs_routing']:+.3f} rho(||k||,route)={row['spearman_keynorm_vs_routing']:+.3f} "
          f"| route tr/held {row['routing_trained']:.3f}/{row['routing_held']:.3f} "
          f"| offspan tr/held {row['offspan_trained']:.4f}/{row['offspan_held']:.4f} "
          f"| ||k|| tr/held {row['keynorm_trained']:.2f}/{row['keynorm_held']:.2f}"
          + (f" | within-held rho off {row['spearman_offspan_within_held']:+.3f} norm {row['spearman_keynorm_within_held']:+.3f}" if 'spearman_offspan_within_held' in row else ""), flush=True)
json.dump(res, open("so/results/diagnostics/key_span_seed0.json", "w"), indent=1)
print("SPAN DONE")
