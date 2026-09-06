"""Entity-generalisation pilot (critic item 7): does the E-000020 seed-0 adapter route and read a bank over
256 entity tokens it never saw in training?  Two held-out sets: NEXT (regex matches 257-512, frequency-confounded)
and INTERLEAVED (odd positions of the first 512, frequency-matched; the trained set is the even positions only
where they coincide with the original 256 -- reported as the in-vocab control on the same bank).  2x2 cells:
subject in {trained, held-out} x object in {trained, held-out}.  Reads on trained medial templates (t1, t9) with
and without the lone-space prefix; routing top-1 from the returned tensor.  Diagnostic only, seed 0."""
import json, os, numpy as np, torch
torch.set_num_threads(1)
from so.llm_adapter import AdapterConfig
from so.data import bank_from_store
from so.mvcc import MVCCStore
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
gk = E8.GPT2Knowledge(cfg); out = E20.train_or_load(gk, 0, 3000); centre = out["centre"]; model = gk.model; model.eval()
tok = gk.tok
ids512 = E8.select_entities(tok, 512)
trained = list(ids512[:256]); nxt = list(ids512[256:512]); inter = list(ids512[1::2])
rng = np.random.default_rng(0)
def run(table_ids, label, held_mask):
    """table_ids: 256 ids (some trained, some held-out); held_mask[i] True if id i is NOT in the trained 256."""
    saved = (model.entity_token_ids.clone(), model.candidate_ids.clone())
    model.entity_token_ids.copy_(torch.as_tensor(table_ids)); model.candidate_ids.copy_(torch.as_tensor(list(table_ids) + [gk.unknown_id]))
    names = [tok.decode([i]) for i in table_ids]
    n = 256; R = 4
    # 600 random facts, distinct keys; markers valid
    store = MVCCStore(marker_dim=centre.shape[0], seed=7, marker_centre=centre)
    keys = set(); facts = []
    while len(facts) < 600:
        s, r, o = int(rng.integers(n)), int(rng.integers(R)), int(rng.integers(n))
        if (s, r) in keys: continue
        keys.add((s, r)); facts.append((s, r, o)); store.write(s, r, o)
    bank = bank_from_store(store); T = bank.tensors(); kid_row = {int(k): i for i, k in enumerate(bank.kid)}
    rows_of = {(s, r): i for i, (s, r) in enumerate(zip(bank.subject, bank.relation))}
    res = {}
    for template in (1, 9):
        for prefix in ("", " "):
            texts = [prefix + E17.TEMPLATES12[r][template].format(s=names[s]) for s, r, o in facts]
            ans, hit = [], []
            for i in range(0, len(texts), 50):
                ids, am, last = E8.encode_texts(tok, texts[i:i+50])
                with torch.no_grad(): cand, _, routing, _ = model(T, ids, am, last)
                a = cand.argmax(-1).numpy(); ans.append(a)
                top = routing[:, 2].argmax(-1).numpy()          # layer-10 resolve
                hit.append(np.array([top[j] == rows_of[(facts[i+j][0], facts[i+j][1])] for j in range(len(a))]))
            ans = np.concatenate(ans); hit = np.concatenate(hit)
            truth = np.array([o for _, _, o in facts]); sh = np.array([held_mask[s] for s, _, _ in facts]); oh = np.array([held_mask[o] for _, _, o in facts])
            cell = {}
            for sn, sm in (("s_trained", ~sh), ("s_held", sh)):
                for on, om in (("o_trained", ~oh), ("o_held", oh)):
                    m = sm & om
                    if m.sum() == 0: continue
                    cell[f"{sn}/{on}"] = {"n": int(m.sum()), "read": float((ans[m] == truth[m]).mean()), "route": float(hit[m].mean()),
                                          "read_given_route": float((ans[m & hit] == truth[m & hit]).mean()) if (m & hit).sum() else None}
            res[f"t{template}{'_space' if prefix else ''}"] = cell
            print(label, f"t{template}{'_space' if prefix else ''}", json.dumps(cell), flush=True)
    model.entity_token_ids.copy_(saved[0]); model.candidate_ids.copy_(saved[1])
    return res
out_all = {}
# Table A: NEXT -- all 256 held-out (frequency-confounded)
out_all["next"] = run(nxt, "NEXT", [True] * 256)
# Table B: INTERLEAVED odd positions of the first 512: 128 of them are inside the trained 256, 128 are not
trained_set = set(trained)
out_all["interleaved"] = run(inter, "INTER", [i not in trained_set for i in inter])
# Table C: in-vocab control on the trained table itself (same bank recipe)
out_all["in_vocab"] = run(trained, "INVOCAB", [False] * 256)
json.dump(out_all, open(os.environ.get("SO_OUT", ".") + "/pilot_entity_seed0.json", "w"), indent=1)
print("PILOT DONE")
