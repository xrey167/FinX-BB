"""End-to-end check of MVCCStore(blank_export='foreign') on the retrained E-000020 seed-0 adapter: the same
100-pod world as blank_attrib.py, BLANK through the real store path under both options, wrong-entity rate per template."""
import copy, json, numpy as np, torch
torch.set_num_threads(1)
from so.llm_adapter import AdapterConfig
from so.data import bank_from_store
from so.mvcc import MVCCStore
from so.world import UNKNOWN
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
gk = E8.GPT2Knowledge(cfg); out = E20.train_or_load(gk, 0, 3000); centre = out["centre"]; model = gk.model; model.eval()
rng = np.random.default_rng(2000)
world, spec = E15.sample_alias_world(rng, E20.EVAL["n_base"], E20.EVAL["n_groups"], E20.EVAL["n_alias_per_group"], gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
res = {}
for option in ("self", "foreign"):
    st = MVCCStore(marker_dim=centre.shape[0], seed=2000, marker_centre=centre, blank_export=option)
    kids = {}
    for f in world.facts:
        if f.key not in spec.alias_of: kids[f.key] = st.write(f.subject, f.relation, f.obj, provenance="fact")
    for f in world.facts:
        if f.key in spec.alias_of: kids[f.key] = st.link(f.subject, f.relation, kids[spec.alias_of[f.key]], provenance="alias")
    for t, ks in spec.groups:
        st.evict(kids[t])
        for a in ks: st.blank(kids[a])
    bank = bank_from_store(st)
    aliases = [a for _, ks in spec.groups for a in ks]
    for template, prefix in ((0, " "), (0, ""), (3, ""), (8, "")):
        a, _, _ = E20._answers(gk, bank, aliases, gk.names, template=template) if not prefix else (None, None, None)
        if prefix:
            texts = [prefix + E17.TEMPLATES12[r][template].format(s=gk.names[s]) for s, r in aliases]; L = []
            for i in range(0, len(texts), 50):
                ids, am, last = E8.encode_texts(gk.tok, texts[i:i+50])
                with torch.no_grad(): cand, _, _, _ = model(bank.tensors(), ids, am, last)
                L.append(cand.argmax(-1).numpy())
            a = np.concatenate(L); a = np.where(a == gk.n_entities, UNKNOWN, a)
        key = f"t{template}{'_space' if prefix else ''}"
        res.setdefault(option, {})[key] = {"unknown": float((a == UNKNOWN).mean()), "wrong_entity": float((a != UNKNOWN).mean())}
        print(option, key, res[option][key], flush=True)
json.dump(res, open(os.environ.get("SO_OUT", ".") + "/blank_option_check_seed0.json", "w"), indent=1)
print("OPTION CHECK DONE")
