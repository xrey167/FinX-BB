"""Q1c -- is read layer 8 used at all on the recorded symlink design?  Routing masses per slot for DIRECT fact
reads (200 base keys) at templates 0 (+space) and 9, plus the answer accuracy with layer 8's injection removed."""
import json, os, numpy as np, torch
torch.set_num_threads(1)
from so.llm_adapter import AdapterConfig
from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
gk = E8.GPT2Knowledge(cfg); out = E20.train_or_load(gk, 0, 3000); centre = out["centre"]; model = gk.model; model.eval()
rng = np.random.default_rng(2000)
world, spec = E15.sample_alias_world(rng, E20.EVAL["n_base"], E20.EVAL["n_groups"], E20.EVAL["n_alias_per_group"], gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
store, kids = E15.load_arm(world, spec, centre, 2000, symlink=True); bank = bank_from_store(store); T = bank.tensors()
kid_row = {int(k): i for i, k in enumerate(bank.kid)}
base_keys = [f.key for f in world.facts if f.key not in spec.alias_of][:200]
rows = np.array([kid_row[kids[k]] for k in base_keys]); truth = np.array([world.index[k] for k in base_keys]); C = len(bank.kid)
res = {}
for template, prefix in ((0, " "), (0, ""), (9, "")):
    texts = [prefix + E17.TEMPLATES12[r][template].format(s=gk.names[s]) for s, r in base_keys]
    for arm in ("normal", "no_layer8"):
        saved = model.inject_gain.data.clone()
        if arm == "no_layer8": model.inject_gain.data[0] = 0.0
        ans, R = [], []
        for i in range(0, len(texts), 50):
            ids, am, last = E8.encode_texts(gk.tok, texts[i:i+50])
            with torch.no_grad(): cand, _, routing, _ = model(T, ids, am, last)
            a = cand.argmax(-1).numpy(); ans.append(np.where(a == gk.n_entities, -1, a)); R.append(routing.numpy())
        model.inject_gain.data.copy_(saved)
        ans = np.concatenate(ans); R = np.concatenate(R); ar = np.arange(len(ans))
        d = {"acc": float((ans == truth).mean()), "unknown": float((ans == -1).mean())}
        for name, slot in (("res8", 0), ("der8", 1), ("res10", 2), ("der10", 3)):
            p = R[:, slot]; d[name] = {"row": float(p[ar, rows].mean()), "null": float(p[:, C].mean())}
        res[f"t{template}{'_space' if prefix else ''}/{arm}"] = d
        print(f"t{template}{'_space' if prefix else ''} {arm:9s} acc {d['acc']:.3f} unk {d['unknown']:.3f} | " + " | ".join(f"{k} row{v['row']:.2f} null{v['null']:.2f}" for k, v in d.items() if isinstance(v, dict)), flush=True)
print("inject_gain", model.inject_gain.data.tolist())
json.dump(res, open(os.environ.get("SO_OUT", ".") + "/q1c_layer8_seed0.json", "w"), indent=1)
print("Q1C DONE")
