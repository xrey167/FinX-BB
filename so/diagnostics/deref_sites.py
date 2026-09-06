"""Q1b -- where does the recorded symlink design dereference?  Per read layer, the deref slot's mass on the
target row, the alias row and the null column (passthrough), for 200 alias queries; plus an arm with passthrough
forced at BOTH layers (no dereference anywhere) and at layer 10 only."""
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
alias_keys = spec.alias_keys
a_rows = np.array([kid_row[kids[k]] for k in alias_keys]); t_rows = np.array([kid_row[kids[spec.alias_of[k]]] for k in alias_keys])
truth = np.array([world.index[spec.alias_of[k]] for k in alias_keys]); C = len(bank.kid)
res = {}
for template, prefix in ((0, " "), (9, "")):
    texts = [prefix + E17.TEMPLATES12[r][template].format(s=gk.names[s]) for s, r in alias_keys]
    for arm, bump in (("normal", (0, 0)), ("pass8", (50, 0)), ("pass10", (0, 50)), ("pass_both", (50, 50))):
        saved = model.deref_pass_bias.data.clone(); model.deref_pass_bias.data += torch.tensor(bump, dtype=torch.float32)
        ans, R = [], []
        for i in range(0, len(texts), 50):
            ids, am, last = E8.encode_texts(gk.tok, texts[i:i+50])
            with torch.no_grad(): cand, _, routing, _ = model(T, ids, am, last)
            a = cand.argmax(-1).numpy(); ans.append(np.where(a == gk.n_entities, -1, a)); R.append(routing.numpy())
        model.deref_pass_bias.data.copy_(saved)
        ans = np.concatenate(ans); R = np.concatenate(R); ar = np.arange(len(ans))
        d = {"acc": float((ans == truth).mean()), "unknown": float((ans == -1).mean())}
        for name, slot in (("res8", 0), ("der8", 1), ("res10", 2), ("der10", 3)):
            p = R[:, slot]
            d[name] = {"alias": float(p[ar, a_rows].mean()), "target": float(p[ar, t_rows].mean()), "null": float(p[:, C].mean())}
        res[f"t{template}{'_space' if prefix else ''}/{arm}"] = d
        print(f"t{template}{'_space' if prefix else ''} {arm:9s} acc {d['acc']:.3f} unk {d['unknown']:.3f} | " + " | ".join(f"{k} a{v['alias']:.2f} t{v['target']:.2f} n{v['null']:.2f}" for k, v in d.items() if isinstance(v, dict)), flush=True)
json.dump(res, open(os.environ.get("SO_OUT", ".") + "/q1b_derefmass_seed0.json", "w"), indent=1)
print("Q1B DONE")
