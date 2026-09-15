"""E-000056 closure, two halves.  (1) The recorded-gate half on the three retrained E-000015 checkpoints: gate
acceptance as a function of chord distance from the training centre (E-000029's shell sampler) -- a single
monotone cap means no rotation of the centre is rejected while a nearer one is accepted, and the lemma (a fixed
acceptor of the marker alone admits at most one monotone epoch transition) applies.  (2) The clock-gate
micro-benchmark: the reader's gate family (Linear-GELU-Linear, sigmoid) given marker (+) current centre, trained on
K random epoch centres with previous-epoch negatives at chord 0.5/0.7/1.0/sqrt2, evaluated on 100 UNSEEN centres,
worst of 3 seeds, hard gate at 0.5; ceiling arm sigma(a*(m.c)-b).  Seconds on CPU."""
import json, os, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
from so.experiments import e000029_marker_geometry as E29
from so.experiments import e000015_symlink_cells as E15
out = {"recorded_gates": {}, "clock_gate": {}}
rng = np.random.default_rng(0)
for seed in (0, 1, 2):
    r = E15.train_or_load(seed, 4000, 1); model, centre = r["model"], np.asarray(r["centre"], dtype=float); centre /= np.linalg.norm(centre)
    row = {}
    for dist in (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.414):
        m = E29.shell(rng, centre, dist, 2000)
        with torch.no_grad(): g = model.gate(torch.as_tensor(m, dtype=torch.float32)).squeeze(-1).numpy()
        row[str(dist)] = float((g > 0.5).mean())
    out["recorded_gates"][seed] = row
    print("recorded gate seed", seed, json.dumps(row), flush=True)
def unit(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
def make(centres, n_per, rng):
    X, Y = [], []
    for c in centres:
        pos = unit(c + rng.normal(scale=0.05, size=(n_per, 16)))
        negs = [E29.shell(rng, c, d, n_per // 4) for d in (0.5, 0.7, 1.0, 1.414)] + [unit(rng.normal(size=(n_per // 2, 16)))]
        neg = np.concatenate(negs)
        X.append(np.concatenate([np.concatenate([pos, np.repeat(c[None], len(pos), 0)], 1), np.concatenate([neg, np.repeat(c[None], len(neg), 0)], 1)]))
        Y.append(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
    return torch.as_tensor(np.concatenate(X), dtype=torch.float32), torch.as_tensor(np.concatenate(Y), dtype=torch.float32)
for K in (4, 16, 64):
    res_K = {}
    for seed in (0, 1, 2):
        rng = np.random.default_rng(100 + seed); torch.manual_seed(seed)
        train_c = unit(rng.normal(size=(K, 16))); test_c = unit(rng.normal(size=(100, 16)))
        Xtr, Ytr = make(train_c, 256, rng); Xte, Yte = make(test_c, 64, rng)
        for arm in ("mlp", "bilinear"):
            if arm == "mlp":
                net = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 1))
                f = lambda X: net(X).squeeze(-1)
            else:
                a = nn.Parameter(torch.tensor(10.0)); b = nn.Parameter(torch.tensor(9.0))
                net = nn.ParameterList([a, b])
                f = lambda X: a * (X[:, :16] * X[:, 16:]).sum(-1) - b
            opt = torch.optim.Adam(net.parameters(), lr=(3e-3 if arm == "mlp" else 5e-2))
            for step in range(3000):
                idx = torch.randint(0, len(Xtr), (512,))
                loss = nn.functional.binary_cross_entropy_with_logits(f(Xtr[idx]), Ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pred = (f(Xte) > 0).float(); far = float(pred[Yte == 0].mean()); frr = float(1 - pred[Yte == 1].mean())
                # FAR split by negative class: previous-epoch at each chord vs uniform
                pred_tr = (f(Xtr) > 0).float(); far_tr = float(pred_tr[Ytr == 0].mean()); frr_tr = float(1 - pred_tr[Ytr == 1].mean())
            res_K.setdefault(arm, []).append({"seed": seed, "FAR_unseen": far, "FRR_unseen": frr, "FAR_train": far_tr, "FRR_train": frr_tr})
    out["clock_gate"][K] = res_K
    for arm, rows in res_K.items():
        print(f"K={K} {arm:8s} worst FAR_unseen {max(r['FAR_unseen'] for r in rows):.3f} worst FRR_unseen {max(r['FRR_unseen'] for r in rows):.3f} | train FAR {max(r['FAR_train'] for r in rows):.3f} FRR {max(r['FRR_train'] for r in rows):.3f}", flush=True)
json.dump(out, open(os.environ.get("SO_OUT", ".") + "/e56_clockgate_v2.json", "w"), indent=1)
print("E56 DONE")
