"""E-000066 -- stale exported-bank replay attack on the current symlink/pod data plane.

Threat model: an actor can retain a previously exported/readable Bank (or its tensors) but does NOT
control the live MVCC store. The current adapter consumes Bank tensors and has no independent current
pod-generation input. Therefore a stale snapshot can be replayed after SHRED/DELETE unless some live
authority sits outside the snapshot.

This experiment is structural and trains nothing. It is a falsification of generation safety, not a
security exploit against GitHub or any external service.

Registered prediction for the CURRENT architecture:
  1. active snapshot resolves root + aliases to object;
  2. fresh export after SHRED/DELETE resolves UNKNOWN;
  3. replaying the PRE-mutation snapshot still resolves the old object;
  4. its tensors are byte-identical to the pre-mutation tensors because the snapshot is detached;
  5. no generation/incarnation field exists in the exported Bank interface to reject it.

If all five hold, CAVI needs a live incarnation/capability boundary between cached memory material and
neural injection. This behavior is expected from snapshots in general; the contribution is diagnosing
it as a missing trust boundary in this neural-memory architecture, not claiming replay prevention is new.

Run: python -m so.experiments.e000066_stale_snapshot_replay --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from so.data import bank_from_store
from so.mvcc import MVCCStore


def tensor_digest(bank) -> str:
    h = hashlib.sha256()
    t = bank.tensors()
    for k in sorted(t):
        x = t[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(x.dtype).encode()); h.update(np.asarray(x.shape,dtype=np.int64).tobytes())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def run(seed: int, aliases: int) -> Dict[str, object]:
    s = MVCCStore(seed=seed)
    root_key=(100,0); obj=77
    root=s.write(*root_key,obj,provenance="pod")
    alias_keys=[]
    for i in range(aliases):
        k=(200+i,0); s.link(*k,root,provenance="alias"); alias_keys.append(k)
    keys=[root_key,*alias_keys]

    stale=bank_from_store(s)                 # attacker-retained old export
    stale_digest_before=tensor_digest(stale)
    active_ok=all(stale.index_view.get(k)==obj for k in keys)

    s.shred(root)
    fresh_shred=bank_from_store(s)
    fresh_shred_closed=all(fresh_shred.index_view.get(k) is None for k in keys)
    stale_after_shred=all(stale.index_view.get(k)==obj for k in keys)
    stale_digest_after_shred=tensor_digest(stale)

    s.resign(root); s.delete(root)
    fresh_delete=bank_from_store(s)
    fresh_delete_closed=all(fresh_delete.index_view.get(k) is None for k in keys)
    stale_after_delete=all(stale.index_view.get(k)==obj for k in keys)
    stale_digest_after_delete=tensor_digest(stale)

    # Current Bank carries no pod incarnation/generation that can be checked against a live authority.
    generation_fields=[name for name in vars(stale) if any(x in name.lower() for x in ("generation","incarnation","epoch","capability"))]
    no_generation_field=len(generation_fields)==0

    checks={
        "active_snapshot_readable":active_ok,
        "fresh_shred_closed":fresh_shred_closed,
        "stale_replay_after_shred":stale_after_shred,
        "fresh_delete_closed":fresh_delete_closed,
        "stale_replay_after_delete":stale_after_delete,
        "snapshot_byte_stable":stale_digest_before==stale_digest_after_shred==stale_digest_after_delete,
        "no_generation_field":no_generation_field,
    }
    return {"seed":seed,"aliases":aliases,"attack_reproduced":all(checks.values()),"checks":checks,
            "generation_like_fields":generation_fields,"stale_digest":stale_digest_before}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,nargs="*",default=[0,1,2,3,4])
    ap.add_argument("--aliases",type=int,nargs="*",default=[1,4,16,64]); ap.add_argument("--results-dir",default="so/results")
    a=ap.parse_args(); rows:List[Dict[str,object]]=[run(s,n) for s in a.seeds for n in a.aliases]
    reproduced=all(bool(r["attack_reproduced"]) for r in rows)
    rec={"experiment":"E-000066","attack":"stale exported Bank replay","all_reproduced":reproduced,"rows":rows,
         "reading":"If true, the current neural-memory interface has no live generation authority; fresh store deletion is correct but an old exported memory snapshot remains replayable."}
    p=Path(a.results_dir);p.mkdir(parents=True,exist_ok=True);(p/"e000066_stale_snapshot_replay.json").write_text(json.dumps(rec,indent=2),encoding="utf-8")
    print(json.dumps({"all_reproduced":reproduced,"runs":len(rows),"sample":rows[0]},indent=2))
    if not reproduced: raise SystemExit(2)

if __name__=="__main__": main()
