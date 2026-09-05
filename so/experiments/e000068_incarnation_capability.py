"""E-000068 -- incarnation-bound one-use capability closes E-000066's stale-snapshot replay boundary.

This is a systems/security control, not a cryptographic novelty claim. It composes established tools
(monotonic generations, HMAC authentication, nonce binding, one-use capabilities) with the canonical
pod identity. The question is whether this gives the neural memory interface the one thing E-000066
shows it lacks: LIVE authority outside the serialized Bank.

Registered contract across UPDATE/REVOKE/RESTORE/SHRED/RESIGN/EVICT/DELETE:
  * a capability issued before a lifecycle transition never verifies afterward;
  * a capability cannot be consumed twice even without a lifecycle transition;
  * a capability bound to nonce A fails for nonce B;
  * an inactive/deleted pod cannot mint a fresh capability;
  * RESTORE/RESIGN create a NEW incarnation; old capabilities never become valid again (ABA-safe);
  * an unrelated pod's capability remains valid when another pod changes.

This prototype deliberately does NOT modify the model yet. E-000069 must put this check on the actual
Bank->adapter injection boundary and show that cached old bank tensors no longer reproduce the old
answer after a live lifecycle transition.

Run: python -m so.experiments.e000068_incarnation_capability --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from so.incarnation import IncarnationAuthority


def nonce(seed: int, tag: str) -> bytes:
    return hashlib.sha256(f"{seed}:{tag}".encode()).digest()[:16]


def run(seed: int) -> Dict[str, object]:
    a=IncarnationAuthority(secret=hashlib.sha256(f"secret:{seed}".encode()).digest())
    a.create(1); a.create(2)
    checks: Dict[str,bool] = {}

    n0=nonce(seed,"initial")
    c0=a.issue(1,n0)
    checks["initial_valid"] = a.verify_and_consume(c0,n0)
    checks["one_use_replay_rejected"] = not a.verify_and_consume(c0,n0)

    c_nonce=a.issue(1,nonce(seed,"nonce-a"))
    checks["wrong_nonce_rejected"] = not a.verify_and_consume(c_nonce,nonce(seed,"nonce-b"))
    checks["right_nonce_still_valid_after_wrong_attempt"] = a.verify_and_consume(c_nonce,nonce(seed,"nonce-a"))

    # Unrelated pod proof: issue before target changes, consume after target changes.
    other_nonce=nonce(seed,"other")
    other=a.issue(2,other_nonce)

    pre_update=a.issue(1,nonce(seed,"pre-update")); old_inc=pre_update.incarnation
    new_inc=a.update(1)
    checks["update_bumps"] = new_inc > old_inc
    checks["pre_update_rejected"] = not a.verify_and_consume(pre_update,pre_update.nonce)
    checks["unrelated_survives_target_update"] = a.verify_and_consume(other,other_nonce)

    pre_revoke=a.issue(1,nonce(seed,"pre-revoke")); revoke_inc=a.revoke(1)
    checks["revoke_bumps"] = revoke_inc > pre_revoke.incarnation
    checks["pre_revoke_rejected"] = not a.verify_and_consume(pre_revoke,pre_revoke.nonce)
    revoke_issue_denied=False
    try: a.issue(1,nonce(seed,"while-revoked"))
    except PermissionError: revoke_issue_denied=True
    checks["revoke_mint_denied"] = revoke_issue_denied

    restore_inc=a.restore(1)
    post_restore=a.issue(1,nonce(seed,"post-restore"))
    checks["restore_new_incarnation"] = restore_inc > revoke_inc and post_restore.incarnation==restore_inc
    checks["old_pre_revoke_never_revives"] = not a.verify_and_consume(pre_revoke,pre_revoke.nonce)
    checks["post_restore_valid"] = a.verify_and_consume(post_restore,post_restore.nonce)

    pre_shred=a.issue(1,nonce(seed,"pre-shred")); shred_inc=a.shred(1)
    checks["pre_shred_rejected"] = not a.verify_and_consume(pre_shred,pre_shred.nonce)
    shred_issue_denied=False
    try: a.issue(1,nonce(seed,"while-shredded"))
    except PermissionError: shred_issue_denied=True
    checks["shred_mint_denied"] = shred_issue_denied

    resign_inc=a.resign(1); post_resign=a.issue(1,nonce(seed,"post-resign"))
    checks["resign_is_new_not_aba"] = resign_inc > shred_inc and post_resign.incarnation==resign_inc
    checks["pre_shred_never_revives"] = not a.verify_and_consume(pre_shred,pre_shred.nonce)
    checks["post_resign_valid"] = a.verify_and_consume(post_resign,post_resign.nonce)

    pre_evict=a.issue(1,nonce(seed,"pre-evict")); evict_inc=a.evict(1)
    checks["pre_evict_rejected"] = not a.verify_and_consume(pre_evict,pre_evict.nonce)
    evict_issue_denied=False
    try: a.issue(1,nonce(seed,"while-evicted"))
    except PermissionError: evict_issue_denied=True
    checks["evict_mint_denied"] = evict_issue_denied

    restored=a.restore(1); pre_delete=a.issue(1,nonce(seed,"pre-delete")); delete_inc=a.delete(1)
    checks["delete_bumps"] = delete_inc > restored
    checks["pre_delete_rejected"] = not a.verify_and_consume(pre_delete,pre_delete.nonce)
    delete_issue_denied=False
    try: a.issue(1,nonce(seed,"after-delete"))
    except PermissionError: delete_issue_denied=True
    checks["delete_mint_denied"] = delete_issue_denied
    delete_restore_denied=False
    try: a.restore(1)
    except KeyError: delete_restore_denied=True
    checks["delete_restore_denied"] = delete_restore_denied

    passed=all(checks.values())
    return {"seed":seed,"pass":passed,"checks":checks,"final_incarnation":a.state(1).incarnation}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--seeds",type=int,nargs="*",default=[0,1,2,3,4]);ap.add_argument("--results-dir",default="so/results")
    a=ap.parse_args(); rows:List[Dict[str,object]]=[run(s) for s in a.seeds]; all_pass=all(bool(r["pass"]) for r in rows)
    rec={"experiment":"E-000068","all_pass":all_pass,"rows":rows,
         "scope":"control-plane prototype only; established security primitives, not novelty",
         "next":"E-000069: enforce this live check at Bank->adapter injection and replay E-000066."}
    p=Path(a.results_dir);p.mkdir(parents=True,exist_ok=True);(p/"e000068_incarnation_capability.json").write_text(json.dumps(rec,indent=2),encoding="utf-8")
    print(json.dumps({"all_pass":all_pass,"runs":len(rows),"sample":rows[0]},indent=2))
    if not all_pass: raise SystemExit(2)

if __name__=="__main__": main()
