"""E90: incomplete source-cone falsification; NOT a trained-LLM/invention claim."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np

EFFECT_FLOOR = 1e-10  # Descriptive only, never the exact-equality gate.


def identical(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def difference(a, b):
    delta = np.abs(a - b)
    return {"byte_identical": identical(a, b), "maxabs": float(delta.max(initial=0)),
            "unequal_coordinates": int(np.count_nonzero(a != b)),
            "material_coordinates": int(np.count_nonzero(delta > EFFECT_FLOOR)),
            "coordinates": int(a.size)}


def normalize(x, kind):
    if kind == "identity":
        return x
    if kind == "layer":
        centered = x - np.mean(x)
        return centered / np.sqrt(np.mean(centered * centered) + 1e-5)
    if kind == "rms":
        return x / np.sqrt(np.mean(x * x) + 1e-5)
    raise ValueError(kind)


def block_case(seed, d=128, block=16, layers=8):
    if d % block or block < 2 or layers < 1:
        raise ValueError("Require d divisible by block>=2 and layers>=1")
    rng = np.random.default_rng(seed)
    return dict(weights=rng.normal(size=(layers, d//block, block, block))*(.5/np.sqrt(block)),
                bias=rng.normal(scale=.15, size=(layers, d)),
                initial=rng.normal(scale=.4, size=d),
                source_old=rng.normal(scale=.8, size=block),
                source_new=rng.normal(scale=.8, size=block), block=block)


def full_blocks(case, source, mode):
    h = case["initial"].copy()
    block = case["block"]
    h[:block] += source
    history = []
    for layer, weights in enumerate(case["weights"]):
        normalized = normalize(h, mode.removeprefix("global_")) if mode.startswith("global_") else h
        out = np.empty_like(h)
        for b, w in enumerate(weights):
            sl = slice(b*block, (b+1)*block)
            x = normalized[sl]
            if mode.startswith("local_"):
                x = normalize(x, mode.removeprefix("local_"))
            out[sl] = np.tanh(w @ x + case["bias"][layer, sl])
        h = out
        history.append(h.copy())
    return np.stack(history)


def replay_local(case, source, old, mode):
    """Ordinary exact cone replay. Both candidate and strong baseline get this."""
    if mode not in {"identity", "local_layer", "local_rms"}:
        raise ValueError("No independence proof for global normalization")
    block = case["block"]
    h = case["initial"][:block].copy() + source
    result = old.copy()
    for layer, weights in enumerate(case["weights"]):
        x = normalize(h, mode.removeprefix("local_")) if mode.startswith("local_") else h
        h = np.tanh(weights[0] @ x + case["bias"][layer, :block])
        result[layer, :block] = h
    return result


def normalization_screen(seed):
    case = block_case(seed)
    block = case["block"]
    rows = {}
    for mode in ("identity", "local_layer", "local_rms", "global_layer", "global_rms"):
        old = full_blocks(case, case["source_old"], mode)
        fresh = full_blocks(case, case["source_new"], mode)
        # Grant exact rebuilt values inside the claimed cone: this oracle patch
        # is stronger than any computed local patch, NOT a timed implementation.
        patch = old.copy()
        patch[:, :block] = fresh[:, :block]
        row = dict(full_rebuild_difference=difference(old, fresh),
                   outside_claimed_cone=difference(old[:, block:], fresh[:, block:]),
                   oracle_local_patch_vs_fresh=difference(patch, fresh),
                   first_layer_material_coordinates=int(np.count_nonzero(np.abs(old[0]-fresh[0]) > EFFECT_FLOOR)),
                   no_op_edit=difference(old, full_blocks(case, case["source_old"], mode)))
        if not mode.startswith("global_"):
            repaired = replay_local(case, case["source_new"], old, mode)
            # NEVER is rebuilt without source injection, from identical exogenous inputs.
            never = full_blocks(case, np.zeros(block), mode)
            deleted = replay_local(case, np.zeros(block), old, mode)
            conventional = replay_local(case, case["source_new"], old, mode)
            row.update(exact_replay_vs_fresh=difference(repaired, fresh),
                       deletion_vs_never=difference(deleted, never),
                       candidate_vs_conventional=difference(repaired, conventional),
                       candidate_affected_matvecs=len(old), conventional_affected_matvecs=len(old),
                       baseline_has_identical_state_and_operators=True)
        rows[mode] = row
    return rows


def routing_case(seed, queries=64, width=16, layers=3):
    rng = np.random.default_rng(1000+seed)
    q = rng.normal(scale=.03, size=(queries, width))
    q[:, 0] = rng.uniform(.9, 1.1, size=queries)
    keys = rng.normal(scale=.05, size=(8, width))
    keys[0] = 0
    keys[0, 0], keys[1, 0] = -5., 1.
    revised = keys.copy()
    revised[0, 0] = 5.
    return dict(queries=q, keys_old=keys, keys_new=revised,
                values=rng.normal(scale=.6, size=(8, width)),
                weights=rng.normal(size=(layers, width, width))*(.55/np.sqrt(width)),
                bias=rng.normal(scale=.1, size=(layers, width)))


def routing_full(case, keys):
    winners = np.argmax(case["queries"] @ keys.T, axis=1)
    h = case["values"][winners].copy()
    states = []
    for w, bias in zip(case["weights"], case["bias"]):
        h = np.tanh(h @ w.T + bias)
        states.append(h.copy())
    return np.stack(states), winners


def routing_screen(seed):
    case = routing_case(seed)
    old, ow = routing_full(case, case["keys_old"])
    fresh, nw = routing_full(case, case["keys_new"])
    selected_before = ow == 0
    changed = np.any(old != fresh, axis=2)
    old_cone = np.broadcast_to(selected_before[None, :], changed.shape)
    repair = old.copy()
    repair[:, selected_before, :] = fresh[:, selected_before, :]
    # Decision-aware reference reruns selection; full-batch operators preserve
    # numerical order. No partial execution speed or minimality is claimed.
    checked, cw = routing_full(case, case["keys_new"])
    decision_cone = (ow == 0) | (cw == 0) | (ow != cw)
    verified = old.copy()
    verified[:, decision_cone, :] = checked[:, decision_cone, :]
    return dict(queries=len(ow), layers=len(old), source_selected_before=int(selected_before.sum()),
                source_selected_after=int((nw == 0).sum()), changed_persistent_vectors=int(changed.sum()),
                vectors_missed_by_old_payload_lineage=int((changed & ~old_cone).sum()),
                old_payload_cone_patch_vs_fresh=difference(repair, fresh),
                decision_aware_vs_fresh=difference(verified, fresh),
                decision_scores_evaluated=int(len(ow)*len(case["keys_new"])),
                reference_is_conventional_recomputation_not_novel=True,
                adversarial_construction_not_frequency_estimate=True)


def denominator_screen(seed):
    rng = np.random.default_rng(2000+seed)
    scores = rng.normal(scale=.2, size=8)
    scores[0], scores[1] = -2., 2.
    revised = scores.copy()
    revised[0] = -1.
    values = rng.normal(size=(8, 16))
    w = rng.normal(size=(16, 16))/4
    def run(s, global_softmax):
        winner = int(np.argmax(s))
        e = np.exp(s-np.max(s))
        coeff = e[winner]/e.sum() if global_softmax else 1.
        h = coeff*values[winner]
        states = []
        for _ in range(3):
            h = np.tanh(w @ h)
            states.append(h.copy())
        return winner, np.stack(states)
    w0, old = run(scores, True)
    w1, new = run(revised, True)
    _, lo = run(scores, False)
    _, ln = run(revised, False)
    return dict(winner_before=w0, winner_after=w1, edited_source=0,
                global_denominator_effect=difference(old, new),
                postselection_renormalization_control=difference(lo, ln))


def averaged_lens_screen(seed):
    """Analytic averaged-Jacobian countermodel, NOT a full LLM J-space test.

    F_c(h)=a*h[0]+c*b*tanh(h[1]), c=+-1. Balanced independent contexts
    yield Jbar=[a,0], but source h[1] is causally used in either context.
    """
    rng = np.random.default_rng(3000+seed)
    a, b = rng.normal(size=(2, 8))
    payload = float(rng.uniform(.5, 1.5))
    stale, never = np.array([.25, payload]), np.array([.25, 0.])
    deriv = b*(1-np.tanh(payload)**2)
    jp, jn = np.column_stack((a, deriv)), np.column_stack((a, -deriv))
    averaged = (jp+jn)/2
    effect = b*np.tanh(payload)
    return dict(stale_vs_never_state=difference(stale, never),
                averaged_lens_stale_vs_never=difference(averaged @ stale, averaged @ never),
                positive_context_logit_effect_maxabs=float(np.max(np.abs(effect))),
                negative_context_logit_effect_maxabs=float(np.max(np.abs(-effect))),
                conditional_jacobian_source_column_maxabs=float(np.max(np.abs(deriv))),
                averaged_jacobian_source_column_maxabs=float(np.max(np.abs(averaged[:, 1]))),
                language_model_jspace_validation=False)


def validate(result):
    for row in result["seeds"]:
        for mode, cell in row["normalization"].items():
            assert cell["no_op_edit"]["byte_identical"]
            if mode.startswith("global_"):
                assert cell["outside_claimed_cone"]["maxabs"] > EFFECT_FLOOR
                assert not cell["oracle_local_patch_vs_fresh"]["byte_identical"]
            else:
                for key in ("oracle_local_patch_vs_fresh", "exact_replay_vs_fresh", "deletion_vs_never", "candidate_vs_conventional"):
                    assert cell[key]["byte_identical"], (mode, key)
        r = row["routing"]
        assert r["source_selected_before"] == 0
        assert r["source_selected_after"] == r["queries"]
        assert r["vectors_missed_by_old_payload_lineage"] == r["changed_persistent_vectors"]
        assert r["decision_aware_vs_fresh"]["byte_identical"]
        d = row["denominator"]
        assert d["winner_before"] == d["winner_after"] != d["edited_source"]
        assert d["global_denominator_effect"]["maxabs"] > EFFECT_FLOOR
        assert d["postselection_renormalization_control"]["byte_identical"]
        j = row["averaged_lens"]
        assert not j["stale_vs_never_state"]["byte_identical"]
        assert j["averaged_lens_stale_vs_never"]["byte_identical"]
        assert j["positive_context_logit_effect_maxabs"] > EFFECT_FLOOR


def run(seeds):
    rows = [dict(seed=s, normalization=normalization_screen(s), routing=routing_screen(s),
                 denominator=denominator_screen(s), averaged_lens=averaged_lens_screen(s)) for s in seeds]
    result = dict(experiment="E-000090", status="negative_result_not_invention",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                  effect_floor_descriptive_only=EFFECT_FLOOR, trained_language_models=0,
                  system_utility_gate="NOT_EVALUATED", seeds=rows)
    validate(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    p.add_argument("--output", type=Path, default=Path("e000090-results.json"))
    args = p.parse_args()
    result = run(args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(experiment=result["experiment"], seeds=args.seeds,
                         countermodel_regressions="PASS", output=str(args.output), major_invention=False), indent=2))

if __name__ == "__main__":
    main()
