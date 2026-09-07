"""E-000051-B: contamination-controlled re-run of E-000051.

The original E-000051 used aliases from two fixed pods in every class-(ii)
bystander set. When either of those pods was itself the deletion target, the
supposed bystander set contained keys from the deleted pod. This wrapper keeps
the original experiment unchanged and changes only that query selection.

For every target pod p, the two alias-providing pods are chosen from groups
whose target differs from p.target. An explicit disjointness assertion makes
this condition falsifiable at runtime.

All banks, readers, features, AUC computation, decision rules and thresholds
remain those of E-000051. This is a correction run, not a new mechanism claim.
"""
from __future__ import annotations

import numpy as np

from so.experiments import e000051_residue_reader as E51


def clean_other_pod_aliases(groups, target, n_other_pods: int = 2):
    """Return aliases from fixed other pods, excluding the current target pod."""
    candidates = [(t, ks) for t, ks in groups if t != target]
    if len(candidates) < n_other_pods:
        raise ValueError("not enough other pods to build a disjoint bystander set")
    chosen = candidates[-n_other_pods:]
    out = [k for _, ks in chosen for k in ks]
    return out


class CleanSetting(E51.Setting):
    """E-000051 Setting with pod-disjoint class-(ii) query keys."""

    def queries(self, p: E51.Pod, cls: str):
        if cls != "ii":
            return super().queries(p, cls)

        w = self.world
        i0 = self.base_pos[p.target]
        after = [self.base_keys[(i0 + 1 + j) % len(self.base_keys)] for j in range(E51.N_AFTER)]
        other_aliases = clean_other_pod_aliases(self.spec.groups, p.target, 2)
        keys = list(self.bystander_base) + other_aliases + after

        overlap = set(keys).intersection(p.keys)
        if overlap:
            raise AssertionError(f"class-(ii) bystander contamination for {p.target}: {sorted(overlap)}")

        objs = np.array([w.index[k] for k in keys])
        if self.reader == "syn":
            return [E51.E15._q1(w, k) for k in keys], objs
        return [(k, E51.GPT2_BYSTANDER_TEMPLATE) for k in keys], objs


def main():
    # Patch only the Setting factory used by the original runner. The original
    # experiment file and its historical record remain untouched.
    E51.Setting = CleanSetting
    return E51.main()


if __name__ == "__main__":
    main()
