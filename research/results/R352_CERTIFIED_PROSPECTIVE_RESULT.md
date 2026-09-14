# R352 Result — Certified Prospective Residual Circuit (CPRC)

Status: **exact selective-binding mechanism passed, but its practical benefit is conditional; not DoD and not a novelty claim.**

## Construction

A ReLU network is partially evaluated with immutable query state fixed while mutable world inputs remain symbolic over the full box `[-1,1]^8`. Interval certificates classify every nonlinear gate:

- always positive -> collapse ReLU to identity;
- always negative -> collapse to zero;
- ambiguous -> retain an unresolved prospective ReLU atom.

The resulting residual program is immutable across world writes and is evaluated later against the current world.

## Result

Across five network regimes, 120 compiled queries/regime and 256 future worlds/query:

- total class mismatches vs full network: **0**;
- maximum absolute numeric error: **6.217e-15**;
- all compiled-program digests stayed byte-identical after 50,000 world writes/regime;
- best certified stable-ReLU fraction: **60.50%**;
- best mean residual/full multiply-count ratio: **0.6853**.

However, the result also falsifies an overly broad performance claim. When few nonlinear gates can be certified stable, the residual program is *larger* than the original network: at bias scale 0 its coefficient count was ~3.15x the full network multiply count. The direct Python residual evaluator was slower in every tested regime. Only when ~60% of gates became certifiably stable did arithmetic count fall below the full network.

Report SHA256: `c358a7d2c91dd6dd46b848d563f6b7405e2bff7743c5e2508951bdc40ccfdf41`.
Artifact ZIP SHA256: `97c02846852123015a61542b39144decd6068c254cd56e6b7174e01e6293f8f8`.

## Decision

Keep **certified selective materialization** as an optional compiler optimization, not as the core PNS representation.

The important architectural result is exactness: a materialization boundary need not be one global layer. It can be pushed through nonlinear regions that are provably insensitive in control-flow shape to the full future world domain, while retaining only genuinely world-sensitive gates.

But CPRC is not universally compact or fast. Production use should require a profitability gate based on certified stable fraction / residual circuit cost, and should fall back to a simpler PNS barrier when symbolic residual complexity grows.

Interval propagation, neural verification and partial evaluation are established prior art. R352 contributes evidence for how they can act as a CKCA/PNS binding-time compiler, not a component-level novelty claim.
