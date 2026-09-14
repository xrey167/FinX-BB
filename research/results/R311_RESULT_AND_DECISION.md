# R311 Result — Phase-Separated Lifetime Domains

Status: **lifetime-factorization gate passed; not DoD and not a standalone novelty claim.**

## Scale

- 50,000 Pods
- 100,000 aliases
- 250,000 plan-cache entries
- 20,000 fact-value updates
- 2,000 alias rebindings
- 4 schema-generation changes

## Exactness

- final plan-validity audit mismatches: **0**
- final J-state-validity audit mismatches: **0**

## The important result

A naïve cache that binds the interpreted query plan to current fact generations invalidated **83,083 plans** during ordinary value updates.

CKVM's phase-separated lifetime design invalidated **0 plans** for those same value updates.

- plan reuse saved fraction versus value-bound naïve design: **100%**
- mean J-state invalidations per value update: **4.15415**
- mean plan invalidations per alias rebind: **2.478**
- mean J-state invalidations per alias rebind: **1.641**
- schema-domain plan invalidations: **995,044** across the four deliberate schema epoch changes
- runtime: **8.177 s**

Report SHA256: `fefdbacd258ea14be0de31e352e7a4dbf5b92716d3e0a4ea19a7f61721ec2934`.
Artifact ZIP SHA256: `a5f79c3acfdb2d3a10aedf481fe18712ad881e9d6488ca914254e5b747d89e74`.

## Architectural decision

CKVM state must carry **typed lifetime domains**, not one undifferentiated freshness token:

- **plan lifetime** = alias-binding generation ∧ schema generation;
- **J-state lifetime** = plan lifetime ∧ exact current value generations actually read;
- **decode-suffix lifetime** = J-state lifetime ∧ any later mutable generations read during output generation.

This prevents ordinary fact writes from destroying reusable language/operation interpretation state while still invalidating value-dependent neural execution exactly.

Prepared-plan dependency management and versioned cache domains are established systems ideas. R311 demonstrates that the same separation is essential for CKCA's low-overhead neural lifecycle contract.
