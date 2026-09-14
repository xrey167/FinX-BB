# R327 Result — Causal Predicate Generations (CPG)

Status: **relational-lifetime mechanism gate passed; not DoD and not a standalone novelty claim.**

## Problem

A graph-derived neural result can become stale even when none of the value generations it previously read changes.

Example:

```text
max risk among suppliers(company X)
```

If a new supplier edge is inserted after the result was cached, a member-only read-set does not contain that new supplier. The cached artifact can therefore appear valid while the relation set it reasoned over has changed.

R327 adds a generation to the relation/predicate set itself.

## Result

- companies: **2,500**
- suppliers: **12,000**
- initial edges/company: **12**
- cached graph queries: **80,000**
- mutations: **30,000**
  - relation insert/delete: **19,453**
  - supplier-risk rewrites: **10,547**
  - same-value risk generation rewrites: **4,751**
- cache admission probes: **240,000**

Member-value generations only:

- semantic stale false serves: **2,036**
- lifetime false accepts: **16,961**

Member generations + predicate/set generation:

- semantic stale false serves: **0**
- lifetime false accepts: **0**

Reference Python costs:

- median mutation: **1,572 ns**
- median combined naive+predicate validity check: **2,386 ns**

Report SHA256: `550853057560bb0b73f34b97aef10167f9925ea4174b1dbe26b160a826edaa6a`.
Artifact ZIP SHA256: `abc5371ae8a4db4689b363c1dd3c4cd837bf26e8495742e0f32f4c960f4b40f7`.

## Architectural decision

Promote **Causal Predicate Generations** for CKVM relation/graph operations:

> A traversal depends both on the generations of values it actually enumerated and on the generation of the relation/predicate set that defined which members could be enumerated.

Edge insert/delete bumps the predicate generation. This gives graph-derived J/KV/semantic artifacts a lifetime over absence/presence structure, not only over values.

Predicate/range versioning and phantom-read protection are established database ideas. The CKCA research target is their use as neural/semantic artifact lifetime dependencies, not standalone novelty.
