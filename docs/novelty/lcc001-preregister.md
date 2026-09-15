# LCC-001 — the boundary between causal audit and lineage certification

Date: 2026-09-05. Parent: `947206b72a75dc3616ca44dc5434a0a293f4bca7`.
Status at registration: NOT EXECUTED. No invention claim.

## Why this follows E-000086R

E-000086R rejects the explicit alias+Pod witness as a major-invention mechanism because an ordinary dependency validator reproduces it. Its remaining directions include neural causal-lineage discovery/certification. This screen asks whether two particularly tempting replacements are even sound: recovering historical source-generation provenance from an unlabelled numerical activation, and promoting finite intervention/local-Jacobian observations to an exact independence certificate over untested mutations or contexts.

This does not repeat E90's concrete normalization/softmax omissions, CRR001's fixed response-basis rank obstruction, or NIC001's finite interaction-order truncation. It concerns an information restriction on the observer. The elementary indistinguishability argument is not claimed as a new mathematical theorem.

## Distinguish two different objects

Historical provenance: which canonical object generation actually produced a cached state. Distinct objects may produce identical numerical values. A generation can be stale even when replacing it changes no floating-point value.

Functional support on a declared domain: whether interventions on a source can change a particular state for some allowed context. This is domain-dependent. A finite audit suite is not the same thing as that domain.

The screen must never infer that the two are interchangeable.

## Arm A: state-only historical provenance collision

Create two different canonical object-generation identities A and B with exactly the same payload and the same deterministic downstream activation. Revoke A, leave B live. Numerical observation, including any deterministic lens of that activation with a fixed downstream model, is identical. Correct lifecycle validity differs. Without external source identity/provenance, an activation-only decision rule cannot be correct on both histories.

Test 16 integer payloads at each of three synthetic identity seeds, plus controls with supplied provenance. The payload collision is intentional equality, not a cryptographic collision or probabilistic claim. Encoding source identity in the state or consulting an authoritative trace escapes this restriction; it is explicit provenance and receives no novelty credit.

## Arm B: finite-probe whole-domain completeness

For three seeds (0,1,2) and probe counts (8,32,128), sample distinct rational probe locations in [0,1], including endpoints. After the complete finite probe set is fixed, choose an open gap containing no probe. Inside the middle half of that gap put a compact triangular ReLU function B:

`B(x) = (relu(x-a) - 2*relu(x-b) + relu(x-c)) / (b-a)`, with `b=(a+c)/2`.

Compare the declared cached output `H0(x,m)=base(x)` and `H1(x,m)=base(x)+m*B(x)*v`, where m is a source-presence flag and v is a nonzero fixed vector. The downstream readout is identical in the two systems. At every audited x, B is identically zero on a neighbourhood: all output values and all local derivatives agree, including the present/revoked m=1/0 comparison. At the untested midpoint b, B=1 and source removal changes H1 by v while H0 is independent.

Use Python Fraction throughout the structural screen. Numerically verify output equality and mixed derivatives through total order four, then verify the untested counterexample. The neighbourhood proof, not the finite derivative checks, establishes agreement of arbitrary finite derivative order. Both models can use the same intermediate gate computations and differ only in the final gate-output connection; parameter/graph inspection is explicitly outside the finite-oracle information restriction.

This establishes a worst-case finite-oracle limitation for a hypothesis class admitting these ReLU gates. It is not a measured attack on the E81/E95 trained readers, not evidence that real-model audits necessarily miss dependencies, and not a probabilistic guarantee for a specified data distribution. A degree/complexity/margin restriction with a proved complete query scheme can escape the argument. Comparing the actual full fresh state for a particular mutation can correctly settle that particular comparison; it does not by itself provide the hoped-for cheap universal certificate.

## Arm C: positive restricted-domain proof control

For the known triangular response, implement an exact white-box interval certificate: on a closed context interval [l,u], source independence holds iff that interval has no intersection with the open nonzero support (a,c). Handle singleton intervals and support endpoints correctly. Validate this rule against an independently evaluated piecewise-linear extremum reference over a finite interval grid including all breakpoints.

This is ordinary program/interval reasoning, not the new invention. Its purpose is to demonstrate the valid escape: declare and prove a scope instead of relabelling successful probes as a universal certificate. It supplies no new runtime authorization mechanism.

## Fixed decisions

If A's observations coincide while validity differs, retire unrestricted numerical-state-only historical provenance recovery.

If B's entire registered finite transcript agrees while the out-of-probe intervention differs, retire finite-probe-only whole-domain completeness certification for the stated function class. Increasing probe count or substituting another deterministic lens cannot turn indistinguishable transcripts into an exact certificate; additional structural assumptions can.

If C's certificate differs from its exact reference, the implementation is invalid and must be corrected with the original error recorded. No narrowing of the interval test set after looking at results.

J-space/J-lens stays independent audit only. No runtime decision may depend on an audit score. No claim that a real Anthropic J-lens was executed by this synthetic screen.

## Unchanged full-system gates

Every held-out real-symlink template must remain >=0.95 in each exact trained-reader attack job. LCC001 has no language reader and therefore has NO capability, deletion-leakage, UNKNOWN, lifecycle propagation, or backbone pass. Synthetic seeds are not trained-reader seeds. No E83/E84/E85/E95 result may be promoted using this screen.

The target still requires >=10x mutation-to-ready over the strongest guarantee-matched baseline at scale, <=5% inference overhead, matched total memory, >=95% fresh/unseen-paraphrase reading and lifecycle propagation, <=2% deleted-object leakage, >=90% UNKNOWN in the declared missing-key scope, exact bypass or <=0.05 nats generic divergence, >=3 trained-reader seeds and preferably >1 backbone. No relaxed screening threshold replaces these gates.

## Prior-art/evidence boundary

The causal-abstraction literature provides a formal framework and intervention-based empirical testing; this screen does not claim to invent either. The original J-lens paper concerns verbalizable representations and remains an audit reference, not a universal provenance certificate. Known formal verification and dependency tracking receive zero novelty credit.

Primary sources inspected before registration:
- https://arxiv.org/html/2106.02997v2
- https://transformer-circuits.pub/2026/workspace/
- https://arxiv.org/html/2305.19521v2

Local CPU execution and any subsequent Actions replication must be distinguished. A submitted workflow is not completed evidence. Record source hashes and all checks in a result JSON; do not count a new countermodel as a useful technical novelty.
