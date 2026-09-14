from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

STAGE = "R325-EPISTEMIC-PLAN-FIREWALL"
VALID_PLANS = int(os.environ.get("SO_R325_VALID", "250000"))
MUTANTS = int(os.environ.get("SO_R325_MUTANTS", "250000"))
REPORT_PATH = Path(os.environ.get("SO_R325_REPORT", "r325_report.json"))
SEED = 3250914

SCHEMA_FIELDS = {"capital", "price", "risk", "status", "quantity"}
SCHEMA_OPS = {"READ", "ADD", "SUB", "MAX", "MIN", "EQ", "SELECT", "RETURN"}
SCHEMA_CONSTS = {"NULL", "UNKNOWN", "TRUE", "FALSE"}


class PlanRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolveSpan:
    dst: str
    start: int
    end: int


@dataclass(frozen=True)
class UserSpan:
    dst: str
    start: int
    end: int


@dataclass(frozen=True)
class SchemaConst:
    dst: str
    name: str


@dataclass(frozen=True)
class DirectPodRef:
    dst: str
    pod_id: int


@dataclass(frozen=True)
class ModelConst:
    dst: str
    value: str


@dataclass(frozen=True)
class ReadField:
    dst: str
    pod_reg: str
    field: str


@dataclass(frozen=True)
class Op:
    dst: str
    opcode: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class Return:
    src: str


Instruction = ResolveSpan | UserSpan | SchemaConst | DirectPodRef | ModelConst | ReadField | Op | Return


@dataclass
class Plan:
    user_text: str
    instructions: list[Instruction]


class TrustedLinker:
    """Only this non-neural component can turn a source span into a PodRef.

    The compiler never gets to invent a canonical address. This is essential:
    otherwise a model with parametric factual memory could smuggle a stale answer
    through a different identity/address even when raw factual literals are banned.
    """

    def __init__(self) -> None:
        self.aliases = {
            "france": 11,
            "germany": 12,
            "supplier alpha": 21,
            "supplier beta": 22,
            "item a": 31,
            "item b": 32,
        }

    def resolve_source_span(self, text: str, start: int, end: int) -> int:
        if start < 0 or end <= start or end > len(text):
            raise PlanRejected("invalid source span")
        raw = text[start:end]
        key = raw.casefold().strip()
        if key not in self.aliases:
            raise PlanRejected(f"untrusted/unresolved source span {raw!r}")
        return self.aliases[key]


class EpistemicPlanFirewall:
    """A pre-world non-interference verifier for LLM-produced CKVM plans.

    The compiler may choose control structure, schema opcodes and source spans.
    It may not introduce new factual literals, direct Pod addresses, or hidden
    entity choices. Every data literal crossing the airlock must be provably
    sourced from user bytes or trusted schema bytes.
    """

    def __init__(self, linker: TrustedLinker) -> None:
        self.linker = linker

    def verify_and_link(self, plan: Plan) -> list[tuple]:
        regs: dict[str, tuple[str, object]] = {}
        linked: list[tuple] = []
        returned = False
        for ins in plan.instructions:
            if returned:
                raise PlanRejected("instructions after RETURN")
            if isinstance(ins, ResolveSpan):
                if ins.dst in regs:
                    raise PlanRejected("SSA destination reused")
                pod = self.linker.resolve_source_span(plan.user_text, ins.start, ins.end)
                regs[ins.dst] = ("pod", pod)
                linked.append(("POD", ins.dst, pod))
            elif isinstance(ins, UserSpan):
                if ins.dst in regs:
                    raise PlanRejected("SSA destination reused")
                if ins.start < 0 or ins.end <= ins.start or ins.end > len(plan.user_text):
                    raise PlanRejected("invalid user literal span")
                literal = plan.user_text[ins.start:ins.end]
                regs[ins.dst] = ("user_literal", literal)
                linked.append(("USER_LITERAL", ins.dst, literal))
            elif isinstance(ins, SchemaConst):
                if ins.dst in regs:
                    raise PlanRejected("SSA destination reused")
                if ins.name not in SCHEMA_CONSTS:
                    raise PlanRejected("unknown schema constant")
                regs[ins.dst] = ("schema_literal", ins.name)
                linked.append(("SCHEMA_LITERAL", ins.dst, ins.name))
            elif isinstance(ins, DirectPodRef):
                raise PlanRejected("model-generated direct PodRef forbidden")
            elif isinstance(ins, ModelConst):
                raise PlanRejected("model-origin factual literal forbidden")
            elif isinstance(ins, ReadField):
                if ins.dst in regs:
                    raise PlanRejected("SSA destination reused")
                if ins.field not in SCHEMA_FIELDS:
                    raise PlanRejected("unknown field")
                if ins.pod_reg not in regs or regs[ins.pod_reg][0] != "pod":
                    raise PlanRejected("READ requires linked source-span Pod")
                regs[ins.dst] = ("borrowed_world", None)
                linked.append(("READ", ins.dst, ins.pod_reg, ins.field))
            elif isinstance(ins, Op):
                if ins.dst in regs:
                    raise PlanRejected("SSA destination reused")
                if ins.opcode not in SCHEMA_OPS - {"READ", "RETURN"}:
                    raise PlanRejected("unknown/disallowed opcode")
                if not ins.args or any(a not in regs for a in ins.args):
                    raise PlanRejected("operator references undefined register")
                # The result provenance is world-borrowed if any input is world-borrowed;
                # otherwise it remains user/schema sourced. No model-latent source exists.
                kinds = {regs[a][0] for a in ins.args}
                kind = "borrowed_world" if "borrowed_world" in kinds else "derived_public_input"
                regs[ins.dst] = (kind, None)
                linked.append((ins.opcode, ins.dst, *ins.args))
            elif isinstance(ins, Return):
                if ins.src not in regs:
                    raise PlanRejected("RETURN undefined register")
                linked.append(("RETURN", ins.src))
                returned = True
            else:
                raise PlanRejected("unknown instruction class")
        if not returned:
            raise PlanRejected("missing RETURN")
        return linked


def span(text: str, needle: str) -> tuple[int, int]:
    s = text.casefold().index(needle.casefold())
    return s, s + len(needle)


def make_valid(rng: random.Random, i: int) -> Plan:
    examples = [
        "What is the capital of France?",
        "Compare the price of item a and item b.",
        "Is supplier alpha risk equal to supplier beta risk?",
        "Return supplier alpha status.",
        "What is the quantity of item a?",
    ]
    text = examples[i % len(examples)]
    inst: list[Instruction] = []
    if "France" in text:
        s, e = span(text, "France")
        inst += [ResolveSpan("p0", s, e), ReadField("v0", "p0", "capital"), Return("v0")]
    elif "item a" in text and "item b" in text:
        sa, ea = span(text, "item a"); sb, eb = span(text, "item b")
        inst += [ResolveSpan("p0", sa, ea), ResolveSpan("p1", sb, eb),
                 ReadField("v0", "p0", "price"), ReadField("v1", "p1", "price"),
                 Op("v2", "SUB", ("v0", "v1")), Return("v2")]
    elif "supplier alpha" in text and "supplier beta" in text:
        sa, ea = span(text, "supplier alpha"); sb, eb = span(text, "supplier beta")
        inst += [ResolveSpan("p0", sa, ea), ResolveSpan("p1", sb, eb),
                 ReadField("v0", "p0", "risk"), ReadField("v1", "p1", "risk"),
                 Op("v2", "EQ", ("v0", "v1")), Return("v2")]
    elif "supplier alpha" in text:
        s, e = span(text, "supplier alpha")
        inst += [ResolveSpan("p0", s, e), ReadField("v0", "p0", "status"), Return("v0")]
    else:
        s, e = span(text, "item a")
        inst += [ResolveSpan("p0", s, e), ReadField("v0", "p0", "quantity"), Return("v0")]
    return Plan(text, inst)


def mutate(p: Plan, rng: random.Random) -> tuple[Plan, str]:
    q = Plan(p.user_text, list(p.instructions))
    mode = rng.randrange(9)
    if mode == 0:
        q.instructions.insert(-1, ModelConst("leak", "Paris"))
        return q, "model_fact_literal"
    if mode == 1:
        q.instructions.insert(0, DirectPodRef("forged", rng.randrange(1, 1000)))
        return q, "direct_pod_forgery"
    if mode == 2:
        # Resolve an answer-like string not present in trusted source spans.
        q.instructions.insert(0, ResolveSpan("bad", 0, min(3, len(q.user_text))))
        return q, "invented_identity_span"
    if mode == 3:
        q.instructions.insert(-1, SchemaConst("badconst", "PARIS"))
        return q, "untrusted_schema_constant"
    if mode == 4:
        # Smuggle a model literal directly into a branch/select argument.
        q.instructions.insert(-1, ModelConst("bias", "42"))
        return q, "model_control_literal"
    if mode == 5:
        # Replace a valid Pod register with a direct forged register.
        q.instructions.insert(0, DirectPodRef("evil", 11))
        for j, ins in enumerate(q.instructions):
            if isinstance(ins, ReadField):
                q.instructions[j] = ReadField(ins.dst, "evil", ins.field)
                break
        return q, "direct_address_substitution"
    if mode == 6:
        q.instructions.insert(-1, Op("x_bad", "RECOVER_FROM_MODEL_MEMORY", tuple()))
        return q, "unknown_memory_opcode"
    if mode == 7:
        # Literal source span outside user bytes.
        q.instructions.insert(-1, UserSpan("badspan", -1, 5))
        return q, "invalid_user_provenance"
    # Duplicate an SSA destination to overwrite trusted provenance.
    dst = next(ins.dst for ins in q.instructions if hasattr(ins, "dst"))
    q.instructions.insert(-1, ModelConst(dst, "smuggle"))
    return q, "ssa_provenance_overwrite"


def main() -> None:
    rng = random.Random(SEED)
    fw = EpistemicPlanFirewall(TrustedLinker())
    valid_rejects = 0
    mutant_accepts = 0
    kinds: dict[str, int] = {}
    linked_pod_errors = 0
    t0 = time.perf_counter_ns()

    for i in range(VALID_PLANS):
        p = make_valid(rng, i)
        try:
            linked = fw.verify_and_link(p)
            # Linked Pod addresses must be produced only from trusted source spans.
            for ins in linked:
                if ins[0] == "POD" and ins[2] not in fw.linker.aliases.values():
                    linked_pod_errors += 1
        except PlanRejected:
            valid_rejects += 1

    for i in range(MUTANTS):
        p = make_valid(rng, i)
        m, kind = mutate(p, rng)
        kinds[kind] = kinds.get(kind, 0) + 1
        try:
            fw.verify_and_link(m)
            mutant_accepts += 1
        except PlanRejected:
            pass

    elapsed_ns = time.perf_counter_ns() - t0
    report = {
        "stage": STAGE,
        "architecture_candidate": "Epistemic Plan Firewall (EPF) + Trusted Span Linker",
        "valid_plans": VALID_PLANS,
        "adversarial_mutant_plans": MUTANTS,
        "valid_plan_false_rejects": valid_rejects,
        "adversarial_plan_false_accepts": mutant_accepts,
        "trusted_linker_address_errors": linked_pod_errors,
        "mutant_type_counts": kinds,
        "mean_verify_link_ns_per_plan_python": elapsed_ns / (VALID_PLANS + MUTANTS),
        "contract_pass": valid_rejects == 0 and mutant_accepts == 0 and linked_pod_errors == 0,
        "mechanism": (
            "The LLM compiler is not permitted to emit factual literals or direct canonical addresses. It can emit only trusted source-span "
            "references, schema opcodes/fields, and user/schema-sourced constants. A non-neural linker maps exact user spans to PodRefs, and "
            "the verifier rejects any plan with model-origin data, direct Pod forgery, unknown memory opcodes, or provenance overwrite."
        ),
        "security_target": (
            "A stale parametric fact must not cross the Semantic Airlock indirectly by being encoded in a compiler-produced constant, "
            "identity choice, direct PodRef, branch constant, or hidden memory opcode."
        ),
        "dod_status": "NOT_DOD; compiler non-interference type gate",
        "claim_boundary": (
            "Capability-safe linking, information-flow typing, source-span grounding, SSA and allowlisted IRs are established techniques. "
            "R325 applies them as a CKCA epistemic membrane; standalone novelty is not claimed."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("epistemic plan firewall failed")


if __name__ == "__main__":
    main()
