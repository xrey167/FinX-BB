from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar

T = TypeVar("T")


class Conflict(RuntimeError):
    """The world changed after a generation-scoped read."""


class Revoked(RuntimeError):
    """The requested canonical Pod has no live current generation."""


class LifetimeViolation(RuntimeError):
    """An expired or unsealed mutable-derived artifact was used."""


@dataclass
class Pod:
    generation: int
    value: int
    live: bool = True


@dataclass(frozen=True)
class Capability:
    pod_id: int
    generation: int
    value: int


@dataclass(frozen=True)
class Plan:
    """Reusable B-plane plan.

    A Plan may contain canonical addresses and operation semantics, but no current
    mutable values or value generations.
    """

    opcode: str
    left_pod: int
    right_pod: int
    schema_generation: int
    alias_generation: int


@dataclass(frozen=True)
class Borrowed(Generic[T]):
    value: T
    generations: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class Sealed(Generic[T]):
    value: T
    factor_id: int
    generations: frozenset[tuple[int, int]]


class AuthorityStore:
    def __init__(self) -> None:
        self.pods: dict[int, Pod] = {}
        self.aliases: dict[str, int] = {}
        self.alias_generation = 1
        self.schema_generation = 1

    def create(self, pod_id: int, value: int) -> None:
        if pod_id in self.pods:
            raise ValueError(f"duplicate pod {pod_id}")
        self.pods[pod_id] = Pod(1, value, True)

    def bind(self, alias: str, pod_id: int) -> None:
        if pod_id not in self.pods:
            raise KeyError(pod_id)
        key = alias.casefold()
        prior = self.aliases.get(key)
        if prior != pod_id:
            self.aliases[key] = pod_id
            self.alias_generation += 1

    def resolve(self, alias: str) -> int:
        return self.aliases[alias.casefold()]

    def acquire(self, pod_id: int) -> Capability:
        p = self.pods[pod_id]
        if not p.live:
            raise Revoked(pod_id)
        return Capability(pod_id, p.generation, p.value)

    def verify_generation(self, dep: tuple[int, int]) -> bool:
        pid, generation = dep
        p = self.pods.get(pid)
        return bool(p is not None and p.live and p.generation == generation)

    def update(self, pod_id: int, value: int) -> tuple[int, int]:
        p = self.pods[pod_id]
        old = (pod_id, p.generation)
        p.generation += 1
        p.value = value
        p.live = True
        return old

    def revoke(self, pod_id: int) -> tuple[int, int]:
        p = self.pods[pod_id]
        old = (pod_id, p.generation)
        p.generation += 1
        p.live = False
        return old


class LifetimeFabric:
    """Minimal CLFD reference implementation.

    Generation leaves are monotonic: once false they can never be revived. A new
    generation creates a different leaf even when payload bytes are identical.
    """

    def __init__(self) -> None:
        self.live: list[bool] = []
        self.children: dict[int, list[int]] = defaultdict(list)
        self.leaf_by_generation: dict[tuple[int, int], int] = {}
        self.factor_by_parents: dict[tuple[int, ...], int] = {}

    def _node(self, live: bool = True) -> int:
        i = len(self.live)
        self.live.append(live)
        return i

    def leaf(self, generation: tuple[int, int]) -> int:
        if generation not in self.leaf_by_generation:
            self.leaf_by_generation[generation] = self._node(True)
        return self.leaf_by_generation[generation]

    def factor(self, generations: Iterable[tuple[int, int]]) -> int:
        parents = tuple(sorted({self.leaf(x) for x in generations}))
        if parents in self.factor_by_parents:
            return self.factor_by_parents[parents]
        f = self._node(all(self.live[p] for p in parents))
        self.factor_by_parents[parents] = f
        for p in parents:
            self.children[p].append(f)
        return f

    def invalidate(self, generation: tuple[int, int]) -> int:
        leaf = self.leaf_by_generation.get(generation)
        if leaf is None or not self.live[leaf]:
            return 0
        q: deque[int] = deque([leaf])
        touched = 0
        while q:
            n = q.popleft()
            if not self.live[n]:
                continue
            self.live[n] = False
            touched += 1
            for c in self.children.get(n, ()):
                if self.live[c]:
                    q.append(c)
        return touched

    def valid(self, factor_id: int) -> bool:
        return self.live[factor_id]


class PlanCache:
    def __init__(self) -> None:
        self._cache: dict[tuple, Plan] = {}
        self.compiles = 0

    def get_or_compile(self, opcode: str, left_pod: int, right_pod: int, authority: AuthorityStore) -> Plan:
        key = (opcode, left_pod, right_pod, authority.schema_generation, authority.alias_generation)
        p = self._cache.get(key)
        if p is None:
            p = Plan(opcode, left_pod, right_pod, authority.schema_generation, authority.alias_generation)
            self._cache[key] = p
            self.compiles += 1
        return p

    @property
    def size(self) -> int:
        return len(self._cache)


class WorldABI:
    """Reference ABI between reusable B-plan state and mutable world state."""

    def __init__(self, authority: AuthorityStore, lifetimes: LifetimeFabric) -> None:
        self.authority = authority
        self.lifetimes = lifetimes

    def transaction(self, plan: Plan) -> "KnowledgeTransaction":
        if plan.schema_generation != self.authority.schema_generation:
            raise Conflict("schema generation changed")
        if plan.alias_generation != self.authority.alias_generation:
            raise Conflict("alias binding generation changed")
        return KnowledgeTransaction(self, plan)

    def expire_generation(self, dep: tuple[int, int]) -> int:
        return self.lifetimes.invalidate(dep)

    def serve(self, sealed: Sealed[T]) -> T:
        if not self.lifetimes.valid(sealed.factor_id):
            raise LifetimeViolation("derived neural artifact lifetime expired")
        # Defense in depth: factor is the fast path; direct authority verification
        # is the reference oracle used by tests/audits.
        if not all(self.authority.verify_generation(d) for d in sealed.generations):
            raise LifetimeViolation("factor/authority mismatch")
        return sealed.value


class KnowledgeTransaction:
    def __init__(self, abi: WorldABI, plan: Plan) -> None:
        self.abi = abi
        self.plan = plan
        self.readset: dict[int, Capability] = {}
        self.committed = False

    def read(self, pod_id: int) -> Borrowed[int]:
        cap = self.abi.authority.acquire(pod_id)
        self.readset[pod_id] = cap
        return Borrowed(cap.value, frozenset([(cap.pod_id, cap.generation)]))

    @staticmethod
    def _union(*values: Borrowed[int]) -> frozenset[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for v in values:
            out.update(v.generations)
        return frozenset(out)

    def execute(self) -> Borrowed[int]:
        # Binding-equivariant execution: after address reads, the operator receives
        # only values, never canonical identities.
        a = self.read(self.plan.left_pod)
        b = self.read(self.plan.right_pod)
        op = self.plan.opcode
        if op == "add8":
            out = (a.value + b.value) & 0xFF
        elif op == "xor8":
            out = (a.value ^ b.value) & 0xFF
        elif op == "max":
            out = max(a.value, b.value)
        elif op == "select_even":
            out = a.value if (a.value & 1) == 0 else b.value
        else:
            raise ValueError(op)
        return Borrowed(out, self._union(a, b))

    def commit(self) -> None:
        deps = [(cap.pod_id, cap.generation) for cap in self.readset.values()]
        if not all(self.abi.authority.verify_generation(dep) for dep in deps):
            raise Conflict("Neural Commit Barrier rejected stale generation read-set")
        self.committed = True

    def seal(self, value: Borrowed[T]) -> Sealed[T]:
        if not self.committed:
            raise LifetimeViolation("cannot seal before successful commit")
        actual = frozenset((c.pod_id, c.generation) for c in self.readset.values())
        if value.generations != actual:
            raise LifetimeViolation("borrowed value lifetime does not match committed read-set")
        f = self.abi.lifetimes.factor(actual)
        return Sealed(value.value, f, actual)
