"""NOV-004 — audit every recorded experiment for the two defect classes found so far.

NOV-001 found the vacuity pattern in E-000105 and NOV-002 found it again in E-000104. NOV-003 found
a second, unrelated class in E-000103: a baseline rebuilding loop-invariant work, which is the
mirror of the representation subsidy — instead of giving the baseline the candidate's representation
for free, it denies the baseline an optimisation the candidate is already using.

Three experiments were read by hand and two defect classes came out of three readings. That rate
makes hand-reading the remaining twenty a poor use of the next day, and a poor instrument besides:
whoever reads them will find what they are looking for. So this file looks mechanically, at all of
them, for exactly the two patterns already demonstrated to occur.

  **Class A — a comparison whose two sides share a source.** Two different names in one scope
  assigned syntactically identical call expressions. `candidate_work = mutation_multiplies(net)` /
  `generic_work = mutation_multiplies(net)`. Whatever predicate then compares those names cannot
  fail. Ledger §31.15.

  **Class B — loop-invariant work inside a loop.** A call assigned inside a `for` whose arguments
  do not mention any name the loop binds or rebinds, so it recomputes the same value on every
  iteration. E-000103's `woodbury_rankk_solution` rebuilding `inv_u` and `middle_inv` across 24
  sessions. Whether that is a defect depends on which arm carries it: on a *baseline* it inflates
  the candidate's margin for free.

**What this is and is not.** Class A is decidable from the syntax and is reported as a finding.
Class B is a *heuristic* — a call can be loop-invariant by this test and still be doing necessary
work, and a genuinely invariant call in the candidate arm is not a defect at all — so it is reported
as a candidate for review, never as a verdict. Confusing the two would be the same error as promoting
on an unresolvable coordinate (§31.46).

**Validity floor.** The scanner must rediscover the two Class A sites already established by hand,
at `e000104:224-225` and `e000105:176-177`. If it does not, it is not detecting the thing it was
written to detect and its silence elsewhere means nothing — the same discipline E-000019 applies to
its probes and PDX-001 applies to its attack. `run()` reports `validity_floor_met` and `main()`
exits non-zero when it fails.

Run:  python -m so.experiments.nov004_instrument_audit
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

# Calls too cheap or too generic to be worth flagging as loop-invariant work.
_IGNORED_CALLS = {
    "len", "range", "int", "float", "str", "list", "tuple", "dict", "set", "sorted", "sum",
    "min", "max", "abs", "enumerate", "zip", "print", "F", "Fraction", "append", "super",
}

# The two sites established by hand in NOV-001 and NOV-002. The scanner must find both.
_KNOWN_CLASS_A = {
    ("e000104_tensor_train_lifecycle_quotient_reduction.py", "local_update_multiplies"),
    ("e000105_stable_gate_cohort_transport_reduction.py", "mutation_multiplies"),
}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _root_name(node: ast.AST) -> str | None:
    """The base name of an assignment target: `aff` for `aff[idx]`, `o` for `o.field`."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _bound_names(loop: ast.For) -> set[str]:
    """Every name the loop moves — rebound *or mutated in place*.

    Rebinding is not the only way a value changes between iterations. `aff[idx] = new` mutates
    through a subscript and rebinds nothing, and `tree.update(...)` mutates through a method. The
    first version of this scanner counted only `ast.Name` stores and therefore reported
    `compose_all(aff, ...)` in E-000097 as loop-invariant when `aff` is written on the line above it.
    Missing in-place mutation is the same error the Class A window makes if it ignores intervening
    writes, and it produced false positives for the same reason.
    """
    names: set[str] = set()
    for n in ast.walk(loop.target):
        if isinstance(n, ast.Name):
            names.add(n.id)
    for stmt in loop.body:
        for n in ast.walk(stmt):
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    root = _root_name(t)
                    if root:
                        names.add(root)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                # a method call may mutate its receiver
                root = _root_name(n.func.value)
                if root:
                    names.add(root)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                names.add(n.id)
            elif isinstance(n, ast.For):
                for t in ast.walk(n.target):
                    if isinstance(t, ast.Name):
                        names.add(t.id)
    return names


def _free_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _rebound_names(stmt: ast.stmt) -> set[str]:
    return {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _compared_together(scope: ast.AST, a: str, b: str) -> bool:
    """Do these two names ever meet in a comparison inside this scope?

    A duplicated call is only a *defect* when something compares its two results: allocating two
    zero vectors, or calling a random generator twice, duplicates the expression without duplicating
    the value. What made e000104 and e000105 vacuous is that a kill predicate read the difference.
    """
    for node in ast.walk(scope):
        if not isinstance(node, ast.Compare):
            continue
        names = _free_names(node)
        if a in names and b in names:
            return True
    return False


# Calls whose two invocations legitimately differ, so duplication says nothing.
_NONDETERMINISTIC = {"randn", "rand", "random", "normal", "integers", "randint", "sample", "time", "randrange",
                     "Event", "Lock", "Thread", "uuid4", "shuffle", "choice"}


def scan_class_a(tree: ast.AST) -> list[dict]:
    """Two names bound to the same call, close enough that nothing can have changed between them.

    Syntactic identity alone is not enough and the first version of this scanner proved it: it
    flagged 87 sites, most of them legitimate. `bank_from_store(store)` before and after a deletion
    is the same expression and a different value, because `store` was mutated in between — that is
    the whole point of the comparison, not a defect in it.

    What makes `candidate_work = mutation_multiplies(net)` / `generic_work = mutation_multiplies(net)`
    vacuous is that the two sit in the same block with nothing between them that could move the
    argument. So: same block, identical call, and every statement in between is an assignment that
    rebinds none of the call's free names.
    """
    findings: list[dict] = []
    for scope in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(scope, field, None)
            if not isinstance(block, list):
                continue
            for i, first in enumerate(block):
                if not isinstance(first, ast.Assign) or len(first.targets) != 1:
                    continue
                t1, v1 = first.targets[0], first.value
                if not isinstance(t1, ast.Name) or not isinstance(v1, ast.Call):
                    continue
                callee = _call_name(v1)
                if callee is None or callee in _IGNORED_CALLS or callee in _NONDETERMINISTIC:
                    continue
                dump, free = ast.dump(v1), _free_names(v1)
                for second in block[i + 1:]:
                    if isinstance(second, ast.Assign) and len(second.targets) == 1:
                        t2, v2 = second.targets[0], second.value
                        if (isinstance(t2, ast.Name) and isinstance(v2, ast.Call)
                                and ast.dump(v2) == dump and t2.id != t1.id):
                            findings.append({
                                "callee": callee,
                                "assigned_to": sorted({t1.id, t2.id}),
                                "lines": [first.lineno, second.lineno],
                                "scope": getattr(scope, "name", type(scope).__name__),
                                "results_are_compared": _compared_together(scope, t1.id, t2.id),
                            })
                            break
                    # anything that could move an argument ends the window
                    if _rebound_names(second) & free:
                        break
                    if not isinstance(second, ast.Assign):
                        break
    return findings


def scan_class_b(tree: ast.AST) -> list[dict]:
    """A call assigned inside a loop whose arguments mention nothing the loop binds."""
    findings: list[dict] = []
    for loop in ast.walk(tree):
        if not isinstance(loop, ast.For):
            continue
        bound = _bound_names(loop)
        for stmt in ast.walk(loop):
            if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
                continue
            call = stmt.value
            name = _call_name(call)
            if name is None or name in _IGNORED_CALLS or name in _NONDETERMINISTIC:
                continue  # a fresh draw every iteration is the point, not a defect
            args_free: set[str] = set()
            for a in list(call.args) + [k.value for k in call.keywords]:
                args_free |= _free_names(a)
            if not args_free:
                continue  # no arguments to be invariant in
            if args_free & bound:
                continue  # depends on something the loop moves
            targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            if not targets:
                continue
            findings.append({
                "callee": name,
                "assigned_to": targets,
                "line": stmt.lineno,
                "loop_line": loop.lineno,
                "invariant_in": sorted(args_free),
            })
    return findings


def audit_file(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        "file": path.name,
        "class_a_shared_source": scan_class_a(tree),
        "class_b_loop_invariant_candidates": scan_class_b(tree),
    }


def run(experiments_dir: Path | None = None) -> dict:
    d = experiments_dir or Path("so/experiments")
    # NOT glob("e0000*.py"): that pattern silently excludes E-000100 onward, which is where both
    # known Class A sites live. The validity floor caught it, which is what a validity floor is for.
    pattern = re.compile(r"^e\d{6}[a-z]?_.*\.py$")
    files = sorted(p for p in d.glob("*.py") if pattern.match(p.name))
    reports = [audit_file(p) for p in files]

    a_sites = [
        {"file": r["file"], **f}
        for r in reports for f in r["class_a_shared_source"]
    ]
    b_sites = [
        {"file": r["file"], **f}
        for r in reports for f in r["class_b_loop_invariant_candidates"]
    ]

    found_known = {(s["file"], s["callee"]) for s in a_sites}
    floor_met = _KNOWN_CLASS_A <= found_known

    confirmed = [s for s in a_sites if s["results_are_compared"]]
    incidental = [s for s in a_sites if not s["results_are_compared"]]
    files_with_a = sorted({s["file"] for s in confirmed})
    files_with_b = sorted({s["file"] for s in b_sites})

    return {
        "experiment": "NOV-004",
        "scope": "every recorded experiment, scanned for the two defect classes already demonstrated",
        "files_scanned": len(files),
        "validity_floor_met": floor_met,
        "known_sites_required": sorted(f"{f}:{c}" for f, c in _KNOWN_CLASS_A),
        "known_sites_found": sorted(
            f"{f}:{c}" for f, c in found_known & _KNOWN_CLASS_A
        ),
        "class_a_confirmed": confirmed,
        "class_a_confirmed_count": len(confirmed),
        "class_a_files": files_with_a,
        "class_a_incidental_duplicates": incidental,
        "class_a_incidental_count": len(incidental),
        "class_b_candidates": b_sites,
        "class_b_files": files_with_b,
        "class_b_count": len(b_sites),
        "decision": (
            "SCANNER_VALID"
            if floor_met
            else "SCANNER_INVALID_KNOWN_SITES_MISSED"
        ),
        "how_to_read_this": (
            "Class A is decidable from the syntax: two names in one scope bound to identical call "
            "expressions, so any predicate comparing them cannot fail. Each is a finding. Class B is "
            "a heuristic and each entry is a candidate for review, never a verdict — a call can be "
            "loop-invariant and still necessary, and an invariant call in the *candidate* arm is not "
            "a defect at all. It becomes one only when it sits on the baseline, where it inflates the "
            "candidate's margin for free, which is what NOV-003 found in E-000103. Reading Class B as "
            "a list of defects would be §31.46's error repeated: scoring something the instrument "
            "cannot resolve."
        ),
        "not_claimed": (
            "Two patterns, chosen because each was demonstrated by hand first. This is not a general "
            "audit of these experiments and finds nothing about arms that are simply mismatched, "
            "domains that hide a crossover (§31.46), or cost coordinates nobody counted. A file "
            "absent from both lists has not been cleared; it has been checked for two things."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("so/results/nov004"))
    parser.add_argument("--experiments-dir", type=Path, default=Path("so/experiments"))
    args = parser.parse_args()
    result = run(args.experiments_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "nov004_instrument_audit.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if not result["validity_floor_met"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
