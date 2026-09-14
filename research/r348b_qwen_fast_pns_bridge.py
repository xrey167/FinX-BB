from __future__ import annotations

from research import r348_qwen_fast_pns_bridge as base

_base_make_rows = base.make_rows


def make_rows(templates, seed):
    rows = _base_make_rows(templates, seed)
    fixed = []
    for text, op, a, b, i in rows:
        # Make canonical operand roles explicit once at the front.  Templates may
        # legitimately mention an operand more than once or reverse surface order.
        prefix = f"Canonical operands: A=pod_{a:03d}; B=pod_{b:03d}. "
        fixed.append((prefix + text, op, a, b, i))
    return fixed


def parse_refs(text: str):
    matches = base.REF_RE.findall(text)
    if len(matches) < 2:
        raise ValueError(text)
    return int(matches[0]), int(matches[1])


base.make_rows = make_rows
base.parse_refs = parse_refs

if __name__ == "__main__":
    raise SystemExit(base.main())
