"""Independent standard-library checker for QCR001 separation witnesses.

Checks exact row algebra and disjointness from a CLOSED rounding-box superset.
It does not authenticate a model trace or rule out every rank-r basis.
Checks remain active under python -O.
"""
from fractions import Fraction
import json
from pathlib import Path
import argparse


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(w: dict) -> None:
    r = w['rank']
    require(type(r) is int and r > 0, 'rank must be a positive integer')
    a = [[Fraction(x) for x in row] for row in w['anchor_rows']]
    b = list(map(Fraction, w['extra_row']))
    c = list(map(Fraction, w['coefficients']))
    lo = list(map(Fraction, w['anchor_low']))
    hi = list(map(Fraction, w['anchor_high']))
    require(all(len(x) == r for x in (a, b, c, lo, hi)), 'invalid vector dimensions')
    require(all(len(row) == r for row in a), 'invalid matrix dimensions')
    anchors = w['anchor_indices']
    require(len(anchors) == r and len(set(anchors)) == r, 'anchors must be distinct')
    require(w['extra_index'] not in anchors, 'extra row must differ from anchors')
    require(all(type(i) is int and i >= 0 for i in anchors + [w['extra_index']]), 'invalid row index')
    require(all(x <= y for x, y in zip(lo, hi)), 'reversed anchor interval')
    el, eh = Fraction(w['extra_low']), Fraction(w['extra_high'])
    require(el <= eh, 'reversed extra interval')
    for j in range(r):
        require(sum(c[i] * a[i][j] for i in range(r)) == b[j], 'row identity is not exact')
    lower = sum(min(t * x, t * y) for t, x, y in zip(c, lo, hi))
    upper = sum(max(t * x, t * y) for t, x, y in zip(c, lo, hi))
    require(lower == Fraction(w['span_low']), 'incorrect lower bound')
    require(upper == Fraction(w['span_high']), 'incorrect upper bound')
    require(upper < el or lower > eh, 'intervals are not strictly disjoint')
    fields = ('original_old', 'absolute_low', 'absolute_high')
    require(all(len(w[f]) == r + 1 for f in fields), 'invalid absolute interval dimensions')
    old, absolute_low, absolute_high = [
        [Fraction(float.fromhex(x)) for x in w[f]] for f in fields]
    require(lo + [el] == [x - y for x, y in zip(absolute_low, old)], 'inexact low-endpoint shift')
    require(hi + [eh] == [x - y for x, y in zip(absolute_high, old)], 'inexact high-endpoint shift')


def walk(x):
    if isinstance(x, dict):
        candidate = x.get('rounding_box_separation')
        if isinstance(candidate, dict) and candidate.get('status') == 'EXACT_SEPARATION':
            require(candidate.get('witness') is not None, 'claimed separation lacks a witness')
            yield candidate['witness']
        for value in x.values():
            yield from walk(value)
    elif isinstance(x, list):
        for value in x:
            yield from walk(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()
    total = 0
    for path in args.files:
        witnesses = list(walk(json.loads(path.read_text())))
        for w in witnesses:
            check(w)
        print(f'{path}: {len(witnesses)} exact witness checks')
        total += len(witnesses)
    print(f'Total exact witnesses: {total}; zero means no certificate was checked.')


if __name__ == '__main__':
    main()
