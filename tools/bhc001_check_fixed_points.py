"""Independent scalar fixed-point checker; standard library only.
This checks finite-grid arithmetic, not model provenance or neural-library tanh.
"""
from fractions import Fraction
import argparse
import json
from pathlib import Path

BITS={'bfloat16':7,'float16':10,'float32':23,'float64':52}


def closest(x, bits):
    spacing=Fraction(1,2**bits)
    location=(x-1)/spacing
    q=location.numerator//location.denominator
    lower=Fraction(1)+q*spacing
    upper=lower+spacing
    dlow,dup=x-lower,upper-x
    if dlow<dup:return lower
    if dup<dlow:return upper
    return lower if q%2==0 else upper


def check(rows):
    if not isinstance(rows,list) or len(rows)!=4:
        raise ValueError('Require all four distinct format witnesses')
    seen=set()
    for r in rows:
        name=r['format']
        if name not in BITS or name in seen:
            raise ValueError('Unknown or duplicate format')
        seen.add(name)
        bits=BITS[name]
        if r['mantissa_bits']!=bits:
            raise ValueError('Wrong format grid')
        lo,hi=Fraction(r['low']),Fraction(r['high'])
        if lo != 1+Fraction(1,2**bits) or hi != lo+Fraction(1,2**bits):
            raise ValueError('Incorrect registered distinct pair')
        mid=lo+(hi-lo)/2
        if Fraction(r['half_step'])!=mid or closest(mid,bits)!=hi or closest(lo,bits)!=lo:
            raise ValueError('Not a fixed-point witness')
        if Fraction(r['real_gap_bound_after_64'])!=(hi-lo)/2**64:
            raise ValueError('Real contraction bound mismatch')
    return dict(exact_scalar_witnesses=len(seen),native_model_authentication=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path)
    args=parser.parse_args()
    print(json.dumps(check(json.loads(args.result.read_text())['scalar'])))

if __name__=='__main__':main()
