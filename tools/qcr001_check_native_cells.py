"""Independent exact RN-cell check for archived QCR001 rational witnesses.

No Torch/NumPy/solver needed. This validates the declared dyadic cell geometry,
not that a trusted model actually emitted the target or produced the basis.
The reference fixed-basis algebra is checked by qcr001_check_certificate.py.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
try:
    from .qcr001_check_certificate import check, walk, require
except ImportError:
    from qcr001_check_certificate import check, walk, require

MANTISSA = {'bfloat16': 7, 'float32': 23}


def pow2(n):
    return F(2**n) if n >= 0 else F(1, 2**(-n))


def decode(bits, mantissa):
    require(type(bits) is int and 0 <= bits < 2**(mantissa+9), 'invalid bits')
    exponent = (bits >> mantissa) & 255
    require(exponent < 255, 'nonfinite values excluded')
    significand = bits & ((1 << mantissa)-1)
    if exponent:
        significand += 1 << mantissa
    magnitude = F(significand)*pow2((exponent-127 if exponent else -126)-mantissa)
    return -magnitude if bits >> (mantissa+8) else magnitude


def round_int(x):
    floor = x.numerator//x.denominator
    tail = x-floor
    return floor+int(tail > F(1,2) or (tail == F(1,2) and floor % 2))


def encode_rn(x, mantissa):
    """Exact rational round-to-nearest-even for finite BF16/FP32 values."""
    x = F(x)
    sign = int(x < 0)
    x = abs(x)
    if x == 0:
        return 0
    exponent = x.numerator.bit_length()-x.denominator.bit_length()
    if x < pow2(exponent):
        exponent -= 1
    if exponent < -126:
        payload = round_int(x/pow2(-126-mantissa))
        require(payload <= 1 << mantissa, 'subnormal overflow')
    else:
        q = round_int(x/pow2(exponent-mantissa))
        if q == 1 << (mantissa+1):
            q >>= 1
            exponent += 1
        require(exponent <= 127, 'overflow endpoint excluded')
        payload = ((exponent+127) << mantissa)+(q-(1 << mantissa))
    return (sign << (mantissa+8)) | payload


def cell(bits, mantissa):
    target = decode(bits, mantissa)
    require(target != 0, 'signed-zero cells excluded')
    exponent = (bits >> mantissa) & 255
    require(exponent > 0, 'subnormal target excluded from this witness contract')
    before, after = (bits-1,bits+1) if target > 0 else (bits+1,bits-1)
    low = (decode(before,mantissa)+target)/2
    high = (target+decode(after,mantissa))/2
    return low, high


def check_native(witness, dtype):
    require(dtype in MANTISSA, 'unsupported native precision')
    check(witness)
    m = MANTISSA[dtype]
    reconstructed = []
    for lo,hi in zip(witness['absolute_low'],witness['absolute_high']):
        low,high = F(float.fromhex(lo)),F(float.fromhex(hi))
        require(low < high, 'empty cell')
        bits = encode_rn((low+high)/2,m)
        require(cell(bits,m) == (low,high), 'bounds are not an entire native RN cell')
        reconstructed.append(bits)
    return reconstructed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files',nargs='+',type=Path)
    parser.add_argument('--require-witnesses',action='store_true')
    args=parser.parse_args()
    results=[]
    for path in args.files:
        data=json.loads(path.read_text())
        witnesses=list(walk(data))
        counts=[len(check_native(w,data['dtype'])) for w in witnesses]
        results.append(dict(file=str(path),witnesses=len(witnesses),native_cells=sum(counts)))
    total=sum(x['witnesses'] for x in results)
    if args.require_witnesses:
        require(total>0, 'no witness checked')
    print(json.dumps(dict(files=results,total_witnesses=total,
                         total_native_cells=sum(x['native_cells'] for x in results)),indent=2))

if __name__=='__main__':
    main()
