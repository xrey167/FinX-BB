"""Post-experiment offline-checker tests, separate from the original29 tests."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from tools import qcr001_check_certificate as checker
from tools.qcr001_check_certificate import check, walk


def fixture():
    return dict(rank=1,anchor_indices=[0],extra_index=1,anchor_rows=[['1']],
        extra_row=['1'],coefficients=['1'],anchor_low=['0'],anchor_high=['1'],
        extra_low='2',extra_high='3',span_low='0',span_high='1',
        original_old=[float(0).hex()]*2,
        absolute_low=[float(0).hex(),float(2).hex()],
        absolute_high=[float(1).hex(),float(3).hex()])


def test_exact_separation_passes():
    check(fixture())


@pytest.mark.parametrize('field,value',[
 ('rank',True),('rank',0),('anchor_indices',[1]),('extra_index',False),
 ('anchor_rows',[['2']]),('extra_row',['2']),('coefficients',['2']),
 ('anchor_low',['2']),('span_low','-1'),('span_high','2'),
 ('extra_low','1'),('original_old',[float(1).hex()]*2),
 ('absolute_high',[float(2).hex(),float(3).hex()])])
def test_tampering_fails(field,value):
    w=fixture();w[field]=value
    with pytest.raises((ValueError,KeyError,TypeError,IndexError)):
        check(w)


def test_optimized_python_keeps_verification_active(tmp_path):
    w=fixture();w['span_high']='999'
    p=tmp_path/'bad.json'
    p.write_text(json.dumps({'rounding_box_separation':{'status':'EXACT_SEPARATION','witness':w}}))
    proc=subprocess.run([sys.executable,'-O',str(Path(checker.__file__)),str(p)],capture_output=True,text=True)
    assert proc.returncode!=0
    assert 'incorrect upper bound' in proc.stderr


def test_missing_certificate_rejected_and_empty_not_counted():
    assert list(walk({'nothing':[]}))==[]
    with pytest.raises(ValueError):
        list(walk({'rounding_box_separation':{'status':'EXACT_SEPARATION','witness':None}}))
