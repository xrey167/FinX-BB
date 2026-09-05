"""A record is only reproducible while the checkpoint it names is still on disk.

E-000020 published three seeds and a forced re-run then replaced two of the three checkpoint files,
so two thirds of that record can no longer be re-scored and nothing in the code objected. These tests
pin the objection.
"""
import json

import pytest

from so import ledger
from so.experiments.e000001b_mini_transformer import _sha256, guard_recorded_checkpoint


@pytest.fixture()
def results(tmp_path, monkeypatch):
    d = tmp_path / "results"
    d.mkdir()
    monkeypatch.setattr(ledger, "RESULTS_DIR", d)
    monkeypatch.delenv("SO_ALLOW_OVERWRITE", raising=False)
    return d


def _ckpt(tmp_path, body=b"weights"):
    p = tmp_path / "e999999_seed0.pt"
    p.write_bytes(body)
    return p


def test_a_checkpoint_no_record_cites_may_be_overwritten(results, tmp_path):
    p = _ckpt(tmp_path)
    (results / "other.json").write_text(json.dumps({"per_seed": [{"checkpoint_sha256": "deadbeef"}]}))
    guard_recorded_checkpoint(p)          # no exception


def test_a_checkpoint_a_record_cites_is_refused(results, tmp_path):
    p = _ckpt(tmp_path)
    (results / "e999999.json").write_text(json.dumps({"per_seed": [{"checkpoint_sha256": _sha256(p)}]}))
    with pytest.raises(SystemExit) as e:
        guard_recorded_checkpoint(p)
    assert "e999999.json" in str(e.value)
    assert "SO_ALLOW_OVERWRITE" in str(e.value)


def test_the_refusal_can_be_overridden_on_purpose(results, tmp_path, monkeypatch):
    p = _ckpt(tmp_path)
    (results / "e999999.json").write_text(json.dumps({"per_seed": [{"checkpoint_sha256": _sha256(p)}]}))
    monkeypatch.setenv("SO_ALLOW_OVERWRITE", "1")
    guard_recorded_checkpoint(p)          # no exception


def test_a_new_checkpoint_is_never_refused(results, tmp_path):
    guard_recorded_checkpoint(tmp_path / "does_not_exist.pt")


def test_changing_the_file_lifts_the_refusal(results, tmp_path):
    """The guard is about the bytes a record names, not about the file name."""
    p = _ckpt(tmp_path)
    (results / "e999999.json").write_text(json.dumps({"per_seed": [{"checkpoint_sha256": _sha256(p)}]}))
    with pytest.raises(SystemExit):
        guard_recorded_checkpoint(p)
    p.write_bytes(b"different weights")
    guard_recorded_checkpoint(p)          # already replaced: the record is beyond saving, do not block work
