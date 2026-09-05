from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.experiments import e000052_bos_two_channel as E52


def test_e000052_generic_templates_are_disjoint():
    assert set(E18.TRAIN_GENERIC).isdisjoint(E17.GENERIC)


def test_e000052_joint_bars_cover_reading_deletion_and_locality():
    required = {
        "on/train/active_correct",
        "on/heldout/active_correct",
        "on/revoke_heldout_min",
        "on/shred_heldout_min",
        "on/broken1_unknown",
        "on/heldout/revoked_deleted_object",
        "on/generic/kl_to_base",
        "off/heldout/active_correct",
    }
    assert required.issubset(E52.CRITERIA)
    assert E52.CRITERIA["on/heldout/active_correct"] == (">=", 0.95)
    assert E52.CRITERIA["on/generic/kl_to_base"] == ("<=", 0.05)
