"""E-000054's surfaces: two tokens per subject, the first on position 0 of every subject-initial template,
no part token inside the object pool, and both renderings distinct over the 256 entities."""

import pytest

pytest.importorskip("transformers")

from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000054_two_token_subjects as E54


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


def test_surfaces_are_two_tokens_with_the_first_on_the_sink(tok):
    ids = E8.select_entities(tok, 256)
    for name, names in E54.surfaces_for(tok, ids).items():
        pos = E54.surface_positions(tok, names)
        assert pos["tokens_per_name_min"] == pos["tokens_per_name_max"] == 2, name
        assert pos["initial_at_0"] and pos["medial_at_ge1"], name
        assert len(set(names)) == 256, name


def test_part_tokens_are_outside_the_object_pool_and_second_keeps_the_entity_token(tok):
    ids = E8.select_entities(tok, 256)
    parts = set(E8.select_entities(tok, 288)[256:])
    assert not parts & set(ids)
    surfaces = E54.surfaces_for(tok, ids)
    single = [tok.decode([i]) for i in ids]
    assert all(s.endswith(single[i]) for i, s in enumerate(surfaces["second"]))
    firsts = {s.split()[0] for s in surfaces["product"]}
    seconds = {s.split()[1] for s in surfaces["product"]}
    assert len(firsts) == 16 and len(seconds) == 16


def test_a_wrong_pool_is_refused(tok):
    ids = E8.select_entities(tok, 257)[1:]
    with pytest.raises(ValueError):
        E54.surfaces_for(tok, ids)
