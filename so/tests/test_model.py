import numpy as np
import torch

from so.data import bank_from_world, encode_queries, sample_training_queries
from so.model import ModelConfig, MutableKnowledgeTransformer
from so.train import TrainConfig, make_centre, routing_loss
from so.world import UNKNOWN, World


def _setup(seed=0, n_cells=200):
    rng = np.random.default_rng(seed)
    world = World.sample(rng, 64, 4, n_cells, 2)
    centre = make_centre(seed, 16)
    bank = bank_from_world(rng, world, centre, 0.1, 0.05, 0.05)
    cfg = ModelConfig(n_entities=64, n_relations=4, n_surface=8)
    torch.manual_seed(seed)
    model = MutableKnowledgeTransformer(cfg)
    return rng, world, centre, bank, cfg, model


def test_forward_shapes_and_routing_normalisation():
    rng, world, centre, bank, cfg, model = _setup()
    qs = sample_training_queries(rng, world, bank, 32, TrainConfig().mix)
    b = encode_queries(qs, bank, world, cfg.max_hops)
    logits, routing, extras = model(bank.tensors(), b.mode, b.start, b.rels, b.hop_valid)
    assert logits.shape == (32, 65)
    assert routing.shape == (32, 3, bank.size + 1)
    sums = routing.sum(-1)
    assert torch.allclose(sums[b.hop_valid], torch.ones_like(sums[b.hop_valid]), atol=1e-5)
    assert torch.all(sums[~b.hop_valid] == 0)
    assert extras["hidden"].shape == (32, cfg.d_model)


def test_inactive_cells_receive_no_routing_mass():
    rng, world, centre, bank, cfg, model = _setup()
    qs = sample_training_queries(rng, world, bank, 16, TrainConfig().mix)
    b = encode_queries(qs, bank, world, cfg.max_hops)
    _, routing, _ = model(bank.tensors(), b.mode, b.start, b.rels, b.hop_valid)
    inactive = torch.as_tensor(~bank.active)
    assert torch.all(routing[:, :, :-1][:, :, inactive] == 0)


def test_cell_mask_blocks_routing():
    rng, world, centre, bank, cfg, model = _setup()
    qs = sample_training_queries(rng, world, bank, 16, TrainConfig().mix)
    b = encode_queries(qs, bank, world, cfg.max_hops)
    mask = torch.zeros(bank.size, dtype=torch.bool)
    _, routing, _ = model(bank.tensors(), b.mode, b.start, b.rels, b.hop_valid, cell_mask=mask)
    # with every cell blocked all valid hops must route to the null cell
    assert torch.all(routing[:, :, -1][b.hop_valid] > 0.999)


def test_routing_targets_are_consistent_with_ground_truth():
    rng, world, centre, bank, cfg, model = _setup()
    qs = sample_training_queries(rng, world, bank, 64, TrainConfig().mix)
    b = encode_queries(qs, bank, world, cfg.max_hops)
    for i, q in enumerate(qs):
        gt = world.answer(q, bank.index_view)
        if q.mode == "fwd":
            for t, e in enumerate(gt.edges):
                p = int(b.route[i, t])
                assert (int(bank.subject[p]), int(bank.relation[p])) == e and bank.usable[p]
            if gt.answer == UNKNOWN and len(gt.edges) < q.hops:
                tgt = int(b.route[i, len(gt.edges)])
                if tgt == -1:
                    cur = q.start
                    for e in gt.edges:
                        cur = bank.index_view[e]
                    assert (cur, q.path[len(gt.edges)]) not in bank.active_pos
                else:
                    assert bank.active[tgt] and not bank.usable[tgt]   # a routable but shredded cell
        if gt.answer == UNKNOWN:
            assert int(b.target[i]) == world.n_entities


def test_routing_loss_is_finite_and_decreases_on_perfect_routing():
    rng, world, centre, bank, cfg, model = _setup()
    qs = sample_training_queries(rng, world, bank, 16, TrainConfig().mix)
    b = encode_queries(qs, bank, world, cfg.max_hops)
    _, routing, _ = model(bank.tensors(), b.mode, b.start, b.rels, b.hop_valid)
    loss = routing_loss(routing, b.route)
    assert torch.isfinite(loss) and loss.item() > 0
    perfect = torch.zeros_like(routing)
    for i in range(routing.shape[0]):
        for t in range(routing.shape[1]):
            r = int(b.route[i, t])
            if r == -2:
                continue
            perfect[i, t, routing.shape[2] - 1 if r == -1 else r] = 1.0
    assert routing_loss(perfect, b.route).item() < 1e-6


def test_no_routing_variant_ignores_bank():
    rng, world, centre, bank, cfg, _ = _setup()
    torch.manual_seed(0)
    model = MutableKnowledgeTransformer(ModelConfig(n_entities=64, n_relations=4, n_surface=8, use_routing=False))
    qs = sample_training_queries(rng, world, bank, 8, TrainConfig().mix)
    b = encode_queries(qs, bank, world, cfg.max_hops)
    l1, _, _ = model(bank.tensors(), b.mode, b.start, b.rels, b.hop_valid)
    empty = dict(bank.tensors()); empty["active"] = torch.zeros_like(empty["active"])
    l2, _, _ = model(empty, b.mode, b.start, b.rels, b.hop_valid)
    assert torch.allclose(l1, l2)


# ------------------------------------------------------- the key channel (E-000028)

def _bank_with_gate(seed=0, n_cells=24):
    import numpy as np
    from so.data import bank_from_world
    from so.train import make_centre
    from so.world import World
    rng = np.random.default_rng(seed)
    world = World.sample(rng, 32, 4, n_cells, 2)
    return bank_from_world(rng, world, make_centre(seed, 16), 0.0, 0.5, 0.0).tensors()


def test_the_reverse_key_names_the_object_by_default():
    """The defect E-000028 measures, pinned so a later change cannot hide it.

    Nothing in the gate touches k_r, so two cells differing only in their object have different
    reverse keys whatever their marker says -- which is what lets a candidate sweep recover a
    shredded object.
    """
    cfg = ModelConfig(n_entities=32, n_relations=4, d_model=32, marker_dim=16, n_core_layers=1, n_heads=2)
    torch.manual_seed(0)
    m = MutableKnowledgeTransformer(cfg).eval()
    b = _bank_with_gate()
    with torch.no_grad():
        enc_a = m.encode_bank(b)
        b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        b2["obj"] = (b2["obj"] + 1) % cfg.n_entities          # same cells, different objects
        enc_b = m.encode_bank(b2)
    assert not torch.allclose(enc_a["k_r"], enc_b["k_r"], atol=1e-4)


def _force_gate(m, value: float):
    """Drive the marker gate to a constant. An untrained gate sits near 0.5 for every marker, so the
    open and closed regimes have to be produced deliberately rather than sampled."""
    last = m.marker_gate[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(value)


def test_gating_the_reverse_key_makes_gate_closed_cells_indistinguishable():
    cfg = ModelConfig(n_entities=32, n_relations=4, d_model=32, marker_dim=16, n_core_layers=1, n_heads=2,
                      gate_reverse_key=True)
    torch.manual_seed(0)
    m = MutableKnowledgeTransformer(cfg).eval()
    b = _bank_with_gate()
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b2["obj"] = (b2["obj"] + 1) % cfg.n_entities              # same cells, different objects

    _force_gate(m, -20.0)                                     # every marker invalid: shredded
    with torch.no_grad():
        assert float(m.gate(b["marker"]).max()) < 1e-6
        assert torch.allclose(m.encode_bank(b)["k_r"], m.encode_bank(b2)["k_r"], atol=1e-6)

    _force_gate(m, 20.0)                                      # every marker valid: the key still addresses
    with torch.no_grad():
        assert float(m.gate(b["marker"]).min()) > 1 - 1e-6
        assert not torch.allclose(m.encode_bank(b)["k_r"], m.encode_bank(b2)["k_r"], atol=1e-4)


def test_without_the_flag_a_closed_gate_leaves_the_reverse_key_naming_the_object():
    """The defect itself, pinned: with the gate shut the key is still a function of the object."""
    cfg = ModelConfig(n_entities=32, n_relations=4, d_model=32, marker_dim=16, n_core_layers=1, n_heads=2)
    torch.manual_seed(0)
    m = MutableKnowledgeTransformer(cfg).eval()
    b = _bank_with_gate()
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b2["obj"] = (b2["obj"] + 1) % cfg.n_entities
    _force_gate(m, -20.0)
    with torch.no_grad():
        assert float(m.gate(b["marker"]).max()) < 1e-6        # nothing readable through the value channel
        assert torch.allclose(m.encode_bank(b)["v_f"], torch.zeros_like(m.encode_bank(b)["v_f"]), atol=1e-6)
        assert not torch.allclose(m.encode_bank(b)["k_r"], m.encode_bank(b2)["k_r"], atol=1e-4)


def test_the_flag_is_off_by_default_so_recorded_models_are_unchanged():
    assert ModelConfig().gate_reverse_key is False
