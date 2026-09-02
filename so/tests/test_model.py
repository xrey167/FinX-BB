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
