"""E-000087 -- RETRACTED. The headline result was a tensor compared with itself.

The experiment claimed that a span-replacement KV erasure primitive leaks the erased entity at 1.0000
from the retained suffix. That number is VACUOUS. P_STUB rewrites only the chunk's own span; the
retained suffix it then reads is BITWISE IDENTICAL to the suffix of the arm where no erasure happened
at all (verified: max abs difference 0.000e+00). The attack therefore matched a tensor against itself,
and 1.0000 was arithmetic, not evidence. The same holds for P_ZERO.

The controls this file declared -- C1 through C4 -- all checked the IDEAL arm. Not one of them asked
whether the TREATMENT arm differed from doing nothing, which is the only control that could have caught
this. That check is now `C0` below and it fails on the original design, which is why the arms are gone.

What the design should have known before running: KVEraser (arXiv:2606.17034) states in its own method
that its steering block is length-preserving and "leaves the suffix cache unchanged", and Leyline
(arXiv:2606.01065) preserves suffix K_nope and V because "that attention is exactly what we want to
keep". Any readout over retained suffix K/V returns the same value erased or not, BY THEIR OWN
EQUATIONS. Pointing an instrument at a quantity the treatment provably does not touch is not a
measurement.

What survives, and it is a null result: decoding the query AGAINST the erased cache -- which does
differ from the un-erased one, since the chunk's own span was replaced -- the erased entity is the
top-1 answer in 0.0 of cases on both backbones, mean rank 2820 (GPT-2) and 916 (Pythia-70m), gain over
the ideal arm +0.66 and +1.66 nats. The primitive works behaviourally on this harness.

Prior art that occupies the space anyway: MEMENTO (arXiv:2604.09852) §6.2.2 evicts a block holding a
5-digit passcode and trains an MLP probe on the retained downstream KV, recovering 26.7% PER DIGIT
against a 10% floor, with a causal control at chance. Note per digit: full recovery is ~0.267^5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch


def _kv_list(pkv) -> List[tuple]:
    if hasattr(pkv, "layers"):
        return [(l.keys, l.values) for l in pkv.layers]
    if hasattr(pkv, "key_cache"):
        return list(zip(pkv.key_cache, pkv.value_cache))
    return [(k, v) for k, v in pkv]


def _slice(kv, lo: int, hi: int) -> torch.Tensor:
    """The retained suffix, flattened: every layer's K and V over positions [lo, hi)."""
    return torch.cat([torch.cat([k[:, :, lo:hi, :].reshape(-1), v[:, :, lo:hi, :].reshape(-1)])
                      for k, v in kv])


@torch.no_grad()
def run(model_name: str, n_entities: int, seed: int) -> Dict[str, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(model_name).float().eval()

    # A candidate set of single-token entities, so the removed content is one token and the attack has
    # a well defined finite domain -- the same reason E-000030's certificate was exhaustive.
    cands, seen = [], set()
    for w in [" Paris", " London", " Berlin", " Madrid", " Rome", " Vienna", " Dublin", " Lisbon",
              " Warsaw", " Prague", " Athens", " Oslo", " Stockholm", " Helsinki", " Moscow", " Cairo",
              " Tokyo", " Beijing", " Delhi", " Sydney", " Toronto", " Boston", " Chicago", " Denver",
              " Miami", " Dallas", " Houston", " Seattle", " Portland", " Atlanta", " Detroit", " Phoenix"]:
        ids = tok.encode(w)
        if len(ids) == 1 and ids[0] not in seen:
            seen.add(ids[0]); cands.append((w, ids[0]))
    # top up from the vocabulary so the candidate set reaches n_entities
    for tid in range(len(tok)):
        if len(cands) >= n_entities + 1:
            break
        s = tok.decode([tid])
        if s.startswith(" ") and s[1:].isalpha() and s[1:2].isupper() and tid not in seen:
            seen.add(tid); cands.append((s, tid))
    # one extra single-token word, never a candidate, is the neutral filler; picking it from the same
    # distribution keeps every chunk the same length, which the assertion below re-checks anyway
    if len(cands) < n_entities + 1:
        raise RuntimeError("not enough single-token entities for the requested candidate set")
    neutral_word = cands[n_entities][0]
    cands = cands[:n_entities]
    N = len(cands)

    PREFIX = "Internal record follows."
    NEUTRAL = " The entry was withheld from this copy."
    SUFFIX = " The auditor reviewed the file and signed the register on the following day."
    QUERY = " The entry named"

    def chunk(word: str) -> str:
        return f" The entry named{word} was filed."

    neutral_ids = tok.encode(chunk(neutral_word))
    def build(word):
        """Token ids and the span boundaries of [prefix | D | S | query]."""
        p = tok.encode(PREFIX)
        d = tok.encode(chunk(word))
        s = tok.encode(SUFFIX)
        q = tok.encode(QUERY)
        return torch.tensor([p + d + s + q]), len(p), len(p) + len(d), len(p) + len(d) + len(s) + len(q)

    # every chunk must be the same length, or removing it shifts positions and the arms stop being
    # comparable; the candidates are single tokens so this holds, and it is checked rather than assumed
    lens = {len(tok.encode(chunk(w))) for w, _ in cands} | {len(neutral_ids)}
    if len(lens) != 1:
        raise RuntimeError(f"chunk lengths differ {lens}; the arms would not be comparable")

    def prefill(ids):
        out = lm(input_ids=ids, use_cache=True)
        return out.past_key_values, _kv_list(out.past_key_values), out.logits

    def answer_from(cache, query_ids):
        """Decode the query AGAINST a given cache. This is what a served request does after an
        erasure: the deleted span is gone from the cache, and the model answers from what remains."""
        out = lm(input_ids=query_ids, past_key_values=cache, use_cache=True)
        return torch.log_softmax(out.logits[0, -1], -1)

    q_ids = torch.tensor([tok.encode(QUERY)])

    def build_body(word):
        """[prefix | D | S] only -- the query is decoded separately, against the erased cache."""
        p = tok.encode(PREFIX); d = tok.encode(chunk(word)); sfx = tok.encode(SUFFIX)
        return torch.tensor([p + d + sfx]), len(p), len(p) + len(d), len(p) + len(d) + len(sfx)

    neutral_body, n_lo, n_hi, n_tot = build_body(neutral_word)
    pkv_neutral, kv_neutral, _ = prefill(neutral_body)
    ideal_answer = answer_from(pkv_neutral, q_ids)

    retained_with, retained_ideal, answer_with, answer_ideal = [], [], [], []
    stub, zero = [], []
    stub_answers = []
    for w, tid in cands:
        body, lo, hi, tot = build_body(w)
        pkv, kv, _ = prefill(body)
        retained_with.append(_slice(kv, hi, tot))                 # suffix computed WITH e present
        retained_ideal.append(_slice(kv_neutral, hi, tot))
        # P_STUB: rewrite only D's span in place, keep the suffix the model already computed, then
        # answer the query against that cache -- exactly what a served request would get.
        for li, (k, v) in enumerate(kv):
            k[:, :, lo:hi, :] = kv_neutral[li][0][:, :, lo:hi, :]
            v[:, :, lo:hi, :] = kv_neutral[li][1][:, :, lo:hi, :]
        stub.append(_slice(kv, hi, tot))
        stub_answers.append(answer_from(pkv, q_ids))
        # P_ZERO on a fresh copy of the same prefill
        pkv2, kv2, _ = prefill(body)
        for k, v in kv2:
            k[:, :, lo:hi, :] = 0.0
            v[:, :, lo:hi, :] = 0.0
        zero.append(_slice(kv2, hi, tot))

    # C0, the control this experiment was missing: the treatment must actually change what is retained.
    # If the erasure arm's retained state equals the no-erasure arm's, any attack on it is comparing a
    # tensor with itself and no number below means anything.
    stub_t = torch.stack(stub)
    R = torch.stack(retained_with)          # attacker's reference set, built the same way as each arm
    c0 = float((stub_t - R).abs().max())
    tgt = torch.arange(N)

    def top1(obs: torch.Tensor) -> float:
        o = obs / obs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        r = R / R.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return float((o @ r.t()).argmax(-1).eq(tgt).float().mean())

    I = torch.stack(retained_ideal)
    # C1: the ideal arm's retained state cannot depend on e
    c1 = bool((I - I[0:1]).abs().max() == 0.0)
    results = {
        "C0_stub_retained_differs_from_no_erasure_maxabs": c0,
        "C0_passes": bool(c0 > 0.0),
        "tensor_attack_is_vacuous": bool(c0 == 0.0),
        "model": model_name,
        "n_entities": N,
        "chance": 1.0 / N,
        "C1_ideal_retained_bit_identical_across_entities": c1,
        "C2_attack_on_ideal_top1": top1(I),
        "P_IDEAL_top1": top1(I),
        "P_STUB_top1": top1(torch.stack(stub)),
        "P_ZERO_top1": top1(torch.stack(zero)),
        "P_NOOP_top1": top1(R),
    }
    # C4: behavioural leakage, from the ERASED cache, against the ideal arm rather than absolutely
    sa = torch.stack(stub_answers)
    ent = torch.tensor([t for _, t in cands])
    idx = torch.arange(N)
    results["behavioural_logprob_gain_on_e_after_stub_vs_ideal"] = float(
        (sa[idx, ent] - ideal_answer[ent]).mean())
    results["behavioural_top1_is_e_after_stub"] = float(sa.argmax(-1).eq(ent).float().mean())
    results["behavioural_rank_of_e_after_stub"] = float(
        (sa > sa[idx, ent][:, None]).sum(-1).float().mean())
    # the tensor-level arms are reported only if C0 passes; on the original design it does not
    results["controls_ok"] = bool(c1 and results["C2_attack_on_ideal_top1"] <= 4.0 / N and c0 > 0.0)
    if c0 == 0.0:
        for k in ("P_STUB_top1", "P_ZERO_top1", "P_NOOP_top1", "P_IDEAL_top1"):
            results[k] = f"VACUOUS ({results[k]}): retained state identical to the no-erasure arm"
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n-entities", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rec = {"experiment": "E-000087", "rows": [run(a.model, a.n_entities, a.seed)],
           "not_claimed": "no novelty claim; an audit of a cache erasure primitive"}
    out = Path(a.results_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"e000087_{a.model.replace('/', '_')}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
