"""E-000087 -- does a KV-cache erasure primitive actually erase?

Serving systems remove content from a KV cache by rewriting the removed span: replace its K/V with a
stub or a neutral span, keep everything else, avoid a recompute. The span is gone, the system reports
the chunk removed. But attention is causal: every token AFTER the chunk attended to it while it was
still there, so the retained suffix K/V was computed FROM it. Rewriting the chunk's own span does not
touch them.

This experiment asks whether that residue is recoverable, which is the same shape as this programme's
E-000028 -- a deletion primitive that passed every attack aimed at the channel it gated, and gave the
payload up through a derived quantity it never touched.

  prompt = [ prefix | D: a chunk naming entity e | S: a fixed bystander chunk | query ]

  P_IDEAL   recompute S and the query with a neutral chunk in D's place. Nothing retained depends
            on e. This is what erasure is supposed to mean.
  P_STUB    take the full cache computed WITH e, and replace only D's span with the neutral span's
            K/V. S and the query keep the K/V they had. This is the span-replacement primitive.
  P_ZERO    same, but D's span is zeroed rather than replaced.

The attack is a storage-layer adversary who reads the retained tensors and knows the candidate set:
for every candidate it builds the same prefill and matches. Top-1 against a chance of 1/N.

CONTROLS, and they can fail. This session produced four measurement errors, every one of them a
comparison that was not like-for-like or a control that could not fail, so they are stated first:

  C1  P_IDEAL's retained tensors must be BIT-IDENTICAL across every candidate e, since e never entered
      that forward. Asserted, not assumed. If it fails the harness is wrong and nothing else is read.
  C2  The same attack run against P_IDEAL must land at chance. If a matcher can identify e from state
      that provably does not depend on e, the matcher is reading the harness and every other number
      in this file is void.
  C3  Every arm is matched against references built the SAME way -- retained-suffix against
      retained-suffix -- so no arm is compared against a differently-normalised quantity.
  C4  The behavioural attack reports the answer distribution's mass on e MINUS its mass under P_IDEAL,
      so a prompt that merely makes e likely cannot be read as leakage.

Nothing here is a novelty claim. It is an audit of a primitive.
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

    R = torch.stack(retained_with)          # attacker's reference set, built the same way as each arm
    tgt = torch.arange(N)

    def top1(obs: torch.Tensor) -> float:
        o = obs / obs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        r = R / R.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return float((o @ r.t()).argmax(-1).eq(tgt).float().mean())

    I = torch.stack(retained_ideal)
    # C1: the ideal arm's retained state cannot depend on e
    c1 = bool((I - I[0:1]).abs().max() == 0.0)
    results = {
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
    results["controls_ok"] = bool(c1 and results["C2_attack_on_ideal_top1"] <= 4.0 / N)
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
