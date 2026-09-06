"""CRR001: fixed output-basis falsification, not a general repair impossibility."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import numpy as np

PROMPTS = [
    "The engineer checked the updated reference before making a decision.",
    "A researcher compared the measurements with the previous laboratory notes.",
]
MODELS = ["distilbert/distilgpt2", "EleutherAI/pythia-70m"]
RANKS = [1, 2, 4, 8, 16]


def identical(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def response(u, v):
    """Exact tanh(.5 log u + .5 log v) - tanh(.5 log u)."""
    return Fraction(2*u*(v-1), (1+u)*(1+u*v))


def fraction_rank(a):
    a = [[Fraction(x) for x in row] for row in a]
    rank = 0
    for col in range(len(a[0])):
        pivot = next((j for j in range(rank, len(a)) if a[j][col]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        value = a[rank][col]
        a[rank] = [x/value for x in a[rank]]
        for j in range(rank+1, len(a)):
            value = a[j][col]
            a[j] = [x-value*y for x, y in zip(a[j], a[rank])]
        rank += 1
        if rank == len(a):
            break
    return rank


def modular_rank_det(a, prime=1000003):
    """Nonzero rational determinant modulo a prime certifies rank over Q."""
    z = []
    for row in a:
        out = []
        for x in row:
            x = Fraction(x)
            if x.denominator % prime == 0:
                raise ValueError("Denominator not invertible modulo chosen prime")
            out.append(x.numerator*pow(x.denominator, -1, prime) % prime)
        z.append(out)
    rank, det = 0, 1
    for col in range(len(z[0])):
        pivot = next((j for j in range(rank, len(z)) if z[j][col]), None)
        if pivot is None:
            continue
        if pivot != rank:
            z[rank], z[pivot] = z[pivot], z[rank]
            det = -det % prime
        value = z[rank][col]
        det = det*value % prime
        inv = pow(value, -1, prime)
        z[rank] = [x*inv % prime for x in z[rank]]
        for j in range(rank+1, len(z)):
            value = z[j][col]
            z[j] = [(x-value*y) % prime for x, y in zip(z[j], z[rank])]
        rank += 1
        if rank == len(z):
            break
    return rank, (det if rank == len(z) == len(z[0]) else 0)


def exact_screen():
    rows = []
    for d in [2, 4, 8, 16, 32, 64]:
        matrix = [[response(i+1, j+2) for j in range(d)] for i in range(d)]
        rank, determinant = modular_rank_det(matrix)
        assert rank == d and determinant != 0
        independent = fraction_rank(matrix) if d <= 16 else None
        if independent is not None:
            assert independent == d
        x = np.array(matrix, dtype=np.float64)
        singular = np.linalg.svd(x, compute_uv=False)
        nonlinear_exact = all(
            response(i+1, j+2) == Fraction((i+1)*(j+2)-1, (i+1)*(j+2)+1)
            - Fraction(i, i+2) for i in range(d) for j in range(d)
        )
        assert nonlinear_exact
        linear = [[Fraction((i+1)*(j+1)) for j in range(d)] for i in range(d)]
        linear_rank, _ = modular_rank_det(linear)
        assert linear_rank == 1
        rows.append(dict(width=d, source_dimension=1, exact_response_rank=rank,
                         modular_determinant=determinant, modulus=1000003,
                         independent_fraction_rank=independent,
                         float64_default_rank=int(np.linalg.matrix_rank(x)),
                         singular_values=singular.tolist(), linear_control_rank=linear_rank,
                         compact_nonlinear_evaluator_exact=nonlinear_exact))
    return rows


def oracle_screen(states, old_index, never_index, ranks=RANKS):
    """Oracle sees ALL fresh states. Optimal Frobenius rank-r, not deployable."""
    states = np.asarray(states, dtype=np.float64)
    delta = states-states[old_index]
    u, s, vt = np.linalg.svd(delta, full_matrices=False)
    total = float(np.linalg.norm(delta))
    results = []
    for r in ranks:
        k = min(r, len(s))
        projected = (u[:, :k]*s[:k]) @ vt[:k]
        # A legitimate no-op needs no carrier reconstruction.
        projected[old_index] = 0
        restored = states[old_index]+projected
        residual = restored-states
        exact = [identical(a, b) for a, b in zip(restored, states)]
        results.append(dict(rank=r, relative_frobenius_error=float(np.linalg.norm(residual)/max(total,1e-300)),
                            maxabs=float(np.max(np.abs(residual))),
                            nontrivial_exact=sum(exact)-int(exact[old_index]),
                            nontrivial_count=len(states)-1,
                            never_exact=exact[never_index],
                            never_maxabs=float(np.max(np.abs(residual[never_index])))))
    full_reconstruction = (u*s) @ vt + states[old_index]
    return dict(coordinates=states.shape[1], revisions=len(states),
                finite_response_maxabs=float(np.max(np.abs(delta))),
                singular_values=s.tolist(),
                numerical_ranks={str(e):int(np.count_nonzero(s > e*s[0])) if s[0] else 0
                                 for e in [1e-6,1e-10,1e-12]},
                full_rank_reconstruction_roundoff_maxabs=float(np.max(np.abs(full_reconstruction-states))),
                oracle=results)


def legacy_cache(cache):
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return tuple((pair[0], pair[1]) for pair in cache)


def snapshots(output):
    kv = [np.concatenate([x.detach().cpu().numpy().ravel() for x in pair])
          for pair in legacy_cache(output.past_key_values)]
    hidden = [x.detach().cpu().numpy().ravel().copy() for x in output.hidden_states[1:]]
    return kv, hidden


def frozen_screen(name, output_dir):
    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    revision = HfApi().model_info(name).sha
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
    model = AutoModel.from_pretrained(name, revision=revision, torch_dtype=torch.float64,
                                     attn_implementation="eager", trust_remote_code=False,
                                     use_safetensors=True).eval()
    model.requires_grad_(False)
    blocks = model.h if hasattr(model, "h") else model.layers
    record = dict(model=name, revision=revision, torch=torch.__version__,
                  transformers=transformers.__version__, dtype=str(next(model.parameters()).dtype),
                  config=model.config.to_dict(), cells=[])
    assert next(model.parameters()).dtype == torch.float64
    amplitudes = np.linspace(-1., 3., 33)
    old_index, never_index = 16, 8
    for prompt_index, text in enumerate(PROMPTS):
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            absent = model(ids, use_cache=True, output_hidden_states=True, return_dict=True)
        absent_kv, absent_hidden = snapshots(absent)
        base_at_source = absent.hidden_states[1][0, 3].detach().clone()
        scale = float(torch.sqrt(torch.mean(base_at_source**2)))*.5
        del absent
        for seed in range(3):
            generator = torch.Generator().manual_seed(seed)
            direction = torch.randn(base_at_source.shape, generator=generator, dtype=torch.float64)
            direction *= scale/torch.sqrt(torch.mean(direction**2))
            current = [1.]
            def inject(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                changed = hidden.clone()
                changed[:, 3, :] += current[0]*direction
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            handle = blocks[0].register_forward_hook(inject)
            kstates, hstates, layer_kv = [], [], []
            try:
                with torch.no_grad():
                    for amplitude in amplitudes:
                        current[0] = float(amplitude)
                        out = model(ids, use_cache=True, output_hidden_states=True, return_dict=True)
                        kv, hidden = snapshots(out)
                        kstates.append(np.concatenate(kv))
                        hstates.append(np.concatenate(hidden))
                        layer_kv.append(kv)
                    current[0] = 1.
                    again = model(ids, use_cache=True, output_hidden_states=True, return_dict=True)
                    ak, ah = snapshots(again)
            finally:
                handle.remove()
            repeat = identical(np.concatenate(ak), kstates[old_index]) and identical(np.concatenate(ah), hstates[old_index])
            no_source = identical(kstates[never_index], np.concatenate(absent_kv)) and identical(hstates[never_index], np.concatenate(absent_hidden))
            layer0 = all(identical(row[0], absent_kv[0]) for row in layer_kv)
            assert repeat and no_source and layer0, "Numerical/causal control failed"
            cell = dict(seed=seed,prompt_index=prompt_index,input_ids=ids.tolist(),
                        source_token_position=3,source_block=0,source_dimension=1,
                        source_direction_rms=scale,amplitudes=amplitudes.tolist(),
                        repeat_forward_exact=repeat,zero_source_equals_absent_exact=no_source,
                        before_injection_kv_unchanged=layer0,
                        kv=oracle_screen(kstates,old_index,never_index),
                        hidden_diagnostics=oracle_screen(hstates,old_index,never_index))
            record["cells"].append(cell)
            print(json.dumps(dict(model=name,seed=seed,prompt=prompt_index,
                                  kv_rank=cell["kv"]["numerical_ranks"],
                                  rank1=cell["kv"]["oracle"][0],rank16=cell["kv"]["oracle"][-1])),flush=True)
            destination = output_dir/(name.replace("/","--")+".json")
            destination.write_text(json.dumps(record,indent=2)+"\n")
    del model
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=MODELS)
    parser.add_argument("--output-dir",type=Path,default=Path("crr001-results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.model:
        frozen_screen(args.model,args.output_dir)
    else:
        data = dict(experiment="CRR-001",source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    python=platform.python_version(),numpy=np.__version__,exact=exact_screen(),
                    status="fixed_basis_boundary_not_invention",trained_reader_gate="NOT_EVALUATED")
        (args.output_dir/"exact.json").write_text(json.dumps(data,indent=2)+"\n")
        print(json.dumps([dict(width=x["width"],rank=x["exact_response_rank"],
                               float_rank=x["float64_default_rank"]) for x in data["exact"]]))

if __name__ == "__main__":
    main()
