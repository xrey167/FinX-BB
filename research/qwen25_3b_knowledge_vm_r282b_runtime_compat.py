from __future__ import annotations

"""Runtime-only compatibility shim for the frozen R282 protocol.

Transformers 4.51.3 expects ``torch_dtype=`` while the frozen R282 source used
``dtype=``.  This wrapper changes only that API spelling at model construction;
all scientific prompts, cases, gates, revision hashes, and measurements remain
owned by the unchanged R282 module.
"""

from research import qwen25_3b_knowledge_vm_r282_causal_register as r282

_ORIGINAL = r282.AutoModelForCausalLM.from_pretrained


def _compat(cls, *args, **kwargs):
    if "dtype" in kwargs:
        if "torch_dtype" in kwargs:
            raise RuntimeError("both dtype and torch_dtype supplied")
        kwargs["torch_dtype"] = kwargs.pop("dtype")
    return _ORIGINAL(*args, **kwargs)


r282.AutoModelForCausalLM.from_pretrained = classmethod(_compat)

if __name__ == "__main__":
    raise SystemExit(r282.main())
