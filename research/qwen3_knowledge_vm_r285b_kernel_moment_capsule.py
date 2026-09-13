from __future__ import annotations

from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

# Transformers 4.51.3 exposes num_key_value_groups on Qwen3Attention but keeps
# the authoritative KV-head count on the frozen model config. R285's scientific
# protocol is unchanged; this is only a runtime-compatibility shim.
if not hasattr(Qwen3Attention, "num_key_value_heads"):
    Qwen3Attention.num_key_value_heads = property(lambda self: int(self.config.num_key_value_heads))

from research.qwen3_knowledge_vm_r285_kernel_moment_capsule import main

if __name__ == "__main__":
    raise SystemExit(main())
