from __future__ import annotations

from pathlib import Path
import os

from research import qwen25_3b_knowledge_vm_r284_factored_ports as r284

# Fast real-model gate using the same lifecycle mechanism as R284 on the already
# revision/hash-pinned Qwen2.5-0.5B backbone. This is a latency/turnaround probe,
# not a substitute for the 3B gate.
r284.MODEL_ID = "Qwen/Qwen2.5-0.5B"
r284.REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
r284.EXPECTED_WEIGHTS = {
    "model.safetensors": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
}
r284.OUT = Path(os.environ.get("SO_R284_FAST_REPORT", "ci-qwen-r284-fast/report.json"))

if __name__ == "__main__":
    raise SystemExit(r284.main())
