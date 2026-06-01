from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
m = torch.load(ROOT / "model_weights.pth", map_location="cpu")
vals = [
    m["kernels"].float().reshape(-1).numpy(),
    m["conv_bias"].float().reshape(-1).numpy(),
    m["linear_w"].float().reshape(-1).numpy(),
    m["linear_b"].float().reshape(-1).numpy(),
]
np.concatenate(vals).astype("float32").tofile(ROOT / "weights.bin")
print("wrote weights.bin")
