# EAGLE-Spine

PyTorch implementation of the SREG / DRE-Gating module for spine centerline geometry. The current model uses a hybrid scale definition (MAD + gamma * median) to compute scale-free gating and a continuity regularizer.

## Contents

- `model/sreg.py`: SREG / DRE-Gating module (v2) implementation.
- `test/sreg_test.py`: Demo/test runner that generates synthetic spine centerlines and prints diagnostic outputs.
- `idea_script/`: Project notes and LaTeX assets.

## Requirements

- Python 3.8+
- PyTorch

## Quick Start

Run the demo/test script:

```bash
python test/sreg_test.py
```

This script generates three synthetic samples (normal, smooth curve, pathological kink), prints `rho` ranges, `gate` behavior, and a scale invariance check.

## Usage

Import the module and run a forward pass:

```python
from model.sreg import SREGGating

sreg = SREGGating(lam_min=0.1, init_tau=2.0, gamma=1.0, gamma_mode="learnable")
out = sreg(c, mask=mask, return_loss=True, detach_geometry=True)

print(out.rho.shape, out.gate.shape, out.scale)
```

## Notes

- `gate` is set to 1 for invalid points; endpoints have `rho = 0`.
- `scale` is computed per-sample using `mad + gamma * median` with a small epsilon for stability.
