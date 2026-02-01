# EAGLE-Spine

EAGLE-Spine (Energy-Adaptive Geometry-Loop-free Enhancement) is a PyTorch research prototype for spine landmark detection that combines scale-free gating with loop-free deformable alignment. The current codebase focuses on two core mechanisms: SREG (Scale-free Relative Energy Gating) and LODA-Conv (Loop-free Orthogonal Deformable Alignment Conv).

## Contents

- `model/sreg.py`: SREG / DRE-Gating module (v2) implementation.
- `model/lda_conv.py`: LODA-Conv module with orthogonal dispersion constraint.
- `test/sreg_test.py`: Demo/test runner that generates synthetic spine centerlines and prints diagnostic outputs.
- `test/lda_conv_test.py`: Demo/test runner for LODA-Conv output and loss stats.
- `idea_script/`: Project notes and LaTeX assets.
- `data/`: Sample dataset with AP X-ray images and CSV annotations.

## Requirements

- Python 3.8+
- PyTorch

## Quick Start

Run the demo/test scripts:

```bash
python test/sreg_test.py
python test/lda_conv_test.py
```

The SREG script generates three synthetic samples (normal, smooth curve, pathological kink), prints `rho` ranges, `gate` behavior, and a scale invariance check. The LODA-Conv script prints output shapes, orthogonality loss, and basic stats.

## Dataset Layout

The repo includes a sample dataset under `data/` with the following structure:

```
data/
  train/
    *.jpg
  train_txt/
    filenames.csv
    landmarks.csv
    angles.csv
```

- `data/train/`: AP spine X-ray images (JPEG). File names match entries in `filenames.csv`.
- `data/train_txt/filenames.csv`: One image file name per line.
- `data/train_txt/landmarks.csv`: Landmark coordinates per image. Each line is a flat list of normalized `(x, y)` pairs in image coordinates (range 0-1). The ordering follows the dataset's vertebra/landmark convention.
- `data/train_txt/angles.csv`: Per-image angle targets. Each line contains three Cobb-related angles in degrees.

## Core Ideas

- SREG: computes a scale-free local turning energy `rho` and applies a hybrid robust scale `S = MAD + gamma * median` to derive gating weights for continuity regularization.
- LODA-Conv: predicts deformable offsets without geometry in the forward path and applies an orthogonal dispersion loss aligned with tangent/normal directions derived from centerline geometry.
- Loop-free geometry: centerline, tangent, normal, and ratio terms are typically stop-gradient to avoid feedback loops.

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
- Suggested training phases (from `idea_script/idea.tex`): warm up with heatmap loss, then enable LODA-Conv with `L_ortho`, and finally enable SREG continuity regularization.
