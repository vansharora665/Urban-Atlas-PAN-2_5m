# 2.5 m PAN TextureNet V8 + SegFormer Roads V9

Standalone public repository for this model only. The trained weights and matching
inference implementation are unchanged from the verified combined release.

```bash
git lfs install
git clone https://github.com/vansharora665/Urban-Atlas-PAN-2_5m.git
cd Urban-Atlas-PAN-2_5m
```

The required `model_weights.pth`, `model_config.json` and `README.md` are at the
repository root, alongside the matching loader. The existing direct-download
mirror is retained to avoid re-uploading an identical large release asset.

Frozen final model bundle. **No new training or accuracy improvement is claimed.**

## Contents

- `model_weights.pth`: the original trained components, losslessly bundled in a Torch-safe container.
- `model_config.json`: display metadata, actual architecture, input schema, metrics with scope, hashes and a direct weights URL.
- `loader.py`, `worker.py`, `runtime/`: matching architecture, preprocessing, tiling and export implementation.
- `requirements.txt`: the versions used for the packaging checks (Python 3.9 environment; Python 3.10/3.11 recommended for a fresh setup).

## Download And Run

```bash
python -m pip install -r requirements.txt
python loader.py --verify --device cpu
python loader.py --input /path/to/image.tif --output /path/to/new_results
```

Expected input: **1 band(s), 2.5 m**, with band order `panchromatic`.
The TIFF must have a projected metre CRS and its original georeferencing. Wrong band counts or resolutions fail explicitly.
No missing spectral band is invented. Source products, training data and companion-band recovery are not redistributed.

Add `--vectors` for class shapefiles with sidecars, generated only on request.
The PAN road expert is part of the normal pipeline. Tiny roofs, shadows and linear boundaries can still be confused with roads.
Results include native-grid class GeoTIFF, marked preview, class percentages and prediction metadata.
The output directory must be new/empty. Leave the original input TIFF available while its linked output VRT is needed.

## Integrating An External Platform

Read `model_config.json` for automatic display/configuration, then register the provided
`loader.py:load_model` adapter. These are **custom composite pipelines**, not an off-the-shelf
HRNet/ResNet state dictionary. Generic platforms must support a custom loader before they
can run this model. A metadata file alone cannot implement the architecture.

```python
from loader import load_model
model = load_model(device="auto")
result = model.predict_tif("image.tif", "new_results", vectors=False)
```

Git LFS stores the repository weights. If you downloaded a pointer or GitHub source ZIP,
the loader automatically fetches the real file from the release and checks SHA256.
Direct download: [pan_2_5m_model_weights.pth](https://github.com/vansharora665/Urban-Atlas-Final-Models/releases/download/v1.0.0/pan_2_5m_model_weights.pth).
Only load checkpoints and Python code from sources you trust: the original PAN teacher
component uses scikit-learn/joblib serialization, as in the retained project.

## Results And Limits

Weak-reference agreement is 92.13% on partial water/urban/vegetation references. Independent overall five-class accuracy is unavailable, so `overall_accuracy` is null. The SegFormer road head reached 28.19% IoU on partial weak road references before geometry filtering, not on a blind geographic test.

Class IDs: 0 water, 1 vehicle roads, 2 built-up, 3 barren/empty land, 4 vegetation.
Unknown/cloud/no-data: 255. Spatial cleanup is a mapping derivative, not ground truth.
Do not claim navigation-ready graphs or guarantees on arbitrary images.

## Attribution And Rights

Vansh Arora (23B0683), Department of Civil Engineering, IIT Bombay.
Guide: Prof. Gopal Patil. See the repository NOTICE.md and retained upstream notices.
CoANet source is from mj129/CoANet, revision b680c4f2b68e06a5ada03c63441a179fd3244490.
Its GPL-3.0 text and explicit non-commercial-research README notice are included.
This academic bundle is not a commercially cleared product.
