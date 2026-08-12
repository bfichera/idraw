# I-DRAW

Image-set Dimensionality Reduction via Autocorrelation of Windows

`idraw` uses a sliding window of two-frame correlation functions (2-FCF) as input features to a pretrained EfficientNet-B2 encoder, then reduces the resulting high-dimensional feature map to an RGB overlay using PCA or UMAP. The output can be used to identify spatially and temporally distinct regions in a time-resolved image stack.

## Installation

```
pip install .
```

Requires Python 3.10–3.12.

## Usage

```python
from idraw.utils.window import WindowMesh
from idraw.networks.EfficientNet_model import EfficientEncoder
from idraw.utils.UMAP_RGB import UMAP
from idraw.utils.PCA_RGB import PCA

# img_stk: np.ndarray of shape (n_frames, height, width)

# 1. Build a sliding window mesh and compute 2-FCFs
window_shape = (n_frames, 24, 24)
step_shape   = (n_frames,  4,  4)
windows = WindowMesh(img_stk, window_shape, step_shape)

# 2. Extract EfficientNet feature embeddings
model = EfficientEncoder(windows, img_stk)
low_res_feature_map, upscaler = model.extract_embedding()

# 3. Reduce to RGB using UMAP (or swap in PCA)
mapper = UMAP(low_res_feature_map, upscaler)
mapper.generate_rgb()

rgb = mapper.rgb            # full-resolution RGB overlay, shape (n_frames, H, W, 3)
low_res = mapper.low_res_rgb  # window-resolution RGB
```

## Example

A complete end-to-end Snakemake workflow is provided in `example/`. See `example/README.md` for instructions.

## Authors

- Nolan Heffner (`nolan.heffner2@gmail.com`)
- Bryan Fichera (`bfichera@anl.gov`)

## Cite

<!-- TODO -->
<!-- Add the citation here -->
If you use I-DRAW in a publication, please cite the following paper: [paper link]
