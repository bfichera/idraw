# UMAP_RGB

**UMAP_RGB** is a Python package for spatially resolving features in imaging data (e.g., microscopy image stacks, X-ray scattering videos). It works by computing a local [Two-Time Correlation Function (2-TCF)](https://en.wikipedia.org/wiki/Photon_correlation_spectroscopy) for each spatial window of an image stack, passing those correlation matrices through a pre-trained EfficientNet-B2 neural network to extract feature embeddings, and then using UMAP (or PCA) to project those embeddings into a 3-channel RGB color space. The result is a color-coded overlay in which spatially distinct dynamical behaviors appear as distinct colors.

**Authors:** Nolan Heffner (nolan.heffner2@gmail.com), Bryan Fichera (bfichera@anl.gov)

---

## Installation

**Python 3.10–3.12 is required.**

```bash
pip install .
```

For development:

```bash
pip install -e .
```

The example script also requires the `noise` package (not a declared dependency):

```bash
pip install noise
```

### Known issue: OpenMP/numba conflict

If you encounter segfaults, set the following environment variable before running:

```bash
export NUMBA_THREADING_LAYER=workqueue
```

---

## Workflow

```
Raw image stack (T x H x W)
         |
         v
1. WindowMesh       — tile the stack into overlapping spatiotemporal windows
         |
         v
2. 2-TCF            — compute a T x T autocorrelation matrix per window
         |
         v
3. EfficientEncoder — pass each 2-TCF through EfficientNet-B2 -> 1408-dim features
         |
         v
4. UMAP / PCA       — reduce 1408-dim features to 3-dim RGB per window
         |
         v
5. Visualize        — display with matplotlib or save a blended MP4 video
```

### Step 1 — Define sliding windows

```python
from UMAP_RGB.utils.window import WindowMesh

# img_stk: numpy array, shape (T, H, W)
window_shape = (T, patch_H, patch_W)   # size of each spatiotemporal window
step_shape   = (T, step_H,  step_W)    # stride between windows

windows = WindowMesh(img_stk, window_shape, step_shape)
```

`window_shape` and `step_shape` control the trade-off between spatial resolution and computation time. Smaller steps produce a finer feature map but require more windows.

A typical choice for a 2D time-series (where you want full temporal depth per window) is:

```python
window_shape = (len(img_stk), patch_H, patch_W)
step_shape   = (len(img_stk), step_H,  step_W)
```

### Step 2 & 3 — Extract feature embeddings

```python
from UMAP_RGB.networks.EfficientNet_model import EfficientEncoder

model = EfficientEncoder(windows, img_stk)
low_res_feature_map, upscaler = model.extract_embedding(full_output=False)
```

`extract_embedding` triggers 2-TCF computation (parallelized with `joblib`), feeds each T×T correlation matrix through EfficientNet-B2 (pretrained on ImageNet, classifier head removed), and returns:

- `low_res_feature_map` — shape `(n_t_windows, n_h_windows, n_w_windows, 1408)`
- `upscaler` — a closure that nearest-neighbor-upscales the low-res feature grid back to the original image dimensions

If a GPU is available it will be used automatically.

### Step 4 — Generate RGB colors

```python
from UMAP_RGB.utils.UMAP_RGB import UMAP

umap_obj = UMAP(low_res_feature_map, upscaler)
umap_obj.generate_rgb(sparsity_mult=20)
```

Key parameters for `generate_rgb`:

| Parameter | Default | Description |
|---|---|---|
| `umap_neighbors` | `15` | UMAP `n_neighbors` — controls local vs. global structure |
| `umap_min_dist` | `0.1` | UMAP `min_dist` — controls cluster compactness |
| `sparsity_mult` | `350` | Subsampling factor; UMAP is fit on at most `N / sparsity_mult` points (capped at 50,000). Smaller values are more accurate but slower. |
| `density_cutoff` | `0.0` | If > 0: percentage of data classified as "standard" (common). Rare/anomalous pixels are colored by UMAP; standard pixels receive a uniform color. Uses IsolationForest internally. |
| `iso_n_estimators` | `100` | Number of trees in the IsolationForest (only used when `density_cutoff > 0`). |

After `generate_rgb`:

- `umap_obj.rgb` — full-resolution RGB array, shape `(T, H, W, 3)`, values in [0, 1]
- `umap_obj.low_res_rgb` — low-resolution RGB at the window grid, shape `(n_t, n_h, n_w, 3)`

#### PCA alternative

If UMAP is too slow, a drop-in PCA replacement is available:

```python
from UMAP_RGB.utils.PCA_RGB import PCA

pca_obj = PCA(low_res_feature_map, upscaler)
pca_obj.generate_rgb(sparsity_mult=20)
```

The interface is identical to `UMAP`; `density_cutoff` and `sparsity_mult` work the same way. `umap_neighbors` and `umap_min_dist` are not applicable.

### Step 5 — Visualize

**Static frame:**

```python
import matplotlib.pyplot as plt
plt.imshow(umap_obj.rgb[0])
plt.show()
```

**All frames as a numpy array:**

`umap_obj.rgb` is a standard `(T, H, W, 3)` float64 array and can be passed to any image or video library directly.

---

## Complete example

See [`examples/umap_rgb_example.py`](examples/umap_rgb_example.py) for a self-contained script that fabricates synthetic data (alternating simplex noise frames) and runs the full pipeline.

```bash
pip install noise   # required for the example only
python examples/umap_rgb_example.py
```

---

## API reference

### `WindowMesh(data, window_shape, step_shape, window_processor=None)`

Wraps `numpy.lib.stride_tricks.sliding_window_view` and manages 2-TCF computation.

- `data` — array of shape `(T, H, W)`
- `window_shape` — `(T_win, H_win, W_win)`
- `step_shape` — `(T_step, H_step, W_step)`
- `window_processor` — optional callable to replace the default 2-TCF computation
- `.windows` — the raw strided view, shape `(n_t, n_h, n_w, T_win, H_win, W_win)`
- `.window_ttcf` — computed on first `__call__`, shape `(n_t, n_h, n_w, T_win, T_win)`

### `EfficientEncoder(window_ttcf, img_stk)`

Loads EfficientNet-B2 with default ImageNet weights. Inference runs on GPU if available.

- `.extract_embedding(full_output=False, preprocess=True)` — triggers 2-TCF computation, runs inference in batches of 512, applies `StandardScaler` to features. Returns `(low_res_feature_map, upscaler)`. If `full_output=True`, returns `(upscaled_map, low_res_feature_map, upscaler)`.

### `UMAP(feature_map, upscaler=None)`

- `.generate_rgb(umap_neighbors, umap_min_dist, density_cutoff, sparsity_mult, iso_n_estimators)` — runs UMAP and assigns RGB colors (see parameters above)
- `.rgb` — `(T, H, W, 3)` float array after `generate_rgb`; `None` before
- `.low_res_rgb` — `(n_t, n_h, n_w, 3)` float array after `generate_rgb`; `None` before

### `PCA(feature_map, upscaler=None)`

Drop-in alternative to `UMAP` using `sklearn.decomposition.PCA`.

- `.generate_rgb(density_cutoff, sparsity_mult, iso_n_estimators)` — same logic as `UMAP.generate_rgb` but uses PCA instead of UMAP
- `.rgb` and `.low_res_rgb` — same semantics as `UMAP`

---

## Project structure

```
UMAP_RGB/
├── pyproject.toml
├── examples/
│   ├── umap_rgb_example.py        # Minimal end-to-end example
│   └── umap_3d_video.mp4          # Example output video
└── src/
    └── UMAP_RGB/
        ├── networks/
        │   └── EfficientNet_model.py   # EfficientNet-B2 feature extractor
        └── utils/
            ├── UMAP_RGB.py             # UMAP class
            ├── PCA_RGB.py              # PCA alternative
            ├── window.py               # WindowMesh + sliding window utilities
            └── ttcf.py                 # Two-Time Correlation Function (autocorr)
```
