import logging
import argparse
from pathlib import Path
import os
import pickle

import numpy as np
import torch
from numba import config
from idraw.utils.window import WindowMesh
from idraw.networks.EfficientNet_model import EfficientEncoder
from idraw.utils.UMAP_RGB import UMAP


def _to_numpy(path: os.PathLike, dtype=np.float32):
    with open(path, 'rb') as fh:
        r = np.fromfile(fh, dtype=dtype)
    return r.reshape(-1, 256, 256)


def load(data_directory: os.PathLike,
         pattern: str,
         shape: tuple,
         dtype=np.float32):
    data = []
    for path in sorted(data_directory.glob(pattern)):
        s = _to_numpy(path, dtype)
        if s is not None:
            data.append(s)
            continue
    return np.array(data)


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=Path)
parser.add_argument('--output-path', type=Path)

cfg = parser.parse_args()
input_dir = cfg.input_dir
output_path = cfg.output_path

window_length = 24
window_stepsize_ratio = 6

logger.info(f"Numba thread layer: {config.THREADING_LAYER}")
logger.info(f"Numba threads: {config.NUMBA_NUM_THREADS}")
logger.info(f"PyTorch intra-op threads: {torch.get_num_threads()}")
logger.info(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")

img_stk = load(input_dir, 'test2_*.bin', (2, 256, 256))
mapper_in = img_stk[:, -1, :, :]
window_shape = (mapper_in.shape[0], window_length, window_length)
step_shape = (mapper_in.shape[0],
              window_length // window_stepsize_ratio,
              window_length // window_stepsize_ratio)
windows = WindowMesh(mapper_in, window_shape, step_shape)

model = EfficientEncoder(windows, mapper_in)
low_res_feature_map, upscaler = model.extract_embedding(
    full_output=False)

mapper = UMAP(low_res_feature_map, upscaler)
mapper.generate_rgb(sparsity_mult=20)

to_pkl = {}
to_pkl['img_stk'] = img_stk
to_pkl['mapper_in'] = mapper_in
to_pkl['window_shape'] = window_shape
to_pkl['step_shape'] = step_shape
to_pkl['window_ttcf'] = windows.window_ttcf
to_pkl['mapper_get_rgb_0'] = mapper.rgb[0]
to_pkl['mapper_rgb'] = mapper.rgb
to_pkl['mapper_low_res_rgb'] = mapper.low_res_rgb

with open(output_path, 'wb') as fh:
    pickle.dump(to_pkl, fh)
