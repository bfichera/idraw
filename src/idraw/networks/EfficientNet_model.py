# I-DRAW
# Copyright (C) 2026  UChicago Argonne, LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np
import torch
from typing import TypeVar
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import torchvision.transforms.functional as F
import logging

from sklearn.preprocessing import StandardScaler

from ..utils.window import WindowMesh

DType = TypeVar("DType", bound=np.generic)

_logger = logging.getLogger(__file__)


class EfficientEncoder:

    def __init__(
            self, window_ttcf: WindowMesh,
            img_stk: np.ndarray[tuple[int, int, int],
                                np.dtype[DType]]) -> None:
        """
        The encoder needs the window_ttcf to perform the feature embeddings,
        and also requires the shape of the image stack to generate the upscaler
        function that is used to convert the small array of feature vectors
        into an overlay for the whole image stack.
        """
        self.window_ttcf = window_ttcf
        self.image_shape = img_stk.shape

        # Loads the best available weights
        weights = EfficientNet_B2_Weights.DEFAULT
        self.model = efficientnet_b2(weights=weights)

        self.img_size = weights.transforms().crop_size
        self.norm_mean = weights.transforms().mean
        self.norm_std = weights.transforms().std

    def _preprocess_numpy_batch(
        self, np_batch: np.ndarray[tuple[int, int, int], np.dtype[DType]]
    ) -> torch.Tensor:
        """
        Normalizes and prepares a 3D NumPy array (a batch of intensity fields)
        for a 3-channel model.
        """
        processed_tensors = []
        for single_np_array in np_batch:
            single_np_array = single_np_array.astype(np.float32)

            min_val, max_val = single_np_array.min(), single_np_array.max()
            if max_val > min_val:
                single_np_array = (single_np_array - min_val) / (max_val -
                                                                 min_val)
            else:
                single_np_array = np.zeros_like(single_np_array)

            tensor = torch.from_numpy(single_np_array)
            tensor = F.resize(tensor.unsqueeze(0),
                              size=self.img_size,
                              antialias=True)
            tensor = tensor.expand(3, -1, -1)
            tensor = F.normalize(tensor,
                                 mean=self.norm_mean,
                                 std=self.norm_std)
            processed_tensors.append(tensor)

        return torch.stack(processed_tensors, dim=0)

    def extract_embedding(self,
                          full_output: bool = False,
                          preprocess: bool = True):
        """
        If full_output is True, then output the upscaled feature map too
        """

        # Prepare pre-trained model
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        _logger.info(f"Using device: {device}")

        self.model.to(device)
        self.model.eval()

        # Strip away classifier layer so we just get feature vectors
        self.model.classifier = torch.nn.Identity()

        # Reformate data to have 3 channels so that EfficientNet can properly
        # process the data
        self.window_ttcf()
        A, B, C, D, E = self.window_ttcf._window_ttcf.shape
        array = self.window_ttcf._window_ttcf
        reshaped_array = array.reshape(-1, D, E)
        if not preprocess:
            net_input = torch.tensor(reshaped_array)
            net_input = net_input.unsqueeze(1)
            net_input = net_input.expand(-1, 3, -1, -1).type(torch.float32)
        else:
            net_input = self._preprocess_numpy_batch(reshaped_array)

        # Generate embeddings
        # Batching
        batch_size = 512
        all_feature_vectors = []

        with torch.no_grad():
            _logger.info(f"Processing {len(net_input)} images"
                         f" in batches of {batch_size}...")

            for i in range(0, len(net_input), batch_size):
                # Define the end of the batch
                end = i + batch_size

                # Critically, we only move the small batch to the GPU
                batch = net_input[i:end]
                batch_gpu = batch.to(device)
                output_embeddings = self.model(batch_gpu)
                all_feature_vectors.append(output_embeddings.cpu())

        feature_vectors_tensor = torch.cat(all_feature_vectors, dim=0)
        feature_vectors = feature_vectors_tensor.numpy()

        _logger.info("Scaling features...")
        scaler = StandardScaler()
        scaled_feature_vecs = scaler.fit_transform(feature_vectors)
        del net_input

        scaled_feature_vecs = scaled_feature_vecs.reshape(
            A, B, C, scaled_feature_vecs.shape[1])

        def upscaler(arr):
            # Assumes arr is shape (A,B,C,L)
            rep_arr = np.repeat(np.repeat(np.repeat(
                arr, self.window_ttcf._step_shape[0], axis=0),
                                          self.window_ttcf._step_shape[1],
                                          axis=1),
                                self.window_ttcf._step_shape[2],
                                axis=2)

            _, window_h, window_w = self.window_ttcf._window_shape[:3]

            pad_h_top = window_h // 2
            pad_w_left = window_w // 2

            pad_h_bottom = max(
                0, self.image_shape[1] - (rep_arr.shape[1] + pad_h_top))
            pad_w_right = max(
                0, self.image_shape[2] - (rep_arr.shape[2] + pad_w_left))

            pad_width = ((0,
                          np.maximum(0,
                                     self.image_shape[0] - rep_arr.shape[0])),
                         (pad_h_top, pad_h_bottom), (pad_w_left,
                                                     pad_w_right), (0, 0))

            padded_arr = np.pad(rep_arr,
                                pad_width,
                                mode='constant',
                                constant_values=0)
            final_arr = padded_arr[:self.image_shape[0], :self.
                                   image_shape[1], :self.image_shape[2], :]
            return final_arr

        if full_output:
            return upscaler(scaled_feature_vecs), scaled_feature_vecs, upscaler
        return scaled_feature_vecs, upscaler
