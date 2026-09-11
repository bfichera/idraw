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
import os
import numpy as np

import numba

# Clustering / Data processing
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA as skPCA


class PCA:
    """
    Main class of the PCA_RGB process. You can get a grasp of what it handles
    from __init__(). This class accepts a feature_map (output of neural network
    step), calculates the RGB array, and contains the upscaler required by
    plotting functions to properly overlay the RGB with the image stack.
    """

    def __init__(self, feature_map: np.ndarray, upscaler=None) -> None:
        """
        Expecting feature_map output from ACAE generate_embeddings function.
        """
        self.feature_map = feature_map
        self.shape = feature_map.shape
        self._rgb = None
        self._low_res_rgb = None
        self.upscaler = upscaler

    @property
    def rgb(self) -> np.ndarray | None:
        return self._rgb

    @property
    def low_res_rgb(self) -> np.ndarray | None:
        return self._low_res_rgb

    def generate_rgb(
        self,
        density_cutoff: float = 0.0,
        sparsity_mult: int = 350,  # Note: for 3d data reduction in umap
        # quality goes as cbrt so this only fuzzies
        # it by ~7 px
        iso_n_estimators: int = 100,
    ) -> None:

        print("Preparing and scaling data...")
        depth, height, width, latent_dim = self.feature_map.shape
        pixels_as_vectors = self.feature_map.reshape(-1, latent_dim)
        scaled_features = StandardScaler().fit_transform(pixels_as_vectors)
        coloring_mode = 'null'

        if density_cutoff > 0.0 and density_cutoff < 100.0:

            print(f"Pruning densest {density_cutoff}%"
                  " of data using Isolation Forest.")
            coloring_mode = 'pruned'

            # fit_predict returns 1 for inliers (standard)
            # and -1 for outliers (unique)
            isf = IsolationForest(n_estimators=iso_n_estimators,
                                  n_jobs=-1,
                                  contamination=(100.0 - density_cutoff) /
                                  100.0,
                                  random_state=42)
            labels = isf.fit_predict(scaled_features)

            is_unique = labels == -1
            is_standard = labels == 1

            unique_vectors = scaled_features[is_unique]
            standard_vector = scaled_features[is_standard][0:1]

            # Handle the edge case where no unique vectors are found
            if len(unique_vectors) == 0:
                print(
                    "Warning: No unique vectors found at this cutoff. Showing"
                    " all points.")
                coloring_mode = 'standard'
                pca_fit_data = scaled_features
            else:
                pca_fit_data = np.vstack([unique_vectors, standard_vector])

        else:

            print("Using standard subsampling for pca fitting.")
            num_fit_pixels = scaled_features.shape[0]
            subsample_size = max(min(50000, num_fit_pixels),
                                 num_fit_pixels // sparsity_mult)
            subsample_indices = np.random.choice(num_fit_pixels,
                                                 subsample_size,
                                                 replace=False)
            pca_fit_data = scaled_features[subsample_indices]

        print(f"Fitting pca on a subsample of {len(pca_fit_data)} vectors...")
        try:
            n_threads = os.cpu_count() - 1 or 1
            numba.set_num_threads(n_threads)
        except Exception as e:
            print(f"Could not set Numba threads: {e}")

        mapper = skPCA(n_components=3)
        mapper.fit(pca_fit_data)

        print("Generating color map...")
        if coloring_mode == 'pruned':
            pruned_embedding = mapper.transform(pca_fit_data)
            pca_colors = MinMaxScaler().fit_transform(pruned_embedding)
            # TODO
            # Does this raise an index error when the mask is applied?
            rgb = np.zeros((height * width, 3))
            standard_color = pca_colors[-1]
            unique_colors = pca_colors[:-1]
            rgb[is_standard] = standard_color
            rgb[is_unique] = unique_colors
        else:  # coloring_mode == 'standard'
            full_embedding = mapper.transform(scaled_features)
            rgb = MinMaxScaler().fit_transform(full_embedding)

        d, h, w = self.feature_map.shape[:3]

        # Correctly reshape the flat (N, 3) color array into
        # a 4D image (D, H, W, 3)
        low_res_rgb_image = rgb.reshape(d, h, w, 3)

        self._low_res_rgb = low_res_rgb_image
        if self.upscaler is not None:
            print("Upscaling RGB map...")
            low_res_rgb_image_np = low_res_rgb_image.copy()
            final_rgb_pixels = self.upscaler(low_res_rgb_image_np)
            self._rgb = final_rgb_pixels
        else:
            self._rgb = low_res_rgb_image
