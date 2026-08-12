# Copyright (C) 2026 Bryan Fichera <bfichera@anl.gov>
# Copyright (C) 2026 Nolan Heffner <nheffne@umich.edu>
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
# along with this program.  If not, see <https://gnu.org>.
from joblib import Parallel, delayed
from typing import Callable, Any, TypeVar
import numpy as np
from tqdm import tqdm

from .ttcf import autocorr

DType = TypeVar("DType", bound=np.generic)


def sliding_window_view(
        data: np.ndarray[Any, np.dtype[DType]], window_shape: tuple[int, int,
                                                                    int],
        step_shape: tuple[int, int, int]) -> np.ndarray[Any, np.dtype[DType]]:
    """
    Creates a sliding window view from a multidimensional data array. Requires
    NumPy version 1.22 or newer.

    Args:
        data (np.ndarray): The input array of arbitrary dimensions.
        window_shape (tuple or list): The size of the sliding window.
        step_shape (tuple or list): The step size for the window.

    Returns:
        np.ndarray: A sliding window view of the data.
    """

    view = np.lib.stride_tricks.sliding_window_view(data,
                                                    window_shape=window_shape)

    slices = tuple(slice(None, None, s) for s in step_shape)

    return view[slices]


class WindowMesh:
    """
    A class to standardize the structure of a sliding window array in UMAP_RGB.
    Mostly a utility wrapper for numpy's sliding_window_view function + handles
    application of window-wise operations, e.g. windowed 2-TCF.
    """

    def __init__(self,
                 data: np.ndarray[tuple[int, int, int], Any],
                 window_shape: tuple[int, int, int],
                 step_shape: tuple[int, int, int],
                 window_processor: Callable = None) -> None:

        self.data = data
        self._window_shape = window_shape
        self._step_shape = step_shape

        # Calculate sliding window
        print("Calculating sliding window view (caution! may be slow...)")
        self._windows = sliding_window_view(
            np.copy(data),
            window_shape,
            step_shape,
        )
        print("Sliding window view calculation finished.")

        # Define default window processor
        self._window_processor = window_processor if (
            window_processor is not None) else self.calc_ttcf
        self._window_ttcf = None

    def calc_ttcf(self, n_proc: int = -1) -> None:
        n_frames = self._window_shape[0]
        self._window_ttcf = np.zeros(
            tuple(list(self._windows.shape[:3]) + [n_frames, n_frames]))

        def wfl(x):
            return x.reshape((x.shape[0], -1))

        def add_to_window_ttcf(triplet):
            i, j, k = triplet
            target_window = self._windows[i, j, k]
            waterfall = wfl(target_window)
            tt = autocorr(waterfall, n_proc=1)
            return i, j, k, tt

        i_values = np.arange(0, self._windows.shape[0])
        j_values = np.arange(0, self._windows.shape[1])
        k_values = np.arange(0, self._windows.shape[2])
        I, J, K = np.meshgrid(i_values, j_values, k_values, indexing='ij')
        triplets = np.stack((I.ravel(), J.ravel(), K.ravel()), axis=-1)

        results = Parallel(n_jobs=n_proc, backend='loky')(
            delayed(add_to_window_ttcf)(triplet)
            for triplet in tqdm(triplets, desc="Processing 2-TCF"))

        print("Consolidating results...")
        for i, j, k, tt_data in tqdm(results):
            self._window_ttcf[i, j, k, :, :] = tt_data

    def __call__(self):
        if self._window_ttcf is None:
            self._window_processor()
        return self._window_ttcf

    @property
    def windows(self) -> np.ndarray:
        return self._windows

    @property
    def window_ttcf(self) -> np.ndarray:
        return self._window_ttcf
