import numpy as np
from typing import TypeVar, Any
from joblib import Parallel, delayed

DType = TypeVar("DType", bound=np.generic)


def autocorr(waterfall: np.ndarray[Any, np.dtype[DType]],
             n_proc: int = 8,
             verbose: int = 0) -> np.ndarray[Any, np.dtype[DType]]:
    # TODO
    # Check return type
    """
    Calculates the autocorrelation of a matrix with itself. When given an image
    stack, it will calculate a 2-index correlation function Corr, where
    Corr(t_1, t_2) gives the correlation between the images at frames t_1 and
    t_2 in the given image stack. autocorr(mat) should be equivalent to
    corr(mat, mat), but I haven't checked.

    Args:
        wfl_1, wfl_2 (np.ndarray): Matrices for comparison
        n_proc (int, optional): Number of threads for parallel processing.
                                Default is 8.
        verbose (int, optional): Controls verbose output. Accepts nonnegative
                                 integers. See documentation for
                                 joblib.Parallel for details. Default is 0.

    Returns:
        np.float32(? need to fact check): Pearson correlation coefficient of
                                          the given matrices.
    """

    I_bar_t = np.mean(waterfall, axis=1)
    I_bar_t1_I_bar_t2 = np.outer(I_bar_t, I_bar_t)

    # TODO
    # Change to specify CORRECT AXES OVER WHICH TO MEAN - should be vector
    # which outer product gives array, no?
    variance_squared = np.mean(np.square(waterfall)) - np.mean(waterfall)**2

    n_frames = waterfall.shape[0]

    ab = np.zeros((n_frames, n_frames))

    def n_in_m_bins(n, m):
        d, r = divmod(n, m)
        res = [[m * j + i for j in range(d)] for i in range(m)]
        for i in range(r):
            res[i].append(res[i][-1] + m)
        return res

    dts_array = n_in_m_bins(n_frames, n_proc)

    def calc_row(waterfall, dts):
        for j in dts:
            ab[j:, j] = np.dot(
                waterfall[j:],
                waterfall[j],
            ) / waterfall.shape[1]
            ab[j, j:] = np.transpose(ab[j:, j])

    Parallel(n_jobs=n_proc, verbose=verbose,
             backend='threading')(delayed(calc_row)(waterfall, dts_array[j])
                                  for j in range(0, n_proc))

    np.fill_diagonal(ab, 0)
    autocorrelation = (ab - I_bar_t1_I_bar_t2) / variance_squared
    np.fill_diagonal(autocorrelation, 0)
    return autocorrelation
