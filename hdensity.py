import numpy as np

import Kernels


class KernelDensityEstimation:
    def __init__(self, random_sample, censor_sample=None):
        """
        Initialize the KernelDensityEstimation with a random sample.

        Parameters:
        - random_sample (array-like): The random sample for kernel density estimation.
        - censor_sample (array-like): A binary array indicating which samples are censored.
                                      Defaults to None, meaning all samples are considered.
        """
        self.random_sample = np.asarray(random_sample)
        self.n = len(random_sample)
        self.kernels = {
            'cauchy': Kernels.cauchy_pdf,
            'cosine': Kernels.cosine_pdf,
            'epanechnikov': Kernels.epanechnikov_pdf,
            'gaussian': Kernels.gaussian_pdf,
            'laplace': Kernels.laplace_pdf,
            'logistic': Kernels.logistic_pdf,
            'quartic': Kernels.quartic_pdf,
            'sigmoid': Kernels.sigmoid_pdf,
            'triangular': Kernels.triangular_pdf,
            'tricube': Kernels.tricube_pdf,
            'triweight': Kernels.triweight_pdf,
            'uniform': Kernels.uniform_pdf,
        }

        if censor_sample is not None:
            self.censor_sample = np.asarray(censor_sample, dtype=int)
        else:
            self.censor_sample = np.ones_like(random_sample, dtype=int)

    def hpdf(self, x_values, bandwidth=1, kernel='gaussian'):
        """
        Compute the kernel density estimate at given points.

        Parameters:
        - x_values (array-like): The points at which to compute the kernel density estimate.
        - bandwidth (float): The bandwidth of the kernel.
        - kernel (str): The name of the kernel function to use.

        Returns:
        - array: The kernel density estimate at each point.
        """
        try:
            pdf = self.kernels[kernel.lower()]
        except KeyError:
            raise ValueError(f"Invalid kernel function: {kernel}. Available kernels: {self.kernels.keys()}")

        if bandwidth <= 0:
            raise ValueError("Bandwidth must be a positive value.")

        x_values = np.asarray(x_values)
        xx_values = x_values.reshape(-1, 1)

        u = (xx_values - self.random_sample) / bandwidth
        return_pdf = np.sum(self.censor_sample * pdf(u), axis=1) / (self.n * bandwidth)
        return return_pdf.reshape(x_values.shape)
