import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF, StepFunction

import Kernels


class CovariateCensorEstimate:
    def __init__(self, time, event, covariates, bandwidth=1, kernel='gaussian', side='right', sorted=False):
        self.time = np.asarray(time)
        self.event = np.asarray(event)
        self.event0 = np.zeros_like(self.event)
        self.event0[self.event == 0] = 1
        self.covariates = np.asarray(covariates)
        self.bandwidth = np.asarray(bandwidth)
        self.kernels = kernel
        self.side = side
        self.sorted = sorted

        if not self.sorted:
            self.sorted = True
            order = np.argsort(self.time)
            self.time, self.event, self.covariates = self.time[order], self.event[order], self.covariates[order]

        self.ecdf_cov = ECDF(self.covariates)
        self.n = len(self.event)
        self.eps = 1e-6
        self.result = np.zeros_like(self.event)

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

        try:
            self.pdf = self.kernels[kernel.lower()]
        except KeyError:
            raise ValueError(f"Invalid kernel function: {kernel}. Available kernels: {self.kernels.keys()}")

        for i in self.covariates:
            func1 = StepFunction(self.time, np.cumsum(self.event * self.pdf(self.time, i, self.bandwidth) / self.n),
                                 ival=0., sorted=self.sorted, side=self.side)
            func0 = StepFunction(self.time, np.cumsum(self.event0 * self.pdf(self.time, i, self.bandwidth) / self.n),
                                 ival=0., sorted=self.sorted, side=self.side)
            func2 = StepFunction(self.time, np.cumsum(self.pdf(self.time, i, self.bandwidth) / self.n), ival=0.,
                                 sorted=self.sorted, side=self.side)

            a1 = (func0(self.time - self.eps) - func0(self.time + self.eps) + func1(self.time - self.eps) - func1(
                self.time + self.eps)) / (1 + self.eps - func2(self.time))
            a = self.ecdf_cov(i + self.eps) - self.ecdf_cov(i - self.eps)

            self.result = self.result + a * (1 - np.exp(np.cumsum(a1)))

        self.func4 = StepFunction(self.time, self.result, ival=0., sorted=self.sorted, side=self.side)

    def cdf(self, x):
        return self.func4(x)
