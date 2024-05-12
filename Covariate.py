import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF, StepFunction

import Kernels


def validate_inputs(time, event, covariates):
    """
    Validates the input time and event arrays.

    Parameters:
        time (array-like): Array of time observations.
        event (array-like): Array of event indicators.

    Raises:
        ValueError: If time and event arrays have different lengths or contain invalid data.
        :param event:
        :param time:
        :param covariates:
    """
    time = np.asarray(time)
    event = np.asarray(event)
    covariates = np.asarray(covariates)

    if len(time) != len(event):
        raise ValueError("The 'time' and 'event' arrays must have the same length.")

    if len(time) != len(covariates):
        raise ValueError("The 'time' and 'covariates' arrays must have the same length.")

    if (not np.all(np.isfinite(time))) and np.any(time < 0):
        raise ValueError("Time values must be non-negative and finite.")

    if not set(np.unique(event)).issubset({0, 1}):
        raise ValueError("Event indicators must be 0 (censored) or 1 (occurred).")


class CovariateCensorEstimateFather:
    """
    Estimates empirical cumulative distribution functions (ECDFs) for right-censored data.
    Right-censoring occurs when the value of an observation is only known to be above a certain threshold,
    common in survival analysis and reliability engineering.

    Attributes:
        time (numpy.ndarray): Array of observed times.
        event (numpy.ndarray): Array indicating whether the event of interest occurred (1) or was censored (0).
        ecdf (ECDF): Empirical cumulative distribution function of observed times.
        ecdf_x (numpy.ndarray): ECDF evaluated at the observed times, used in CDF calculations.
        eps (float): A small constant to prevent division by zero in further calculations.
        sorted (bool): Flag indicating if the input data is sorted.
        side (str): Specifies the side for the ECDF ('right' by default).
    """

    def __init__(self, time, event, covariates, x0, bandwidth, kernel, side, sorted):
        """
        Initializes the CensorEstimate object.

        Parameters:
            time (array-like): The observed time data.
            event (array-like): Event occurrence indicator (1 if occurred, 0 if censored).
            side (str, optional): Side to use for the ECDF calculation ('right' by default).
            sorted (bool, optional): Specifies whether the time data is pre-sorted.
        """
        validate_inputs(time, event, covariates)
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

        self.time = np.asarray(time)
        self.event = np.asarray(event)
        self.covariates = np.asarray(covariates)
        self.x0 = np.asarray(x0)
        self.side = side
        self.sorted = sorted
        self.time1 = self.time[self.event == 1]
        self.time0 = self.time[self.event == 0]
        self.bandwidth = np.asarray(bandwidth)
        self.n = len(self.time)
        self.ecdf_cov = ECDF(self.covariates)

        if not sorted:
            order = np.argsort(self.time)
            self.time, self.event, covariates = self.time[order], self.event[order], self.covariates[order]


class CovariateCensorEstimate(CovariateCensorEstimateFather):
    """
    Extends CensorEstimate to compute the cumulative hazard function for right-censored data.
    The cumulative hazard function is a fundamental component in survival analysis, representing
    the cumulative risk of the event occurring over time.

    Inherits from CensorEstimate.

    Attributes:
        y (StepFunction): Step function representing the cumulative hazard over time.
    """

    def __init__(self, time, event, covariates, x0, bandwidth=1, kernels='gaussian', side='right', sorted=False):
        """
        Initializes the CumulativeHazard object.

        Parameters:
            time (array-like): The observed time data.
            event (array-like): Event occurrence indicator (1 if occurred, 0 if censored).
            risk (str, optional): Type of risk function to use ('Lambda' or 'Lambda1'). 'Lambda'
                                  uses the standard ECDF, while 'Lambda1' uses an alternative
                                  ECDF considering censored times as -infinity.
            side (str, optional): Side to use for the ECDF calculation ('right' by default).
            sorted (bool, optional): Specifies whether the time data is pre-sorted.

        Raises:
            ValueError: If an invalid risk function type is provided.
        """
        super().__init__(time, event, covariates, x0, bandwidth, kernels, side, sorted)
        y1 = np.cumsum(self.pdf(self.time0), self.x0, self.bandwidth)
        self.func1 = StepFunction(self.time0, y1, ival=0., sorted=self.sorted, side=self.side)
        y2 = np.cumsum(self.pdf(self.time1), self.x0, self.bandwidth)
        self.func2 = StepFunction(self.time1, y2, ival=0., sorted=self.sorted, side=self.side)

    def calculate_cumulative_hazard(self, x):
        return lambda x: (self.func1(x) + self.func2(x))/self.n

    def covariate_estimate(self, x):
        y = np.cumsum(
            (self.ecdf(self.time + self.eps) - self.ecdf(self.time - self.eps)) / (1 - self.ecdf_x + self.e))
        return self.func1(x) + self.func2(x)


