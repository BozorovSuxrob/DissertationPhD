import numpy as np
from statsmodels.distributions.empirical_distribution import StepFunction

import Kernels
from CensorEstimate import CensorEstimate, CumulativeHazard


class ACLEstimate(CensorEstimate):
    """
    Extends CensorEstimate to estimate the cumulative distribution function (CDF) and
    the probability density function (PDF) for right-censored data. Utilizes adjusted
    CDFs and kernel density estimation for accurate estimations in the presence of
    censored data.

    Inherits from CensorEstimate.
    """

    def __init__(self, time, event, side='right', sorted=False):
        """
        Initializes the ACLEstimate object, inheriting from CensorEstimate.

        Parameters:
            time (array-like): The observed time data.
            event (array-like): Event occurrence indicator (1 if occurred, 0 if censored).
            side (str, optional): Side to use for the ECDF calculation ('right' by default).
            sorted (bool, optional): Specifies whether the time data is pre-sorted.
        """
        super().__init__(time, event, side=side, sorted=sorted)

    def cdf(self, x, dis='F'):
        """
        Computes the CDF at points x for the specified distribution function.

        Parameters:
            x (array-like): Points at which to evaluate the CDF.
            dis (str): Type of distribution function to use ('F' or 'G').

        Returns:
            numpy.ndarray: CDF values at the specified points x.

        Raises:
            ValueError: If an invalid distribution function type is provided.
        """
        x = np.asarray(x)

        # Compute the CDF based on the specified distribution function
        if dis == 'F':
            # F distribution, considering the mean event occurrence
            y = 1 - (1 - self.ecdf_x) ** np.mean(self.event)
        elif dis == 'G':
            # G distribution, considering the mean event non-occurrence
            y = 1 - (1 - self.ecdf_x) ** (1 - np.mean(self.event))
        else:
            raise ValueError(f"Invalid distribution function: {dis}. Available options: F, G.")

        # Create a step function for the calculated CDF values
        f = StepFunction(self.time, y, ival=0., sorted=self.sorted, side=self.side)
        return f(x)

    def pdf(self, x, bandwidth=1, kernel='gaussian'):
        """
        Computes the PDF at points x for right-censored data using kernel density estimation.

        Parameters:
            x (array-like): Points at which to evaluate the PDF.
            bandwidth (float, optional): Bandwidth parameter for kernel density estimation, default is 1.
            kernel (str, optional): Type of kernel to use in the density estimation, default is 'gaussian'.

        Returns:
            numpy.ndarray: PDF values at the specified points x.

        Notes:
            The PDF is adjusted for right-censored data, taking into account the CDF values.
        """
        x = np.asarray(x)

        h1 = self.kde1.hpdf(x, bandwidth, kernel) / (1 - self.cdf(x, dis='G') + self.eps)
        return h1


class RREstimate(CensorEstimate):
    """
    Extends CensorEstimate to estimate the cumulative distribution function (CDF) and
    the probability density function (PDF) for right-censored data. Utilizes adjusted
    CDFs and kernel density estimation for accurate estimations in the presence of
    censored data.

    Inherits from CensorEstimate.
    """

    def __init__(self, time, event, side='right', sorted=False):
        """
        Initializes the RREstimate object, inheriting from CensorEstimate.

        Parameters:
            time (array-like): The observed time data.
            event (array-like): Event occurrence indicator (1 if occurred, 0 if censored).
            side (str, optional): Side to use for the ECDF calculation ('right' by default).
            sorted (bool, optional): Specifies whether the time data is pre-sorted.
        """
        super().__init__(time, event, side=side, sorted=sorted)
        lam = CumulativeHazard(self.time, self.event, risk='Lambda', side=self.side, sorted=self.sorted).cdf(self.time)
        lam1 = CumulativeHazard(self.time, self.event, risk='Lambda1', side=self.side, sorted=self.sorted).cdf(
            self.time)
        self.l2 = lam1 / (lam + self.eps)

    def cdf(self, x, dis='F'):
        """
        Computes the CDF at points x for the specified distribution function.

        Parameters:
            x (array-like): Points at which to evaluate the CDF.
            dis (str): Type of distribution function to use ('F' or 'G').

        Returns:
            numpy.ndarray: CDF values at the specified points x.

        Raises:
            ValueError: If an invalid distribution function type is provided.
        """
        x = np.asarray(x)

        if dis == 'F':
            y = 1 - (1 - self.ecdf_x) ** self.l2
        elif dis == 'G':
            y = 1 - (1 - self.ecdf_x) ** (1 - self.l2)
        else:
            raise ValueError(f"Invalid distribution function: {dis}. Available options: F, G.")

        f = StepFunction(self.time, y, ival=0., sorted=self.sorted, side=self.side)
        return f(x)

    def pdf(self, x, bandwidth=1, kernel='gaussian'):
        """
        Computes the PDF at points x for right-censored data using kernel density estimation.

        Parameters:
            x (array-like): Points at which to evaluate the PDF.
            bandwidth (float, optional): Bandwidth parameter for kernel density estimation, default is 1.
            kernel (str, optional): Type of kernel to use in the density estimation, default is 'gaussian'.

        Returns:
            numpy.ndarray: PDF values at the specified points x.

        Notes:
            The PDF is adjusted for right-censored data, taking into account the CDF values.
        """
        x = np.asarray(x)

        h1 = self.kde1.hpdf(x, bandwidth, kernel) / (1 - self.cdf(x, dis='G') + self.eps)
        return h1


class PREstimate(CensorEstimate):
    """
    Extends CensorEstimate to estimate the cumulative distribution function (CDF) and
    the probability density function (PDF) for right-censored data. Utilizes adjusted
    CDFs and kernel density estimation for accurate estimations in the presence of
    censored data.

    Inherits from CensorEstimate.
    """

    def __init__(self, time, event, side='right', sorted=False):
        """
        Initializes the ACLEstimate object, inheriting from CensorEstimate.

        Parameters:
            time (array-like): The observed time data.
            event (array-like): Event occurrence indicator (1 if occurred, 0 if censored).
            side (str, optional): Side to use for the ECDF calculation ('right' by default).
            sorted (bool, optional): Specifies whether the time data is pre-sorted.
        """
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
        super().__init__(time, event, side=side, sorted=sorted)

    def cdf(self, x, dis='F', bandwidth=1, kernel='gaussian'):
        """
        Computes the CDF at points x for the specified distribution function.

        Parameters:
            x (array-like): Points at which to evaluate the CDF.
            dis (str): Type of distribution function to use ('F' or 'G').

        Returns:
            numpy.ndarray: CDF values at the specified points x.

        Raises:
            ValueError: If an invalid distribution function type is provided.
            :param x:
            :param dis:
            :type bandwidth: object
            :param kernel:
        """
        x = np.asarray(x)
        p1 = self.kde1.hpdf(self.time, bandwidth=bandwidth, kernel=kernel)
        p2 = self.kde.hpdf(self.time, bandwidth=bandwidth, kernel=kernel)
        lam = CumulativeHazard(self.time, self.event, risk='Lambda', side=self.side, sorted=self.sorted)
        l2 = np.cumsum((lam.cdf(self.time + self.eps) - lam.cdf(self.time - self.eps)) * (p1 / (p2 + self.eps))) / (
                lam.cdf(self.time) + self.eps)
        # Compute the CDF based on the specified distribution function
        if dis == 'F':
            # F distribution, considering the mean event occurrence
            y = 1 - (1 - self.ecdf_x) ** l2
        elif dis == 'G':
            # G distribution, considering the mean event non-occurrence
            y = 1 - (1 - self.ecdf_x) ** (1 - l2)
        else:
            raise ValueError(f"Invalid distribution function: {dis}. Available options: F, G.")

        # Create a step function for the calculated CDF values
        f = StepFunction(self.time, y, ival=0., sorted=self.sorted, side=self.side)
        return f(x)

    def pdf(self, x, bandwidth=1, kernel='gaussian'):
        """
        Computes the PDF at points x for right-censored data using kernel density estimation.

        Parameters:
            x (array-like): Points at which to evaluate the PDF.
            bandwidth (float, optional): Bandwidth parameter for kernel density estimation, default is 1.
            kernel (str, optional): Type of kernel to use in the density estimation, default is 'gaussian'.

        Returns:
            numpy.ndarray: PDF values at the specified points x.

        Notes:
            The PDF is adjusted for right-censored data, taking into account the CDF values.
        """
        x = np.asarray(x)

        h1 = self.kde1.hpdf(x, bandwidth, kernel) / (1 - self.cdf(x, dis='G') + self.eps)
        return h1
