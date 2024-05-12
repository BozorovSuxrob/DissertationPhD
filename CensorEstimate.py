import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF, StepFunction

from hdensity import KernelDensityEstimation


def validate_inputs(time, event):
    """
    Validates the input time and event arrays.

    Parameters:
        time (array-like): Array of time observations.
        event (array-like): Array of event indicators.

    Raises:
        ValueError: If time and event arrays have different lengths or contain invalid data.
    """
    time = np.asarray(time)
    event = np.asarray(event)

    if len(time) != len(event):
        raise ValueError("The 'time' and 'event' arrays must have the same length.")

    if (not np.all(np.isfinite(time))) and np.any(time < 0):
        raise ValueError("Time values must be non-negative and finite.")

    if not set(np.unique(event)).issubset({0, 1}):
        raise ValueError("Event indicators must be 0 (censored) or 1 (occurred).")


class CensorEstimate:
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

    def __init__(self, time, event, side='right', sorted=False):
        """
        Initializes the CensorEstimate object.

        Parameters:
            time (array-like): The observed time data.
            event (array-like): Event occurrence indicator (1 if occurred, 0 if censored).
            side (str, optional): Side to use for the ECDF calculation ('right' by default).
            sorted (bool, optional): Specifies whether the time data is pre-sorted.
        """
        validate_inputs(time, event)
        self.time = np.asarray(time)
        self.event = np.asarray(event)

        if not sorted:
            order = np.argsort(self.time)
            self.time, self.event = self.time[order], self.event[order]

        self.ecdf = ECDF(self.time, side=side)
        self.ecdf_x = self.ecdf(self.time)
        self.kde = KernelDensityEstimation(self.time)
        self.kde1 = KernelDensityEstimation(self.time, self.event)
        self.eps = 1e-6
        self.sorted = sorted
        self.side = side

    def update_data(self, additional_time, additional_event):
        """
        Updates the time and event data with additional observations.

        Parameters:
            additional_time (array-like): Additional time observations to add.
            additional_event (array-like): Additional event indicators to add.
        """
        validate_inputs(additional_time, additional_event)
        self.time = np.concatenate([self.time, np.asarray(additional_time)])
        self.event = np.concatenate([self.event, np.asarray(additional_event)])
        order = np.argsort(self.time)
        self.time, self.event = self.time[order], self.event[order]
        self.ecdf = ECDF(self.time, side=self.side)
        self.ecdf_x = self.ecdf(self.time)

    def get_summary_statistics(self):
        """
        Returns summary statistics of the censored data.

        Returns:
            dict: Summary statistics including mean, median, and event rate.
        """
        summary = {
            "mean_time": np.mean(self.time),
            "median_time": np.median(self.time),
            "event_rate": np.mean(self.event)
        }
        return summary

    # Additional methods as required...


class CumulativeHazard(CensorEstimate):
    """
    Extends CensorEstimate to compute the cumulative hazard function for right-censored data.
    The cumulative hazard function is a fundamental component in survival analysis, representing
    the cumulative risk of the event occurring over time.

    Inherits from CensorEstimate.

    Attributes:
        y (StepFunction): Step function representing the cumulative hazard over time.
    """

    def __init__(self, time, event, risk='Lambda', side='right', sorted=False):
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
        super().__init__(time, event, side, sorted)
        self.y = self.calculate_cumulative_hazard(risk)

    def calculate_cumulative_hazard(self, risk):
        """
        Calculates the cumulative hazard based on the specified risk function.

        Parameters:
            risk (str): Type of risk function ('Lambda' or 'Lambda1').

        Returns:
            StepFunction: Step function representing the cumulative hazard over time.

        Raises:
            ValueError: If an invalid risk function type is provided.
        """
        if risk == 'Lambda':
            y = np.cumsum(
                (self.ecdf(self.time + self.eps) - self.ecdf(self.time - self.eps)) / (1 - self.ecdf_x + self.eps))
        elif risk == 'Lambda1':
            ecdf1 = ECDF(np.where(self.event == 0, -np.inf, self.time), side=self.side)
            y = np.cumsum(
                (ecdf1(self.time + self.eps) - ecdf1(self.time - self.eps)) / (1 - self.ecdf_x + self.eps))
        else:
            raise ValueError(f"Invalid risk function '{risk}'. Available options: 'Lambda', 'Lambda1'")

        return StepFunction(self.time, y, ival=0., sorted=self.sorted, side=self.side)

    def cdf(self, x):
        """
        Computes the cumulative hazard function at points x.

        Parameters:
            x (array-like): Points at which to evaluate the cumulative hazard.

        Returns:
            numpy.ndarray: Cumulative hazard values at the specified points x.
        """
        return self.y(x)
