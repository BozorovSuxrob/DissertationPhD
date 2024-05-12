import numpy as np
import scipy.stats as st


def informative_sample(distribution, sample_size, censoring_count):
    """
    Generate an informative sample from a given distribution.

    Args:
        distribution: The probability distribution to sample from.
        sample_size: The size of the sample to generate.
        censoring_count: The count of censored data points.

    Returns:
        T: An array of sampled data points.
        e: An array indicating whether each data point is censored (0) or observed (1).
    """
    theta = censoring_count / (sample_size - censoring_count)

    if theta != 0:
        while True:
            f_observed = distribution.isf(1 - st.uniform.rvs(size=sample_size))
            g_observed = distribution.isf(np.power(1 - st.uniform.rvs(size=sample_size), 1 / theta))
            censor_sample = np.ones(sample_size)
            censor_sample[g_observed < f_observed] = 0

            if sample_size - np.sum(censor_sample) == censoring_count:
                random_sample = np.minimum(f_observed, g_observed)
                return random_sample, censor_sample
    elif theta == 0:
        return distribution.isf(1 - st.uniform.rvs(size=sample_size)), np.ones(sample_size)

    else:
        raise ValueError("Unable to generate informative sample with specified censoring count.")
