# All work
import numpy as np


def uniform_pdf(u, loc=0, scale=1):
    """
    Calculate the probability density function (PDF) of a uniform distribution
    over a specified interval.

    The uniform distribution PDF is constant within the interval [loc-scale, loc+scale]
    and zero outside. For an interval [loc-scale, loc+scale], the PDF is 1 / (2*scale)
    when loc-scale <= u <= loc+scale, and 0 otherwise.

    Parameters:
    - u (array-like): Input values.
    - loc (float, optional): Lower bound of the interval loc-scale. Default is -1.
    - scale (float, optional): Upper bound of the interval loc+scale. Default is 1.

    Returns:
    - array: PDF values corresponding to the input values.

    Raises:
    Raises:
    - ValueError: If `scale` is non-positive.
    - TypeError: If `loc` or `scale` is not a numeric type.
    """
    if not (isinstance(loc, (int, float)) and isinstance(scale, (int, float))):
        raise TypeError("Location (loc) and scale parameters must be numeric.")

    if scale <= 0:
        raise ValueError("Scale parameter must be positive.")

    u = np.asarray(u)
    transformed_u = (u - loc) / scale

    pdf_values = (1 / (2 * scale)) * ((transformed_u >= loc - scale) & (transformed_u <= loc + scale))
    return pdf_values


def triangular_pdf(u, loc=0, scale=1):
    """
    Calculate the probability density function (PDF) of a triangular distribution.

    The standard triangular distribution is defined on the interval [-1, 1] with
    a peak at 0. This function allows for a general triangular distribution
    by shifting (loc) and scaling (scale) the standard distribution.

    Parameters:
    - u (array-like): Input values where the PDF is to be evaluated.
    - loc (float, optional): Location parameter (peak of the distribution). Default is 0.
    - scale (float, optional): Scale parameter (half-width at half-maximum). Default is 1.

    Returns:
    - pdf_values (array): PDF values corresponding to the input values.

    Raises:
    - ValueError: If `scale` is non-positive.
    - TypeError: If `loc` or `scale` is not a numeric type.
    """
    if not (isinstance(loc, (int, float)) and isinstance(scale, (int, float))):
        raise TypeError("Location (loc) and scale parameters must be numeric.")

    if scale <= 0:
        raise ValueError("Scale parameter must be positive.")

    u = np.asarray(u)
    # Transform u to the standard triangular distribution
    transformed_u = (u - loc) / scale

    # Calculate the absolute values of the transformed input
    uu = np.abs(transformed_u)

    # Calculate the PDF: It's (1 - |transformed_u|)/scale where |transformed_u| <= 1, and 0 otherwise
    pdf_values = np.where(uu <= 1, (1 - uu) / scale, 0)

    return pdf_values


def tricube_pdf(u, loc=0, scale=1):
    """
    Calculate the probability density function (PDF) of a Tricube distribution.

    The standard Tricube distribution is defined on the interval [-1, 1]. This function
    allows for a general Tricube distribution by shifting (loc) and scaling (scale)
    the standard distribution.

    Parameters:
    - u (array-like): Input values where the PDF is to be evaluated.
    - loc (float, optional): Location parameter (central point of the distribution). Default is 0.
    - scale (float, optional): Scale parameter (defines the spread of the distribution). Default is 1.

    Returns:
    - pdf_values (array): PDF values corresponding to the input values.

    Raises:
    - ValueError: If `scale` is non-positive.
    - TypeError: If `loc` or `scale` is not a numeric type.
    """
    if not (isinstance(loc, (int, float)) and isinstance(scale, (int, float))):
        raise TypeError("Location (loc) and scale parameters must be numeric.")

    if scale <= 0:
        raise ValueError("Scale parameter must be positive.")

    u = np.asarray(u)
    # Transform u to the standard Tricube distribution
    transformed_u = (u - loc) / scale

    # Calculate the absolute values of the transformed input
    uu = np.abs(transformed_u)

    # Calculate the PDF: It's (70/81) * (1 - |transformed_u|^3)^3 where |transformed_u| <= 1, and 0 otherwise
    pdf_values = (70 / (scale * 81)) * np.where(uu <= 1, (1 - uu ** 3) ** 3, 0)

    return pdf_values


def gaussian_pdf(u, loc=0, scale=1):
    """
    Calculate the probability density function (PDF) of a normal distribution.

    This function calculates the PDF of a normal distribution with given mean and
    standard deviation. The standard normal distribution is a special case with
    mean = 0 and std = 1.

    Parameters:
    - u (array-like): Input values where the PDF is to be evaluated.
    - mean (float, optional): Mean (µ) of the normal distribution. Default is 0.
    - std (float, optional): Standard deviation (σ) of the normal distribution. Default is 1.

    Returns:
    - pdf_values (array): PDF values corresponding to the input values.

    Raises:
    - ValueError: If `std` is non-positive.
    - TypeError: If `mean` or `std` is not a numeric type.
    """
    # if not (isinstance(loc, (int, float)) and isinstance(scale, (int, float))):
    # raise TypeError("loc and scale must be numeric.")

    if scale <= 0:
        raise ValueError("Standard deviation must be positive.")

    u = np.asarray(u)
    # Adjust u for the mean and standard deviation
    adjusted_u = (u - loc) / scale

    # Normal distribution PDF formula
    pdf_values = (1.0 / (scale * np.sqrt(2.0 * np.pi))) * np.exp(-np.square(adjusted_u) / 2)

    return pdf_values


def laplace_pdf(u, loc=0, scale=1):
    """
    Calculate the probability density function (PDF) of a Laplace distribution.

    The Laplace distribution is characterized by a location parameter (μ) and a scale
    parameter (b). The PDF is given by: f(u|μ, b) = (1 / (2b)) * exp(-|u - μ| / b).

    Parameters:
    - u (array-like): Input values where the PDF is to be evaluated.
    - loc (float, optional): Location parameter (μ) of the Laplace distribution. Default is 0.
    - scale (float, optional): Scale parameter (b) of the Laplace distribution. Default is 1.

    Returns:
    - array: PDF values corresponding to the input values.

    Raises:
    - ValueError: If `scale` is not positive.
    """
    if not (isinstance(loc, (int, float)) and isinstance(scale, (int, float))):
        raise TypeError("Location (loc) and scale parameters must be numeric.")

    if scale <= 0:
        raise ValueError("Scale parameter must be positive.")

    u = np.asarray(u)
    # Adjust u for the location parameter
    adjusted_u = np.abs(u - loc)

    # Laplace distribution PDF formula
    pdf_values = (1 / (2 * scale)) * np.exp(-adjusted_u / scale)

    return pdf_values


def cosine_pdf(u, loc=0, scale=1):
    """
    Calculate the probability density function (PDF) of a cosine distribution.

    The standard cosine distribution is defined on the interval [-1, 1] with a peak at 0.
    This function allows for a general cosine distribution by shifting (loc) and scaling
    (scale) the standard distribution.

    Parameters:
    - u (array-like): Input values where the PDF is to be evaluated.
    - loc (float, optional): Location parameter, shifting the center of the distribution.
                             Default is 0.
    - scale (float, optional): Scale parameter, affecting the spread of the distribution.
                               Default is 1.

    Returns:
    - pdf_values (array): PDF values corresponding to the input values.

    Raises:
    - ValueError: If `scale` is non-positive.
    """
    if not (isinstance(loc, (int, float)) and isinstance(scale, (int, float))):
        raise TypeError("Location (loc) and scale parameters must be numeric.")

    if scale <= 0:
        raise ValueError("Scale parameter must be positive.")

    u = np.asarray(u)
    # Transform u to the standard cosine distribution
    transformed_u = np.abs((u - loc) / scale)

    # Calculate the PDF within the transformed range
    pdf_values = np.where(transformed_u <= 1, (np.pi / (scale * 4)) * np.cos(transformed_u * np.pi / 2), 0)

    return pdf_values


def epanechnikov_pdf(u, loc=0, scale=1):
    """
    Calculate the probability density function (PDF) of an Epanechnikov distribution.

    The standard Epanechnikov distribution is defined on the interval [-1, 1] with a peak at 0.
    This function allows for a general Epanechnikov distribution by shifting (loc) and scaling
    (scale) the standard distribution.

    Parameters:
    - u (array-like): Input values where the PDF is to be evaluated.
    - loc (float, optional): Location parameter, shifting the center of the distribution.
                             Default is 0.
    - scale (float, optional): Scale parameter, affecting the spread of the distribution.
                               Default is 1.

    Returns:
    - pdf_values (array): PDF values corresponding to the input values.

    Raises:
    - ValueError: If `scale` is non-positive.
    """
    if not (isinstance(loc, (int, float)) and isinstance(scale, (int, float))):
        raise TypeError("Location (loc) and scale parameters must be numeric.")

    if scale <= 0:
        raise ValueError("Scale parameter must be positive.")

    u = np.asarray(u)
    # Transform u to the standard Epanechnikov distribution
    transformed_u = (u - loc) / scale

    # Calculate the PDF within the transformed range
    pdf_values = np.where(np.abs(transformed_u) <= 1, (0.75 / scale) * (1 - np.square(transformed_u)), 0)

    return pdf_values


def quartic_pdf(u, loc=0, scale=1):
    """
    Calculate the probability density function (PDF) of a Quartic distribution.

    The standard Quartic distribution is defined on the interval [-1, 1] with a peak at 0.
    This function allows for a general Quartic distribution by shifting (loc) and scaling
    (scale) the standard distribution.

    Parameters:
    - u (array-like): Input values where the PDF is to be evaluated.
    - loc (float, optional): Location parameter, shifting the center of the distribution.
                             Default is 0.
    - scale (float, optional): Scale parameter, affecting the spread of the distribution.
                               Default is 1.

    Returns:
    - pdf_values (array): PDF values corresponding to the input values.

    Raises:
    - ValueError: If `scale` is non-positive.
    """
    if not (isinstance(loc, (int, float)) and isinstance(scale, (int, float))):
        raise TypeError("Location (loc) and scale parameters must be numeric.")

    if scale <= 0:
        raise ValueError("Scale parameter must be positive.")

    u = np.asarray(u)
    # Transform u to the standard Quartic distribution
    transformed_u = (u - loc) / scale

    # Calculate the PDF within the transformed range
    pdf_values = np.where(np.abs(transformed_u) <= 1, (0.9375 / scale) * np.square(1 - np.square(transformed_u)), 0)

    return pdf_values


def triweight_pdf(u, loc=0, scale=1):
    """
    Calculate the probability density function (PDF) of a Triweight distribution.

    The standard Triweight distribution is defined on the interval [-1, 1] with a peak at 0.
    This function allows for a general Triweight distribution by shifting (loc) and scaling
    (scale) the standard distribution.

    Parameters:
    - u (array-like): Input values where the PDF is to be evaluated.
    - loc (float, optional): Location parameter, shifting the center of the distribution.
                             Default is 0.
    - scale (float, optional): Scale parameter, affecting the spread of the distribution.
                               Default is 1.

    Returns:
    - pdf_values (array): PDF values corresponding to the input values.

    Raises:
    - ValueError: If `scale` is non-positive.
    """
    if not (isinstance(loc, (int, float)) and isinstance(scale, (int, float))):
        raise TypeError("Location (loc) and scale parameters must be numeric.")

    if scale <= 0:
        raise ValueError("Scale parameter must be positive.")

    u = np.asarray(u)
    # Transform u to the standard Triweight distribution
    transformed_u = (u - loc) / scale

    # Calculate the PDF within the transformed range
    pdf_values = np.where(np.abs(transformed_u) <= 1, (1.09375 / scale) * np.power(1 - np.square(transformed_u), 3), 0)

    return pdf_values


def logistic_pdf(u, loc=0, scale=1):
    """
    Calculate the probability density function (PDF) of a Logistic distribution.

    The Logistic distribution is characterized by a location parameter (μ) and a scale
    parameter (s). The PDF is given by: f(u|μ, s) = e^(-(u - μ)/s) / [s(1 + e^(-(u - μ)/s))^2].

    Parameters:
    - u (array-like): Input values where the PDF is to be evaluated.
    - loc (float, optional): Location parameter (μ) of the Logistic distribution. Default is 0.
    - scale (float, optional): Scale parameter (s) of the Logistic distribution. Default is 1.

    Returns:
    - pdf_values (array): PDF values corresponding to the input values.

    Raises:
    - ValueError: If `scale` is non-positive.
    """
    if not (isinstance(loc, (int, float)) and isinstance(scale, (int, float))):
        raise TypeError("Location (loc) and scale parameters must be numeric.")

    if scale <= 0:
        raise ValueError("Scale parameter must be positive.")

    u = np.asarray(u)
    # Adjust u for the location and scale parameters
    z = (u - loc) / scale

    # Logistic distribution PDF formula
    pdf_values = (0.5 / scale) / (1 + np.cosh(z))

    return pdf_values


def sigmoid_pdf(u, loc=0, scale=1):
    """
    Calculate the probability density function (PDF) of a Sigmoid distribution.

    The Sigmoid distribution is a special case of the Logistic distribution. It is often
    used in machine learning for activation functions. This function can be generalized
    by including a location parameter (loc) and a scale parameter (scale).

    Parameters:
    - u (array-like): Input values where the PDF is to be evaluated.
    - loc (float, optional): Location parameter, shifting the center of the distribution.
                             Default is 0.
    - scale (float, optional): Scale parameter, affecting the spread of the distribution.
                               Default is 1.

    Returns:
    - pdf_values (array): PDF values corresponding to the input values.
    """
    if not (isinstance(loc, (int, float)) and isinstance(scale, (int, float))):
        raise TypeError("Location (loc) and scale parameters must be numeric.")

    if scale <= 0:
        raise ValueError("Scale parameter must be positive.")
    u = np.asarray(u)
    # Adjust u for the location and scale parameters
    z = (u - loc) / scale

    # Sigmoid distribution PDF formula
    pdf_values = 1 / (np.pi * scale * np.cosh(z))

    return pdf_values


def cauchy_pdf(u, loc=0, scale=1):
    """
    Calculate the probability density function (PDF) of a Cauchy distribution.

    The Cauchy distribution is characterized by a location parameter (x0) and a scale
    parameter (γ). The PDF is given by: f(u|x0, γ) = 1 / [πγ(1 + ((u - x0)/γ)^2)].

    Parameters:
    - u (array-like): Input values where the PDF is to be evaluated.
    - loc (float, optional): Location parameter (x0) of the Cauchy distribution. Default is 0.
    - scale (float, optional): Scale parameter (γ) of the Cauchy distribution. Default is 1.

    Returns:
    - pdf_values (array): PDF values corresponding to the input values.

    Raises:
    - ValueError: If `scale` is non-positive.
    """
    if not (isinstance(loc, (int, float)) and isinstance(scale, (int, float))):
        raise TypeError("Location (loc) and scale parameters must be numeric.")

    if scale <= 0:
        raise ValueError("Scale parameter must be positive.")

    u = np.asarray(u)
    # Adjust u for the location and scale parameters
    z = (u - loc) / scale

    # Cauchy's distribution PDF formula
    pdf_values = 1.0 / (np.pi * scale * np.cosh(z))

    return pdf_values
