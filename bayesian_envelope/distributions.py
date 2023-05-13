import numpy as np
from scipy import stats
import scipy.special as sc


def truncated_inverse_gamma(
    shape: float, scale: float, lower: float, upper: float, rng, max_iter=10e2, grid_points=1000, max_upper=10e5
):
    """
    Random sampling of a truncated inverse gamma distribution.
    We follow the `shape and scale parameterisation <https://en.wikipedia.org/wiki/Inverse-gamma_distribution>`_
    This is equivalent to sampling from a gamma distribution and take the reciprocal.
    The sampling strategy consists in running a rejection sampler until a certain number of iterations
    is reached. Then, we resort to a discrete approximation.

    Parameters
    ----------
    shape, scale : float
        The shape and scale parameters of the distribution.
    lower, upper : float
        The lower and upper bounds of the support.
    rng : np.random._generator.Generator
        The random number generator.
    max_iter : int
        The maximum number of iterations for the rejection sampler.
    grid_points : int
        Number of points in the grid.
    max_upper : float
        Maximum upper bound of the support for the discrete approximation.

    Returns
    -------
    x : np.ndarray
        The sampled random variable.
    """

    np.testing.assert_array_less(lower, upper, err_msg="Lower bound should be smaller than upper bound.")
    assert lower >= 0.0, "Lower bound cannot be negative."

    try:
        x = rejection_sampler_truncated_inverse_gamma(shape, scale, lower, upper, rng, max_iter)
    except ValueError:
        if upper == np.inf:
            upper = max_upper
        x = discrete_truncated_inverse_gamma(shape, scale, lower, upper, rng, grid_points)

    return x


def discrete_truncated_inverse_gamma(shape, scale, lower, upper, rng, grid_points):
    """
    Discrete approximation sampling of a truncated inverse gamma distribution.

    Parameters
    ----------
    shape, scale : float
        The shape and scale parameters of the distribution.
    lower, upper : float
        The lower and upper bounds of the support.
    rng : np.random._generator.Generator
        The random number generator.
    grid_points : int
        Number of points in the grid.

    Returns
    -------
    x : np.ndarray
        The sampled random variable.
    """
    x = np.linspace(start=lower, stop=upper, num=grid_points)
    log_pdf = stats.invgamma.logpdf(x, shape, 1.0 / scale)
    prob = np.exp(log_pdf - np.max(log_pdf))
    prob /= np.sum(prob)
    return rng.choice(a=x, size=1, replace=True, p=prob, shuffle=False)


def rejection_sampler_truncated_inverse_gamma(shape, rate, lower, upper, rng, max_iter):
    """
    Rejection sampler for a truncated inverse gamma.

    Parameters
    ----------
    shape, scale : float
        The shape and scale parameters of the distribution.
    lower, upper : float
        The lower and upper bounds of the support.
    rng : np.random._generator.Generator
        The random number generator.
    max_iter : int
        The maximum number of iterations for the rejection sampler.

    Returns
    -------
    x : np.ndarray
        The sampled random variable.
    """
    x = np.power(rng.gamma(shape, np.power(rate, -1.0)), -1.0)
    counter = 0
    while x < lower or x > upper:
        x = 1.0 / rng.gamma(shape, np.power(rate, -1.0))
        counter += 1
        if counter >= max_iter:
            raise ValueError

    return x


def discrete_bingham(A: np.ndarray, B: np.ndarray, rng, N: int):
    """
    Random sampling of a Bingham distribution from a discrete approximation.

    Parameters
    ----------
    A, B : np.ndarray
        The parameters.
    rng : np.random._generator.Generator
        The random number generator.
    N : int
        The number of points in the grid.

    Returns
    -------
    X : np.ndarray
        The sampled random varibales.
    """

    a = -1.0 * (A[0, 0] + B[1, 1] - A[1, 1] - B[0, 0])
    b = B[0, 1] + B[1, 0] - A[0, 1] - A[1, 0]
    S = 2 * np.pi * (np.array([j / N for j in np.arange(1, N + 1)]) - 0.5 / N)
    log_prob = a * np.cos(S) ** 2 + b * np.cos(S) * np.sin(S) - a
    prob = np.exp(log_prob - np.max(log_prob))
    prob /= np.sum(prob)

    # Select a random x
    x = rng.choice(a=S, size=1, replace=False, p=prob)

    # Build the matrix
    x_1 = np.array([np.cos(x), np.sin(x)])
    x_2 = np.array([np.sin(x), -np.cos(x)]) * (-1.0) ** rng.binomial(n=1, p=0.5)

    return np.column_stack((x_1, x_2))


def get_m_from_w(w, rng):
    """
    This function create an orthogonal matrix from the initial cosine of an angle.
    """
    assert w <= 1.0 and w >= -1.0, "w must be included in [-1., 1]"
    if w >= 0:
        x_1 = np.array([w, np.sqrt(1.0 - w**2)]) * (-1.0) ** rng.binomial(n=1, p=0.5)
        x_2 = np.array([x_1[1], -x_1[0]]) * (-1.0) ** rng.binomial(n=1, p=0.5)
    else:
        x_1 = np.array([np.absolute(w), -np.sqrt(1.0 - w**2.0)]) * (-1.0) ** rng.binomial(n=1, p=0.5)
        x_2 = np.array([x_1[1], -x_1[0]]) * (-1.0) ** rng.binomial(n=1, p=0.5)
    return np.column_stack((x_1, x_2))


def bingham(A, B, rng, max_iter=1e5, N=4e5):

    # Check validity of input
    assert A.shape == (2, 2), "Parameter A is misspecified. Dimensions are not (2, 2)"
    assert B.shape == (2, 2), "Parameter B is misspecified. Dimensions are not (2, 2)"
    try:
        W = rejection_sampler_bingham(A, B, rng, max_iter)
    except ValueError:
        W = discrete_bingham(A, B, rng, N)
    return W


def rejection_sampler_bingham(A, B, rng, max_iter=1e5):
    """'
    This function samples from a Bingham distribution
    with dimension 2x2, given the parameters A and B.
    The input are 2x2 numpy arrays
    """

    # Set boolean flags
    is_accepted = False
    a_positive = False

    # Compute relevant parameters
    a = -1.0 * (A[0, 0] + B[1, 1] - A[1, 1] - B[0, 0])
    b = B[0, 1] + B[1, 0] - A[0, 1] - A[1, 0]

    # Change the sign of a
    if a > 0:
        a_positive = True
        a = -a
        b = -b

    # Iteration counter and random uniform
    count_iter = 0

    # Constant for the sampler
    BETA = 0.573
    GAMMA = 0.223

    if b < 0:
        log_k_1 = (
            -np.log(2) + sc.betaln(0.5 - 1.5 * GAMMA, 0.5 - 0.5 * GAMMA) + 2 * np.log(BETA) - GAMMA * np.log(a * b)
        )
        log_k_2 = -np.log(2) + sc.betaln(0.5 - GAMMA, 0.5) + np.log(BETA) - 0.5 * b - GAMMA * np.log(-a)
        max_log_k = np.max(np.array([log_k_1, log_k_2]))
        k_1 = np.exp(log_k_1 - max_log_k)
        k_2 = np.exp(log_k_2 - max_log_k)

        while is_accepted is False and count_iter <= max_iter:
            bin = rng.binomial(n=1, p=k_1 / (k_1 + k_2))
            if bin == 1.0:
                x = np.sqrt(rng.beta(0.5 - 1.5 * GAMMA, 0.5 - 0.5 * GAMMA))
                lr = (
                    (a * np.power(x, 2.0) + b * x * np.sqrt(1 - x**2))
                    - 2.0 * np.log(BETA)
                    + GAMMA * np.log(-a * x**2)
                    + GAMMA * np.log(-b * x * np.sqrt(1.0 - x**2))
                )
            else:
                x = np.sqrt(rng.beta(0.5 - GAMMA, 0.5))
                lr = (
                    (a * x**2.0 - b * x * np.sqrt(1.0 - x**2))
                    - 2 * np.log(BETA)
                    + 0.5 * b
                    + GAMMA * np.log(-a * x**2)
                )
                x = -x

            u = rng.uniform()
            is_accepted = np.log(u) < lr
            if is_accepted:
                w = x
            count_iter += 1
    else:
        log_k_1 = -np.log(2.0) + sc.betaln(0.5 - GAMMA, 0.5) + np.log(BETA) + 0.5 * b - GAMMA * np.log(-a)
        log_k_2 = (
            -np.log(2.0)
            + sc.betaln(0.5 - 1.5 * GAMMA, 0.5 - 0.5 * GAMMA)
            + 2 * np.log(BETA)
            - GAMMA * np.log(-a * b)
        )
        max_log_k = np.max(np.array([log_k_1, log_k_2]))
        k_1 = np.exp(log_k_1 - max_log_k)
        k_2 = np.exp(log_k_2 - max_log_k)

        while is_accepted is False and count_iter <= max_iter:
            bin = rng.binomial(n=1, p=k_1 / (k_1 + k_2))
            if bin == 1:
                x = np.sqrt(rng.beta(0.5 - GAMMA, 0.5))
                lr = (
                    (a * x**2 + b * x * np.sqrt(1.0 - x**2))
                    - np.log(BETA)
                    - 0.5 * b
                    + GAMMA * np.log(-a * x**2.0)
                )
            else:
                x = np.sqrt(rng.beta(0.5 - 1.5 * GAMMA, 0.5 - 0.5 * GAMMA))
                lr = (
                    (a * x**2 - b * x * np.sqrt(1.0 - x**2.0))
                    - 2 * np.log(BETA)
                    + GAMMA * np.log(-a * x**2.0)
                    + GAMMA * np.log(b * x * np.sqrt(1.0 - x**2.0))
                )
                x = -x

            u = rng.uniform()
            is_accepted = np.log(u) < lr
            if is_accepted:
                w = x
            count_iter += 1

    if is_accepted:
        Z = get_m_from_w(w, rng)
        if a_positive:
            Z = Z[:, [1, 0]]
    else:
        raise ValueError

    return Z