import pytest
import numpy as np
from ..distributions import truncated_inverse_gamma

def test_gamma_boundaries():
    "Check behaviour for boundaries values"
    rng = np.random.default_rng()
    alpha = 2.1
    psi = 1.9
    with pytest.raises(AssertionError):
        truncated_inverse_gamma(alpha, psi, -0.31, 0.31, rng)
    with pytest.raises(AssertionError):
        truncated_inverse_gamma(alpha, psi, 102., 0., rng)
    with pytest.raises(AssertionError):
        truncated_inverse_gamma(alpha, psi, 0., 0., rng)
    truncated_inverse_gamma(alpha, psi, 0., np.inf, rng)
    truncated_inverse_gamma(alpha, psi, 0., 10e3, rng)

def test_gamma_mean_variance():
    "Check if the samples are reasonable by checking first and second moments"
    alpha = np.array(3.1)
    psi = np.array(1.9)
    true_mean = psi / (alpha - 1.)
    true_var = psi**2. / ((alpha - 1)**2. * (alpha - 2.))
    n_iter = 10000
    samples = np.empty((n_iter, ))
    rng = np.random.default_rng()
    for i in range(n_iter):
        samples[i] = truncated_inverse_gamma(alpha, psi, 0., np.inf, rng)
    mc_mean = np.mean(samples)
    mc_var = np.var(samples)
    np.testing.assert_allclose(mc_mean, true_mean, rtol=1e-01, atol=1e-01, err_msg="The Monte Carlo mean is too different from the true value")
    np.testing.assert_allclose(mc_var, true_var, rtol=1e-01, atol=1e-01, err_msg="The Monte Carlo mean is too different from the true value")

    