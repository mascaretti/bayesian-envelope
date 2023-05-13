import numpy as np
from scipy.stats import ortho_group, matrix_normal
from .distributions import bingham, truncated_inverse_gamma
from .utils import sign_update
from more_itertools import distinct_combinations
from tqdm import tqdm
import arviz as az


def gibbs_sampler(rng, Y, X, u, prior, init_vals, n_iter):
    """
    Gibbs sampler for the Bayesian formulation of the envelope model.

    For a more complete reference see:
    https://doi.org/10.1214/16-AOS1449
    Mascaretti, https://it.pearson.com/content/dam/region-core/italy/pearson-italy/pdf/Docenti/Universit%C3%A0/Sis-2022-4c-low.pdf

    Parameters
    ----------
    rng : np.random._generator.Generator
        The random number generator.
    Y : np.ndarray
        The response variables
    X : np.ndarray
        The covariates
    prior : dict
        The dictionary of the hyperparameters
    init_vals : dict
        The dictionary of the initial points
    n_iter : int
        The number of iterations for the gibbs sampler.

    Returns
    -------
    mcmc : az.data.inference_data.InferenceData
        The MCMC Chain.
    """
    # For now, we consider u to be between 0 and r
    r = Y.shape[1]
    p = X.shape[1]
    n = X.shape[0]
    assert u >= 0 and u <= r, "u should be greater or equal than 0 and smaller or equal than r"
    assert n == Y.shape[0], "Datasets misspecified" 

    # Storing MCMC:
    mcmc = dict()
    mcmc["mu"] = np.full((1, n_iter, r), np.nan)
    mcmc["O"] = np.full((1, n_iter, r, r), np.nan)
    mcmc["omega"] = np.full((1, n_iter, r), np.nan)
    mcmc["omega_0"] = np.full((1, n_iter, r), np.nan)
    mcmc["eta"] = np.full((1, n_iter, r, p), np.nan)
    mcmc["Sigma"] = np.full((1, n_iter, r, r), np.nan)
    mcmc["Beta"] = np.full((1, n_iter, r, p), np.nan)

    # Initialise values
    mu = init_vals.get("mu")
    O = init_vals.get("O")
    eta = init_vals.get("eta")
    omega = init_vals.get("omega")
    omega_0 = init_vals.get("omega_0")

    assert omega.shape == (u,), f"The shape of omega is {omega.shape}"
    assert omega_0.shape == (r-u,)
    assert O.shape == (r, r)
    assert mu.shape == (r, )
    assert eta.shape == (u, p)

    Gamma = O[:, 0:u]
    Gamma_0 = O[:, u:r]
    Beta = Gamma @ eta
    Sigma = Gamma @ np.diag(omega) @ Gamma.T + Gamma_0 @ np.diag(omega_0) @ Gamma_0.T

    # Check the prior
    # mu_0, Sigma_0, e, C, D, G, alpha, psi, alpha_0, psi_0,
    mu_0 = prior.get("mu_0")
    Sigma_0 = prior.get("Sigma_0")
    e = prior.get("e")
    C = prior.get("C")
    D = prior.get("D")
    G = prior.get("G")
    alpha = prior.get("alpha")
    psi = prior.get("psi")
    alpha_0 = prior.get("alpha_0")
    psi_0 = prior.get("psi_0")

    assert mu_0.shape == (r, ), "Prior mean for mu has wrong dimensions"
    assert Sigma_0.shape == (r, r), "Prior covariance for mu has wrong dimensions"
    try:
        np.linalg.cholesky(Sigma_0)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Prior covariance for mu is not positive semidefinite")
        
    try:
        np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Prior hyperam C for eta is not positive semidefinite")
    assert C.shape == (p, p), f"Prior hyperam C for eta has wrong dimensions: {C.shape} instead of {p}x{p}"
    assert e.shape == (r, p), f"Prior hyperam e for eta has wrong dimensions: {e.shape} instead of {r}x{p}"

    try:
        np.linalg.cholesky(D)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Prior hyperparam D for O is not positive semidefinite")
    assert D.shape == (r, r), f"Prior hyperam D for O has wrong dimensions: {D.shape} instead of {r}x{r}"

    try:
        np.linalg.cholesky(G)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Prior hyperparam G for O is not positive semidefinite")
    assert D.shape == (r, r), f"Prior hyperam G for O has wrong dimensions: {D.shape} instead of {r}x{r}"

    assert alpha > 0, "Shape parameter alpha must be positive"
    assert psi > 0, "Scale parameter psi must be positive"

    assert alpha_0 > 0, "Shape parameter alpha_0 must be positive"
    assert psi_0 > 0, "Scale parameter psi_0 must be positive"   

    # Compute invariant quantities -----
    Y_bar = np.mean(Y, axis=0)
    Y_c = Y - Y_bar
    e_tilde = (Y_c.T @ X + e @ C) @ np.linalg.inv(X.T @ X + C)
    G_tilde = Y_c.T @ Y_c + e @ C @ e.T - e_tilde @ (X.T @ X + C) @ e_tilde.T

    # Compute omega rates
    omega_rates = np.diag(Gamma.T @ G_tilde @ Gamma)
    omega_0_rates = np.diag(Gamma_0.T @ (Y_c.T @ Y) @ Gamma_0)

    for citer in tqdm(range(n_iter)):

        # Update Omega -------
        omega_rates = np.diag(Gamma.T @ G_tilde @ Gamma)
        for h in range(u):
        # Upper bound
            if h == 0:
                upper = np.inf
            else:
                upper = omega[h - 1]
        # Lower Bound
            if h == u - 1:
                lower = 0.
            else:
                lower = omega[h + 1]
            omega[h] = truncated_inverse_gamma(0.5*(n+2*alpha-1), 0.5*(omega_rates[h]+2*psi), lower, upper, rng)
        mcmc.get("omega")[0, citer, 0:u] = omega

        # Update Omega_0 ------------
        omega_0_rates = np.diag(Gamma_0.T @ (Y_c.T @ Y) @ Gamma_0)
        for h in range(r - u):
            # Upper Bound:
            if h == 0:
                upper = np.inf
            else:
                upper = omega_0[h - 1]
            # Lower Bound:
            if h == (r - u) - 1:
                lower = 0.
            else:
                lower = omega_0[h + 1]
            omega_0[h] = truncated_inverse_gamma(0.5*(n+2*alpha_0-1), 0.5*(omega_0_rates[h]+2*psi_0), lower, upper, rng)
        mcmc.get("omega_0")[0, citer, 0:(r-u)] = omega_0

        # Update Gammas -----
        pairs = list(distinct_combinations([j for j in range(r)], 2))
        for pair in pairs:
            N = O[:, pair]
            l, h = pair
            if 0 <= l < u and u <= h < r:
                A = 0.5 * N.T @ (G_tilde / omega[l] + G / D[l, l]) @ N
                B = 0.5 * N.T @ (Y_c.T @ Y_c / omega_0[h - u] + G / D[h, h]) @ N
            elif 0 <= l < u and 0 <= h < u:
                A = 0.5 * N.T @ (G_tilde / omega[l] + G / D[l, l]) @ N
                B = 0.5 * N.T @ (G_tilde / omega[h] + G / D[h, h]) @ N
            else:
                A = 0.5 * N.T @ (Y_c.T @ Y_c / omega_0[l - u] + G / D[l, l]) @ N
                B = 0.5 * N.T @ (Y_c.T @ Y_c / omega_0[h - u]  + G / D[h, h]) @ N
            O[:, pair] = sign_update(N @ bingham(A, B, rng))
                    
        Gamma = O[:, 0:u]
        Gamma_0 = O[:, u:r]
        mcmc.get("O")[0, citer] = O

        # Update Eta --------------
        if u > 0 and u <= r:
            eta = matrix_normal(Gamma.T @ e_tilde, np.diag(omega), np.linalg.inv(X.T @ X + C)).rvs(random_state=rng.integers(0, 1000000))
        elif u == 0:
            eta = eta = np.array([]).reshape((u, p))
        else:
            raise ValueError()
        mcmc.get("eta")[0, citer, 0:u, :] = eta

        # Compute Sigma -----
        # assert Gamma.shape == (r, u), f"The dimensions for Gamma of component {j} at iteration {i} are not ({r}, {u})"
        # assert np.diag(omega).shape == (u, u), f"The dimensions of Omega for component {j} at iteration {i} are not ({u}, {u})"
        # assert Gamma_0.shape == (r, r - u), f"The dimensions for Gamma of component {j} at iteration {i} are not ({r}, {r - u})"
        # assert np.diag(omega_0).shape == (r - u, r - u), f"The dimensions of Omega_0 for component {j} at iteration {i} are not ({r - u}, {r - u})"

        Sigma = Gamma @ np.diag(omega) @ Gamma.T + Gamma_0 @ np.diag(omega_0) @ Gamma_0.T
        mcmc.get("Sigma")[0, citer] = Sigma

        # Compute beta -----
        # assert eta.shape == (u, p), f"The dimension of eta (component {j}, iteration {i}) are not ({u}, {p})"
        Beta = Gamma @ eta
        mcmc.get("Beta")[0, citer] = Beta

        # Update mu --------
        Sigma_0_inv = np.linalg.inv(Sigma_0)
        Sigma_1_inv = np.linalg.inv(Sigma / n)
        Sigma_c = np.linalg.inv(Sigma_0_inv + Sigma_1_inv)
        mu_c = Sigma_c @ (Sigma_0_inv @ mu_0 + Sigma_1_inv @ Y_bar)
        mu = rng.multivariate_normal(mu_c, Sigma_c)
        mcmc.get("mu")[0, citer] = mu

    return az.from_dict(mcmc)
