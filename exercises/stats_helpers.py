"""Helper functions for statistics exercises."""

import numpy as np
from scipy.stats import binom


def binomial_posterior(
    data: np.ndarray,
    grid_points: int = 10,
    prior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the posterior distribution with binomial likelihood.

    The posterior is computed via Bayes' Theorem and is evaluated on a
    grid of `grid_points` points spanning the interval [0, 1].
    The prior is assumed to be uniform if not provided.

    Parameters
    ----------
    data : np.ndarray
        A boolean array where True represents success.
    grid_points : int
        Number of points in the grid used to compute the posterior.
    prior : np.ndarray, optional
        Prior distribution over the grid points. If None, a uniform prior
        is assumed.

    Returns
    -------
    np.ndarray
        The grid of points where the posterior is evaluated.
    np.ndarray
        The posterior distribution evaluated on the grid points.

    """
    p_grid = np.linspace(0, 1, grid_points)
    n = len(data)
    k = np.sum(data)
    likelihood = binom.pmf(k, n, p_grid)  # Binomial likelihood
    posterior = likelihood if prior is None else likelihood * prior
    posterior /= np.sum(posterior)  # Normalize the posterior
    return p_grid, posterior



def binomial_ppd(
        posterior: np.ndarray,
        n_trials: int,
        n_samples: int = 10_000,
        rng: np.random.Generator | None = None
) -> np.ndarray:
    """Compute the posterior predictive distribution for a binomial model.

    The posterior predictive distribution is computed by sampling from the
    posterior distribution and then sampling from the binomial distribution
    with the sampled probability. The process is repeated `n_samples` times.

    Parameters
    ----------
    posterior : np.ndarray
        The posterior distribution.
    n_trials : int
        Number of trials in the binomial distribution.
    n_samples : int
        Number of samples to draw from the posterior.
    rng : np.random.Generator, optional
        Random number generator to use for sampling. If None, the default
        numpy random number generator is used.

    Returns
    -------
    np.ndarray
        Samples from the posterior predictive distribution.

    """
    if rng is None:
        rng = np.random.default_rng()
    
    p_grid = np.linspace(0, 1, len(posterior))
    samples = rng.choice(p_grid, size=n_samples, replace=True, p=posterior)
    return binom.rvs(n_trials, samples, size=n_samples, random_state=rng)
