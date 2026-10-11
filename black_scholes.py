import numpy as np
from scipy.stats import norm

def black_scholes_call_delta(S, K, tau, sigma, r=0.0):

    S, tau = np.broadcast_arrays(
        np.asarray(S, dtype=float),
        np.asarray(tau, dtype=float)
    )
    safe_tau = np.maximum(tau, 1e-12)
    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * safe_tau
    ) / (sigma * np.sqrt(safe_tau))

    delta = norm.cdf(d1)

    # At expiration, use the call's payoff slope
    delta = np.where(
        tau > 0,
        delta,
        np.where(S > K, 1.0, np.where(S < K, 0.0, 0.5)),
    )

    return delta

def black_scholes_hedge(path, strike=1.0, sigma=0.2, T=30/365, r=0.0):

    path = np.asarray(path, dtype=float)

    n_paths, n_prices = path.shape
    n_steps = n_prices - 1

    dt = T / n_steps
    times = np.arange(n_steps) * dt
    time_remaining = T - times

    hedge_positions = black_scholes_call_delta(
        S = path[:, :-1],
        K = strike,
        tau=time_remaining[np.newaxis, :],
        sigma=sigma,
        r=r
    )

    return hedge_positions