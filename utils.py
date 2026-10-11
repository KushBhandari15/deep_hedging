import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

def simulate_gbm(
    n_paths: int,
    n_steps: int,
    s0: float = 1.0,
    sigma: float = 0.2,
    T: float = 30/365,
    mu: float = 0.0
):
    dt = T / n_steps

    z = np.random.normal(
        size=(n_paths, n_steps)
    )
    increments = (
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * z
    )
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = s0

    S[:, 1:] = s0 * np.exp(
        np.cumsum(increments, axis=1)
    )
    return S

def european_call_option(paths, strike = 1):
    
    terminal_prices = paths[:, -1]
    payoff = np.maximum(terminal_prices - strike, 0)

    return payoff

def black_scholes_delta_torch(S, K, tau, sigma, r=0.0):

    tau = torch.as_tensor(
            tau,
            dtype=S.dtype,
            device=S.device,
    )
    d1 = (
        torch.log(S / K)
        + (r + 0.5 * sigma ** 2) * tau
    ) / (sigma * torch.sqrt(tau))

    return torch.special.ndtr(d1)

def evaluate_hedge(
        paths,
        hedge_positions,
        strike = 1.0,
        transaction_cost_rate = 0.001,
):
    
    price_changes = paths[:, 1:] - paths[:, :-1]

    trading_gains = torch.sum(
        price_changes * hedge_positions,
        dim = 1,
    )
    previous_positions = torch.cat(
        [
            torch.zeros_like(hedge_positions[:, :1]),
            hedge_positions[:, :-1],
        ],
        dim=1
    )
    trades = hedge_positions - previous_positions

    transaction_costs = torch.sum(
        transaction_cost_rate 
        * paths[:, :-1]
        * torch.abs(trades),
        dim = 1
    )
    option_payoff = torch.clamp(
        paths[:, -1] - strike, 
        min= 0.0,
    )

    net_pnl = (
        trading_gains
        - transaction_costs
        - option_payoff
    )

    return {
        "trading_gains": trading_gains,
        "transaction_costs": transaction_costs,
        "option_payoff": option_payoff,
        "net_pnl": net_pnl,
    }

validation_paths_np = simulate_gbm(
    n_paths=10_000,
    n_steps=30,
    s0=1.0,
    sigma=0.2,
    T=30 / 365,
    mu=0.0,
)

validation_paths = torch.tensor(
    validation_paths_np,
    dtype=torch.float32,
)

# Baseline: ordinary Black-Scholes delta hedging.
bs_positions_np = black_scholes_hedge(validation_paths_np)

bs_positions = torch.tensor(
    bs_positions_np,
    dtype=torch.float32,
)
with torch.no_grad():
    standard_positions = standard_deep_hedge(
        model=model,
        paths=validation_paths,
        strike=1.0,
        sigma=0.2,
        T=30 / 365,
    )

    band_positions = no_transaction_band_hedge(
        model=band_model,
        paths=validation_paths,
        strike=1.0,
        sigma=0.2,
        T=30 / 365,
    )

bs_results = evaluate_hedge(
    paths=validation_paths,
    hedge_positions=bs_positions,
    strike=1.0,
    transaction_cost_rate=0.001,
)

standard_results = evaluate_hedge(
    validation_paths,
    standard_positions,
    strike=1.0,
    transaction_cost_rate=0.001,
)

band_results = evaluate_hedge(
    validation_paths,
    band_positions,
    strike=1.0,
    transaction_cost_rate=0.001,
)
# Strategy that never trades or holds any shares.
no_hedge_positions = torch.zeros_like(bs_positions)

no_hedge_results = evaluate_hedge(
    paths=validation_paths,
    hedge_positions=no_hedge_positions,
    strike=1.0,
    transaction_cost_rate=0.001,
)

report_results("Never hedge", no_hedge_results)
report_results("Black-Scholes", bs_results)
report_results("Standard Deep Hedging NN", standard_results)
report_results("No-Transaction Band NN", band_results)