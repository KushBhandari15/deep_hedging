import numpy as np
import torch
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

def report_results(name, results):
    pnl = results["net_pnl"]

    print(f"\n{name}")
    print(f"Exponential loss: {torch.exp(-pnl).mean().item():.6f}")
    print(f"Mean P&L:         {pnl.mean().item():.6f}")
    print(f"Mean transaction costs: "
          f"{results['transaction_costs'].mean().item():.6f}")