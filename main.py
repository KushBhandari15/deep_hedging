import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from black_scholes import black_scholes_hedge
from no_transaction_band import no_transaction_band_hedge
from standard_deep_hedge import standard_deep_hedge
import utils as helper
    
def compare_model():

    validation_paths_np = helper.simulate_gbm(
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
            model=helper.band_model,
            paths=validation_paths,
            strike=1.0,
            sigma=0.2,
            T=30 / 365,
        )

    bs_results = helper.evaluate_hedge(
        paths=validation_paths,
        hedge_positions=bs_positions,
        strike=1.0,
        transaction_cost_rate=0.001,
    )

    standard_results = helper.evaluate_hedge(
        validation_paths,
        standard_positions,
        strike=1.0,
        transaction_cost_rate=0.001,
    )

    band_results = helper.evaluate_hedge(
        validation_paths,
        band_positions,
        strike=1.0,
        transaction_cost_rate=0.001,
    )
    # Strategy that never trades or holds any shares.
    no_hedge_positions = torch.zeros_like(bs_positions)

    no_hedge_results = helper.evaluate_hedge(
        paths=validation_paths,
        hedge_positions=no_hedge_positions,
        strike=1.0,
        transaction_cost_rate=0.001,
    )

    helper.report_results("Never hedge", no_hedge_results)
    helper.report_results("Black-Scholes", bs_results)
    helper.report_results("Standard Deep Hedging NN", standard_results)
    helper.report_results("No-Transaction Band NN", band_results)