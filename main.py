import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import black_scholes as black_scholes_hedge_funcs
import no_transaction_band as no_transaction_band_funcs
import standard_deep_hedge as standard_deep_funcs
import utils as helper

def compare_model(standard_model, transaction_band_model):

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
    bs_positions_np = black_scholes_hedge_funcs.black_scholes_hedge(validation_paths_np)

    bs_positions = torch.tensor(
        bs_positions_np,
        dtype=torch.float32,
    )
    with torch.no_grad():
        standard_positions = standard_deep_funcs.standard_deep_hedge(
            model=standard_model,
            paths=validation_paths,
            strike=1.0,
            sigma=0.2,
            T=30 / 365,
        )

        band_positions = no_transaction_band_funcs.no_transaction_band_hedge(
            model=transaction_band_model,
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

def main():
    # Train the standard deep hedging model
    standard_model, standard_loss_history = standard_deep_funcs.train_standard_hedge_model(
        n_paths=10_000,
        epochs=200,
        strike=1.0,
        sigma=0.2,
        T=30 / 365,
        transaction_cost_rate=0.001,
        lambda_=1.0
    )

    # Train the no-transaction band model
    transaction_band_model, band_loss_history = no_transaction_band_funcs.train_no_transaction_band_model(
        n_paths=10_000,
        epochs=200,
        strike=1.0,
        sigma=0.2,
        T=30 / 365,
        transaction_cost_rate=0.001
    )

    # Compare the models
    compare_model(standard_model, transaction_band_model)

if __name__ == "__main__":
    main()
