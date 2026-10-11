import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class StandardDeepHedgingNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, features, bs_delta):
        
        adjustments = self.network(features).squeeze(-1)
        hedge_positions = bs_delta + adjustments
        return hedge_positions

def standard_deep_hedge(
        model,
        paths,
        strike = 1.0,
        sigma = 0.2,
        T = 30 / 365,
        r = 0.0
):
    n_paths, n_prices = paths.shape
    n_steps = n_prices - 1
    dt = T / n_steps

    current_position = torch.zeros(
        n_paths,
        dtype=paths.dtype,
        device=paths.device
    )

    all_positions = []

    for step in range(n_steps):

        S = paths[:, step]
        tau = T - step * dt
        bs_delta = black_scholes_delta_torch(
            S, strike, tau, sigma, r
        )

        features = torch.stack(
            [
                torch.log(S/strike),
                torch.full_like(S, tau),
                torch.full_like(S, sigma),
                current_position
            ],
            dim = 1
        )
        target_position = model(features, bs_delta)
        all_positions.append(target_position)

        current_position = target_position

    return torch.stack(all_positions, dim=1)

def train_standard_hedge_model(n_paths = 10000, 
                                epochs = 200, 
                                strike = 1.0,
                                sigma = 0.2,
                                T = 30/365,
                                transaction_cost_rate = 0.001,
                                lambda_ = 1.0
                            ):

    
    model = StandardDeepHedgingNN()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    loss_history = []

    for epoch in range(epochs):

        paths_np = simulate_gbm(
            n_paths=n_paths,
            n_steps=30,
            s0=strike,
            sigma=sigma,
            T = T,
            mu=0.0 
        )
        paths = torch.tensor(paths_np, dtype=torch.float32)

        hedge_positions = standard_deep_hedge(model=model, paths=paths, strike=strike, sigma=sigma, T=T)

        performance = evaluate_hedge(paths, hedge_positions, strike, transaction_cost_rate)

        net_pnl = performance["net_pnl"]

        loss = torch.mean(torch.exp(-lambda_ * net_pnl))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(
                f"Iteration: {epoch + 1} / {epochs}, "
                f"Loss: {loss.item():.6f}"
            )

    return model, loss_history

model, loss_history = train_standard_hedge_model()
window = 10

smoothed_loss = np.convolve(
    loss_history,
    np.ones(window) / window,
    mode="valid",
)

plt.plot(
    range(1, len(loss_history) + 1),
    loss_history,
    alpha=0.3,
    label="Loss per epoch",
)

plt.plot(
    range(window, len(loss_history) + 1),
    smoothed_loss,
    label="10-epoch moving average",
)

plt.xlabel("Epoch")
plt.ylabel("Exponential utility loss")
plt.title("Standard Deep Hedging Training")
plt.legend()
plt.show()