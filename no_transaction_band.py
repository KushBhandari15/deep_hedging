import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class NoTransactionBandNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(32, 2)
        nn.init.constant_(self.output.bias, -3.0)


    def forward(self, features, bs_delta):

        raw_widths = self.output(self.hidden(features))
        widths = F.softplus(raw_widths)

        lower_band = bs_delta - widths[:, 0]
        upper_band = bs_delta + widths[:, 1]

        return lower_band, upper_band

def no_transaction_band_hedge(
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
            ],
            dim = 1
        )
        lower_band, upper_band = model(features, bs_delta)
        next_position = torch.minimum(
            torch.maximum(current_position, lower_band),
            upper_band,
        )

        all_positions.append(next_position)
        current_position = next_position

    return torch.stack(all_positions, dim=1)

def train_no_transaction_band_model(
    n_paths=10000,
    epochs=200,
    strike=1.0,
    sigma=0.2,
    T=30 / 365,
    transaction_cost_rate=0.001,
    lambda_=1.0,
):

    model = NoTransactionBandNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_history = []

    for epoch in range(epochs):

        # 1. Generate fresh simulated stock-price paths.
        paths_np = simulate_gbm(
            n_paths=n_paths,
            n_steps=30,
            s0=1.0,
            sigma=sigma,
            T=T,
            mu=0.0,
        )

        paths = torch.tensor(
            paths_np,
            dtype=torch.float32,
        )

        # 2. Choose hedge positions using the band strategy.
        hedge_positions = no_transaction_band_hedge(
            model=model,
            paths=paths,
            strike=strike,
            sigma=sigma,
            T=T,
        )

        # 3. Evaluate gains, trading costs and option liability.
        performance = evaluate_hedge(
            paths=paths,
            hedge_positions=hedge_positions,
            strike=strike,
            transaction_cost_rate=transaction_cost_rate,
        )

        net_pnl = performance["net_pnl"]

        # 4. Minimize exponential-utility loss.
        loss = torch.mean(torch.exp(-lambda_ * net_pnl))

        # 5. Update the network.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(
                f"Iteration: {epoch + 1}/{epochs}, "
                f"Loss: {loss.item():.6f}"
            )

    return model, loss_history

band_model, band_loss_history = (
    train_no_transaction_band_model()
)

plt.plot(band_loss_history)
plt.xlabel("Epoch")
plt.ylabel("Exponential utility loss")
plt.title("No-Transaction Band Network")
plt.show()

band_model.eval()

log_moneyness = torch.linspace(-0.15, 0.15, 300)
S = torch.exp(log_moneyness)

tau = torch.full_like(S, 15 / 365)
sigma_values = torch.full_like(S, 0.2)

features = torch.stack(
    [log_moneyness, tau, sigma_values],
    dim=1,
)

with torch.no_grad():
    bs_delta = black_scholes_delta_torch(
        S, 1.0, 15 / 365, 0.2
    )

    lower_band, upper_band = band_model(
        features, bs_delta
    )

plt.plot(
    log_moneyness.numpy(),
    bs_delta.numpy(),
    label="Black-Scholes delta",
)
plt.plot(
    log_moneyness.numpy(),
    lower_band.numpy(),
    label="Lower band",
)
plt.plot(
    log_moneyness.numpy(),
    upper_band.numpy(),
    label="Upper band",
)

plt.xlabel("Log-moneyness: log(S/K)")
plt.ylabel("Hedge ratio")
plt.title("Learned No-Transaction Band")
plt.legend()
plt.grid(True)
plt.show()