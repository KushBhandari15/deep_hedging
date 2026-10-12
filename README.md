# Deep Hedging

This project compares four ways to hedge a short European call option: never hedging, Black–Scholes delta hedging, a neural network that adjusts the Black–Scholes hedge, and a neural network that learns a no-transaction band around the Black–Scholes hedge.

The code simulates stock-price paths, trains the two neural strategies while accounting for trading costs, and compares their performance on a separate batch of simulated paths. It is a small educational experiment, not a calibrated trading or pricing system.

## The problem the code models

The seller of a call option receives a payoff obligation at expiry. For stock price `S_T` and strike `K`, that liability is:

```text
call payoff = max(S_T - K, 0)
```

To reduce the risk of owing this payoff, the seller can hold shares. A hedge position `h_t` means holding that many shares during the next time interval. A positive position gains when the share price rises; a negative position loses when it rises. Rebalancing the position incurs a transaction cost.

For each simulated path, the project calculates:

```text
trading gains       = sum_t h_t (S_(t+1) - S_t)
transaction costs   = c sum_t S_t |h_t - h_(t-1)|
net P&L              = trading gains - transaction costs - call payoff
```

Here `c` is the transaction-cost rate, and the initial previous position is zero. The implementation is a seller-side hedge: the option payoff is subtracted from the gains and costs.

## How one run works

Running `main.py` performs these steps:

1. Train the standard deep hedge on simulated paths.
2. Train the no-transaction-band hedge on newly simulated paths.
3. Simulate a fresh validation batch and evaluate all four strategies on it.
4. Print the exponential loss, mean P&L, and mean transaction costs for each strategy.

The default run uses 10,000 paths per training epoch, 200 epochs, 30 hedge intervals over 30 days, initial stock and strike of 1, volatility of 20%, and transaction-cost rate of 0.001 (0.1%). These values are specified in `main.py`; training both networks can take some time. The training functions also return their loss histories, although the main script currently only uses the trained models.

## The simulated market

`utils.py` generates paths with geometric Brownian motion. For each time step it draws a standard normal random value and applies:

```text
S_(t+1) = S_t exp((mu - sigma²/2) dt + sigma sqrt(dt) Z_t)
```

The defaults are `mu = 0`, `sigma = 0.2`, `S_0 = 1`, and `T = 30/365` years. Each generated path has 31 prices: the starting price plus 30 subsequent prices. With zero interest rate, zero drift is also the risk-neutral drift used for this experiment. The simulator samples new random paths during every training epoch; it does not set a random seed, so runs are not exactly reproducible.

## The four strategies

### Never hedge

This baseline holds zero shares throughout. It pays no transaction costs, but leaves the entire call payoff unhedged. It helps show the scale of the liability and the value or cost of hedging.

### Black–Scholes delta

`black_scholes.py` computes the call's Black–Scholes delta, the theoretical number of shares to hold per option. For a call, delta is the standard normal cumulative distribution evaluated at `d1`; it is usually between zero and one. `black_scholes_hedge` recomputes delta at each time and stock price.

This is the analytic benchmark. It does not account for transaction costs when choosing positions, though its realized performance is charged the same transaction costs during evaluation as the learned strategies.

### Standard deep hedge

`StandardDeepHedgingNN` in `standard_deep_hedge.py` uses a feed-forward neural network to adjust the Black–Scholes delta at each time step. Its inputs are:

- log-moneyness, `log(S/K)`;
- time remaining until expiry;
- volatility;
- the position held in the previous interval.

The network outputs a `tanh`-bounded adjustment, which is added to the analytic delta. The hedge function loops over time, computes the Black–Scholes delta and features, asks the network for the new position, and carries that position forward as the next step's previous position.

### No-transaction-band hedge

`NoTransactionBandNN` in `no_transaction_band.py` takes log-moneyness, time remaining, and volatility as inputs. It predicts two nonnegative widths using `softplus`, which converts any real-valued output to a positive value. Those widths define lower and upper thresholds around Black–Scholes delta.

At each time step, the strategy keeps its previous position if it lies inside the band. If it is below the lower threshold, it moves up to the lower threshold; if it is above the upper threshold, it moves down to the upper threshold. This rule is intended to reduce small trades and their transaction costs. The two output biases start at -3, which starts the widths relatively narrow before training changes them.

## How training works

Each strategy's training function creates a network and an Adam optimizer. In every epoch, it simulates a fresh batch of paths, generates hedge positions, and calculates path-by-path net P&L including transaction costs and the option liability. It minimizes the average exponential loss:

```text
loss = mean(exp(-lambda * net P&L))
```

This objective penalizes poor outcomes more strongly than good outcomes of equal size, so it is sensitive to downside risk. `lambda` controls that sensitivity and defaults to 1. The code prints the loss every 20 epochs and returns `(trained_model, loss_history)`.

## Evaluation output

`compare_model` in `main.py` generates 10,000 new paths after training. It evaluates all four strategies on the same paths, which makes the comparison fairer than using separate validation data for each strategy. For each strategy, the output reports:

- **Exponential loss:** average `exp(-net P&L)`; lower is better for the objective used in training.
- **Mean P&L:** average seller-side net P&L after costs and option payoff.
- **Mean transaction costs:** average rebalancing cost per path.

These are sample estimates from one random validation batch. They can vary between runs. A low training loss alone does not prove good performance on other market assumptions or future data.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Runs training for both neural strategies, then compares all four strategies. |
| `utils.py` | Simulates paths, computes Torch Black–Scholes deltas, calculates hedge P&L and costs, and prints metrics. |
| `black_scholes.py` | Implements the analytic call delta and its dynamic hedging schedule using NumPy and SciPy. |
| `standard_deep_hedge.py` | Defines and trains the neural strategy that adjusts Black–Scholes delta. |
| `no_transaction_band.py` | Defines and trains the neural strategy that trades only when outside learned thresholds. |
| `main.ipynb` | Notebook version or exploration of the project; it is separate from the `main.py` run flow. |

The strategy modules put their standalone training plots behind `if __name__ == "__main__"`. Importing a module therefore defines its functions and classes without launching training. `main.py` explicitly calls the training functions when the script is run.

## Setup and run

Use Python 3. Install the libraries used by the scripts:

```bash
python3 -m pip install numpy scipy torch matplotlib
```

Then, from the project directory, run:

```bash
python3 main.py
```

To train just one strategy and display its loss plot, run `python3 standard_deep_hedge.py` or `python3 no_transaction_band.py`. Those standalone commands train using the function defaults. The no-transaction-band script also plots its learned thresholds at a sample time and volatility.

## Important assumptions and limitations

- The option is a European call, the underlying is a single stock, and the evaluation horizon is 30 days.
- The model uses constant volatility, zero interest rate, and zero stock drift by default. It does not model jumps, volatility smiles, dividends, funding, or market impact.
- A proportional transaction cost is charged on the stock notional traded at each hedge date. There is no separate option premium, financing account, discounting, or terminal stock position adjustment.
- The neural models and the Black–Scholes benchmark use the same simulated paths for evaluation, but training and validation paths are freshly generated and are not seeded.
- Results depend on the chosen parameters, random draws, and training setup. They should be treated as a coding and modeling experiment rather than investment advice or evidence of production readiness.
