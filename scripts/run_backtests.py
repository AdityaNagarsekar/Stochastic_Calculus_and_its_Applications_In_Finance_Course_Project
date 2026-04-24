"""
Extended backtests — run after run_p1.py and run_p3.py have completed.

Backtest 1 — Real Nifty path: legacy RL PPO vs BSM(rv21) on actual price paths
Backtest 1b — Fair real-path test: LSTM-RL vs BSM(LSTM sigma) on actual paths
Backtest 2 — TC sensitivity:  MAE vs kappa for BSM daily and RL
Backtest 3 — Moneyness:       pricing horse race at OTM / ATM / ITM
Backtest 6 — Real path hedge: full terminal PnL comparison on real paths

Usage:
    python scripts/run_backtests.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from bsm import bsm_price, bsm_delta, implied_vol
from data_pipeline import load_nifty, make_sequences, split_data, normalize_splits
from lstm_model import VolLSTM, predict
from pricing_pipeline import backtest_pricing, summarise_backtest
from backtest import analyse_hedging_results, paired_bootstrap_mae_diff
import torch

RESULTS = Path(__file__).parent.parent / "results"
MODELS  = Path(__file__).parent.parent / "models"
RESULTS.mkdir(exist_ok=True)

R      = 0.065      # Indian risk-free rate
KAPPA  = 0.001      # baseline transaction cost
N_STEP = 21         # steps per hedging episode
T_OPT  = 21 / 252

PROC = Path(__file__).parent.parent / "data" / "processed"

# ── load data once ───────────────────────────────────────────────────────────
print("Loading data and models...")
df       = load_nifty()
X, y, dates = make_sequences(df, seq_len=60)
splits   = split_data(X, y, dates)
splits, scaler = normalize_splits(splits)

rl_model = PPO.load(str(MODELS / "ppo_hedge_v1"))
dr_model_path = MODELS / "ppo_domain_random"
if not dr_model_path.with_suffix(".zip").exists():
    raise FileNotFoundError(
        "models/ppo_domain_random.zip not found. "
        "Run scripts/run_p1.py first to generate the domain-randomized RL model."
    )
dr_model = PPO.load(str(dr_model_path))

lstm_rl_path = MODELS / "ppo_lstm_hedge_v1"
if not lstm_rl_path.with_suffix(".zip").exists():
    raise FileNotFoundError(
        "models/ppo_lstm_hedge_v1.zip not found. "
        "Run scripts/run_integration.py first to generate the LSTM-RL model."
    )
lstm_rl_model = PPO.load(str(lstm_rl_path))

lstm = VolLSTM(input_size=6, hidden1=64, hidden2=32)
lstm.load_state_dict(torch.load(str(MODELS / "lstm_vol_best.pt"), map_location="cpu"))
lstm.eval()
lstm_vols_test = predict(lstm, splits["test"]["X"])

# Load val LSTM predictions (saved by run_p3.py); fall back to re-computing
val_preds_path = PROC / "lstm_val_preds.npy"
if val_preds_path.exists():
    lstm_vols_val = np.load(val_preds_path)
else:
    lstm_vols_val = predict(lstm, splits["val"]["X"])

test_dates  = splits["test"]["dates"]
test_prices = df.loc[test_dates, "Close"].values.astype(float)
hist_vols_test = df.loc[test_dates, "rv21"].values.astype(float)

val_dates   = splits["val"]["dates"]
val_prices  = df.loc[val_dates, "Close"].values.astype(float)
hist_vols_val = df.loc[val_dates, "rv21"].values.astype(float)

# ── Combine val + test for larger sample (val: Aug 2020–Oct 2022, test: Oct 2022–Dec 2024)
# Neither BSM delta nor the RL model was trained on real prices, so using
# the val period here does not introduce any data leakage.
combined_dates  = pd.concat([pd.Series(val_dates),  pd.Series(test_dates)],
                             ignore_index=True)
combined_prices = np.concatenate([val_prices,  test_prices])
combined_hist   = np.concatenate([hist_vols_val, hist_vols_test])
combined_lstm   = np.concatenate([lstm_vols_val, lstm_vols_test])
combined_true   = np.concatenate([splits["val"]["y"],  splits["test"]["y"]])

# Keep test-only arrays for moneyness backtest (consistent K_offset computation)
test_true = splits["test"]["y"]
hist_vols = hist_vols_test
lstm_vols = lstm_vols_test
log_rets  = df.loc[test_dates, "log_ret"].values.astype(float)

print(f"Test set: {len(test_dates)} dates, "
      f"{test_dates.min().date()} – {test_dates.max().date()}")
print(f"Val+Test combined: {len(combined_dates)} dates, "
      f"{combined_dates.min().date()} – {combined_dates.max().date()}")


# =============================================================================
# Helpers
# =============================================================================

def _run_bsm_on_path(price_path: np.ndarray, K: float, T: float,
                      r: float, sigma: float, kappa: float) -> tuple[float, float]:
    """
    Run BSM daily delta hedge on a given price path.
    price_path: array of length N_STEP+1  (S0, S1, ..., S_N)
    Returns (terminal_pnl, total_tc).
    """
    n    = len(price_path) - 1
    dt   = T / n
    S    = price_path[0]
    portfolio = bsm_price(S, K, T, r, sigma)
    delta = 0.0
    tc_sum = 0.0

    for t in range(n):
        tau      = max(T - t * dt, 1e-9)
        new_d    = bsm_delta(S, K, tau, r, sigma)
        tc       = kappa * abs(new_d - delta) * S
        tc_sum  += tc
        portfolio -= tc
        delta    = new_d

        S_new    = price_path[t + 1]
        portfolio += delta * (S_new - S)
        S        = S_new

    payoff = max(S - K, 0.0)
    return float(portfolio - payoff), float(tc_sum)


def _run_rl_on_path(price_path: np.ndarray, K: float, T: float,
                    r: float, sigma: float, kappa: float,
                    model, sigma_obs: bool = False) -> tuple[float, float]:
    """
    Run RL agent on a given price path using the pre-trained policy.
    The agent observes (S/K, tau/T, delta_prev) and optionally sigma/0.20.
    Returns (terminal_pnl, total_tc).
    """
    n         = len(price_path) - 1
    dt        = T / n
    S         = price_path[0]
    portfolio = bsm_price(S, K, T, r, sigma)
    delta     = 0.0
    tc_sum    = 0.0

    for t in range(n):
        tau   = max(T - t * dt, 1e-9)
        obs_items = [S / K, tau / T, delta]
        if sigma_obs:
            obs_items.append(sigma / 0.20)
        obs   = np.array(obs_items, dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        new_d = float(np.clip(action[0], 0.0, 1.0))

        tc        = kappa * abs(new_d - delta) * S
        tc_sum   += tc
        portfolio -= tc
        delta     = new_d

        S_new     = price_path[t + 1]
        portfolio += delta * (S_new - S)
        S         = S_new

    payoff = max(S - K, 0.0)
    return float(portfolio - payoff), float(tc_sum)


def _extract_real_paths(prices: np.ndarray, window: int = N_STEP,
                        stride: int = N_STEP) -> list[np.ndarray]:
    """Extract non-overlapping windows of length window+1 from price series."""
    paths = []
    for i in range(0, len(prices) - window, stride):
        paths.append(prices[i : i + window + 1])
    return paths


def _extract_window_starts(values: np.ndarray, window: int = N_STEP,
                           stride: int = N_STEP,
                           min_value: float = 1e-6) -> list[float]:
    """Pick the start-of-window scalar aligned with _extract_real_paths."""
    starts = []
    for i in range(0, len(values) - window, stride):
        starts.append(float(max(values[i], min_value)))
    return starts


def _regime_switched_real_path_hedge(
    price_path: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma_signal: float,
    kappa: float,
    sigma_high_cutoff: float,
    bsm_sigma: float,
    rl_model,
) -> tuple[float, float, str]:
    """
    Conservative hybrid:
      - low/mid vol: BSM daily with the supplied sigma signal
      - high vol:    domain-randomized RL policy

    The gate uses only start-of-window information, matching the real-path
    setup used elsewhere in this script.
    """
    if sigma_signal >= sigma_high_cutoff:
        pnl, tc = _run_rl_on_path(price_path, K, T, r, bsm_sigma, kappa, rl_model)
        return pnl, tc, "rl_high_vol"
    pnl, tc = _run_bsm_on_path(price_path, K, T, r, bsm_sigma, kappa)
    return pnl, tc, "bsm_low_mid_vol"


# =============================================================================
# Backtests 1 & 6 — RL and BSM daily on real Nifty price paths
# =============================================================================
print("\n" + "=" * 60)
print("BACKTESTS 1 & 6 — Hedging on real Nifty price paths")
print("=" * 60)

# Use the combined val+test series for more statistical power (~50 windows)
real_paths = _extract_real_paths(combined_prices, window=N_STEP, stride=N_STEP)
print(f"Extracted {len(real_paths)} non-overlapping 21-day windows (val+test period)")

bsm_pnl_real, bsm_tc_real = [], []
rl_pnl_real,  rl_tc_real  = [], []
bsm_pnl_real_lstm, bsm_tc_real_lstm = [], []
rl_pnl_real_lstm,  rl_tc_real_lstm  = [], []
hybrid_pnl_real, hybrid_tc_real = [], []
real_start_moneyness = []
real_sigma_levels = []
real_lstm_sigma_levels = []
hybrid_gate_labels = []

# Use the start-of-window vol estimate so all strategies only see information
# available at hedge inception on that 21-day real-price window.
hist_sigma_windows = _extract_window_starts(combined_hist, window=N_STEP, stride=N_STEP)
lstm_sigma_windows = _extract_window_starts(combined_lstm, window=N_STEP, stride=N_STEP)

for path, sigma_hist, sigma_lstm in zip(real_paths, hist_sigma_windows, lstm_sigma_windows):
    S0 = path[0]
    K  = round(S0 / 50) * 50   # nearest 50-pt ATM strike (Nifty convention)
    real_start_moneyness.append(float(S0 / K))
    real_sigma_levels.append(float(sigma_hist))
    real_lstm_sigma_levels.append(float(sigma_lstm))

    p_bsm, tc_bsm = _run_bsm_on_path(path, K, T_OPT, R, sigma_hist, KAPPA)
    p_rl,  tc_rl  = _run_rl_on_path(path, K, T_OPT, R, sigma_hist, KAPPA, rl_model)
    p_bsm_lstm, tc_bsm_lstm = _run_bsm_on_path(path, K, T_OPT, R, sigma_lstm, KAPPA)
    p_rl_lstm,  tc_rl_lstm  = _run_rl_on_path(
        path, K, T_OPT, R, sigma_lstm, KAPPA, lstm_rl_model, sigma_obs=True
    )
    p_hybrid, tc_hybrid, hybrid_label = _regime_switched_real_path_hedge(
        path,
        K,
        T_OPT,
        R,
        sigma_lstm,
        KAPPA,
        sigma_high_cutoff=0.25,
        bsm_sigma=sigma_lstm,
        rl_model=dr_model,
    )

    bsm_pnl_real.append(p_bsm)
    bsm_tc_real.append(tc_bsm)
    rl_pnl_real.append(p_rl)
    rl_tc_real.append(tc_rl)
    bsm_pnl_real_lstm.append(p_bsm_lstm)
    bsm_tc_real_lstm.append(tc_bsm_lstm)
    rl_pnl_real_lstm.append(p_rl_lstm)
    rl_tc_real_lstm.append(tc_rl_lstm)
    hybrid_pnl_real.append(p_hybrid)
    hybrid_tc_real.append(tc_hybrid)
    hybrid_gate_labels.append(hybrid_label)

bsm_pnl_real = np.array(bsm_pnl_real)
rl_pnl_real  = np.array(rl_pnl_real)
bsm_pnl_real_lstm = np.array(bsm_pnl_real_lstm)
rl_pnl_real_lstm  = np.array(rl_pnl_real_lstm)
hybrid_pnl_real   = np.array(hybrid_pnl_real)

summary_real = pd.DataFrame({
    "Strategy":  [
        "BSM daily (rv21 sigma, real paths)",
        "RL PPO baseline (rv21 sigma, real paths)",
        "BSM daily (LSTM sigma, real paths)",
        "RL PPO (LSTM sigma, real paths)",
        "Hybrid (BSM-LSTM low/mid, DR-RL high)",
    ],
    "Mean PnL":  [
        np.mean(bsm_pnl_real),
        np.mean(rl_pnl_real),
        np.mean(bsm_pnl_real_lstm),
        np.mean(rl_pnl_real_lstm),
        np.mean(hybrid_pnl_real),
    ],
    "Std PnL":   [
        np.std(bsm_pnl_real),
        np.std(rl_pnl_real),
        np.std(bsm_pnl_real_lstm),
        np.std(rl_pnl_real_lstm),
        np.std(hybrid_pnl_real),
    ],
    "MAE":       [
        np.mean(np.abs(bsm_pnl_real)),
        np.mean(np.abs(rl_pnl_real)),
        np.mean(np.abs(bsm_pnl_real_lstm)),
        np.mean(np.abs(rl_pnl_real_lstm)),
        np.mean(np.abs(hybrid_pnl_real)),
    ],
    "Mean TC":   [
        np.mean(bsm_tc_real),
        np.mean(rl_tc_real),
        np.mean(bsm_tc_real_lstm),
        np.mean(rl_tc_real_lstm),
        np.mean(hybrid_tc_real),
    ],
}).set_index("Strategy").round(4)

print(summary_real.to_string())
summary_real.to_csv(RESULTS / "bt_real_paths_summary.csv")

real_analysis = {
    "legacy_bsm_rv21": analyse_hedging_results(
        bsm_pnl_real,
        np.asarray(bsm_tc_real),
        moneyness=np.asarray(real_start_moneyness),
        sigma_proxy=np.asarray(real_sigma_levels),
    ),
    "legacy_rl_ppo": analyse_hedging_results(
        rl_pnl_real,
        np.asarray(rl_tc_real),
        moneyness=np.asarray(real_start_moneyness),
        sigma_proxy=np.asarray(real_sigma_levels),
    ),
    "fair_bsm_lstm_sigma": analyse_hedging_results(
        bsm_pnl_real_lstm,
        np.asarray(bsm_tc_real_lstm),
        moneyness=np.asarray(real_start_moneyness),
        sigma_proxy=np.asarray(real_lstm_sigma_levels),
    ),
    "fair_rl_lstm_sigma": analyse_hedging_results(
        rl_pnl_real_lstm,
        np.asarray(rl_tc_real_lstm),
        moneyness=np.asarray(real_start_moneyness),
        sigma_proxy=np.asarray(real_lstm_sigma_levels),
    ),
    "hybrid_bsm_lstm_dr_highvol": analyse_hedging_results(
        hybrid_pnl_real,
        np.asarray(hybrid_tc_real),
        moneyness=np.asarray(real_start_moneyness),
        sigma_proxy=np.asarray(real_lstm_sigma_levels),
    ),
    "paired_bootstrap": {
        "legacy_bsm_rv21_vs_rl_ppo": paired_bootstrap_mae_diff(bsm_pnl_real, rl_pnl_real),
        "fair_bsm_lstm_vs_rl_lstm": paired_bootstrap_mae_diff(
            bsm_pnl_real_lstm, rl_pnl_real_lstm
        ),
        "fair_bsm_lstm_vs_hybrid": paired_bootstrap_mae_diff(
            bsm_pnl_real_lstm, hybrid_pnl_real
        ),
    },
    "hybrid_gate_counts": {
        "bsm_low_mid_vol": int(sum(label == "bsm_low_mid_vol" for label in hybrid_gate_labels)),
        "rl_high_vol": int(sum(label == "rl_high_vol" for label in hybrid_gate_labels)),
    },
}
with open(RESULTS / "bt_real_paths_analysis.json", "w") as f:
    json.dump(real_analysis, f, indent=2)
print("Saved bt_real_paths_analysis.json")

# Plot — terminal PnL distributions on real paths
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(bsm_pnl_real, bins=30, alpha=0.6, label="BSM daily", density=True)
ax.hist(rl_pnl_real,  bins=30, alpha=0.6, label="RL (PPO)",  density=True)
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_xlabel("Terminal hedging PnL (₹)")
ax.set_ylabel("Density")
ax.set_title("Hedging on real Nifty paths — BSM daily vs RL")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "bt_real_paths_pnl.png", dpi=150)
plt.close(fig)
print("Saved bt_real_paths_pnl.png")

# Plot — fair LSTM-sigma real-path comparison
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(bsm_pnl_real_lstm, bins=30, alpha=0.6, label="BSM daily (LSTM sigma)", density=True)
ax.hist(rl_pnl_real_lstm,  bins=30, alpha=0.6, label="RL PPO (LSTM sigma)", density=True)
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_xlabel("Terminal hedging PnL (₹)")
ax.set_ylabel("Density")
ax.set_title("Fair real-path hedge — BSM vs LSTM-RL with matched sigma signal")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "bt_real_paths_lstm_fair_pnl.png", dpi=150)
plt.close(fig)
print("Saved bt_real_paths_lstm_fair_pnl.png")

# Plot — cumulative MAE over time
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(np.cumsum(np.abs(bsm_pnl_real)), label="BSM daily", lw=2)
ax.plot(np.cumsum(np.abs(rl_pnl_real)),  label="RL (PPO)",  lw=2, ls="--")
ax.set_xlabel("Episode (21-day window)")
ax.set_ylabel("Cumulative absolute PnL error")
ax.set_title("Cumulative hedging error — real Nifty paths")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "bt_real_paths_cumulative.png", dpi=150)
plt.close(fig)
print("Saved bt_real_paths_cumulative.png")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(np.cumsum(np.abs(bsm_pnl_real_lstm)), label="BSM daily (LSTM sigma)", lw=2)
ax.plot(np.cumsum(np.abs(rl_pnl_real_lstm)),  label="RL PPO (LSTM sigma)", lw=2, ls="--")
ax.set_xlabel("Episode (21-day window)")
ax.set_ylabel("Cumulative absolute PnL error")
ax.set_title("Cumulative hedging error — fair LSTM-sigma real-path test")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "bt_real_paths_lstm_fair_cumulative.png", dpi=150)
plt.close(fig)
print("Saved bt_real_paths_lstm_fair_cumulative.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(bsm_pnl_real_lstm, bins=30, alpha=0.5, label="BSM daily (LSTM sigma)", density=True)
ax.hist(hybrid_pnl_real, bins=30, alpha=0.5, label="Hybrid regime switch", density=True)
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_xlabel("Terminal hedging PnL (₹)")
ax.set_ylabel("Density")
ax.set_title("Hybrid real-path hedge vs BSM-LSTM")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "bt_real_paths_hybrid_pnl.png", dpi=150)
plt.close(fig)
print("Saved bt_real_paths_hybrid_pnl.png")


# =============================================================================
# Backtest 2 — Transaction cost sensitivity
# =============================================================================
print("\n" + "=" * 60)
print("BACKTEST 2 — Transaction cost sensitivity sweep")
print("=" * 60)

kappas     = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01]
bsm_maes   = []
rl_maes    = []
bsm_costs  = []
rl_costs   = []

sigma_fixed = float(np.median(hist_vols))   # representative vol for sweep

for kap in kappas:
    b_pnls, b_tcs, r_pnls, r_tcs = [], [], [], []
    for path, sigma in zip(real_paths, hist_sigma_windows):
        S0 = path[0]
        K  = round(S0 / 50) * 50
        p_b, tc_b = _run_bsm_on_path(path, K, T_OPT, R, sigma, kap)
        p_r, tc_r = _run_rl_on_path( path, K, T_OPT, R, sigma, kap, rl_model)
        b_pnls.append(p_b); b_tcs.append(tc_b)
        r_pnls.append(p_r); r_tcs.append(tc_r)

    bsm_maes.append(np.mean(np.abs(b_pnls)))
    rl_maes.append(np.mean(np.abs(r_pnls)))
    bsm_costs.append(np.mean(b_tcs))
    rl_costs.append(np.mean(r_tcs))
    print(f"  kappa={kap:.4f}  BSM MAE={bsm_maes[-1]:.4f}  RL MAE={rl_maes[-1]:.4f}")

df_tc = pd.DataFrame({
    "kappa":    kappas,
    "bsm_mae":  bsm_maes,
    "rl_mae":   rl_maes,
    "bsm_cost": bsm_costs,
    "rl_cost":  rl_costs,
})
df_tc.to_csv(RESULTS / "bt_tc_sensitivity.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot([k * 100 for k in kappas], bsm_maes, "o-", label="BSM daily", lw=2)
ax.plot([k * 100 for k in kappas], rl_maes,  "s--", label="RL (PPO)", lw=2)
ax.set_xlabel("Transaction cost κ (%)")
ax.set_ylabel("MAE of terminal PnL")
ax.set_title("Hedging MAE vs transaction cost")
ax.legend()

ax = axes[1]
ax.plot([k * 100 for k in kappas], bsm_costs, "o-", label="BSM daily", lw=2)
ax.plot([k * 100 for k in kappas], rl_costs,  "s--", label="RL (PPO)", lw=2)
ax.set_xlabel("Transaction cost κ (%)")
ax.set_ylabel("Mean transaction cost paid")
ax.set_title("Transaction costs paid vs κ")
ax.legend()

plt.tight_layout()
fig.savefig(RESULTS / "bt_tc_sensitivity.png", dpi=150)
plt.close(fig)
print("Saved bt_tc_sensitivity.png")


# =============================================================================
# Backtest 3 — Moneyness stress test (pricing horse race at ITM/ATM/OTM)
# =============================================================================
print("\n" + "=" * 60)
print("BACKTEST 3 — Moneyness stress test")
print("=" * 60)

moneyness_levels = {
    "OTM (S/K=0.95)": 0.95,
    "ATM (S/K=1.00)": 1.00,
    "ITM (S/K=1.05)": 1.05,
}

all_summaries = {}
proxy_rng = np.random.default_rng(42)
# Use combined val+test for moneyness stress test (more observations, same logic)
bt_prices_all = combined_prices
bt_dates_all  = combined_dates
bt_hist_all   = combined_hist
bt_lstm_all   = combined_lstm
bt_true_all   = combined_true

for label, mk in moneyness_levels.items():
    # Strike implied by moneyness: K = S / mk → so S/K = mk
    strikes = np.array([round(float(s) / mk / 50) * 50 for s in bt_prices_all])

    # Proxy market price at true forward vol with 4% risk premium
    proxy_vols = np.array([
        max(float(bt_true_all[i]) * 1.04 + proxy_rng.normal(0, 0.005), 0.01)
        for i in range(len(bt_prices_all))
    ])
    mkt_prices = np.array([
        bsm_price(float(bt_prices_all[i]), float(strikes[i]), T_OPT, R, proxy_vols[i])
        for i in range(len(bt_prices_all))
    ])

    # Lagged implied vol: invert yesterday's market price to price today's option.
    # This avoids the circular error of inverting today's price and immediately
    # re-pricing at that same vol (which gives zero error by construction).
    raw_impl = np.array([
        implied_vol(float(mkt_prices[i]), float(bt_prices_all[i]),
                    float(strikes[i]), T_OPT, R)
        for i in range(len(bt_prices_all))
    ])
    # Shift forward by 1: use yesterday's implied vol to price today's option
    lagged_impl = np.concatenate([[raw_impl[0]], raw_impl[:-1]])

    df_bt = backtest_pricing(
        prices=bt_prices_all,
        dates=bt_dates_all,
        hist_vols=bt_hist_all,
        lstm_vols=bt_lstm_all,
        market_prices=mkt_prices,
        K_offsets=strikes,
        T=T_OPT, r=R,
        lagged_impl_vols=lagged_impl,
    )

    summ = summarise_backtest(df_bt)
    all_summaries[label] = summ["mae"]
    print(f"\n{label}")
    print(summ.to_string())

# Combine into one comparison table
df_moneyness = pd.DataFrame(all_summaries).T
df_moneyness.columns.name = "Method"
df_moneyness.index.name   = "Moneyness"
print("\nMAE by moneyness and method:")
print(df_moneyness.round(4).to_string())
df_moneyness.to_csv(RESULTS / "bt_moneyness.csv")

# Grouped bar chart
fig, ax = plt.subplots(figsize=(10, 5))
x       = np.arange(len(df_moneyness))
methods = df_moneyness.columns.tolist()
width   = 0.25

for j, method in enumerate(methods):
    ax.bar(x + j * width, df_moneyness[method],
           width, label=method, alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(df_moneyness.index)
ax.set_ylabel("Mean absolute pricing error")
ax.set_title("Pricing MAE by moneyness and vol source")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "bt_moneyness.png", dpi=150)
plt.close(fig)
print("Saved bt_moneyness.png")


# =============================================================================
# Save consolidated results
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

consolidated = {
    "real_path_hedging": {
        "legacy_bsm_rv21": {"mae": float(np.mean(np.abs(bsm_pnl_real))),
                            "std": float(np.std(bsm_pnl_real))},
        "legacy_rl_ppo":   {"mae": float(np.mean(np.abs(rl_pnl_real))),
                            "std": float(np.std(rl_pnl_real))},
        "fair_bsm_lstm_sigma": {"mae": float(np.mean(np.abs(bsm_pnl_real_lstm))),
                                "std": float(np.std(bsm_pnl_real_lstm))},
        "fair_rl_lstm_sigma":  {"mae": float(np.mean(np.abs(rl_pnl_real_lstm))),
                                "std": float(np.std(rl_pnl_real_lstm))},
        "hybrid_bsm_lstm_dr_highvol": {"mae": float(np.mean(np.abs(hybrid_pnl_real))),
                                       "std": float(np.std(hybrid_pnl_real))},
    },
    "tc_sensitivity": df_tc.to_dict(orient="list"),
    "moneyness":      {k: v.to_dict() for k, v in all_summaries.items()},
    "real_path_analysis": real_analysis,
}
with open(RESULTS / "bt_consolidated.json", "w") as f:
    json.dump(consolidated, f, indent=2)
print("Saved bt_consolidated.json")

print("\nAll backtest outputs in results/")
print("  bt_real_paths_pnl.png       — PnL distributions on real paths")
print("  bt_real_paths_cumulative.png — cumulative MAE over time")
print("  bt_real_paths_lstm_fair_pnl.png — fair LSTM-sigma PnL distributions")
print("  bt_real_paths_lstm_fair_cumulative.png — fair LSTM-sigma cumulative MAE")
print("  bt_real_paths_hybrid_pnl.png — hybrid regime-switch PnL distributions")
print("  bt_tc_sensitivity.png       — MAE and cost vs kappa")
print("  bt_moneyness.png            — pricing MAE by OTM/ATM/ITM")
