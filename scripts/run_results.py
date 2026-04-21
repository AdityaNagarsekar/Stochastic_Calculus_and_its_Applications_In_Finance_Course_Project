"""
Integration experiment + unified report grid.

Loads trained PPO agent and VolLSTM, feeds LSTM vol schedules into
the RL environment, compares hedging quality vs constant-vol baseline.

Usage:
    python scripts/run_results.py
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
import matplotlib.image as mpimg
import torch

from stable_baselines3 import PPO
from env import HedgingEnv
from lstm_model import VolLSTM, predict
from data_pipeline import load_nifty, make_sequences, split_data, normalize_splits

RESULTS = Path(__file__).parent.parent / "results"
MODELS  = Path(__file__).parent.parent / "models"
RESULTS.mkdir(exist_ok=True)

ENV_PARAMS = dict(S0=100, K=100, T=21/252, r=0.05,
                  sigma=0.20, n_steps=21, kappa=0.001)
N_EPS = 500


# ==========================================================================
# Load checkpoints
# ==========================================================================
print("=" * 60)
print("Loading checkpoints")
print("=" * 60)

rl_model = PPO.load(str(MODELS / "ppo_hedge_v1"))
print("Loaded ppo_hedge_v1")

lstm = VolLSTM(input_size=4, hidden1=64, hidden2=32)
lstm.load_state_dict(torch.load(str(MODELS / "lstm_vol_best.pt"), map_location="cpu"))
lstm.eval()
print("Loaded lstm_vol_best.pt")


# ==========================================================================
# Build LSTM vol schedules from test set
# ==========================================================================
print("\nBuilding LSTM vol schedules...")

df_nifty = load_nifty()
X_all, y_all, dates_all = make_sequences(df_nifty, seq_len=60)
splits = split_data(X_all, y_all, dates_all)
splits, _ = normalize_splits(splits)

lstm_vols_test = predict(lstm, splits["test"]["X"])
print(f"LSTM vol range: [{lstm_vols_test.min():.3f}, {lstm_vols_test.max():.3f}]  "
      f"mean={lstm_vols_test.mean():.3f}")


# ==========================================================================
# (a) RL with constant vol — baseline
# ==========================================================================
print("\n" + "=" * 60)
print("INTEGRATION — RL with constant vol (baseline)")
print("=" * 60)

from backtest import evaluate_rl_agent
results_const = evaluate_rl_agent(rl_model, HedgingEnv, ENV_PARAMS, n_episodes=N_EPS)


# ==========================================================================
# (b) RL with LSTM vol schedule
# ==========================================================================
print("\nINTEGRATION — RL with LSTM vol schedule")

rng = np.random.default_rng(0)
n_test = len(lstm_vols_test)
terminal_pnls_lstm = []
episode_tcs_lstm   = []

for ep in range(N_EPS):
    start    = rng.integers(0, n_test - ENV_PARAMS["n_steps"])
    schedule = lstm_vols_test[start : start + ENV_PARAMS["n_steps"]]
    params   = {**ENV_PARAMS, "sigma_schedule": schedule}
    env      = HedgingEnv(**params)

    obs, _ = env.reset()
    done   = False
    while not done:
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    terminal_pnls_lstm.append(info.get("terminal_pnl", np.nan))
    episode_tcs_lstm.append(info.get("episode_tc", np.nan))

    if (ep + 1) % 100 == 0:
        print(f"  {ep+1}/{N_EPS} episodes done")

pnl_lstm = np.array(terminal_pnls_lstm)
tc_lstm  = np.array(episode_tcs_lstm)


# ==========================================================================
# Summary
# ==========================================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

pnl_const = results_const["terminal_pnl"]
tc_const  = results_const["episode_tc"]

summary = pd.DataFrame({
    "Setup":     ["RL — constant σ", "RL — LSTM σ schedule"],
    "Mean PnL":  [float(np.nanmean(pnl_const)), float(np.nanmean(pnl_lstm))],
    "Std PnL":   [float(np.nanstd(pnl_const)),  float(np.nanstd(pnl_lstm))],
    "MAE":       [float(np.nanmean(np.abs(pnl_const))),
                  float(np.nanmean(np.abs(pnl_lstm)))],
    "Mean TC":   [float(np.nanmean(tc_const)), float(np.nanmean(tc_lstm))],
}).set_index("Setup")

print(summary.round(5).to_string())
summary.to_csv(RESULTS / "integration_summary.csv")
print("Saved integration_summary.csv")


# ==========================================================================
# Plots
# ==========================================================================

# PnL comparison
fig, ax = plt.subplots(figsize=(9, 5))
for label, vals in [("RL — constant σ",      pnl_const),
                     ("RL — LSTM σ schedule", pnl_lstm)]:
    ax.hist(vals[~np.isnan(vals)], bins=60, alpha=0.55, label=label, density=True)
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_xlabel("Terminal hedging PnL")
ax.set_ylabel("Density")
ax.set_title("Integration: RL agent under constant vs LSTM-forecast vol")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "integration_pnl.png", dpi=150)
plt.close(fig)
print("Saved integration_pnl.png")


# ==========================================================================
# Unified report grid — all plots from both projects
# ==========================================================================
print("\nBuilding unified report grid...")

plot_files = [
    ("p3_realised_vol.png",      "Nifty Realised Vol"),
    ("p3_lstm_loss.png",         "LSTM Loss Curves"),
    ("p3_lstm_scatter.png",      "Predicted vs Actual Vol"),
    ("p3_vol_forecast.png",      "Vol Forecast (test)"),
    ("p3_vega_surface.png",      "Vega Sensitivity"),
    ("p3_pricing_error_bar.png", "Pricing Horse Race"),
    ("p3_lstm_vs_implied.png",   "LSTM vs Implied Vol"),
    ("p1_pnl_comparison.png",    "Terminal PnL Distributions"),
    ("p1_hedge_ratios.png",      "Hedge Ratio Paths"),
    ("p1_cost_comparison.png",   "Transaction Costs"),
    ("p1_error_boxplot.png",     "Hedging Error Boxplot"),
    ("integration_pnl.png",      "Integration Result"),
]

available = [(p, t) for p, t in plot_files if (RESULTS / p).exists()]
print(f"  Found {len(available)}/{len(plot_files)} plots")

if available:
    cols = 4
    rows = (len(available) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    for ax, (fname, title) in zip(axes, available):
        img = mpimg.imread(RESULTS / fname)
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(available):]:
        ax.axis("off")
    plt.suptitle("Full Report — Projects 1 & 3", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS / "report_grid.png", dpi=120)
    plt.close(fig)
    print("Saved report_grid.png")

print("\nIntegration complete. All outputs in results/")
