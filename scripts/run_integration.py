"""
Integration experiment — RL agent trained with LSTM vol schedule.

Trains a PPO agent where each episode samples its volatility from the
distribution of LSTM-predicted vols rather than a fixed σ=0.20.
This exposes the agent to realistic vol regimes during training and should
improve hedging under vol uncertainty.

Prerequisite: run run_p3.py first (generates data/processed/lstm_test_preds.npy)
              and run_p1.py  first (generates models/ppo_hedge_v1 baseline)

Usage:
    PYTHONPATH="" /opt/anaconda3/envs/scaf/bin/python scripts/run_integration.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from env import HedgingEnv
from baselines import run_bsm_hedge, record_baseline_metrics
from backtest import evaluate_rl_agent

RESULTS = Path(__file__).parent.parent / "results"
MODELS  = Path(__file__).parent.parent / "models"
PROC    = Path(__file__).parent.parent / "data" / "processed"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# ── Load LSTM predictions ─────────────────────────────────────────────────────
lstm_preds_path = PROC / "lstm_test_preds.npy"
if not lstm_preds_path.exists():
    raise FileNotFoundError(
        "data/processed/lstm_test_preds.npy not found. "
        "Run run_p3.py first to generate LSTM predictions."
    )

lstm_vols_test  = np.load(PROC / "lstm_test_preds.npy")
lstm_vols_train = np.load(PROC / "lstm_train_preds.npy") \
    if (PROC / "lstm_train_preds.npy").exists() else lstm_vols_test

# Combine train + test predictions for the full vol distribution
lstm_vols_all = np.concatenate([lstm_vols_train, lstm_vols_test])
lstm_vols_all = lstm_vols_all[(lstm_vols_all > 0.03) & (lstm_vols_all < 0.80)]

print(f"LSTM vol pool: {len(lstm_vols_all)} values  "
      f"min={lstm_vols_all.min():.3f}  "
      f"mean={lstm_vols_all.mean():.3f}  "
      f"max={lstm_vols_all.max():.3f}")


# ── LSTM-sigma env ────────────────────────────────────────────────────────────

class LSTMHedgingEnv(HedgingEnv):
    """
    HedgingEnv that draws σ from LSTM predictions at the start of each episode.

    The agent is trained across the full distribution of LSTM-predicted vols
    rather than a single fixed σ=0.20, making it vol-aware.
    """

    def __init__(self, lstm_vols: np.ndarray, **kwargs):
        # Initialise with median vol; reset() overrides it each episode
        super().__init__(sigma=float(np.median(lstm_vols)), **kwargs)
        self._lstm_vols = lstm_vols

    def reset(self, seed=None, options=None):
        # Draw a random LSTM vol for this episode
        rng = np.random.default_rng(seed)
        self.sigma = float(rng.choice(self._lstm_vols))
        return super().reset(seed=seed, options=options)


BASE_PARAMS = dict(S0=100, K=100, T=21/252, r=0.05, n_steps=21, kappa=0.001)
N_EPISODES  = 1000


# ── Train LSTM-RL agent ───────────────────────────────────────────────────────
print("=" * 60)
print("Training LSTM-RL agent (500k timesteps)")
print("=" * 60)

vec_env_lstm = make_vec_env(
    lambda: LSTMHedgingEnv(lstm_vols_all, **BASE_PARAMS),
    n_envs=4,
)
eval_env_lstm = LSTMHedgingEnv(lstm_vols_all, **BASE_PARAMS)

eval_cb_lstm = EvalCallback(
    eval_env_lstm,
    best_model_save_path=str(MODELS),
    log_path=str(RESULTS),
    eval_freq=10_000,
    n_eval_episodes=200,
    deterministic=True,
    verbose=0,
)

lstm_rl_model = PPO(
    "MlpPolicy", vec_env_lstm,
    learning_rate=1e-4,
    n_steps=512,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    policy_kwargs=dict(net_arch=[128, 64]),
    verbose=1,
)
lstm_rl_model.learn(total_timesteps=500_000, callback=eval_cb_lstm)
lstm_rl_model.save(str(MODELS / "ppo_lstm_hedge_v1"))
print("Saved models/ppo_lstm_hedge_v1")


# ── Evaluate all agents ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Evaluation — comparing all strategies")
print("=" * 60)

# Load baseline RL (constant vol)
baseline_rl = PPO.load(str(MODELS / "ppo_hedge_v1"))

# Evaluate on standard env (σ=0.20) for fair comparison
std_params = {**BASE_PARAMS, "sigma": 0.20}

daily_errors,  daily_costs  = run_bsm_hedge(std_params, n_episodes=N_EPISODES, rebalance_freq=1)
weekly_errors, weekly_costs = run_bsm_hedge(std_params, n_episodes=N_EPISODES, rebalance_freq=5)

baseline_rl_results = evaluate_rl_agent(baseline_rl, HedgingEnv, std_params, n_episodes=N_EPISODES)
lstm_rl_results     = evaluate_rl_agent(lstm_rl_model, HedgingEnv, std_params, n_episodes=N_EPISODES)

bsm_d  = record_baseline_metrics(daily_errors,  daily_costs,  "BSM daily")
bsm_w  = record_baseline_metrics(weekly_errors, weekly_costs, "BSM weekly")

rl_std_pnl   = baseline_rl_results["terminal_pnl"]
rl_lstm_pnl  = lstm_rl_results["terminal_pnl"]
rl_std_costs = baseline_rl_results["episode_tc"]
rl_lstm_costs= lstm_rl_results["episode_tc"]

metrics_all = [
    bsm_d,
    bsm_w,
    {
        "name":       "RL (constant σ)",
        "mean_error": float(np.mean(rl_std_pnl)),
        "std_error":  float(np.std(rl_std_pnl)),
        "mae":        float(np.mean(np.abs(rl_std_pnl))),
        "mean_cost":  float(np.mean(rl_std_costs)),
    },
    {
        "name":       "RL (LSTM σ)",
        "mean_error": float(np.mean(rl_lstm_pnl)),
        "std_error":  float(np.std(rl_lstm_pnl)),
        "mae":        float(np.mean(np.abs(rl_lstm_pnl))),
        "mean_cost":  float(np.mean(rl_lstm_costs)),
    },
]

print("\nResults:")
for m in metrics_all:
    print(f"  {m['name']:20s}  MAE={m['mae']:.4f}  "
          f"mean={m['mean_error']:+.4f}  std={m['std_error']:.4f}  "
          f"cost={m['mean_cost']:.4f}")

with open(RESULTS / "integration_metrics.json", "w") as f:
    json.dump(metrics_all, f, indent=2)
print("Saved integration_metrics.json")


# ── Plots ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Plots")
print("=" * 60)

# Terminal PnL distributions — all four strategies
fig, ax = plt.subplots(figsize=(10, 5))
for label, vals in [("BSM daily",       daily_errors),
                    ("BSM weekly",      weekly_errors),
                    ("RL (constant σ)", rl_std_pnl),
                    ("RL (LSTM σ)",     rl_lstm_pnl)]:
    ax.hist(vals, bins=60, alpha=0.50, label=label, density=True)
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_xlabel("Terminal hedging PnL")
ax.set_ylabel("Density")
ax.set_title("Terminal PnL — BSM vs RL (constant σ) vs RL (LSTM σ)")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "integration_pnl.png", dpi=150)
plt.close(fig)
print("Saved integration_pnl.png")

# MAE bar chart
fig, ax = plt.subplots(figsize=(8, 4))
names = [m["name"] for m in metrics_all]
maes  = [m["mae"]  for m in metrics_all]
colors = ["#4c72b0", "#55a868", "#c44e52", "#dd8452"]
bars = ax.bar(names, maes, color=colors, width=0.5)
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
ax.set_ylabel("MAE (terminal hedging PnL)")
ax.set_title("Hedging MAE comparison")
plt.tight_layout()
fig.savefig(RESULTS / "integration_mae_bar.png", dpi=150)
plt.close(fig)
print("Saved integration_mae_bar.png")

# Transaction cost comparison
fig, ax = plt.subplots(figsize=(8, 4))
costs = [np.mean(daily_costs), np.mean(weekly_costs),
         np.mean(rl_std_costs), np.mean(rl_lstm_costs)]
bars = ax.bar(names, costs, color=colors, width=0.5)
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
ax.set_ylabel("Mean transaction cost per episode")
ax.set_title("Transaction cost comparison")
plt.tight_layout()
fig.savefig(RESULTS / "integration_tc.png", dpi=150)
plt.close(fig)
print("Saved integration_tc.png")

print("\nIntegration experiment complete. Outputs in results/")
print("  integration_metrics.json")
print("  integration_pnl.png")
print("  integration_mae_bar.png")
print("  integration_tc.png")
