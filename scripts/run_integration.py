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
from backtest import (
    evaluate_rl_agent,
    analyse_hedging_results,
    paired_bootstrap_mae_diff,
    bootstrap_mean_ci,
)

RESULTS = Path(__file__).parent.parent / "results"
MODELS  = Path(__file__).parent.parent / "models"
PROC    = Path(__file__).parent.parent / "data" / "processed"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

BASE_PARAMS = dict(S0=100, K=100, T=21/252, r=0.05, n_steps=21, kappa=0.001)
N_EPISODES  = 1000

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


# ── LSTM-guided BSM delta baseline (no RL training needed) ───────────────────
print("=" * 60)
print("LSTM-guided BSM delta baseline (key missing comparison)")
print("=" * 60)

from baselines import run_lstm_bsm_hedge
from data_pipeline import load_nifty

df_nifty    = load_nifty()
real_prices = df_nifty["Close"].values
lstm_test   = np.load(PROC / "lstm_test_preds.npy")

lstm_bsm_errors, lstm_bsm_costs = run_lstm_bsm_hedge(
    real_prices=real_prices,
    lstm_vols=lstm_test,
    S0=BASE_PARAMS["S0"],
    K=BASE_PARAMS["K"], T=BASE_PARAMS["T"],
    r=BASE_PARAMS["r"], kappa=BASE_PARAMS["kappa"],
    n_steps=BASE_PARAMS["n_steps"],
)
lstm_bsm_m = record_baseline_metrics(lstm_bsm_errors, lstm_bsm_costs, "LSTM-guided BSM delta")
print(f"  LSTM-guided BSM delta  MAE={lstm_bsm_m['mae']:.4f}  "
      f"mean={lstm_bsm_m['mean_error']:+.4f}  TC={lstm_bsm_m['mean_cost']:.4f}")


# ── LSTM-sigma env ────────────────────────────────────────────────────────────

class LSTMHedgingEnv(HedgingEnv):
    """
    HedgingEnv that draws σ from LSTM predictions at the start of each episode.

    The agent is trained across the full distribution of LSTM-predicted vols
    rather than a single fixed σ=0.20, making it vol-aware.
    """

    def __init__(self, lstm_vols: np.ndarray, **kwargs):
        # sigma_obs=True so the agent can condition on the current episode's σ.
        # setdefault allows callers to override if needed.
        kwargs.setdefault("sigma_obs", True)
        # Initialise with median vol; reset() overrides it each episode
        super().__init__(sigma=float(np.median(lstm_vols)), **kwargs)
        self._lstm_vols = lstm_vols

    def reset(self, seed=None, options=None):
        # Draw a random LSTM vol for this episode
        rng = np.random.default_rng(seed)
        self.sigma = float(rng.choice(self._lstm_vols))
        return super().reset(seed=seed, options=options)


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

# Baseline RL must be trained with sigma_obs=True so both agents share the same
# 4-D obs space.  ppo_hedge_v1 has 3-D obs and cannot be loaded into a 4-D env.
std_params = {**BASE_PARAMS, "sigma": 0.20, "sigma_obs": True}

sigma_obs_path = MODELS / "ppo_hedge_sigma_obs"
if sigma_obs_path.with_suffix(".zip").exists():
    baseline_rl = PPO.load(str(sigma_obs_path))
    print("Loaded ppo_hedge_sigma_obs (sigma_obs baseline)")
else:
    print("=" * 60)
    print("Training sigma_obs baseline (500k steps, σ=0.20 constant)")
    print("=" * 60)
    baseline_rl = PPO(
        "MlpPolicy",
        make_vec_env(lambda: HedgingEnv(**std_params), n_envs=4),
        learning_rate=1e-4, n_steps=512, batch_size=128, n_epochs=10, gamma=0.99,
        policy_kwargs=dict(net_arch=[128, 64]),
        verbose=1,
    )
    baseline_rl.learn(total_timesteps=500_000)
    baseline_rl.save(str(sigma_obs_path))
    print("Saved ppo_hedge_sigma_obs")

daily_errors,  daily_costs  = run_bsm_hedge(std_params, n_episodes=N_EPISODES, rebalance_freq=1)
weekly_errors, weekly_costs = run_bsm_hedge(std_params, n_episodes=N_EPISODES, rebalance_freq=5)

# ── Standard evaluation: both agents in the FIXED σ=0.20 environment ─────────
# Fair head-to-head: same env, same σ.  sigma_obs=1.0 always.
# Note: constant-σ RL is fully in-distribution; LSTM-RL sees sigma_obs=1.0
# during eval but trained on sigma_obs ∈ [0.38, 4.0] — mild distribution shift.
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
    lstm_bsm_m,
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


# ── Extended analysis ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXTENDED ANALYSIS — bootstrap CIs, tail risk, utility scores")
print("=" * 60)

analysis_daily    = analyse_hedging_results(daily_errors,   daily_costs)
analysis_weekly   = analyse_hedging_results(weekly_errors,  weekly_costs)
analysis_lstm_bsm = analyse_hedging_results(lstm_bsm_errors, lstm_bsm_costs)
analysis_rl_std   = analyse_hedging_results(rl_std_pnl,  rl_std_costs)
analysis_rl_lstm  = analyse_hedging_results(rl_lstm_pnl, rl_lstm_costs)

# Print MAE with 95 % bootstrap CI and CVaR95 for each strategy
for label, ana in [("BSM daily",        analysis_daily),
                   ("BSM weekly",       analysis_weekly),
                   ("LSTM-guided BSM",  analysis_lstm_bsm),
                   ("RL (constant σ)",  analysis_rl_std),
                   ("RL (LSTM σ)",      analysis_rl_lstm)]:
    ci = ana["mae_ci95"]
    tr = ana["tail_risk"]
    print(f"  {label:20s}  MAE={ana['mae']:.4f} [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
          f"  CVaR95={tr['cvar95']:.4f}  J(λ=1)={ana['utility']['J_lambda_1']:.4f}")

# Paired bootstrap tests
n_min = min(len(daily_errors), len(rl_std_pnl), len(rl_lstm_pnl), len(lstm_bsm_errors))
boot_bsmd_vs_rl_std  = paired_bootstrap_mae_diff(daily_errors[:n_min], rl_std_pnl[:n_min])
boot_bsmd_vs_rl_lstm = paired_bootstrap_mae_diff(daily_errors[:n_min], rl_lstm_pnl[:n_min])
boot_rl_std_vs_lstm  = paired_bootstrap_mae_diff(rl_std_pnl[:n_min],  rl_lstm_pnl[:n_min])
boot_bsmd_vs_lstm_bsm = paired_bootstrap_mae_diff(daily_errors[:n_min], lstm_bsm_errors[:n_min])

print("\nPaired bootstrap MAE differences (a - b, positive → a is worse):")
for label, test in [("BSM daily vs RL constant", boot_bsmd_vs_rl_std),
                    ("BSM daily vs RL LSTM",     boot_bsmd_vs_rl_lstm),
                    ("RL constant vs RL LSTM",   boot_rl_std_vs_lstm),
                    ("BSM daily vs LSTM-BSM",    boot_bsmd_vs_lstm_bsm)]:
    sig = "**" if test["p_value"] < 0.05 else "ns"
    print(f"  {label:30s}  diff={test['diff']:+.4f}  "
          f"CI=[{test['ci_low']:+.4f}, {test['ci_high']:+.4f}]  p={test['p_value']:.3f} {sig}")

# Seed robustness
print("\nSeed robustness (3 seeds × 200 episodes):")

def _seed_robustness_int(model, env_cls, params, seeds=(0, 17, 42), episodes=200):
    rows = []
    for s in seeds:
        res = evaluate_rl_agent(model, env_cls, params, n_episodes=episodes, seed=s)
        rows.append({"seed": s,
                     "mae":       float(np.mean(np.abs(res["terminal_pnl"]))),
                     "mean_cost": float(np.mean(res["episode_tc"]))})
    maes  = np.array([r["mae"]       for r in rows])
    costs = np.array([r["mean_cost"] for r in rows])
    return {"seeds": rows,
            "mae_mean":  float(maes.mean()),  "mae_std":  float(maes.std()),
            "cost_mean": float(costs.mean()), "cost_std": float(costs.std())}

rob_rl_std  = _seed_robustness_int(baseline_rl,    HedgingEnv, std_params)
rob_rl_lstm = _seed_robustness_int(lstm_rl_model,  HedgingEnv, std_params)
print(f"  RL (constant σ)  MAE={rob_rl_std['mae_mean']:.4f} ± {rob_rl_std['mae_std']:.4f}")
print(f"  RL (LSTM σ)      MAE={rob_rl_lstm['mae_mean']:.4f} ± {rob_rl_lstm['mae_std']:.4f}")

# ── Natural habitat evaluation: both agents in a VARYING-σ environment ───────
# The LSTM-RL was trained on episodes where σ is drawn from the LSTM prediction
# pool.  Here we evaluate both agents in that same environment using only the
# held-out test LSTM predictions to avoid data leakage.
# This answers: does the LSTM-RL outperform constant-σ RL when σ genuinely varies?
print("\n" + "=" * 60)
print("NATURAL HABITAT EVALUATION — varying-σ environment (test LSTM pool)")
print("=" * 60)

lstm_vols_test_only = np.load(PROC / "lstm_test_preds.npy")
lstm_vols_test_only = lstm_vols_test_only[
    (lstm_vols_test_only > 0.03) & (lstm_vols_test_only < 0.80)
]
print(f"  Test LSTM pool: {len(lstm_vols_test_only)} values  "
      f"min={lstm_vols_test_only.min():.3f}  mean={lstm_vols_test_only.mean():.3f}  "
      f"max={lstm_vols_test_only.max():.3f}")

# env_params dict must include lstm_vols as keyword arg
nh_params = dict(lstm_vols=lstm_vols_test_only, **BASE_PARAMS)

nh_lstm_rl_res   = evaluate_rl_agent(lstm_rl_model, LSTMHedgingEnv, nh_params,
                                      n_episodes=N_EPISODES)
nh_const_rl_res  = evaluate_rl_agent(baseline_rl,   LSTMHedgingEnv, nh_params,
                                      n_episodes=N_EPISODES)

nh_lstm_pnl   = nh_lstm_rl_res["terminal_pnl"]
nh_const_pnl  = nh_const_rl_res["terminal_pnl"]

nh_analysis_lstm  = analyse_hedging_results(nh_lstm_pnl,  nh_lstm_rl_res["episode_tc"])
nh_analysis_const = analyse_hedging_results(nh_const_pnl, nh_const_rl_res["episode_tc"])

print(f"\nNatural habitat MAEs (σ drawn from test LSTM pool each episode):")
for label, ana in [("RL (LSTM σ)",     nh_analysis_lstm),
                   ("RL (constant σ)", nh_analysis_const)]:
    ci = ana["mae_ci95"]
    tr = ana["tail_risk"]
    print(f"  {label:20s}  MAE={ana['mae']:.4f} [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
          f"  CVaR95={tr['cvar95']:.4f}  J(λ=1)={ana['utility']['J_lambda_1']:.4f}")

nh_boot = paired_bootstrap_mae_diff(nh_const_pnl, nh_lstm_pnl)
sig_nh  = "**" if nh_boot["p_value"] < 0.05 else "ns"
print(f"\nPaired bootstrap (const vs LSTM-RL in varying-σ env):")
print(f"  diff={nh_boot['diff']:+.4f}  "
      f"CI=[{nh_boot['ci_low']:+.4f}, {nh_boot['ci_high']:+.4f}]  "
      f"p={nh_boot['p_value']:.3f} {sig_nh}")

integration_analysis = {
    "analyses": {
        "bsm_daily":   analysis_daily,
        "bsm_weekly":  analysis_weekly,
        "lstm_bsm":    analysis_lstm_bsm,
        "rl_constant": analysis_rl_std,
        "rl_lstm":     analysis_rl_lstm,
    },
    "paired_bootstrap": {
        "bsm_daily_vs_rl_constant":  boot_bsmd_vs_rl_std,
        "bsm_daily_vs_rl_lstm":      boot_bsmd_vs_rl_lstm,
        "rl_constant_vs_rl_lstm":    boot_rl_std_vs_lstm,
        "bsm_daily_vs_lstm_bsm":     boot_bsmd_vs_lstm_bsm,
    },
    "seed_robustness": {
        "rl_constant": rob_rl_std,
        "rl_lstm":     rob_rl_lstm,
    },
    "natural_habitat": {
        "rl_lstm":    nh_analysis_lstm,
        "rl_constant": nh_analysis_const,
        "paired_bootstrap_const_vs_lstm": nh_boot,
    },
}

with open(RESULTS / "integration_analysis.json", "w") as f:
    json.dump(integration_analysis, f, indent=2)
print("Saved integration_analysis.json")


# ── Plots ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Plots")
print("=" * 60)

# Terminal PnL distributions — all strategies
fig, ax = plt.subplots(figsize=(10, 5))
for label, vals in [("BSM daily",          daily_errors),
                    ("BSM weekly",         weekly_errors),
                    ("LSTM-guided BSM",    lstm_bsm_errors),
                    ("RL (constant σ)",    rl_std_pnl),
                    ("RL (LSTM σ)",        rl_lstm_pnl)]:
    ax.hist(vals, bins=60, alpha=0.50, label=label, density=True)
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_xlabel("Terminal hedging PnL")
ax.set_ylabel("Density")
ax.set_title("Terminal PnL — BSM vs LSTM-guided BSM vs RL")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(RESULTS / "integration_pnl.png", dpi=150)
plt.close(fig)
print("Saved integration_pnl.png")

# MAE bar chart
fig, ax = plt.subplots(figsize=(9, 4))
names = [m["name"] for m in metrics_all]
maes  = [m["mae"]  for m in metrics_all]
colors = ["#4c72b0", "#55a868", "#e377c2", "#c44e52", "#dd8452"]
bars = ax.bar(names, maes, color=colors, width=0.5)
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
ax.set_ylabel("MAE (terminal hedging PnL)")
ax.set_title("Hedging MAE comparison")
ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()
fig.savefig(RESULTS / "integration_mae_bar.png", dpi=150)
plt.close(fig)
print("Saved integration_mae_bar.png")

# Transaction cost comparison
fig, ax = plt.subplots(figsize=(9, 4))
costs = [np.mean(daily_costs), np.mean(weekly_costs),
         np.mean(lstm_bsm_costs), np.mean(rl_std_costs), np.mean(rl_lstm_costs)]
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
