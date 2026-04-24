"""
Train and evaluate BSM-anchored RL controller variants.

Goal:
    Beat the current real-path baseline "BSM daily (LSTM sigma)" by keeping
    BSM as the structural hedge and using RL only for execution control.

Usage:
    python scripts/run_anchor_models.py
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
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from anchor_env import BSMAnchorHedgingEnv, DiscreteBSMAnchorHedgingEnv
from bsm import bsm_delta, bsm_price
from backtest import analyse_hedging_results, paired_bootstrap_mae_diff
from data_pipeline import load_nifty, make_sequences, split_data, normalize_splits
from lstm_model import VolLSTM, predict


RESULTS = Path(__file__).parent.parent / "results"
MODELS = Path(__file__).parent.parent / "models"
PROC = Path(__file__).parent.parent / "data" / "processed"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

BASE_PARAMS = dict(S0=100, K=100, T=21 / 252, r=0.05, n_steps=21, kappa=0.001)
REAL_R = 0.065
N_STEP = 21
TOTAL_TIMESTEPS = 250_000


def _load_lstm_splits():
    df = load_nifty()
    X, y, dates = make_sequences(df, seq_len=60)
    splits = split_data(X, y, dates)
    splits, _ = normalize_splits(splits)

    lstm = VolLSTM(input_size=6, hidden1=64, hidden2=32)
    lstm.load_state_dict(torch.load(str(MODELS / "lstm_vol_best.pt"), map_location="cpu"))
    lstm.eval()

    def _pred(split_name: str) -> np.ndarray:
        path = PROC / f"lstm_{split_name}_preds.npy"
        if path.exists():
            return np.load(path)
        return predict(lstm, splits[split_name]["X"])

    return df, splits, {
        "train": _pred("train"),
        "val": _pred("val"),
        "test": _pred("test"),
    }


def _extract_real_paths(prices: np.ndarray, window: int = N_STEP, stride: int = N_STEP):
    return [prices[i: i + window + 1] for i in range(0, len(prices) - window, stride)]


def _extract_window_starts(values: np.ndarray, window: int = N_STEP, stride: int = N_STEP,
                           min_value: float = 1e-6):
    return [float(max(values[i], min_value)) for i in range(0, len(values) - window, stride)]


def _run_bsm_on_path(price_path: np.ndarray, K: float, T: float,
                     r: float, sigma: float, kappa: float):
    n = len(price_path) - 1
    dt = T / n
    S = price_path[0]
    portfolio = bsm_price(S, K, T, r, sigma)
    delta = 0.0
    tc_sum = 0.0
    for t in range(n):
        tau = max(T - t * dt, 1e-9)
        new_delta = bsm_delta(S, K, tau, r, sigma)
        tc = kappa * abs(new_delta - delta) * S
        tc_sum += tc
        portfolio -= tc
        delta = new_delta
        S_new = price_path[t + 1]
        portfolio += delta * (S_new - S)
        S = S_new
    payoff = max(S - K, 0.0)
    return float(portfolio - payoff), float(tc_sum)


def _run_partial_bsm_on_path(price_path: np.ndarray, K: float, T: float, r: float,
                             sigma: float, kappa: float, alpha: float, theta: float):
    n = len(price_path) - 1
    dt = T / n
    S = price_path[0]
    portfolio = bsm_price(S, K, T, r, sigma)
    delta = 0.0
    tc_sum = 0.0
    for t in range(n):
        tau = max(T - t * dt, 1e-9)
        delta_target = float(bsm_delta(S, K, tau, r, sigma))
        gap = delta_target - delta
        if abs(gap) <= theta:
            new_delta = delta
        else:
            new_delta = float(np.clip(delta + alpha * gap, 0.0, 1.0))
        tc = kappa * abs(new_delta - delta) * S
        tc_sum += tc
        portfolio -= tc
        delta = new_delta
        S_new = price_path[t + 1]
        portfolio += delta * (S_new - S)
        S = S_new
    payoff = max(S - K, 0.0)
    return float(portfolio - payoff), float(tc_sum)


def _run_anchor_model_on_path(price_path: np.ndarray, K: float, T: float, r: float,
                              sigma: float, kappa: float, model, env_cls, env_params: dict):
    eval_env_params = dict(env_params)
    eval_env_params.pop("sigma_pool", None)
    eval_env_params.pop("sigma_range", None)
    eval_env_params["sigma"] = sigma
    eval_env_params["kappa"] = kappa
    env = env_cls(**eval_env_params)
    n = len(price_path) - 1
    dt = T / n
    S = float(price_path[0])
    portfolio = bsm_price(S, K, T, r, sigma)
    delta = 0.0
    tc_sum = 0.0
    delta_path = []
    bsm_path = []

    for t in range(n):
        env.S = S
        env.t = t
        env.delta_prev = delta
        env.sigma = sigma
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        new_delta = env.action_to_delta(action, obs)
        tau = max(T - t * dt, 1e-9)
        delta_bsm = bsm_delta(S, K, tau, r, sigma)
        tc = kappa * abs(new_delta - delta) * S
        tc_sum += tc
        portfolio -= tc
        S_new = float(price_path[t + 1])
        portfolio += new_delta * (S_new - S)
        S = S_new
        delta = new_delta
        delta_path.append(float(new_delta))
        bsm_path.append(float(delta_bsm))

    payoff = max(S - K, 0.0)
    return float(portfolio - payoff), float(tc_sum), delta_path, bsm_path


def _episode_state_means(delta_paths, bsm_paths, sigmas):
    turnovers = []
    delta_rmse = []
    for deltas, bsm_deltas in zip(delta_paths, bsm_paths):
        d = np.asarray(deltas, dtype=float)
        b = np.asarray(bsm_deltas, dtype=float)
        turnovers.append(float(np.sum(np.abs(np.diff(d)))) if d.size > 1 else 0.0)
        delta_rmse.append(float(np.sqrt(np.mean((d - b) ** 2))) if d.size else np.nan)
    return {
        "turnover_mean": float(np.nanmean(turnovers)),
        "delta_rmse_mean": float(np.nanmean(delta_rmse)),
        "sigma_mean": float(np.mean(sigmas)),
    }


def _train_or_load(name: str, env_cls, env_kwargs: dict, algo: str = "ppo"):
    model_path = MODELS / name
    if model_path.with_suffix(".zip").exists():
        print(f"Loading existing {name}")
        return PPO.load(str(model_path)) if algo == "ppo" else DQN.load(str(model_path))

    print("=" * 60)
    print(f"Training {name} ({TOTAL_TIMESTEPS:,} timesteps)")
    print("=" * 60)
    if algo == "ppo":
        vec_env = make_vec_env(lambda: env_cls(**env_kwargs), n_envs=4)
    else:
        vec_env = env_cls(**env_kwargs)
    eval_env = env_cls(**env_kwargs)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS),
        log_path=str(RESULTS),
        eval_freq=10_000,
        n_eval_episodes=200,
        deterministic=True,
        verbose=0,
    )
    if algo == "ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,
            n_steps=512,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            policy_kwargs=dict(net_arch=[128, 64]),
            verbose=1,
        )
    else:
        model = DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=2_000,
            batch_size=128,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.30,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[128, 64]),
            verbose=1,
        )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_cb)
    model.save(str(model_path))
    print(f"Saved {model_path.name}")
    return model


def _evaluate_real_paths(model_name: str, model, env_cls, env_kwargs: dict,
                         real_paths, sigma_windows, kappa: float):
    pnl = []
    tc = []
    delta_paths = []
    bsm_paths = []
    sigma_used = []
    for path, sigma in zip(real_paths, sigma_windows):
        S0 = path[0]
        K = round(S0 / 50) * 50
        p, c, d_path, b_path = _run_anchor_model_on_path(
            path, K, BASE_PARAMS["T"], REAL_R, sigma, kappa, model, env_cls, env_kwargs
        )
        pnl.append(p)
        tc.append(c)
        delta_paths.append(d_path)
        bsm_paths.append(b_path)
        sigma_used.append(float(sigma))
    pnl = np.asarray(pnl)
    tc = np.asarray(tc)
    summary = analyse_hedging_results(pnl, tc, sigma_proxy=np.asarray(sigma_used))
    summary["extras"] = _episode_state_means(delta_paths, bsm_paths, sigma_used)
    summary["model_name"] = model_name
    return summary, pnl, tc


def _evaluate_partial_bsm(real_paths, sigma_windows, alpha: float, theta: float):
    pnl = []
    tc = []
    for path, sigma in zip(real_paths, sigma_windows):
        K = round(path[0] / 50) * 50
        p, c = _run_partial_bsm_on_path(path, K, BASE_PARAMS["T"], REAL_R, sigma,
                                        BASE_PARAMS["kappa"], alpha, theta)
        pnl.append(p)
        tc.append(c)
    pnl = np.asarray(pnl)
    tc = np.asarray(tc)
    summary = analyse_hedging_results(pnl, tc, sigma_proxy=np.asarray(sigma_windows))
    summary["model_name"] = f"partial_bsm_lstm_a{alpha:.2f}_t{theta:.2f}"
    summary["alpha"] = float(alpha)
    summary["theta"] = float(theta)
    return summary, pnl, tc


def main():
    df, splits, lstm_preds = _load_lstm_splits()
    train_pool = np.asarray(lstm_preds["train"], dtype=float)
    train_pool = train_pool[(train_pool > 0.03) & (train_pool < 0.80)]
    train_plus_val_pool = np.concatenate([lstm_preds["train"], lstm_preds["val"]])
    train_plus_val_pool = train_plus_val_pool[(train_plus_val_pool > 0.03) & (train_plus_val_pool < 0.80)]

    print(f"Train LSTM sigma pool: n={train_pool.size} mean={train_pool.mean():.3f}")
    print(f"Train+Val LSTM sigma pool: n={train_plus_val_pool.size} mean={train_plus_val_pool.mean():.3f}")

    configs = [
        {
            "name": "ppo_anchor_fraction_dr_v1",
            "algo": "ppo",
            "env_cls": BSMAnchorHedgingEnv,
            "env": {**BASE_PARAMS, "sigma_range": (0.10, 0.40), "action_mode": "fraction",
                    "anchor_penalty": 0.00, "gamma_obs": True},
        },
        {
            "name": "ppo_anchor_fraction_dr_reg_v1",
            "algo": "ppo",
            "env_cls": BSMAnchorHedgingEnv,
            "env": {**BASE_PARAMS, "sigma_range": (0.10, 0.40), "action_mode": "fraction",
                    "anchor_penalty": 0.05, "gamma_obs": True},
        },
        {
            "name": "ppo_anchor_fraction_lstm_v1",
            "algo": "ppo",
            "env_cls": BSMAnchorHedgingEnv,
            "env": {**BASE_PARAMS, "sigma_pool": train_pool, "action_mode": "fraction",
                    "anchor_penalty": 0.02, "gamma_obs": True},
        },
        {
            "name": "ppo_anchor_residual_lstm_v1",
            "algo": "ppo",
            "env_cls": BSMAnchorHedgingEnv,
            "env": {**BASE_PARAMS, "sigma_pool": train_plus_val_pool, "action_mode": "residual",
                    "residual_scale": 0.12, "anchor_penalty": 0.02, "gamma_obs": True},
        },
        {
            "name": "dqn_anchor_discrete_lstm_v1",
            "algo": "dqn",
            "env_cls": DiscreteBSMAnchorHedgingEnv,
            "env": {**BASE_PARAMS, "sigma_pool": train_pool, "anchor_penalty": 0.01,
                    "gamma_obs": True, "action_levels": [0.0, 0.25, 0.5, 0.75, 1.0]},
        },
        {
            "name": "dqn_anchor_discrete_lstm_fine_v1",
            "algo": "dqn",
            "env_cls": DiscreteBSMAnchorHedgingEnv,
            "env": {**BASE_PARAMS, "sigma_pool": train_plus_val_pool, "anchor_penalty": 0.01,
                    "gamma_obs": True, "action_levels": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]},
        },
    ]

    models = {}
    for cfg in configs:
        models[cfg["name"]] = _train_or_load(cfg["name"], cfg["env_cls"], cfg["env"], cfg["algo"])

    test_dates = splits["test"]["dates"]
    val_dates = splits["val"]["dates"]
    test_prices = df.loc[test_dates, "Close"].values.astype(float)
    val_prices = df.loc[val_dates, "Close"].values.astype(float)
    combined_prices = np.concatenate([val_prices, test_prices])
    combined_lstm = np.concatenate([lstm_preds["val"], lstm_preds["test"]])
    real_paths = _extract_real_paths(combined_prices, window=N_STEP, stride=N_STEP)
    lstm_sigma_windows = _extract_window_starts(combined_lstm, window=N_STEP, stride=N_STEP)
    val_paths = _extract_real_paths(val_prices, window=N_STEP, stride=N_STEP)
    val_sigmas = _extract_window_starts(lstm_preds["val"], window=N_STEP, stride=N_STEP)
    test_paths = _extract_real_paths(test_prices, window=N_STEP, stride=N_STEP)
    test_sigmas = _extract_window_starts(lstm_preds["test"], window=N_STEP, stride=N_STEP)

    bsm_pnl = []
    bsm_tc = []
    for path, sigma in zip(real_paths, lstm_sigma_windows):
        S0 = path[0]
        K = round(S0 / 50) * 50
        p, c = _run_bsm_on_path(path, K, BASE_PARAMS["T"], REAL_R, sigma, BASE_PARAMS["kappa"])
        bsm_pnl.append(p)
        bsm_tc.append(c)
    bsm_pnl = np.asarray(bsm_pnl)
    bsm_tc = np.asarray(bsm_tc)
    bsm_summary = analyse_hedging_results(bsm_pnl, bsm_tc, sigma_proxy=np.asarray(lstm_sigma_windows))
    bsm_summary["model_name"] = "bsm_lstm_real_paths"

    rows = [{
        "name": "bsm_lstm_real_paths",
        "mae": bsm_summary["mae"],
        "mean_error": bsm_summary["mean_error"],
        "std_error": bsm_summary["std_error"],
        "mean_cost": bsm_summary["mean_cost"],
    }]
    full = {"bsm_lstm_real_paths": bsm_summary}

    model_pnls = {"bsm_lstm_real_paths": bsm_pnl}

    for cfg in configs:
        summary, pnl, tc = _evaluate_real_paths(
            cfg["name"], models[cfg["name"]], cfg["env_cls"], cfg["env"],
            real_paths, lstm_sigma_windows, BASE_PARAMS["kappa"]
        )
        full[cfg["name"]] = summary
        model_pnls[cfg["name"]] = pnl
        rows.append({
            "name": cfg["name"],
            "mae": summary["mae"],
            "mean_error": summary["mean_error"],
            "std_error": summary["std_error"],
            "mean_cost": summary["mean_cost"],
            "delta_rmse_mean": summary["extras"]["delta_rmse_mean"],
            "turnover_mean": summary["extras"]["turnover_mean"],
        })

    for cfg in configs:
        full[f"compare__bsm_lstm_vs__{cfg['name']}"] = paired_bootstrap_mae_diff(
            bsm_pnl, model_pnls[cfg["name"]]
        )

    grid = []
    for alpha in [0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00]:
        for theta in [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            val_summary, val_pnl, _ = _evaluate_partial_bsm(val_paths, val_sigmas, alpha, theta)
            test_summary, test_pnl, _ = _evaluate_partial_bsm(test_paths, test_sigmas, alpha, theta)
            combined_summary, combined_pnl, _ = _evaluate_partial_bsm(real_paths, lstm_sigma_windows, alpha, theta)
            grid.append({
                "alpha": alpha,
                "theta": theta,
                "val_mae": val_summary["mae"],
                "test_mae": test_summary["mae"],
                "combined_mae": combined_summary["mae"],
                "combined_mean_error": combined_summary["mean_error"],
                "combined_cost": combined_summary["mean_cost"],
                "combined_summary": combined_summary,
                "combined_pnl": combined_pnl,
                "test_pnl": test_pnl,
            })

    best_by_val = min(grid, key=lambda row: (row["val_mae"], row["test_mae"]))
    best_by_test = min(grid, key=lambda row: row["test_mae"])
    heuristic_row = {
        "name": f"partial_bsm_lstm(alpha={best_by_val['alpha']:.2f}, theta={best_by_val['theta']:.2f})",
        "mae": best_by_val["combined_mae"],
        "mean_error": best_by_val["combined_mean_error"],
        "std_error": best_by_val["combined_summary"]["std_error"],
        "mean_cost": best_by_val["combined_cost"],
    }
    rows.append(heuristic_row)
    full["partial_bsm_lstm_best_by_val"] = best_by_val["combined_summary"]
    full["partial_bsm_lstm_best_by_val"]["selection"] = {
        "alpha": float(best_by_val["alpha"]),
        "theta": float(best_by_val["theta"]),
        "val_mae": float(best_by_val["val_mae"]),
        "test_mae": float(best_by_val["test_mae"]),
        "combined_mae": float(best_by_val["combined_mae"]),
    }
    full["partial_bsm_lstm_best_test_candidate"] = {
        "alpha": float(best_by_test["alpha"]),
        "theta": float(best_by_test["theta"]),
        "val_mae": float(best_by_test["val_mae"]),
        "test_mae": float(best_by_test["test_mae"]),
        "combined_mae": float(best_by_test["combined_mae"]),
    }
    full["compare__bsm_lstm_vs__partial_bsm_lstm_best_by_val"] = paired_bootstrap_mae_diff(
        bsm_pnl, best_by_val["combined_pnl"]
    )

    df_rows = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    df_rows.to_csv(RESULTS / "anchor_model_real_path_summary.csv", index=False)
    with open(RESULTS / "anchor_model_results.json", "w") as f:
        json.dump(full, f, indent=2)

    print("\nReal-path ranking:")
    print(df_rows.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_rows["name"], df_rows["mae"], color="#2c7fb8")
    ax.axhline(bsm_summary["mae"], color="black", ls="--", lw=1)
    ax.set_ylabel("Real-path MAE")
    ax.set_title("BSM-anchor controller variants on real Nifty paths")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    fig.savefig(RESULTS / "anchor_model_real_path_mae.png", dpi=150)
    plt.close(fig)
    print("Saved anchor_model_real_path_mae.png")


if __name__ == "__main__":
    main()
