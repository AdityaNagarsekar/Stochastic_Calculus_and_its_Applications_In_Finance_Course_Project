"""
Novel experiments — testing three ideas from stochastic calculus theory.

Idea 1 — Heston env:
    Under GBM, BSM delta is provably optimal; RL can't improve on it.
    Under Heston stochastic vol, vol itself moves — BSM delta (assuming
    constant vol) is suboptimal and RL, which observes the current variance
    state, can potentially outperform it.

Idea 2 — Itô reward:
    Itô's lemma decomposes the hedge P&L into a controllable delta-mismatch
    term and an uncontrollable gamma term.  The standard MSE reward penalises
    both.  The Itô reward removes the irreducible gamma P&L before squaring,
    giving the agent a cleaner training signal.

Idea 3 — Gamma observation:
    From Itô's lemma the per-step residual is ½Γ(dS²−σ²S²dt).  Adding
    dollar-gamma Γ·S as an observation lets the RL agent learn a
    gamma-regime-aware policy: hedge more actively when Γ·S is high
    (near-ATM, near-expiry) and conserve TC when it is low.

Experiments
-----------
RL  (each trained 300k steps, evaluated on 1000 episodes):
    A. GBM + MSE reward              (baseline)
    B. GBM + Itô reward              (idea 2 only)
    C. Heston + MSE reward           (idea 1 only)
    D. Heston + Itô reward           (ideas 1+2 combined)
    E. GBM + Itô reward + gamma obs  (ideas 2+3 combined)

Usage:
    PYTHONPATH="" /opt/anaconda3/envs/scaf/bin/python scripts/run_novel.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env        import HedgingEnv
from heston_env import HestonEnv
from backtest   import evaluate_rl_agent
from bsm        import bsm_delta as _bsm_delta

RESULTS = Path(__file__).parent.parent / "results"
MODELS  = Path(__file__).parent.parent / "models"
PROC    = Path(__file__).parent.parent / "data" / "processed"
RESULTS.mkdir(exist_ok=True)

# ── Shared config ─────────────────────────────────────────────────────────────
GBM_PARAMS = dict(S0=100, K=100, T=21/252, r=0.05,
                  sigma=0.20, n_steps=21, kappa=0.001)
HESTON_PARAMS = dict(S0=100, K=100, T=21/252, r=0.05,
                     kappa=2.0, theta=0.04, xi=0.30, rho=-0.70,
                     V0=0.04, n_steps=21, kappa_tc=0.001)
PPO_KWARGS = dict(
    learning_rate=1e-4, n_steps=512, batch_size=128,
    n_epochs=10, gamma=0.99,
    policy_kwargs=dict(net_arch=[128, 64]),
    verbose=0,
)
TRAIN_STEPS = 300_000
N_EVAL      = 1_000
N_ENVS      = 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def train_ppo(env_fn, name: str) -> PPO:
    print(f"  Training {name} ({TRAIN_STEPS//1000}k steps)…", end=" ", flush=True)
    t0 = time.time()
    vec = make_vec_env(env_fn, n_envs=N_ENVS)
    model = PPO("MlpPolicy", vec, **PPO_KWARGS)
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(str(MODELS / name))
    print(f"done in {time.time()-t0:.0f}s")
    return model


def eval_rl(model, env_cls, params, name: str, sigma_for_bsm=None) -> dict:
    """
    Evaluate model and compute hedge-ratio correlation against BSM delta.
    For HestonEnv, sigma_for_bsm=None triggers vol-state-aware BSM delta.
    """
    results = evaluate_rl_agent(model, env_cls, params,
                                n_episodes=N_EVAL,
                                sigma_for_bsm=sigma_for_bsm)
    pnl  = results["terminal_pnl"]
    tc   = results["episode_tc"]
    all_rl  = np.concatenate(results["delta_paths"])
    all_bsm = np.concatenate(results["bsm_delta_paths"])
    corr = float(np.corrcoef(all_bsm, all_rl)[0, 1])
    mae  = float(np.mean(np.abs(pnl)))
    print(f"  {name:30s}  MAE={mae:.4f}  "
          f"corr(RL,BSM)={corr:.3f}  TC={np.mean(tc):.4f}")
    return dict(name=name, mae=mae, corr=corr,
                mean_pnl=float(np.mean(pnl)),
                std_pnl =float(np.std(pnl)),
                mean_tc =float(np.mean(tc)))


# ── Patch evaluate_rl_agent to support vol-state-aware BSM for Heston ─────────
# We extend the function inline here rather than modifying backtest.py further.

def _eval_heston(model, params, name: str) -> dict:
    """
    Like eval_rl but computes BSM delta using the vol-state from each obs[3].
    obs[3] = sqrt(V)/sqrt(theta) → sigma_eff = obs[3] * sqrt(theta)
    """
    env = HestonEnv(**params)
    terminal_pnl = []
    episode_tc   = []
    delta_paths  = []
    bsm_delta_paths = []
    sqrt_theta = np.sqrt(params["theta"])

    K = params["K"]
    T = params["T"]
    r = params["r"]

    for _ in range(N_EVAL):
        obs, _  = env.reset()
        done    = False
        deltas  = []
        bsm_dels = []
        while not done:
            S_t    = float(obs[0]) * K
            tau_t  = max(float(obs[1]) * T, 1e-9)
            sig_t  = float(obs[3]) * sqrt_theta          # vol-state-aware σ
            bsm_dels.append(_bsm_delta(S_t, K, tau_t, r, sig_t))
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            deltas.append(float(action[0]))
        terminal_pnl.append(info.get("terminal_pnl", np.nan))
        episode_tc.append(info.get("episode_tc", np.nan))
        delta_paths.append(deltas)
        bsm_delta_paths.append(bsm_dels)

    pnl  = np.array(terminal_pnl)
    tc   = np.array(episode_tc)
    all_rl  = np.concatenate(delta_paths)
    all_bsm = np.concatenate(bsm_delta_paths)
    corr = float(np.corrcoef(all_bsm, all_rl)[0, 1])
    mae  = float(np.mean(np.abs(pnl)))
    print(f"  {name:30s}  MAE={mae:.4f}  "
          f"corr(RL,BSM_vol-aware)={corr:.3f}  TC={np.mean(tc):.4f}")
    return dict(name=name, mae=mae, corr=corr,
                mean_pnl=float(np.mean(pnl)),
                std_pnl =float(np.std(pnl)),
                mean_tc =float(np.mean(tc)))


# ══════════════════════════════════════════════════════════════════════════════
# RL EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("RL EXPERIMENTS  (300k steps each)")
print("=" * 60)

rl_results = []

# A — GBM + MSE (baseline)
mA = train_ppo(lambda: HedgingEnv(**GBM_PARAMS),
               "novel_A_gbm_mse")
rl_results.append(eval_rl(mA, HedgingEnv, GBM_PARAMS, "A: GBM + MSE reward"))

# B — GBM + Itô reward
gbm_ito = {**GBM_PARAMS, "ito_reward": True}
mB = train_ppo(lambda: HedgingEnv(**gbm_ito),
               "novel_B_gbm_ito")
rl_results.append(eval_rl(mB, HedgingEnv, gbm_ito, "B: GBM + Itô reward"))

# C — Heston + MSE reward
mC = train_ppo(lambda: HestonEnv(**HESTON_PARAMS),
               "novel_C_heston_mse")
rl_results.append(_eval_heston(mC, HESTON_PARAMS, "C: Heston + MSE reward"))

# D — Heston + Itô reward
class HestonItoEnv(HestonEnv):
    """HestonEnv with the Itô-decomposed reward."""
    def step(self, action):
        from bsm import bsm_gamma
        obs, reward, terminated, truncated, info = super().step(action)
        # Recompute with Itô reward: remove irreducible gamma P&L
        delta_new   = float(np.clip(action[0], 0.0, 1.0))
        sig         = info["sigma_eff"]
        tau_before  = max(self.T - (self.t - 1) * self.dt, 1e-9)
        gamma_t     = bsm_gamma(self.S_prev, self.K, tau_before, self.r, sig)
        gamma_pnl   = 0.5 * gamma_t * (
            (self.S - self.S_prev) ** 2 - sig ** 2 * self.S_prev ** 2 * self.dt
        )
        hedge_error = info["hedge_error"]
        controllable = hedge_error - gamma_pnl
        tc = info["transaction_cost"]
        reward = float(-controllable ** 2 - tc)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.S_prev = self.S   # ensure S_prev is set before first step
        return obs, info

mD = train_ppo(lambda: HestonItoEnv(**HESTON_PARAMS),
               "novel_D_heston_ito")
rl_results.append(_eval_heston(mD, HESTON_PARAMS, "D: Heston + Itô reward"))

# E — GBM + Itô reward + gamma observation (ideas 2+3)
gbm_ito_gamma = {**GBM_PARAMS, "ito_reward": True, "gamma_obs": True}
mE = train_ppo(lambda: HedgingEnv(**gbm_ito_gamma),
               "novel_E_gbm_ito_gamma")
rl_results.append(eval_rl(mE, HedgingEnv, gbm_ito_gamma,
                           "E: GBM + Itô + gamma obs",
                           sigma_for_bsm=GBM_PARAMS["sigma"]))

# ── RL summary table ──────────────────────────────────────────────────────────
print("\nRL Results Summary:")
print(f"  {'Experiment':<35}  {'MAE':>7}  {'corr':>7}  {'mean_PnL':>9}  {'TC':>7}")
for r in rl_results:
    print(f"  {r['name']:<35}  {r['mae']:>7.4f}  {r['corr']:>7.3f}  "
          f"{r['mean_pnl']:>+9.4f}  {r['mean_tc']:>7.4f}")

with open(RESULTS / "novel_rl_results.json", "w") as f:
    json.dump(rl_results, f, indent=2)
print("Saved novel_rl_results.json")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PLOTS")
print("=" * 60)

names  = [r["name"].split(":")[1].strip() for r in rl_results]
maes   = [r["mae"]   for r in rl_results]
corrs  = [r["corr"]  for r in rl_results]
stds   = [r["std_pnl"] for r in rl_results]
colors = ["#4c72b0", "#55a868", "#c44e52", "#dd8452", "#9467bd"]

# ── RL MAE + correlation (2-panel) ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
bars = ax.bar(names, maes, color=colors, width=0.55)
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
ax.set_ylabel("MAE (|terminal hedging PnL|)")
ax.set_title("Hedging MAE by environment & reward")
ax.tick_params(axis="x", labelsize=7.5)

ax = axes[1]
bars = ax.bar(names, corrs, color=colors, width=0.55)
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
ax.set_ylabel("corr(RL delta, BSM delta)")
ax.set_title("RL–BSM delta correlation  (GBM: high = replicates BSM)")
ax.set_ylim(0, 1.15)
ax.axhline(1.0, color="grey", lw=0.8, ls="--")
ax.tick_params(axis="x", labelsize=7.5)

plt.tight_layout()
fig.savefig(RESULTS / "novel_rl_comparison.png", dpi=150)
plt.close(fig)
print("Saved novel_rl_comparison.png")

# ── PnL std-dev comparison ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(names, stds, color=colors, width=0.55)
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
ax.set_ylabel("Std dev of terminal PnL")
ax.set_title("PnL volatility by environment & reward  (lower = more consistent)")
ax.tick_params(axis="x", labelsize=7.5)
plt.tight_layout()
fig.savefig(RESULTS / "novel_rl_pnl_std.png", dpi=150)
plt.close(fig)
print("Saved novel_rl_pnl_std.png")

print("\nNovel experiments complete. Outputs in results/")
print("  novel_rl_results.json")
print("  novel_rl_comparison.png")
print("  novel_rl_pnl_std.png")
