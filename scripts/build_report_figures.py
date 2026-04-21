"""
Gather and generate all figures for report.tex.

Copies relevant existing plots from results/ and generates new ones:
  fig_flow_diagram.png    -- project architecture (matplotlib, clean)
  fig_heston_paths.png    -- Heston vs GBM sample paths + variance process
  fig_gamma_surface.png   -- dollar-gamma Γ·S as a function of S/K, τ/T
  fig_ito_decomp.png      -- Itô decomposition: total vs controllable error
  fig_delta_dist.png      -- BSM delta distribution across moneyness/tau

All output goes to report_figures/ at the project root.

Usage:
    PYTHONPATH="" /opt/anaconda3/envs/scaf/bin/python scripts/build_report_figures.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import norm

ROOT   = Path(__file__).parent.parent
RFIGS  = ROOT / "report_figures"
RES    = ROOT / "results"
PROC   = ROOT / "data" / "processed"
RFIGS.mkdir(exist_ok=True)

# ── colour palette (consistent across all figures) ───────────────────────────
C_BSM  = "#2166ac"   # blue
C_RL   = "#d6604d"   # red-orange
C_LSTM = "#4dac26"   # green
C_GARC = "#f4a582"   # light orange
C_HEST = "#762a83"   # purple
C_ITO  = "#1a9641"   # dark green
C_GAMMA = "#e66101"  # amber

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 180,
})

# ─────────────────────────────────────────────────────────────────────────────
# Helper: copy existing result figures
# ─────────────────────────────────────────────────────────────────────────────

def cp(src_name, dst_name):
    src = RES / src_name
    dst = RFIGS / dst_name
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  copied  {src_name} → {dst_name}")
    else:
        print(f"  MISSING {src_name}")

print("=== Copying existing figures ===")
cp("p1_hedge_ratios.png",      "fig_p1_hedge_ratios.png")
cp("p1_pnl_comparison.png",    "fig_p1_pnl_dist.png")
cp("p1_error_boxplot.png",     "fig_p1_error_boxplot.png")
cp("bt_tc_sensitivity.png",    "fig_tc_sensitivity.png")
cp("bt_real_paths_pnl.png",    "fig_real_paths_pnl.png")
cp("novel_rl_comparison.png",  "fig_novel_comparison.png")
cp("novel_rl_pnl_std.png",     "fig_novel_pnl_std.png")
cp("p3_lstm_loss.png",         "fig_p3_loss.png")
cp("p3_vol_forecast.png",      "fig_p3_vol_forecast.png")
cp("p3_lstm_scatter.png",      "fig_p3_scatter.png")
cp("p3_pricing_error_bar.png", "fig_p3_pricing_error.png")
cp("p3_vega_surface.png",      "fig_p3_vega.png")
cp("integration_mae_bar.png",  "fig_integration_mae.png")
cp("integration_pnl.png",      "fig_integration_pnl.png")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Flow diagram
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Generating fig_flow_diagram.png ===")

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis("off")

def box(ax, x, y, w, h, label, color, fontsize=9, subtext=None):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.08",
                          fc=color + "22", ec=color, lw=1.5)
    ax.add_patch(rect)
    if subtext:
        ax.text(x, y + 0.12, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=color)
        ax.text(x, y - 0.2, subtext, ha="center", va="center",
                fontsize=fontsize - 1, color="gray")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=color)

def arrow(ax, x1, y1, x2, y2, color="#555555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                connectionstyle="arc3,rad=0.0"))

def label_arrow(ax, x1, y1, x2, y2, txt, color="#555555"):
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
    ax.text(mx + 0.05, my + 0.1, txt, fontsize=7.5, color=color, ha="center")

# ── row 0: data source ───────────────────────────────────────────────────────
box(ax, 6.5, 6.3, 2.8, 0.7, "NIFTY 50 Daily Prices  (2010–2024)", "#444444", fontsize=9)

# ── COMPONENT 2 (top-right) ──────────────────────────────────────────────────
box(ax, 10.2, 5.1, 2.8, 0.7, "Feature Engineering",
    C_LSTM, subtext="rv5, rv21, rv_ratio, mom5, ...")
box(ax, 10.2, 3.9, 2.8, 0.7, "VolLSTM  (64→32→1)",
    C_LSTM, subtext="Stacked LSTM + Softplus")
box(ax, 10.2, 2.7, 2.8, 0.7, "GARCH(1,1)  benchmark",
    C_GARC, subtext="rolling expanding window")
box(ax, 10.2, 1.5, 2.8, 0.7, "σ̂_lstm  vol forecast",
    C_LSTM, subtext="RMSE 0.0187  (-27% vs GARCH)")

# ── BSM backbone (centre) ────────────────────────────────────────────────────
box(ax, 6.5, 3.2, 2.8, 0.95, "BSM Formula",
    "#7B2D8B", fontsize=10,
    subtext="V, Δ*, Γ, Vega  shared by all")

# ── COMPONENT 1 (top-left) ───────────────────────────────────────────────────
box(ax, 2.8, 5.1, 2.8, 0.7, "HedgingEnv",
    C_RL, subtext="GBM / Heston SDE")
box(ax, 2.8, 3.9, 2.8, 0.7, "PPO Agent",
    C_RL, subtext="π(o_t) → δ_t ∈ [0,1]")
box(ax, 2.8, 2.7, 2.8, 0.7, "Reward",
    C_RL, subtext="−ε²−κ|Δδ|S  or  Itô form")
box(ax, 2.8, 1.5, 2.8, 0.7, "Obs: [S/K, τ/T, δ_prev, ΓS*]",
    C_GAMMA, subtext="*optional gamma obs (Exp E)")

# ── Integration box (bottom centre) ─────────────────────────────────────────
box(ax, 6.5, 1.5, 2.8, 0.7, "Integration",
    C_HEST, subtext="RL + LSTM σ schedule")

# ── results (very bottom) ────────────────────────────────────────────────────
box(ax, 6.5, 0.5, 5.5, 0.6,
    "Results:  MAE · corr(RL,BSM) · TC · RMSE(vol)",
    "#333333", fontsize=9)

# ── arrows ───────────────────────────────────────────────────────────────────
# data → feature eng
arrow(ax, 7.0, 5.95, 9.0, 5.45, "#444444")
# data → hedging env
arrow(ax, 6.0, 5.95, 4.2, 5.45, "#444444")

# comp2 chain
arrow(ax, 10.2, 4.75, 10.2, 4.25, C_LSTM)
arrow(ax, 10.2, 3.55, 10.2, 3.05, C_GARC)   # lstm → garch branch
ax.annotate("", xy=(10.2, 2.05), xytext=(10.2, 3.55),
            arrowprops=dict(arrowstyle="-|>", color=C_LSTM, lw=1.4))

# sigma_lstm → BSM
arrow(ax, 9.2, 1.75, 7.8, 2.85, C_LSTM)
# sigma_lstm → integration
arrow(ax, 9.2, 1.5, 7.8, 1.5, C_HEST)

# comp1 chain
arrow(ax, 2.8, 4.75, 2.8, 4.25, C_RL)
arrow(ax, 2.8, 3.55, 2.8, 3.05, C_RL)
arrow(ax, 2.8, 2.35, 2.8, 1.85, C_RL)

# BSM → hedging env
arrow(ax, 5.5, 3.5, 4.2, 4.85, "#7B2D8B")
# BSM → reward
arrow(ax, 5.5, 2.9, 4.2, 2.85, "#7B2D8B")
# BSM → obs (gamma)
arrow(ax, 5.5, 2.75, 4.2, 1.75, C_GAMMA)

# comp1 obs → PPO
ax.annotate("", xy=(2.8, 4.25), xytext=(2.8, 1.85),
            arrowprops=dict(arrowstyle="-|>", color=C_RL, lw=1.0,
                            connectionstyle="arc3,rad=-0.5"))

# everything → results
arrow(ax, 2.8, 1.15, 4.5, 0.65, C_RL)
arrow(ax, 10.2, 1.15, 8.5, 0.65, C_LSTM)
arrow(ax, 6.5, 1.15, 6.5, 0.8, C_HEST)

# ── component labels ─────────────────────────────────────────────────────────
ax.text(2.8, 6.55, "COMPONENT 1 — RL Hedging", ha="center",
        fontsize=10, fontweight="bold", color=C_RL,
        bbox=dict(fc=C_RL+"18", ec=C_RL, boxstyle="round,pad=0.3", lw=1.2))
ax.text(10.2, 6.55, "COMPONENT 2 — LSTM Forecasting", ha="center",
        fontsize=10, fontweight="bold", color=C_LSTM,
        bbox=dict(fc=C_LSTM+"18", ec=C_LSTM, boxstyle="round,pad=0.3", lw=1.2))
ax.text(6.5, 3.9, "BSM BACKBONE", ha="center",
        fontsize=8, fontstyle="italic", color="#7B2D8B")

plt.tight_layout()
fig.savefig(RFIGS / "fig_flow_diagram.png")
plt.close(fig)
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Heston vs GBM simulated paths + variance process
# ─────────────────────────────────────────────────────────────────────────────
print("=== Generating fig_heston_paths.png ===")

rng = np.random.default_rng(42)
S0, K, T, r = 100.0, 100.0, 21/252, 0.05
n_steps = 21
dt = T / n_steps
kappa, theta, xi, rho, V0 = 2.0, 0.04, 0.30, -0.70, 0.04
sigma_gbm = np.sqrt(theta)   # same starting vol
n_paths = 5

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# --- GBM paths ---
ax = axes[0]
for i in range(n_paths):
    Z = rng.standard_normal(n_steps)
    S = np.zeros(n_steps + 1); S[0] = S0
    for t in range(n_steps):
        S[t+1] = S[t] * np.exp((r - 0.5*sigma_gbm**2)*dt + sigma_gbm*np.sqrt(dt)*Z[t])
    ax.plot(range(n_steps+1), S, lw=1.4, alpha=0.8)
ax.axhline(K, color="black", ls="--", lw=1, label="Strike K")
ax.set_title("GBM: 5 sample paths\n($\\sigma=0.20$ constant)")
ax.set_xlabel("Step"); ax.set_ylabel("Stock price S")
ax.legend(); ax.grid(alpha=0.3)

# --- Heston paths + variance ---
ax  = axes[1]
axv = axes[2]
rho_perp = np.sqrt(1 - rho**2)

for i in range(n_paths):
    S = np.zeros(n_steps + 1); S[0] = S0
    V = np.zeros(n_steps + 1); V[0] = V0
    for t in range(n_steps):
        Z1 = rng.standard_normal()
        Z2 = rng.standard_normal()
        W1, W2 = Z1, rho*Z1 + rho_perp*Z2
        sqV = np.sqrt(max(V[t], 1e-9))
        V[t+1] = max(V[t] + kappa*(theta - V[t])*dt + xi*sqV*np.sqrt(dt)*W2, 1e-6)
        S[t+1] = S[t] * np.exp((r - 0.5*V[t])*dt + sqV*np.sqrt(dt)*W1)
    ax.plot(range(n_steps+1), S, lw=1.4, alpha=0.8)
    axv.plot(range(n_steps+1), np.sqrt(V)*100, lw=1.4, alpha=0.8)

ax.axhline(K, color="black", ls="--", lw=1, label="Strike K")
ax.set_title("Heston: 5 sample paths\n($\\kappa=2,\\,\\theta=0.04,\\,\\xi=0.30,\\,\\rho=-0.70$)")
ax.set_xlabel("Step"); ax.set_ylabel("Stock price S")
ax.legend(); ax.grid(alpha=0.3)

axv.axhline(np.sqrt(theta)*100, color="black", ls="--", lw=1,
            label=f"$\\sqrt{{\\theta}}={np.sqrt(theta)*100:.0f}\\%$")
axv.set_title("Heston: instantaneous vol $\\sqrt{V_t}$ (%)")
axv.set_xlabel("Step"); axv.set_ylabel("Ann. vol (%)")
axv.legend(); axv.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(RFIGS / "fig_heston_paths.png")
plt.close(fig)
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Dollar-gamma surface: Γ·S as a function of S/K and τ/T
# ─────────────────────────────────────────────────────────────────────────────
print("=== Generating fig_gamma_surface.png ===")

from scipy.stats import norm as spnorm

def bsm_gamma(S, K, tau, r, sigma):
    if tau <= 0: return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    return spnorm.pdf(d1) / (S * sigma * np.sqrt(tau))

moneyness = np.linspace(0.70, 1.30, 80)
tau_norm  = np.linspace(0.02, 1.0,  80)
M, TN = np.meshgrid(moneyness, tau_norm)
GS = np.zeros_like(M)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        s = M[i, j] * 100.0
        t = TN[i, j] * (21/252)
        GS[i, j] = bsm_gamma(s, 100.0, t, 0.05, 0.20) * s

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# heatmap
im = axes[0].contourf(M, TN, GS, levels=30, cmap="YlOrRd")
fig.colorbar(im, ax=axes[0], label="Dollar-gamma  $\\Gamma \\cdot S$")
axes[0].set_xlabel("Moneyness  $S/K$")
axes[0].set_ylabel("Normalised time-to-expiry  $\\tau/T$")
axes[0].set_title("Dollar-gamma surface\n"
                  "High = ATM + near-expiry; Low = deep ITM/OTM or long tenor")
axes[0].axvline(1.0, color="white", lw=1.2, ls="--", label="ATM")
axes[0].legend(fontsize=8)

# slices at three τ values
for tau_frac, ls, lbl in [(0.95, "-", "$\\tau/T=0.95$  (just started)"),
                           (0.5,  "--", "$\\tau/T=0.50$  (mid-life)"),
                           (0.05, ":",  "$\\tau/T=0.05$  (near expiry)")]:
    t = tau_frac * (21/252)
    gs_slice = np.array([bsm_gamma(m*100, 100, t, 0.05, 0.20)*m*100
                         for m in moneyness])
    axes[1].plot(moneyness, gs_slice, ls=ls, lw=2, label=lbl)
axes[1].set_xlabel("Moneyness  $S/K$")
axes[1].set_ylabel("Dollar-gamma  $\\Gamma \\cdot S$")
axes[1].set_title("Dollar-gamma slices\n(observation added in Experiment E)")
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].axvline(1.0, color="gray", lw=0.8, ls="--")

plt.tight_layout()
fig.savefig(RFIGS / "fig_gamma_surface.png")
plt.close(fig)
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Itô decomposition illustration over one episode
# ─────────────────────────────────────────────────────────────────────────────
print("=== Generating fig_ito_decomp.png ===")

def bsm_price(S, K, tau, r, sigma):
    if tau <= 0: return max(S-K, 0.0)
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    return S*spnorm.cdf(d1) - K*np.exp(-r*tau)*spnorm.cdf(d2)

def bsm_delta(S, K, tau, r, sigma):
    if tau <= 0: return 1.0 if S > K else 0.0
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    return float(spnorm.cdf(d1))

rng2 = np.random.default_rng(17)
n_steps = 21
dt = (21/252) / n_steps
sigma, r, S0, K = 0.20, 0.05, 100.0, 100.0

total_errors, gamma_parts, ctrl_parts = [], [], []
for ep in range(500):
    S = S0 * rng2.uniform(0.90, 1.10)
    for t in range(n_steps):
        tau = max((n_steps - t) * dt, 1e-9)
        delta_t = bsm_delta(S, K, tau, r, sigma)
        V_before = bsm_price(S, K, tau, r, sigma)
        Z = rng2.standard_normal()
        S_new = S * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
        tau_new = max((n_steps - t - 1)*dt, 1e-9)
        V_after  = bsm_price(S_new, K, tau_new, r, sigma)
        dS = S_new - S
        dV = V_after - V_before
        he = delta_t * dS - dV
        gam = bsm_gamma(S, K, tau, r, sigma)
        gamma_pnl = 0.5 * gam * (dS**2 - sigma**2 * S**2 * dt)
        controllable = he - gamma_pnl
        total_errors.append(he)
        gamma_parts.append(gamma_pnl)
        ctrl_parts.append(controllable)
        S = S_new

total_errors = np.array(total_errors)
gamma_parts  = np.array(gamma_parts)
ctrl_parts   = np.array(ctrl_parts)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
bins = np.linspace(-0.3, 0.3, 60)

axes[0].hist(total_errors, bins=bins, color=C_RL, alpha=0.75, density=True)
axes[0].axvline(0, color="black", lw=1, ls="--")
axes[0].set_title("Total hedging error  $\\varepsilon_t$")
axes[0].set_xlabel("Error"); axes[0].set_ylabel("Density")
axes[0].text(0.05, 0.95, f"std={total_errors.std():.4f}", transform=axes[0].transAxes,
             va="top", fontsize=8)

axes[1].hist(gamma_parts, bins=bins, color=C_GAMMA, alpha=0.75, density=True)
axes[1].axvline(0, color="black", lw=1, ls="--")
axes[1].set_title("Irreducible gamma term\n$\\frac{1}{2}\\Gamma[(\\Delta S)^2 - \\sigma^2 S^2 \\Delta t]$")
axes[1].set_xlabel("Error")
axes[1].text(0.05, 0.95, f"std={gamma_parts.std():.4f}", transform=axes[1].transAxes,
             va="top", fontsize=8)

axes[2].hist(ctrl_parts, bins=bins, color=C_ITO, alpha=0.75, density=True)
axes[2].axvline(0, color="black", lw=1, ls="--")
axes[2].set_title("Controllable part\n$\\varepsilon_t - \\text{gamma term}$")
axes[2].set_xlabel("Error")
axes[2].text(0.05, 0.95, f"std={ctrl_parts.std():.4f}", transform=axes[2].transAxes,
             va="top", fontsize=8)

for ax in axes:
    ax.grid(alpha=0.3)

plt.suptitle("Itô P\\&L decomposition (500 BSM-delta episodes): "
             "subtracting the irreducible term tightens the controllable distribution",
             fontsize=9)
plt.tight_layout()
fig.savefig(RFIGS / "fig_ito_decomp.png")
plt.close(fig)
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 5. LSTM vs GARCH: side-by-side scatter plots with diagonal
# ─────────────────────────────────────────────────────────────────────────────
print("=== Generating fig_lstm_garch_scatter.png ===")

y_true = np.load(PROC / "y_test.npy")
lstm_pred = np.load(PROC / "lstm_test_preds.npy")

# Load GARCH preds if saved, else skip
garch_path = ROOT / "data" / "processed" / "garch_test_preds.npy"
has_garch  = garch_path.exists()

if has_garch:
    garch_pred = np.load(garch_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    lims = [0.05, 0.65]
    for ax, pred, label, color in [
            (axes[0], lstm_pred,  "LSTM",      C_LSTM),
            (axes[1], garch_pred, "GARCH(1,1)", C_GARC)]:
        ax.scatter(y_true, pred, s=8, alpha=0.4, color=color)
        ax.plot(lims, lims, "k--", lw=1, label="Perfect forecast")
        rmse = np.sqrt(np.mean((pred - y_true)**2))
        mae  = np.mean(np.abs(pred - y_true))
        ax.set_title(f"{label}  (RMSE={rmse:.4f}, MAE={mae:.4f})")
        ax.set_xlabel("Realised vol (ann.)"); ax.set_ylabel("Forecast vol (ann.)")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle("Forecast vs realised volatility — test set (NIFTY 50)", fontsize=10)
    plt.tight_layout()
    fig.savefig(RFIGS / "fig_lstm_garch_scatter.png")
    plt.close(fig)
    print("  done (with GARCH)")
else:
    # LSTM only — recreate cleaner version than existing scatter
    fig, ax = plt.subplots(figsize=(5.5, 5))
    lims = [0.05, 0.65]
    ax.scatter(y_true, lstm_pred, s=8, alpha=0.4, color=C_LSTM)
    ax.plot(lims, lims, "k--", lw=1.2, label="Perfect forecast")
    rmse = np.sqrt(np.mean((lstm_pred - y_true)**2))
    mae  = np.mean(np.abs(lstm_pred - y_true))
    ax.set_title(f"LSTM forecast vs realised vol\n(RMSE={rmse:.4f}, MAE={mae:.4f})")
    ax.set_xlabel("Realised vol (ann.)"); ax.set_ylabel("LSTM forecast (ann.)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(RFIGS / "fig_lstm_garch_scatter.png")
    plt.close(fig)
    print("  done (LSTM only — garch_test_preds.npy not found)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. BSM delta across moneyness/tau (theoretical reference figure)
# ─────────────────────────────────────────────────────────────────────────────
print("=== Generating fig_bsm_delta_surface.png ===")

moneyness2 = np.linspace(0.70, 1.30, 120)
tau_vals   = [0.95, 0.70, 0.40, 0.10, 0.02]  # fraction of T
sigma_val  = 0.20; T_val = 21/252

fig, ax = plt.subplots(figsize=(7, 4.5))
for tf in tau_vals:
    tau = tf * T_val
    deltas = [bsm_delta(m*100, 100, tau, 0.05, sigma_val) for m in moneyness2]
    ax.plot(moneyness2, deltas, lw=2, label=f"$\\tau/T={tf:.2f}$")
ax.axvline(1.0, color="gray", lw=0.8, ls="--", label="ATM ($S=K$)")
ax.set_xlabel("Moneyness $S/K$")
ax.set_ylabel("BSM delta $\\Delta^* = \\Phi(d_1)$")
ax.set_title("BSM delta as a function of moneyness and time-to-expiry\n"
             "(RL agent learns to replicate this surface; corr=0.960)")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(RFIGS / "fig_bsm_delta_surface.png")
plt.close(fig)
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary comparison: all 5 novel experiments (cleaner bar chart)
# ─────────────────────────────────────────────────────────────────────────────
print("=== Generating fig_novel_summary.png ===")

import json
with open(RES / "novel_rl_results.json") as f:
    novel = json.load(f)

labels  = ["A\nGBM+MSE", "B\nGBM+Itô", "C\nHeston\n+MSE", "D\nHeston\n+Itô",
           "E\nGBM+Itô\n+ΓS"]
maes    = [r["mae"]      for r in novel]
corrs   = [r["corr"]     for r in novel]
tcs     = [r["mean_tc"]  for r in novel]
std_pnl = [r["std_pnl"]  for r in novel]
colors  = [C_BSM, C_ITO, C_HEST, C_GARC, C_GAMMA]

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
x = np.arange(len(labels))
w = 0.6

for ax, vals, ylabel, title, fmt in [
        (axes[0,0], maes,    "MAE (|terminal PnL|)", "Hedging MAE", "%.3f"),
        (axes[0,1], corrs,   "corr(RL, BSM delta)",  "RL–BSM delta correlation", "%.3f"),
        (axes[1,0], tcs,     "Mean TC per episode",  "Transaction costs", "%.4f"),
        (axes[1,1], std_pnl, "Std dev of terminal PnL", "PnL volatility", "%.3f"),
]:
    bars = ax.bar(x, vals, width=w, color=colors, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt=fmt, padding=2, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)

# add corr ylim
axes[0,1].set_ylim(0, 1.15)
axes[0,1].axhline(1.0, color="gray", lw=0.8, ls="--")

plt.suptitle("Novel experiment results — 5 configurations, 1,000 evaluation episodes each",
             fontsize=10, fontweight="bold")
plt.tight_layout()
fig.savefig(RFIGS / "fig_novel_summary.png")
plt.close(fig)
print("  done")

print(f"\n{'='*50}")
print(f"All figures written to  {RFIGS}")
print(f"Total:  {len(list(RFIGS.glob('*.png')))} files")
