"""
Project 3 — LSTM Volatility Forecasting + Binomial Pricing
Run end-to-end: data pipeline → LSTM train → pricing horse race → plots

Usage:
    python scripts/run_p3.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — saves files, no popup windows
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
from arch import arch_model

from data_pipeline import (
    load_nifty, make_sequences, split_data,
    normalize_splits, save_processed,
)
from nse_options import build_options_dataset, load_options_dataset
from lstm_model import VolLSTM, train, predict, evaluate
from pricing_pipeline import (
    price_option_three_ways,
    vega_sensitivity_analysis,
    backtest_pricing,
    summarise_backtest,
)
from bsm import bsm_price, crr_binomial, implied_vol

RESULTS = Path(__file__).parent.parent / "results"
MODELS  = Path(__file__).parent.parent / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


# ==========================================================================
# Phase 1 — Data pipeline
# ==========================================================================
print("=" * 60)
print("PHASE 1 — Data pipeline")
print("=" * 60)

df = load_nifty(start="2010-01-01", end="2024-12-31")
if len(df) < 100:
    raise RuntimeError(f"Data load failed — only {len(df)} rows. Check internet / yfinance.")
print(f"Loaded {len(df)} rows  |  cols: {list(df.columns)}")
print(df.tail(3).to_string())

# Realised vol plot
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(df.index, df["rv21"],      label="21d realised vol", lw=1)
ax.plot(df.index, df["target_vol"],label="target (fwd 21d)", lw=1, ls="--", alpha=0.7)
ax.set_title("Nifty 50 — Rolling realised vol")
ax.set_ylabel("Annualised sigma")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "p3_realised_vol.png", dpi=150)
plt.close(fig)
print("Saved p3_realised_vol.png")

SEQ_LEN = 60
X, y, dates = make_sequences(df, seq_len=SEQ_LEN)
print(f"\nSequences — X: {X.shape}  y: {y.shape}")

splits = split_data(X, y, dates)
splits, scaler = normalize_splits(splits)

# Feature normalisation sanity check
flat = splits["train"]["X"].reshape(-1, splits["train"]["X"].shape[-1])
print(f"Feature means after normalisation: {flat.mean(axis=0).round(3)}")
print(f"Feature stds  after normalisation: {flat.std(axis=0).round(3)}")

for k, v in splits.items():
    print(f"  {k}: {v['X'].shape}  "
          f"{v['dates'].min().date()} – {v['dates'].max().date()}")

save_processed(splits, scaler)


# ==========================================================================
# Phase 2 — LSTM training
# ==========================================================================
print("\n" + "=" * 60)
print("PHASE 2 — LSTM training")
print("=" * 60)

model, history = train(
    splits["train"]["X"], splits["train"]["y"],
    splits["val"]["X"],   splits["val"]["y"],
    epochs=100,
    lr=1e-3,
    batch_size=64,
    patience=15,
    hidden1=64,
    hidden2=32,
    dropout=0.20,
    checkpoint_name="lstm_vol_best.pt",
)

# Loss curves
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(np.sqrt(history["train_loss"]), label="Train RMSE")
ax.plot(np.sqrt(history["val_loss"]),   label="Val RMSE")
ax.set_xlabel("Epoch")
ax.set_ylabel("RMSE (annualised sigma)")
ax.set_title("LSTM training curves")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "p3_lstm_loss.png", dpi=150)
plt.close(fig)
print("Saved p3_lstm_loss.png")

# Test-set evaluation
y_pred_test = predict(model, splits["test"]["X"])
y_true_test = splits["test"]["y"]

metrics = evaluate(y_true_test, y_pred_test)
print(f"\nTest metrics:")
print(f"  RMSE              : {metrics['rmse']:.4f}")
print(f"  MAE               : {metrics['mae']:.4f}")
print(f"  Directional acc.  : {metrics['directional_accuracy']:.2%}")

# ── GARCH(1,1) benchmark ──────────────────────────────────────────────────────
# Fit on train log-returns (unscaled), forecast a 21-day forward vol for each
# test observation using a rolling/recursive scheme.
print("\nFitting GARCH(1,1) benchmark on train set …")

# Raw (unscaled) log-returns aligned with LSTM splits
train_dates = splits["train"]["dates"]
val_dates   = splits["val"]["dates"]
test_dates_garch = splits["test"]["dates"]

# log_ret for train+val (fit set) then test (eval set)
all_dates   = pd.concat([
    pd.Series(train_dates), pd.Series(val_dates), pd.Series(test_dates_garch)
]).reset_index(drop=True)
df_ret      = df.loc[df.index.isin(all_dates), "log_ret"].sort_index()

n_train_val = len(train_dates) + len(val_dates)
ret_fit     = df_ret.iloc[:n_train_val]
ret_test    = df_ret.iloc[n_train_val:]

# Fit GARCH(1,1) on training data (returns in percent — arch convention)
garch_spec  = arch_model(ret_fit * 100, vol="Garch", p=1, q=1, dist="normal")
garch_res   = garch_spec.fit(disp="off")

# One-step-ahead conditional vol for each test day, then annualise
# arch forecasts variance in (return %)^2 — convert: σ_ann = sqrt(var)/100*sqrt(252)
n_test = len(ret_test)
garch_fc = garch_res.forecast(horizon=1, reindex=False,
                               start=0, align="target")
# Re-fit expanding window for each test step (gives realistic out-of-sample view)
garch_preds = np.empty(n_test, dtype=np.float32)
for i in range(n_test):
    r_window = pd.concat([ret_fit, ret_test.iloc[:i]]) * 100
    spec_i   = arch_model(r_window, vol="Garch", p=1, q=1, dist="normal",
                          rescale=False)
    res_i    = spec_i.fit(disp="off", show_warning=False)
    # 1-step conditional variance forecast
    fc_var   = res_i.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
    # GARCH gives 1-day variance in (pct-return)^2; convert to annualised sigma
    garch_preds[i] = float(np.sqrt(fc_var) / 100 * np.sqrt(252))

garch_metrics = evaluate(y_true_test, garch_preds)
print(f"\nGARCH(1,1) test metrics:")
print(f"  RMSE              : {garch_metrics['rmse']:.4f}")
print(f"  MAE               : {garch_metrics['mae']:.4f}")
print(f"  Directional acc.  : {garch_metrics['directional_accuracy']:.2%}")

# ── Naive persistence baseline (rv21) ─────────────────────────────────────────
# rv21 at date t = trailing 21-day realised vol = σ([t-20, t]).
# target_vol at date t = forward 21-day realised vol = σ([t+1, t+21]).
# These windows overlap by 18 days (86%), so rv21 is a strong predictor of
# target_vol purely through vol clustering — not circularity.  Including it
# as an explicit baseline contextualises both LSTM and the pricing horse race.
rv21_naive = df.loc[splits["test"]["dates"], "rv21"].values.astype(np.float32)
rv21_metrics = evaluate(y_true_test, rv21_naive)

print(f"\nBenchmark comparison (test set):")
print(f"  {'Model':<20} {'RMSE':>8} {'MAE':>8} {'Dir.Acc':>10}")
print(f"  {'LSTM':<20} {metrics['rmse']:>8.4f} {metrics['mae']:>8.4f} "
      f"{metrics['directional_accuracy']:>10.2%}")
print(f"  {'GARCH(1,1)':<20} {garch_metrics['rmse']:>8.4f} "
      f"{garch_metrics['mae']:>8.4f} "
      f"{garch_metrics['directional_accuracy']:>10.2%}")
print(f"  {'Naive rv21':<20} {rv21_metrics['rmse']:>8.4f} "
      f"{rv21_metrics['mae']:>8.4f} "
      f"{rv21_metrics['directional_accuracy']:>10.2%}")

# Save combined metrics
combined_metrics = {"lstm": metrics, "garch": garch_metrics, "rv21_naive": rv21_metrics}

# ── India VIX baseline ────────────────────────────────────────────────────
PROC = Path(__file__).parent.parent / "data" / "processed"
print("\n--- India VIX baseline ---")
from data_pipeline import load_india_vix

try:
    vix_series = load_india_vix()
    # Align India VIX to test dates
    test_dates_idx = splits["test"]["dates"]
    # Forward-fill gaps (VIX not available on all dates)
    vix_aligned = vix_series.reindex(test_dates_idx, method="ffill")
    n_valid = vix_aligned.notna().sum()
    print(f"  India VIX aligned: {n_valid}/{len(test_dates_idx)} dates")

    if n_valid > 10:
        vix_preds = vix_aligned.fillna(vix_aligned.mean()).values.astype(np.float32)
        vix_metrics = evaluate(y_true_test, vix_preds)
        print(f"  India VIX  RMSE={vix_metrics['rmse']:.4f}  "
              f"MAE={vix_metrics['mae']:.4f}  "
              f"dir_acc={vix_metrics['directional_accuracy']:.4f}")

        # Save VIX preds for backtest
        np.save(PROC / "vix_test_preds.npy", vix_preds)

        # Add to combined metrics
        combined_metrics["india_vix"] = vix_metrics
    else:
        print("  India VIX: insufficient aligned data — skipping")
        vix_preds = None
except Exception as e:
    print(f"  India VIX download failed: {e}")
    vix_preds = None

with open(RESULTS / "p3_lstm_metrics.json", "w") as f:
    json.dump(combined_metrics, f, indent=2)

# Save predictions for integration experiment (run_integration.py loads these)
np.save(PROC / "lstm_test_preds.npy",  y_pred_test.astype(np.float32))
np.save(PROC / "lstm_train_preds.npy",
        predict(model, splits["train"]["X"]).astype(np.float32))
# Save GARCH preds so build_report_figures.py can generate the side-by-side scatter
np.save(PROC / "garch_test_preds.npy", garch_preds.astype(np.float32))
print("Saved LSTM + GARCH predictions to data/processed/")

# Save val predictions for extended backtests (run_backtests.py uses val+test period)
np.save(PROC / "lstm_val_preds.npy",
        predict(model, splits["val"]["X"]).astype(np.float32))
print("Saved lstm_val_preds.npy")

# Predicted vs actual scatter — LSTM, GARCH, and naive rv21
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, preds, title in [
    (axes[0], y_pred_test, f"LSTM  (MAE={metrics['mae']:.4f})"),
    (axes[1], garch_preds, f"GARCH(1,1)  (MAE={garch_metrics['mae']:.4f})"),
    (axes[2], rv21_naive,  f"Naive rv21  (MAE={rv21_metrics['mae']:.4f})"),
]:
    lims = [min(y_true_test.min(), preds.min()),
            max(y_true_test.max(), preds.max())]
    ax.scatter(y_true_test, preds, alpha=0.25, s=5)
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Realised fwd vol (target)")
    ax.set_ylabel("Forecast")
    ax.set_title(title)
plt.suptitle("Predicted vs Actual vol (test set)", fontsize=12)
plt.tight_layout()
fig.savefig(RESULTS / "p3_lstm_scatter.png", dpi=150)
plt.close(fig)
print("Saved p3_lstm_scatter.png")

# Vol forecast time series — LSTM vs GARCH vs rv21 vs realised
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(splits["test"]["dates"], y_true_test,  label="Realised fwd vol",  lw=1.5)
ax.plot(splits["test"]["dates"], y_pred_test,  label="LSTM forecast",     lw=1.5, ls="--")
ax.plot(splits["test"]["dates"], garch_preds,  label="GARCH(1,1)",        lw=1.2, ls=":",  alpha=0.85)
ax.plot(splits["test"]["dates"], rv21_naive,   label="Naive rv21",        lw=1.0, ls="-.", alpha=0.75)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_title("Vol Forecast: LSTM vs GARCH(1,1) vs Naive rv21 vs Realised (test set)")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "p3_vol_forecast.png", dpi=150)
plt.close(fig)
print("Saved p3_vol_forecast.png")


# ==========================================================================
# Phase 3 — Pricing pipeline
# ==========================================================================
print("\n" + "=" * 60)
print("PHASE 3 — Pricing pipeline")
print("=" * 60)

# Vega sensitivity analysis
sigma_range = np.linspace(0.05, 0.60, 100)
S_ref, K_ref, r_ref = 100.0, 100.0, 0.065

df_vega = vega_sensitivity_analysis(
    S_ref, K_ref, r_ref, sigma_range,
    maturities=[7/252, 21/252, 63/252],
)

fig, ax = plt.subplots(figsize=(9, 5))
for T_days, grp in df_vega.groupby("T_days"):
    ax.plot(grp["sigma"], grp["vega"], label=f"T = {T_days}d", lw=2)
ax.set_xlabel("Sigma")
ax.set_ylabel("Vega (∂V/∂σ)")
ax.set_title("Vega vs sigma by maturity")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "p3_vega_surface.png", dpi=150)
plt.close(fig)
print("Saved p3_vega_surface.png")

row = df_vega[df_vega["T_days"] == 21].iloc[
    (df_vega[df_vega["T_days"] == 21]["sigma"] - 0.20).abs().argsort().iloc[0]
]
print(f"ATM 1m vega={row['vega']:.3f}  →  1ppt vol error ≈ price error {row['price_err_1pct']:.4f}")

# ── CRR binomial convergence to BSM ──────────────────────────────────────────
# Show how the CRR price converges to the BSM analytical limit as N grows.
# Test across three representative cases: OTM, ATM, ITM.
print("\nCRR binomial convergence vs BSM (T=21d, σ=0.20, r=0.065):")

conv_cases = [
    ("OTM (S/K=0.95)", 95.0,  100.0),
    ("ATM (S/K=1.00)", 100.0, 100.0),
    ("ITM (S/K=1.05)", 105.0, 100.0),
]
N_values = [5, 10, 25, 50, 100, 200, 500, 1000]
T_conv, sigma_conv, r_conv = 21/252, 0.20, 0.065

conv_rows = []
for label, S_c, K_c in conv_cases:
    bsm_ref = bsm_price(S_c, K_c, T_conv, r_conv, sigma_conv)
    for N in N_values:
        crr = crr_binomial(S_c, K_c, T_conv, r_conv, sigma_conv, N)
        conv_rows.append({
            "case":      label,
            "N":         N,
            "bsm_price": bsm_ref,
            "crr_price": crr,
            "abs_error": abs(crr - bsm_ref),
            "rel_error": abs(crr - bsm_ref) / bsm_ref if bsm_ref > 1e-8 else np.nan,
        })

df_conv = pd.DataFrame(conv_rows)
print(df_conv.to_string(index=False))
df_conv.to_csv(RESULTS / "p3_binomial_convergence.csv", index=False)
print("Saved p3_binomial_convergence.csv")

# Plot: abs error vs N for each moneyness case
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
for label, _, _ in conv_cases:
    grp = df_conv[df_conv["case"] == label]
    ax.plot(grp["N"], grp["abs_error"], "o-", lw=2, label=label)
ax.set_xlabel("Number of steps N")
ax.set_ylabel("Absolute error vs BSM")
ax.set_title("CRR binomial convergence — absolute error")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for label, _, _ in conv_cases:
    grp = df_conv[df_conv["case"] == label]
    ax.plot(grp["N"], grp["rel_error"] * 100, "o-", lw=2, label=label)
ax.set_xlabel("Number of steps N")
ax.set_ylabel("Relative error (%)")
ax.set_title("CRR binomial convergence — relative error")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

plt.suptitle("CRR Binomial → BSM (T=21d, σ=20%, r=6.5%)", fontsize=12)
plt.tight_layout()
fig.savefig(RESULTS / "p3_binomial_convergence.png", dpi=150)
plt.close(fig)
print("Saved p3_binomial_convergence.png")

# Sample single-option three-way pricing
test_prices = df.loc[splits["test"]["dates"], "Close"].values
hist_vols   = df.loc[splits["test"]["dates"], "rv21"].values

S_sample = float(test_prices[0])
result = price_option_three_ways(
    S=S_sample,
    K=round(S_sample / 100) * 100,
    T=21/252, r=r_ref,
    sigma_hist=float(hist_vols[0]),
    sigma_lstm=float(y_pred_test[0]),
)
print("\nSample option — three-way pricing:")
print(json.dumps(result, indent=2))


# ==========================================================================
# Phase 4 — Horse race backtest
# ==========================================================================
print("\n" + "=" * 60)
print("PHASE 4 — Horse race backtest")
print("=" * 60)

T_opt, r_opt = 21/252, 0.065
test_dates = splits["test"]["dates"]

# --- Try real NSE option prices first ---
df_opts = load_options_dataset()

if df_opts is None:
    print("No cached option data found. Fetching from NSE Bhav copies...")
    print("(This downloads ~540 daily files — takes ~3 min with polite delays)")
    df_opts = build_options_dataset(df, start=str(test_dates.min().date()),
                                        end=str(test_dates.max().date()))

use_real = df_opts is not None and len(df_opts) > 50

if use_real:
    from bsm import implied_vol as _implied_vol

    df_opts.index = pd.to_datetime(df_opts["date"]).dt.tz_localize(None)
    test_dates_plain = pd.to_datetime(test_dates).tz_localize(None)

    # Sanity filter: only keep rows where the implied vol (assuming T=21/252)
    # falls in [8%, 35%] — the realistic range for a genuine 21-day Nifty ATM
    # call. Options outside this range are either near-expiry (IV<8%) or
    # exotic strikes/events (IV>35%).
    def _iv_ok(row):
        iv = _implied_vol(row["market_price"], row["spot"],
                          row["atm_strike"], T_opt, r_opt)
        return 0.08 <= iv <= 0.35

    df_opts_valid = df_opts[df_opts.apply(_iv_ok, axis=1)]
    use_real = len(df_opts_valid) >= 50     # demote to proxy if too few rows

    if use_real:
        common = test_dates_plain[test_dates_plain.isin(df_opts_valid.index)]
        idx    = [list(test_dates_plain).index(d) for d in common]

        market_prices = df_opts_valid.loc[common, "market_price"].values
        bt_prices     = test_prices[idx]
        bt_hist       = hist_vols[idx]
        bt_lstm       = y_pred_test[idx]
        bt_garch      = garch_preds[idx]
        bt_dates      = common
        bt_k_offsets  = df_opts_valid.loc[common, "atm_strike"].values.astype(float)

        n_filtered = len(df_opts) - len(df_opts_valid)
        print(f"Using {len(common)} real NSE option prices "
              f"(dropped {n_filtered} near-expiry / anomalous rows)")
        use_real = False   # signal proxy fallback below
        print(f"Only {len(df_opts_valid)} NSE prices passed the IV sanity check "
              f"(need ≥50) — falling back to proxy market prices")

if not use_real:
    # Proxy: BSM at realised forward vol + realistic implied vol premium.
    # Nifty options typically trade at 2-5% vol premium over realised vol
    # (volatility risk premium). We add that plus a small bid-ask spread.
    if not (df_opts is not None and len(df_opts) > 50):
        print("NSE fetch failed or insufficient data — using realistic proxy market prices")
    print("  (proxy = BSM at realised_vol × 1.04 + N(0, 0.5%) noise)")
    rng       = np.random.default_rng(42)
    test_true = splits["test"]["y"]
    # Use nearest-50-point strikes (Nifty convention); keep these for K_offsets
    proxy_strikes = np.array([
        round(float(test_prices[i]) / 50) * 50
        for i in range(len(test_prices))
    ])
    market_prices = np.array([
        bsm_price(
            float(test_prices[i]),
            float(proxy_strikes[i]),
            T_opt, r_opt,
            max(float(test_true[i]) * 1.04 + rng.normal(0, 0.005), 0.01),
        )
        for i in range(len(test_prices))
    ])
    bt_prices    = test_prices
    bt_hist      = hist_vols
    bt_lstm      = y_pred_test
    bt_garch     = garch_preds
    bt_dates     = test_dates
    bt_k_offsets = proxy_strikes

# Diagnostic: confirm LSTM vols are non-trivially non-zero
print(f"\nLSTM vol stats before backtest: "
      f"min={bt_lstm.min():.4f}  max={bt_lstm.max():.4f}  "
      f"mean={bt_lstm.mean():.4f}  zeros={np.sum(bt_lstm == 0)}")
assert bt_lstm.max() > 0.01, (
    f"bt_lstm appears to be all zeros (max={bt_lstm.max():.6f}). "
    "The LSTM model may not have been trained correctly."
)

# Compute 1-day-lagged implied vols to avoid circular pricing error
# (same-day inversion gives zero error by construction)
raw_impl = np.array([
    implied_vol(float(market_prices[i]), float(bt_prices[i]),
                float(bt_k_offsets[i]), T_opt, r_opt)
    for i in range(len(bt_prices))
])
lagged_impl = np.concatenate([[raw_impl[0]], raw_impl[:-1]])

df_bt = backtest_pricing(
    prices=bt_prices,
    dates=bt_dates,
    hist_vols=bt_hist,
    lstm_vols=bt_lstm,
    market_prices=market_prices,
    K_offsets=bt_k_offsets,
    T=T_opt,
    r=r_opt,
    garch_vols=bt_garch,
    lagged_impl_vols=lagged_impl,
)

summary = summarise_backtest(df_bt)
print("\nPricing horse race — summary:")
print(summary.to_string())

df_bt.to_csv(RESULTS / "p3_pricing_backtest.csv", index=False)
summary.to_csv(RESULTS / "p3_pricing_summary.csv")
print("Saved p3_pricing_backtest.csv, p3_pricing_summary.csv")

# India VIX pricing backtest (if available)
if vix_preds is not None:
    df_vix_backtest = backtest_pricing(
        prices=bt_prices, dates=bt_dates,
        hist_vols=bt_hist, lstm_vols=bt_lstm,
        market_prices=market_prices, K_offsets=bt_k_offsets,
        T=T_opt, r=r_opt, garch_vols=bt_garch,
        vix_vols=vix_preds,
    )
    sum_vix = summarise_backtest(df_vix_backtest)
    print("\nPricing results (with India VIX):")
    print(sum_vix[["mae","mean_error"]].to_string())
    df_vix_backtest.to_csv(RESULTS / "p3_pricing_vix_backtest.csv", index=False)

with open(RESULTS / "p3_lstm_metrics.json", "w") as f:
    json.dump(combined_metrics, f, indent=2)
print("Saved updated p3_lstm_metrics.json")

# Pricing error bar chart
fig, ax = plt.subplots(figsize=(7, 4))
summary["mae"].plot.bar(ax=ax, width=0.5)
ax.set_ylabel("Mean absolute pricing error")
ax.set_title("Pricing horse race — MAE by vol source")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
fig.savefig(RESULTS / "p3_pricing_error_bar.png", dpi=150)
plt.close(fig)
print("Saved p3_pricing_error_bar.png")

# LSTM vs implied vol scatter
lstm_rows = df_bt[df_bt["method"] == "lstm"]
impl_rows = df_bt[df_bt["method"] == "implied"]
merged = lstm_rows[["date", "sigma"]].merge(
    impl_rows[["date", "sigma"]], on="date", suffixes=("_lstm", "_impl")
)
if not merged.empty:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(merged["sigma_impl"], merged["sigma_lstm"], alpha=0.3, s=5)
    lims = [0, max(merged["sigma_impl"].max(), merged["sigma_lstm"].max()) * 1.05]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Implied sigma")
    ax.set_ylabel("LSTM sigma")
    ax.set_title("LSTM forecast vs Implied vol")
    plt.tight_layout()
    fig.savefig(RESULTS / "p3_lstm_vs_implied.png", dpi=150)
    plt.close(fig)
    print("Saved p3_lstm_vs_implied.png")

print("\nProject 3 complete. All outputs in results/")
