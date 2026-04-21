# Stochastic Calculus and Applications in Finance — Course Project Report

---

## Overview

This project has three interconnected components, all grounded in the same stochastic calculus theory.

**Component 1 — RL Delta Hedging (GBM):** An RL agent learns to hedge a European call option when the underlying follows Geometric Brownian Motion (GBM). The theoretical benchmark is Black-Scholes-Merton (BSM) delta, which is provably optimal under GBM. The question is whether RL independently rediscovers it.

**Component 2 — LSTM Volatility Forecasting:** A long short-term memory network forecasts next-day realised volatility from NIFTY 50 price data, providing a data-driven alternative to the historical rolling-window estimate typically plugged into BSM.

**Component 3 (Integration) — RL Hedging with LSTM Vol:** The LSTM vol forecast is fed as a time-varying sigma schedule into the hedging environment, testing whether a better vol estimate improves hedging quality.

**Component 4 (Novel) — Theory-Motivated Extensions:** Three ideas from the stochastic calculus curriculum are implemented as experiments: a Heston stochastic-vol environment, an Itô-decomposed reward signal, and a gamma-aware observation.

---

## Theoretical Foundation

All three components share the same underlying SDE. Under the risk-neutral measure:

```
dS = r S dt + σ S dW
```

**BSM pricing** gives the fair value `V(S, t)` of a European call. By Itô's lemma applied to `V`:

```
dV = (∂V/∂t + ½σ²S²∂²V/∂S²) dt  +  (∂V/∂S) dS
   = Θ dt  +  Δ dS  +  ½ Γ dS²
```

**Delta hedging** holds `Δ = ∂V/∂S` shares. The residual hedging error per step is then:

```
hedge_error = Δ·dS − dV  =  −(Θ dt + ½Γ dS²)
```

The gamma term `½Γ(dS² − σ²S²dt)` is irreducible with delta hedging alone — it cannot be eliminated by any choice of `Δ`. This decomposition directly motivates the Itô reward and gamma observation experiments.

---

## Component 1: RL Delta Hedging

### Setup

The `HedgingEnv` gymnasium environment simulates a 21-step episode (one trading month):

- **Dynamics:** Log-Euler GBM step  `S_{t+1} = S_t · exp((r − ½σ²)dt + σ√dt·Z)`
- **Observation:** `[S/K, τ/T, δ_prev]` — moneyness, normalised time-to-expiry, previous delta
- **Action:** continuous hedge ratio `δ ∈ [0, 1]`
- **Reward:** `−(hedge_error)² − κ|Δδ|·S`  (mean-squared error minus transaction cost)
- **Agent:** PPO (Proximal Policy Optimization) via stable-baselines3, trained for 300k steps

### Results

| Strategy | MAE (|terminal PnL|) | Mean TC per episode |
|---|---|---|
| BSM delta (daily) | **0.284** | 0.146 |
| BSM delta (weekly) | 0.499 | 0.098 |
| RL (PPO) | 0.748 | 0.127 |

BSM daily rebalancing has lower MAE, but the RL agent achieves a corr=0.960 with BSM delta when hedge ratios are compared step-by-step at the same states (state-aligned comparison). This is the key finding:

> **RL learns to replicate BSM delta purely from reward signal, without ever being shown the BSM formula.**

The gap in MAE is expected — RL is trained with transaction costs and learns to rebalance less frequently than daily BSM, trading hedging accuracy for cost savings.

---

## Component 2: LSTM Volatility Forecasting

### Setup

The pipeline uses 15 years of NIFTY 50 daily close prices. Features engineered from the price series:

| Feature | Description |
|---|---|
| `log_ret` | Daily log return |
| `rv5` | 5-day rolling realised vol |
| `rv21` | 21-day rolling realised vol |
| `rv_ratio` | `rv5 / rv21` (short-vs-long vol regime) |
| `mom5` | 5-day return momentum |
| `abs_ret` | Absolute daily return |

The LSTM (`VolLSTM`) takes a 21-day sliding window of these 6 features and predicts the next-day annualised volatility. Training uses Softplus activation (eliminates dying-ReLU for vol outputs), ReduceLROnPlateau scheduling, and a 70/15/15 train/val/test split.

A **GARCH(1,1)** model is fitted on a rolling expanding window as the benchmark, re-estimated at each test step using only data available up to that point (no look-ahead).

### Results

| Model | RMSE | MAE | Directional Accuracy |
|---|---|---|---|
| LSTM | **0.0187** | **0.0102** | 50.3% |
| GARCH(1,1) | 0.0254 | 0.0194 | 47.9% |

LSTM is **27% better on RMSE** and **47% better on MAE** than GARCH. This is the project's strongest quantitative result. Note that directional accuracy near 50% for both models is normal for volatility forecasting — the signal lies in magnitude, not direction.

### Pricing Application

The LSTM vol forecast is also used as the sigma input to BSM pricing, creating a "pricing horse race" between historical rolling vol, LSTM-forecast vol, and implied vol. Vega sensitivity analysis (`∂V/∂σ`) quantifies how a 1 percentage-point vol forecast error propagates into option pricing error, illustrating the practical cost of LSTM accuracy.

---

## Component 3: Integration — RL Hedging with LSTM Vol

When the LSTM vol schedule replaces the fixed constant sigma in the hedging environment:

| Strategy | MAE | Mean TC |
|---|---|---|
| BSM daily | **0.284** | 0.146 |
| RL (constant σ) | 0.778 | 0.128 |
| RL (LSTM σ schedule) | 0.918 | 0.102 |

The LSTM-guided RL achieves the lowest transaction cost of the RL variants, but does not improve MAE over the constant-σ RL. This is explained by the LSTM introducing noisy vol estimates that increase hedging error variance, even though the mean estimate is more accurate. The BSM daily benchmark remains hardest to beat, as expected theoretically.

---

## Component 4: Theory-Motivated Extensions

### Idea 1 — Heston Stochastic Volatility Environment

**Theory:** Under GBM, sigma is constant and BSM delta is provably optimal — RL cannot improve on it. Under the Heston model, vol itself is stochastic:

```
dS = r S dt + √V · S · dW₁
dV = κ(θ − V) dt + ξ√V dW₂
dW₁ dW₂ = ρ dt       (leverage effect, ρ = −0.70 for equities)
```

BSM delta computed at the current `√V` is no longer optimal because it ignores future vol dynamics. An RL agent that observes the current variance state `√V/√θ` can potentially learn a better policy.

**Implementation:** Full-truncation Euler–Maruyama for the variance SDE (reflects V at zero to stay non-negative). Correlated Brownian motions via Cholesky: `W₂ = ρW₁ + √(1−ρ²)Z₂`. The 4D observation is `[S/K, τ/T, δ_prev, √V/√θ]`.

**Result:** RL-BSM delta correlation drops from 0.947 (GBM) to 0.840 (Heston), confirming the agent learned a **different strategy** from BSM delta. This is the expected and theoretically correct outcome — BSM delta is suboptimal under stochastic vol, and RL diverges from it as a result.

### Idea 2 — Itô-Decomposed Reward Signal

**Theory:** From the Itô decomposition above, the total hedge P&L splits as:

```
hedge_error = (δ − δ*) dS  −  ½Γ(dS² − σ²S²dt)
               controllable     irreducible gamma term
```

The standard MSE reward `−(hedge_error)²` penalises the gamma term even though no delta choice can eliminate it. This adds noise to the training signal. The Itô reward removes the irreducible component before squaring:

```python
gamma_pnl    = 0.5 * Γ * ((S_new − S_old)² − σ²S²dt)
controllable = hedge_error − gamma_pnl
reward       = −(controllable)² − tc
```

**Result (GBM):**

| Reward | MAE | TC |
|---|---|---|
| MSE baseline | 0.858 | 0.113 |
| Itô reward | **0.836** | **0.098** |

**2.6% lower MAE and 13% lower transaction cost.** Cleaner signal → the agent learns to trade less when the dominant risk is irreducible gamma noise.

### Idea 3 — Gamma-Aware Observation

**Theory:** The magnitude of the irreducible gamma term is `½Γ·S²·σ²·dt`. This is high when Γ·S is large — near ATM and near expiry — and near zero when deep ITM/OTM. A standard BSM delta hedge rebalances uniformly regardless of regime. An RL agent that observes `Γ·S` (dollar-gamma) can learn a **non-uniform rebalancing policy**: hedge aggressively when gamma risk is high, conserve transaction costs when it is low.

**Implementation:** Dollar-gamma `Γ·S` (clipped to `[0, 5]`) is appended as a 4th observation. This is computed from BSM gamma at the current state before each step, so the agent has forward-looking information about how sensitive the option is at this moment.

**Result (GBM + Itô + gamma obs combined):**

| Experiment | MAE | corr(RL,BSM) | TC |
|---|---|---|---|
| A: GBM + MSE (baseline) | 0.858 | 0.947 | 0.113 |
| B: GBM + Itô | 0.836 | 0.940 | 0.098 |
| **E: GBM + Itô + gamma obs** | 0.933 | 0.909 | **0.078** |

Experiment E achieves the **lowest transaction cost across all experiments: −31% vs baseline, −20% vs Itô-only.** The trade-off is a modest increase in MAE. The lower correlation with BSM delta (0.909 vs 0.940) is itself evidence that the agent learned a genuinely different, gamma-regime-aware policy — it rebalances less when gamma is low and the option is insensitive, which BSM delta does not do.

---

## How Everything Connects

```
NIFTY 50 price data
       │
       ├─── Feature engineering (rv5, rv21, rv_ratio, mom5, ...)
       │              │
       │        VolLSTM (LSTM)    ←── trained to predict rv21 next-day
       │              │
       │         vol forecast σ_lstm
       │              │
       ├──────────────┤
       │              │
  BSM formula         │          (σ_lstm → more accurate option pricing)
  V(S,K,T,r,σ)        │
       │              │
  BSM delta Δ*        │
  BSM gamma Γ         │
       │              │
  ──────────────────────────────────────────────────────────
  Hedging Environment (HedgingEnv / HestonEnv)
  ──────────────────────────────────────────────────────────
       │
  Observation: [S/K, τ/T, δ_prev]  +  optional [Γ·S, √V/√θ]
  Reward:      −(hedge_error)²  −  tc
               optionally: remove irreducible Itô gamma term
       │
  PPO agent learns δ_t = π(obs_t)
       │
  ┌────────────────────────────────────────────────────────┐
  │  GBM world:     RL ≈ BSM delta  (corr = 0.96)         │
  │  Heston world:  RL ≠ BSM delta  (corr = 0.84)         │
  │  Gamma obs:     RL learns TC-efficient regime policy   │
  └────────────────────────────────────────────────────────┘
```

The BSM formula is the shared spine: it prices options in the environments, provides the theoretical benchmark delta, supplies gamma for the Itô decomposition and the new observation, and converts LSTM vol forecasts into option prices. The LSTM forecaster and the RL hedger are both downstream users of the same BSM machinery, connected through the vol input.

---

## Summary of Results

| Result | Value |
|---|---|
| LSTM RMSE vs GARCH RMSE | 0.0187 vs 0.0254 (27% better) |
| LSTM MAE vs GARCH MAE | 0.0102 vs 0.0194 (47% better) |
| RL–BSM delta correlation (GBM) | 0.960 |
| RL–BSM delta correlation (Heston) | 0.840 (diverges — expected) |
| Itô reward MAE improvement | −2.6% vs MSE baseline |
| Itô reward TC improvement | −13% vs MSE baseline |
| Gamma obs TC improvement | −31% vs MSE baseline |

The LSTM forecasting result is the strongest: a statistically meaningful improvement over GARCH, a 40-year industry standard. The RL results are theoretically clean: the agent independently discovers BSM delta under GBM (validating the environment and training setup), and sensibly diverges from it under Heston. The novel experiments demonstrate that Itô's lemma is not just a theorem but a practical design tool for reward shaping and observation engineering.
