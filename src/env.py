"""
Project 1 — Gym environment for delta-hedging under GBM.

State  : (moneyness S/K, normalised time-to-expiry tau/T, current delta)
         optionally + dollar-gamma Γ·S (gamma_obs=True, idea 3)
         optionally + observed sigma proxy (sigma_obs=True)
Action : new hedge ratio in [0, 1]
Reward : −(hedging error)² − transaction cost

By Itô's lemma the irreducible hedging error per step is ½Γ(dS²−σ²S²dt).
Giving the agent Γ·S as an observation lets it learn when gamma risk is
high (near-ATM, near-expiry) and adjust rebalancing accordingly.

Optionally accepts a sigma_schedule (array of per-step vol forecasts from
the LSTM) to enable the Project 1 + 3 integration experiment.

sigma_range : tuple (lo, hi) — if provided, σ is sampled uniformly from
              [lo, hi] at the start of each episode (domain randomization).
              Overrides the fixed sigma parameter each episode.
"""

import gymnasium as gym
import numpy as np
from bsm import bsm_price, bsm_delta, bsm_gamma


class HedgingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 21 / 252,          # 21 trading days ≈ 1 month
        r: float = 0.05,
        sigma: float = 0.20,
        n_steps: int = 21,
        kappa: float = 0.001,          # proportional transaction cost rate
        sigma_schedule=None,           # optional array of per-step LSTM vols
        ito_reward: bool = False,      # use Itô-decomposed reward (idea 2)
        gamma_obs: bool = False,       # append dollar-gamma Γ·S to obs (idea 3)
        sigma_obs: bool = False,       # append sigma proxy to obs
        sigma_range: tuple = None,     # (lo, hi) → sample σ each episode
    ):
        super().__init__()

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.kappa = kappa
        self.sigma_schedule = sigma_schedule  # None → constant vol
        self.ito_reward = ito_reward
        self.gamma_obs  = gamma_obs
        self.sigma_obs  = sigma_obs
        self.sigma_range = sigma_range

        # Observation: [moneyness, tau_normalised, delta_prev]
        # plus optional [gamma*S] and/or [sigma_proxy].
        low = [0.2, 0.0, 0.0]
        high = [5.0, 1.0, 1.0]
        if gamma_obs:
            # Normalized dollar-gamma: phi(d1)/phi(0) in [0, 1].
            # Raw gamma*S at ATM near-expiry ≈ 32 with σ=0.20 — well above [0,5],
            # so the old bound was clipping away the most important regime.
            # phi(d1)/phi(0) = gamma*S*sigma*sqrt(tau) / 0.3989, always in [0,1].
            low.append(0.0)
            high.append(1.5)   # 1.5 gives headroom for numerical noise
        if sigma_obs:
            # Sigma proxy normalized to a 20% reference volatility.
            low.append(0.0)
            high.append(5.0)
        self.observation_space = gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
        )
        # Action: continuous hedge ratio in [0, 1]
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        # Episode tracking (set in reset)
        self.t = 0
        self.S = S0
        self.delta_prev = 0.0
        self.episode_tc = 0.0        # cumulative transaction cost
        self.episode_errors = []     # per-step hedging errors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sigma(self) -> float:
        if self.sigma_schedule is not None:
            idx = min(self.t, len(self.sigma_schedule) - 1)
            return float(self.sigma_schedule[idx])
        return self.sigma

    def _get_obs(self) -> np.ndarray:
        tau = max(self.T - self.t * self.dt, 1e-9)
        sig = self._get_sigma()
        # Clip all components to declared observation_space bounds — prevents
        # out-of-range observations under domain randomization (high-σ paths).
        obs = [
            float(np.clip(self.S / self.K, 0.2, 5.0)),
            float(np.clip(tau / self.T,    0.0, 1.0)),
            float(np.clip(self.delta_prev, 0.0, 1.0)),
        ]
        if self.gamma_obs:
            gamma_t = bsm_gamma(self.S, self.K, tau, self.r, sig)
            # phi(d1)/phi(0) = gamma*S*sigma*sqrt(tau) / 0.3989
            # Bounded in [0, 1] by construction; 1.0 at ATM regardless of tau or sig.
            phi_d1_norm = gamma_t * self.S * sig * np.sqrt(tau) / 0.3989
            obs.append(float(np.clip(phi_d1_norm, 0.0, 1.5)))
        if self.sigma_obs:
            obs.append(float(np.clip(sig / 0.20, 0.0, 5.0)))
        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.sigma_range is not None:
            lo, hi = self.sigma_range
            self.sigma = float(self.np_random.uniform(lo, hi))

        # Randomise starting spot ±10 % to generalise across moneyness
        self.S = self.S0 * self.np_random.uniform(0.90, 1.10)
        self.S_prev = self.S   # tracked as instance var so it's always accessible
        self.t = 0
        self.delta_prev = 0.0
        self.episode_tc = 0.0
        self.episode_errors = []

        # Portfolio starts funded at the BSM price of the option (at t=0 spot)
        sig0 = self._get_sigma()
        self.portfolio = bsm_price(self.S, self.K, self.T, self.r, sig0)

        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        delta_new = float(np.clip(action[0], 0.0, 1.0))
        sig = self._get_sigma()
        tau_before = max(self.T - self.t * self.dt, 1e-9)

        # Snapshot S before the GBM move — stored as instance var so it
        # never relies on Z being in scope to recover.
        self.S_prev = self.S

        # Option value before the stock move
        V_before = bsm_price(self.S_prev, self.K, tau_before, self.r, sig)

        # Transaction cost for the rebalancing trade
        tc = self.kappa * abs(delta_new - self.delta_prev) * self.S_prev
        self.episode_tc += tc

        # --- GBM step ---
        Z = self.np_random.standard_normal()
        self.S = self.S_prev * np.exp(
            (self.r - 0.5 * sig ** 2) * self.dt
            + sig * np.sqrt(self.dt) * Z
        )
        self.t += 1

        tau_after = max(self.T - self.t * self.dt, 1e-9)
        sig_after = self._get_sigma()  # possibly updated vol for next step
        V_after = bsm_price(self.S, self.K, tau_after, self.r, sig_after)

        # Hedging error: what the delta position earned vs how the option moved
        delta_S = self.S - self.S_prev
        delta_V = V_after - V_before
        hedge_error = delta_new * delta_S - delta_V
        self.episode_errors.append(hedge_error)

        # Track running portfolio value: gains from delta position minus costs
        self.portfolio += delta_new * delta_S - tc

        if self.ito_reward:
            # Itô decomposition: hedge P&L = (δ − δ*) dS − ½Γ(dS² − σ²S²dt)
            # The gamma term is irreducible with delta hedging alone; only the
            # delta-mismatch part is controllable.  Remove it before squaring
            # so the agent isn't penalised for variance it cannot eliminate.
            gamma_t    = bsm_gamma(self.S_prev, self.K, tau_before, self.r, sig)
            gamma_pnl  = 0.5 * gamma_t * (
                (self.S - self.S_prev) ** 2 - sig ** 2 * self.S_prev ** 2 * self.dt
            )
            controllable = hedge_error - gamma_pnl
            reward = float(-controllable ** 2 - tc)
        else:
            reward = float(-hedge_error ** 2 - tc)

        self.delta_prev = delta_new
        terminated = self.t >= self.n_steps

        info = {
            "hedge_error": hedge_error,
            "transaction_cost": tc,
            "sigma_used": sig,
        }
        if terminated:
            payoff = max(self.S - self.K, 0.0)
            info["episode_tc"] = self.episode_tc
            # terminal_pnl: how much the hedged portfolio beats/misses the payoff
            info["terminal_pnl"] = float(self.portfolio - payoff)

        return self._get_obs(), reward, terminated, False, info
