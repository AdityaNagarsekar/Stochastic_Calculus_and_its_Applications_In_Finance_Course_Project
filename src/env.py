"""
Project 1 — Gym environment for delta-hedging under GBM.

State  : (moneyness S/K, normalised time-to-expiry tau/T, current delta)
         optionally + dollar-gamma Γ·S (gamma_obs=True, idea 3)
Action : new hedge ratio in [0, 1]
Reward : −(hedging error)² − transaction cost

By Itô's lemma the irreducible hedging error per step is ½Γ(dS²−σ²S²dt).
Giving the agent Γ·S as an observation lets it learn when gamma risk is
high (near-ATM, near-expiry) and adjust rebalancing accordingly.

Optionally accepts a sigma_schedule (array of per-step vol forecasts from
the LSTM) to enable the Project 1 + 3 integration experiment.
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

        # Observation: [moneyness, tau_normalised, delta_prev] + optional [gamma*S]
        if gamma_obs:
            self.observation_space = gym.spaces.Box(
                low=np.array( [0.2, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([5.0, 1.0, 1.0, 5.0], dtype=np.float32),
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=np.array([0.2, 0.0, 0.0], dtype=np.float32),
                high=np.array([5.0, 1.0, 1.0], dtype=np.float32),
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
        obs = [self.S / self.K, tau / self.T, self.delta_prev]
        if self.gamma_obs:
            # Dollar-gamma Γ·S: high near ATM + near expiry, low otherwise.
            # Clipped to [0, 5] to match observation_space bounds.
            gamma_t = bsm_gamma(self.S, self.K, tau, self.r, sig)
            obs.append(float(np.clip(gamma_t * self.S, 0.0, 5.0)))
        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

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
