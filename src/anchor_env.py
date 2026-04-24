"""
BSM-anchored RL environments.

These environments do not ask RL to invent the hedge ratio from scratch.
Instead, the analytic BSM delta is treated as an anchor and the policy only
controls how aggressively to move toward, or around, that anchor.
"""

import gymnasium as gym
import numpy as np

from bsm import bsm_delta, bsm_gamma, bsm_price


class BSMAnchorHedgingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 21 / 252,
        r: float = 0.05,
        sigma: float = 0.20,
        n_steps: int = 21,
        kappa: float = 0.001,
        sigma_range: tuple | None = None,
        sigma_pool: np.ndarray | None = None,
        action_mode: str = "fraction",   # fraction | residual
        residual_scale: float = 0.15,
        anchor_penalty: float = 0.0,
        gamma_obs: bool = True,
    ):
        super().__init__()
        if action_mode not in {"fraction", "residual"}:
            raise ValueError(f"Unsupported action_mode: {action_mode}")

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.kappa = kappa
        self.sigma_range = sigma_range
        self.sigma_pool = None if sigma_pool is None else np.asarray(sigma_pool, dtype=float)
        self.action_mode = action_mode
        self.residual_scale = residual_scale
        self.anchor_penalty = anchor_penalty
        self.gamma_obs = gamma_obs

        # Obs: moneyness, tau/T, delta_prev, delta_bsm, hedge_gap, sigma/0.20, gamma_norm
        low = [0.2, 0.0, 0.0, 0.0, -1.0, 0.0]
        high = [5.0, 1.0, 1.0, 1.0, 1.0, 5.0]
        if gamma_obs:
            low.append(0.0)
            high.append(1.5)
        self.observation_space = gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
        )

        if self.action_mode == "fraction":
            act_low = np.array([0.0], dtype=np.float32)
            act_high = np.array([1.0], dtype=np.float32)
        else:
            act_low = np.array([-1.0], dtype=np.float32)
            act_high = np.array([1.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=act_low, high=act_high)

        self.t = 0
        self.S = S0
        self.delta_prev = 0.0
        self.episode_tc = 0.0
        self.portfolio = 0.0

    def _sample_sigma(self) -> float:
        if self.sigma_pool is not None and self.sigma_pool.size:
            idx = int(self.np_random.integers(0, self.sigma_pool.size))
            return float(self.sigma_pool[idx])
        if self.sigma_range is not None:
            lo, hi = self.sigma_range
            return float(self.np_random.uniform(lo, hi))
        return float(self.sigma)

    def _current_tau(self) -> float:
        return max(self.T - self.t * self.dt, 1e-9)

    def _bsm_delta(self, S: float | None = None, tau: float | None = None) -> float:
        S_use = self.S if S is None else S
        tau_use = self._current_tau() if tau is None else tau
        return float(bsm_delta(S_use, self.K, tau_use, self.r, self.sigma))

    def _gamma_norm(self) -> float:
        tau = self._current_tau()
        gamma_t = bsm_gamma(self.S, self.K, tau, self.r, self.sigma)
        phi_d1_norm = gamma_t * self.S * self.sigma * np.sqrt(tau) / 0.3989
        return float(np.clip(phi_d1_norm, 0.0, 1.5))

    def _get_obs(self) -> np.ndarray:
        tau = self._current_tau()
        delta_bsm = self._bsm_delta(self.S, tau)
        obs = [
            float(np.clip(self.S / self.K, 0.2, 5.0)),
            float(np.clip(tau / self.T, 0.0, 1.0)),
            float(np.clip(self.delta_prev, 0.0, 1.0)),
            float(np.clip(delta_bsm, 0.0, 1.0)),
            float(np.clip(delta_bsm - self.delta_prev, -1.0, 1.0)),
            float(np.clip(self.sigma / 0.20, 0.0, 5.0)),
        ]
        if self.gamma_obs:
            obs.append(self._gamma_norm())
        return np.array(obs, dtype=np.float32)

    def action_to_delta(self, action: np.ndarray | list | tuple, obs: np.ndarray | None = None) -> float:
        action_val = float(np.asarray(action, dtype=float)[0])
        if obs is None:
            obs = self._get_obs()
        delta_prev = float(obs[2])
        delta_bsm = float(obs[3])
        if self.action_mode == "fraction":
            alpha = float(np.clip(action_val, 0.0, 1.0))
            delta_new = delta_prev + alpha * (delta_bsm - delta_prev)
        else:
            eps = float(np.clip(action_val, -1.0, 1.0))
            delta_new = delta_bsm + self.residual_scale * eps
        return float(np.clip(delta_new, 0.0, 1.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sigma = self._sample_sigma()
        self.S = self.S0 * self.np_random.uniform(0.90, 1.10)
        self.t = 0
        self.delta_prev = 0.0
        self.episode_tc = 0.0
        self.portfolio = bsm_price(self.S, self.K, self.T, self.r, self.sigma)
        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        obs_before = self._get_obs()
        delta_bsm = float(obs_before[3])
        delta_new = self.action_to_delta(action, obs_before)
        tau_before = self._current_tau()
        S_prev = self.S
        V_before = bsm_price(S_prev, self.K, tau_before, self.r, self.sigma)

        tc = self.kappa * abs(delta_new - self.delta_prev) * S_prev
        self.episode_tc += tc

        Z = self.np_random.standard_normal()
        self.S = S_prev * np.exp(
            (self.r - 0.5 * self.sigma ** 2) * self.dt
            + self.sigma * np.sqrt(self.dt) * Z
        )
        self.t += 1

        tau_after = self._current_tau()
        V_after = bsm_price(self.S, self.K, tau_after, self.r, self.sigma)
        hedge_error = delta_new * (self.S - S_prev) - (V_after - V_before)
        self.portfolio += delta_new * (self.S - S_prev) - tc

        reward = -hedge_error ** 2 - tc
        if self.anchor_penalty > 0.0:
            reward -= self.anchor_penalty * (delta_new - delta_bsm) ** 2

        self.delta_prev = delta_new
        terminated = self.t >= self.n_steps
        info = {
            "hedge_error": float(hedge_error),
            "transaction_cost": float(tc),
            "sigma_used": float(self.sigma),
            "delta_bsm": float(delta_bsm),
            "delta_new": float(delta_new),
        }
        if terminated:
            payoff = max(self.S - self.K, 0.0)
            info["episode_tc"] = float(self.episode_tc)
            info["terminal_pnl"] = float(self.portfolio - payoff)
        return self._get_obs(), float(reward), terminated, False, info


class DiscreteBSMAnchorHedgingEnv(BSMAnchorHedgingEnv):
    """
    Discrete-action controller around the BSM anchor.

    Each action picks a fraction of the current BSM hedge gap to close:
      action 0 -> hold
      action 1 -> close 25% of the gap
      ...
    """

    def __init__(
        self,
        *args,
        action_levels=None,
        **kwargs,
    ):
        kwargs["action_mode"] = "fraction"
        super().__init__(*args, **kwargs)
        if action_levels is None:
            action_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
        self.action_levels = np.asarray(action_levels, dtype=float)
        self.action_space = gym.spaces.Discrete(len(self.action_levels))

    def action_to_delta(self, action, obs: np.ndarray | None = None) -> float:
        if obs is None:
            obs = self._get_obs()
        idx = int(action)
        alpha = float(self.action_levels[idx])
        delta_prev = float(obs[2])
        delta_bsm = float(obs[3])
        delta_new = delta_prev + alpha * (delta_bsm - delta_prev)
        return float(np.clip(delta_new, 0.0, 1.0))
