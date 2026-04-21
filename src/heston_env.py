"""
Heston stochastic volatility environment for delta hedging.

Under GBM (constant vol), BSM delta is provably optimal — RL cannot beat it.
Under Heston, vol itself evolves stochastically, so BSM delta (which assumes
constant vol) becomes suboptimal.  An RL agent that observes the current
variance state can potentially outperform naive BSM delta.

Dynamics
--------
  dS = r S dt + sqrt(V) S dW_1
  dV = kappa (theta - V) dt + xi sqrt(V) dW_2
  dW_1 dW_2 = rho dt          (leverage effect; negative for equities)

Observation space: 4-D
  [S/K,  tau/T,  delta_prev,  sqrt(V)/sqrt(theta)]
The last component encodes the current vol regime relative to the long-run
average — this is the extra signal RL can use that BSM ignores.

Option pricing inside the env uses BSM with the *current* sqrt(V) as sigma.
This is an approximation (true Heston price requires numerical integration)
but is accurate for short maturities and provides a tractable training signal.
"""

import numpy as np
import gymnasium as gym
from bsm import bsm_price, bsm_delta


class HestonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        S0:     float = 100.0,
        K:      float = 100.0,
        T:      float = 21 / 252,
        r:      float = 0.05,
        # Heston parameters (calibrated to typical equity index behaviour)
        kappa:  float = 2.0,     # mean-reversion speed
        theta:  float = 0.04,    # long-run variance (≈ 20% ann. vol)
        xi:     float = 0.30,    # vol-of-vol
        rho:    float = -0.70,   # leverage correlation
        V0:     float = 0.04,    # initial variance (same as theta → stationary start)
        n_steps: int  = 21,
        kappa_tc: float = 0.001, # proportional transaction cost
    ):
        super().__init__()

        self.S0      = S0
        self.K       = K
        self.T       = T
        self.r       = r
        self.kappa   = kappa
        self.theta   = theta
        self.xi      = xi
        self.rho     = rho
        self.V0      = V0
        self.n_steps = n_steps
        self.dt      = T / n_steps
        self.kappa_tc = kappa_tc

        # Cholesky factor for correlated Brownian motions
        self._rho_perp = np.sqrt(max(1.0 - rho ** 2, 1e-9))

        # Observation: [moneyness, tau_norm, delta_prev, vol_state]
        self.observation_space = gym.spaces.Box(
            low=np.array( [0.2, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([5.0, 1.0, 1.0, 5.0], dtype=np.float32),
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        # Episode state
        self.t          = 0
        self.S          = S0
        self.V          = V0
        self.delta_prev = 0.0
        self.episode_tc = 0.0

    # ── internal ─────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        tau = max(self.T - self.t * self.dt, 1e-9)
        return np.array([
            self.S / self.K,
            tau / self.T,
            self.delta_prev,
            np.sqrt(max(self.V, 1e-9)) / np.sqrt(self.theta),
        ], dtype=np.float32)

    def _sigma_eff(self) -> float:
        return float(np.sqrt(max(self.V, 1e-6)))

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomise starting spot ±10 % (same as HedgingEnv)
        self.S = self.S0 * self.np_random.uniform(0.90, 1.10)
        # Randomise starting variance ±25 % around V0
        self.V = float(np.clip(
            self.V0 * self.np_random.uniform(0.75, 1.25),
            1e-4, 4.0 * self.theta,
        ))
        self.t          = 0
        self.delta_prev = 0.0
        self.episode_tc = 0.0
        self.S_prev     = self.S

        sig0 = self._sigma_eff()
        self.portfolio = bsm_price(self.S, self.K, self.T, self.r, sig0)

        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action)

        delta_new   = float(np.clip(action[0], 0.0, 1.0))
        sig         = self._sigma_eff()
        tau_before  = max(self.T - self.t * self.dt, 1e-9)

        S_prev      = self.S
        self.S_prev = self.S   # stored for HestonItoEnv subclass
        V_prev      = self.V

        V_before = bsm_price(S_prev, self.K, tau_before, self.r, sig)

        # Transaction cost
        tc = self.kappa_tc * abs(delta_new - self.delta_prev) * S_prev
        self.episode_tc += tc

        # ── Heston Euler–Maruyama step ────────────────────────────────────────
        Z1 = self.np_random.standard_normal()
        Z2 = self.np_random.standard_normal()
        W1 = Z1
        W2 = self.rho * Z1 + self._rho_perp * Z2   # correlated increment

        sqV = np.sqrt(max(V_prev, 1e-9))

        # Variance: full-truncation Euler (reflect V at 0 to stay non-negative)
        V_new = V_prev + self.kappa * (self.theta - V_prev) * self.dt \
                       + self.xi * sqV * np.sqrt(self.dt) * W2
        self.V = max(V_new, 1e-6)

        # Stock price (log-Euler, exact for GBM part)
        self.S = S_prev * np.exp(
            (self.r - 0.5 * V_prev) * self.dt
            + sqV * np.sqrt(self.dt) * W1
        )
        self.t += 1

        tau_after  = max(self.T - self.t * self.dt, 1e-9)
        sig_after  = self._sigma_eff()
        V_after    = bsm_price(self.S, self.K, tau_after, self.r, sig_after)

        delta_S     = self.S - S_prev
        delta_V_opt = V_after - V_before
        hedge_error = delta_new * delta_S - delta_V_opt

        self.portfolio += delta_new * delta_S - tc

        reward     = float(-hedge_error ** 2 - tc)
        self.delta_prev = delta_new
        terminated = self.t >= self.n_steps

        info = {"hedge_error": hedge_error, "transaction_cost": tc,
                "sigma_eff": sig, "V": self.V}
        if terminated:
            payoff = max(self.S - self.K, 0.0)
            info["episode_tc"]    = self.episode_tc
            info["terminal_pnl"]  = float(self.portfolio - payoff)

        return self._get_obs(), reward, terminated, False, info
