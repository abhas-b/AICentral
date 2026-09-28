"""
optrl.envs
==========
Environments used throughout the tutorial.

StrategyBandit        Part 2  - each strategy is a slot-machine arm; pulling an
                               arm returns a random historical weekly P&L.
ContextualBandit      Part 3  - walks through weeks in time order, shows the
                               market features, pays the chosen strategy's P&L.
OptionsSelectionEnv   Parts 5-9 - a full Gymnasium MDP: positions carry over,
                               can be closed early, pay transaction costs.
DiscretizeObs         Part 5  - wrapper turning the MDP's observation into a
                               single integer state for tabular Q-learning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from .market import MarketData, TRADING_DAYS
from .strategies import (STRATEGY_NAMES, ACTION_NAMES, N_ACTIONS, RISK_BUDGET, MULT,
                         build_legs, bs_price_scalar, effective_iv,
                         risk_per_contract, entry_cost_per_contract)


# --------------------------------------------------------------------------- #
#  Part 2: plain multi-armed bandit                                            #
# --------------------------------------------------------------------------- #
class StrategyBandit:
    """k-armed bandit. Arm i pays a random draw from strategy i's historical
    weekly P&L (so every arm has a realistic, fat-tailed distribution).

    Pass `pnl_table` (a DataFrame, one column per action) or `means`/`stds`
    for a simple Gaussian bandit.
    """

    def __init__(self, pnl_table: pd.DataFrame | None = None, means=None, stds=None,
                 seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        if pnl_table is not None:
            self.names = list(pnl_table.columns)
            self._data = [pnl_table[c].to_numpy() for c in self.names]
            self.true_means = np.array([d.mean() for d in self._data])
        else:
            self.names = [f"arm_{i}" for i in range(len(means))]
            self._means, self._stds = np.asarray(means, float), np.asarray(stds, float)
            self._data = None
            self.true_means = self._means.copy()
        self.k = len(self.names)
        self.best_arm = int(np.argmax(self.true_means))

    def pull(self, arm: int) -> float:
        if self._data is not None:
            d = self._data[arm]
            return float(d[self.rng.integers(len(d))])
        return float(self.rng.normal(self._means[arm], self._stds[arm]))


# --------------------------------------------------------------------------- #
#  Part 3: contextual bandit                                                   #
# --------------------------------------------------------------------------- #
class ContextualBandit:
    """Walks through weekly decision points in chronological order.

        ctx = cb.reset()            # features for week 0
        ctx, reward, done = cb.step(action)

    Only the chosen action's P&L is revealed (bandit feedback), exactly like
    live trading: you never learn what the other strategies *would* have made.
    """

    def __init__(self, features: pd.DataFrame, pnl_table: pd.DataFrame):
        assert features.index.equals(pnl_table.index)
        self.X = features.to_numpy(float)
        self.R = pnl_table.to_numpy(float)
        self.dates = pnl_table.index
        self.names = list(pnl_table.columns)
        self.k = self.R.shape[1]
        self.t = 0

    def reset(self):
        self.t = 0
        return self.X[0]

    def step(self, action: int):
        r = self.R[self.t, action]
        self.t += 1
        done = self.t >= len(self.X)
        ctx = None if done else self.X[self.t]
        return ctx, float(r), done

    def __len__(self):
        return len(self.X)


# --------------------------------------------------------------------------- #
#  Parts 5-9: the full MDP                                                     #
# --------------------------------------------------------------------------- #
OBS_CLIP = 20.0


class _Position:
    __slots__ = ("strategy", "legs", "contracts", "expiry_idx", "open_idx", "entry_value")

    def __init__(self, strategy, legs, contracts, expiry_idx, open_idx, entry_value):
        self.strategy, self.legs, self.contracts = strategy, legs, contracts
        self.expiry_idx, self.open_idx, self.entry_value = expiry_idx, open_idx, entry_value


class OptionsSelectionEnv(gym.Env):
    """Choose which option strategy to hold (or none) at each decision point.

    Action (Discrete(8)) = the position you WANT to hold after this decision:
        0 = no_trade (be flat; closes any open position)
        k = strategy k (1..7). If you already hold k you just keep holding it;
            if you hold something else it is closed (paying costs) and k opened.

    Timing: every step covers `step_days` trading days. New positions use
    options expiring `dte` trading days later. With the defaults (weekly steps,
    2-week options) a position lives for up to 2 decisions - so today's choice
    affects next week's state. That is what makes this an MDP, not a bandit.

    Observation (float32 vector):
        market features (z-scored)  |  position one-hot (8)  |
        fraction of life left       |  unrealized P&L / RISK_BUDGET

    Reward: change in account equity over the step (P&L net of all costs),
        divided by RISK_BUDGET, minus an optional risk penalty.
    """

    metadata = {"render_modes": []}

    def __init__(self, market: MarketData, features: list[str] | None = None,
                 step_days: int = 5, dte: int = 10, episode_len: int = 52,
                 random_start: bool = True, start_dow: int | None = 0,
                 obs_stats: tuple[np.ndarray, np.ndarray] | None = None,
                 risk_penalty: float = 0.0, cost_multiplier: float = 1.0,
                 seed: int | None = None):
        super().__init__()
        self.market = market
        self.df = market.df
        self.features = features or MarketData.FEATURES
        self.step_days, self.dte, self.episode_len = step_days, dte, episode_len
        self.random_start, self.start_dow = random_start, start_dow
        self.risk_penalty, self.cost_multiplier = risk_penalty, cost_multiplier

        # raw arrays for speed
        self._S = self.df["close"].to_numpy(float)
        self._vix = self.df["vix"].to_numpy(float)
        self._dte_earn = self.df["days_to_earnings"].to_numpy(int)
        self._F = self.df[self.features].to_numpy(float)
        if obs_stats is None:                       # z-score with THIS data's stats
            obs_stats = (np.nanmean(self._F, 0), np.nanstd(self._F, 0) + 1e-8)
        self.obs_stats = obs_stats
        self._Fz = np.nan_to_num((self._F - obs_stats[0]) / obs_stats[1])

        n_obs = len(self.features) + N_ACTIONS + 2
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, (n_obs,), np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.action_names = ACTION_NAMES
        last = len(self.df) - (episode_len * step_days + dte + 2)
        starts = np.arange(0, max(last, 1))
        if start_dow is not None:
            starts = starts[self.df["dow"].to_numpy()[starts] == start_dow]
        self._starts = starts
        if seed is not None:
            self.reset(seed=seed)

    # ---------------------------------------------------------------- pricing
    def _value(self, pos: _Position, i: int) -> float:
        """Mark-to-market $ value of the position at the close of day i."""
        days_left = pos.expiry_idx - i
        S = self._S[i]
        if days_left <= 0:
            v = sum(q * (max(S - K, 0.0) if k == "C" else max(K - S, 0.0))
                    for q, k, K in pos.legs)
        else:
            sig = float(effective_iv(self._vix[i], days_left, self._dte_earn[i]))
            T = days_left / TRADING_DAYS
            v = sum(q * bs_price_scalar(S, K, T, sig, k) for q, k, K in pos.legs)
        return v * MULT * pos.contracts

    def _leg_prices(self, pos, i):
        days_left = max(pos.expiry_idx - i, 0)
        sig = float(effective_iv(self._vix[i], max(days_left, 1e-9), self._dte_earn[i]))
        T = days_left / TRADING_DAYS
        return [bs_price_scalar(self._S[i], K, T, sig, k) for _, k, K in pos.legs]

    def _open(self, strategy_idx: int, i: int) -> float:
        """Open strategy (1..7) at close of day i. Returns cost paid ($)."""
        name = STRATEGY_NAMES[strategy_idx - 1]
        S = self._S[i]
        sig = float(effective_iv(self._vix[i], self.dte, self._dte_earn[i]))
        T = self.dte / TRADING_DAYS
        legs = [(q, k, float(K)) for q, k, K in build_legs(name, S, sig, T)]
        prem = [bs_price_scalar(S, K, T, sig, k) for _, k, K in legs]
        risk1 = float(risk_per_contract(name, legs, prem, S, sig, T))
        contracts = RISK_BUDGET / risk1
        entry_value = sum(q * p for (q, _, _), p in zip(legs, prem)) * MULT * contracts
        self.pos = _Position(strategy_idx, legs, contracts, i + self.dte, i, entry_value)
        self.cash -= entry_value
        cost = float(entry_cost_per_contract(prem)) * contracts * self.cost_multiplier
        self.cash -= cost
        return cost

    def _close(self, i: int) -> float:
        """Close the open position at close of day i. Returns cost paid ($)."""
        pos = self.pos
        self.cash += self._value(pos, i)
        cost = 0.0
        if pos.expiry_idx > i:                     # early exit pays the spread again
            cost = float(entry_cost_per_contract(self._leg_prices(pos, i))) \
                * pos.contracts * self.cost_multiplier
            self.cash -= cost
        self.pos = None
        return cost

    def _equity(self, i: int) -> float:
        return self.cash + (self._value(self.pos, i) if self.pos else 0.0)

    # ------------------------------------------------------------- gym API
    def _obs(self):
        i = self.i
        onehot = np.zeros(N_ACTIONS, np.float32)
        onehot[self.pos.strategy if self.pos else 0] = 1.0
        if self.pos:
            life = (self.pos.expiry_idx - i) / self.dte
            unreal = (self._value(self.pos, i) - self.pos.entry_value) / RISK_BUDGET
        else:
            life, unreal = 0.0, 0.0
        obs = np.concatenate([self._Fz[i].astype(np.float32), onehot,
                              np.array([life, unreal], np.float32)])
        return np.clip(obs, -OBS_CLIP, OBS_CLIP)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)                    # seeds self.np_random
        start = (options or {}).get("start")
        if start is None:
            start = int(self.np_random.choice(self._starts)) if self.random_start else int(self._starts[0])
        self.i, self.t = start, 0
        self.cash, self.pos = 0.0, None
        return self._obs(), self._info(0.0, 0.0)

    def step(self, action: int):
        action = int(action)
        i0 = self.i
        eq0 = self._equity(i0)
        cost = 0.0
        current = self.pos.strategy if self.pos else 0
        if action != current:
            if self.pos:
                cost += self._close(i0)
            if action != 0:
                cost += self._open(action, i0)

        # let the market move `step_days` days, settling at expiry
        for d in range(1, self.step_days + 1):
            self.i = i0 + d
            if self.pos and self.i >= self.pos.expiry_idx:
                self._close(self.i)                # expiry: cash-settle, no cost
        eq1 = self._equity(self.i)

        pnl = eq1 - eq0                            # includes costs paid this step
        r = pnl / RISK_BUDGET
        reward = r - self.risk_penalty * r * r
        self.t += 1
        terminated = False
        truncated = self.t >= self.episode_len
        return self._obs(), float(reward), terminated, truncated, self._info(pnl, cost, action)

    def _info(self, pnl, cost, action=None):
        i = self.i
        return {"date": self.df.index[i], "pnl": pnl, "cost": cost,
                "equity": self._equity(i), "action": action,
                "position": ACTION_NAMES[self.pos.strategy] if self.pos else "no_trade",
                "regime": self.df["regime_name"].iloc[i] if "regime_name" in self.df else None}


# --------------------------------------------------------------------------- #
#  Part 5: discretised observation for tabular methods                         #
# --------------------------------------------------------------------------- #
class DiscretizeObs(gym.ObservationWrapper):
    """Turns the continuous observation into ONE integer:

        market bucket  = iv_rank (low/mid/high) x trend (down/flat/up)
                         x term structure (contango/flat/backwardation)  -> 27
        position       = which strategy is held (0 = flat)               -> 8
        state id       = market_bucket * 8 + position                    -> 216
    """

    IV_EDGES = (25.0, 50.0)          # iv_rank
    TREND_EDGES = (-0.3, 0.3)        # ema_trend, percent
    TS_EDGES = (0.92, 1.0)           # vix / vix3m

    def __init__(self, env: OptionsSelectionEnv):
        super().__init__(env)
        self.observation_space = spaces.Discrete(27 * N_ACTIONS)
        self.buckets = self.market_buckets(env.unwrapped.df)     # precompute once

    @classmethod
    def market_buckets(cls, df: pd.DataFrame) -> np.ndarray:
        a = np.digitize(df["iv_rank"], cls.IV_EDGES)
        b = np.digitize(df["ema_trend"], cls.TREND_EDGES)
        c = np.digitize(df["term_structure"], cls.TS_EDGES)
        return np.asarray((a * 3 + b) * 3 + c)

    def market_bucket(self, i: int) -> int:
        return int(self.buckets[i])

    def observation(self, obs):
        e = self.env.unwrapped
        pos = e.pos.strategy if e.pos else 0
        return self.market_bucket(e.i) * N_ACTIONS + pos

    @staticmethod
    def describe(state: int) -> str:
        m, pos = divmod(state, N_ACTIONS)
        ab, c = divmod(m, 3)
        a, b = divmod(ab, 3)
        return (f"IVR={['low', 'mid', 'high'][a]}, trend={['down', 'flat', 'up'][b]}, "
                f"term={['contango', 'flat', 'backwrd'][c]}, holding={ACTION_NAMES[pos]}")


# --------------------------------------------------------------------------- #
#  Part 9: an MDP built from YOUR backtest table (no option repricing needed) #
# --------------------------------------------------------------------------- #
class PnLTableEnv(gym.Env):
    """Weekly env driven by a table of per-strategy P&L (your own backtests).

    Use this when your backtester gives you "P&L of strategy k if opened on
    date t" but you cannot re-price positions day by day.

    State  = z-scored features  +  one-hot of LAST action (so switching is visible)
    Reward = (P&L of chosen strategy - switch_cost if you changed strategy) / scale
    The switching cost makes yesterday's choice matter -> a (small) MDP.
    """

    metadata = {"render_modes": []}

    def __init__(self, features: pd.DataFrame, pnl_table: pd.DataFrame,
                 episode_len: int = 52, switch_cost: float = 25.0, scale: float = 1000.0,
                 obs_stats=None, random_start: bool = True, seed: int | None = None):
        super().__init__()
        assert features.index.equals(pnl_table.index), "features and P&L must share dates"
        self.X = features.to_numpy(float)
        self.R = pnl_table.to_numpy(float)
        self.dates = pnl_table.index
        self.action_names = list(pnl_table.columns)
        self.k = self.R.shape[1]
        if obs_stats is None:
            obs_stats = (self.X.mean(0), self.X.std(0) + 1e-8)
        self.obs_stats = obs_stats
        self.Xz = np.clip(np.nan_to_num((self.X - obs_stats[0]) / obs_stats[1]), -OBS_CLIP, OBS_CLIP)
        self.episode_len = min(episode_len, len(self.X) - 1)
        self.switch_cost, self.scale, self.random_start = switch_cost, scale, random_start
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, (self.X.shape[1] + self.k,), np.float32)
        self.action_space = spaces.Discrete(self.k)
        if seed is not None:
            self.reset(seed=seed)

    def _obs(self):
        oh = np.zeros(self.k, np.float32)
        oh[self.prev] = 1.0
        return np.concatenate([self.Xz[self.t].astype(np.float32), oh])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        hi = len(self.X) - self.episode_len
        start = (options or {}).get("start")
        if start is None:
            start = int(self.np_random.integers(0, hi)) if self.random_start else 0
        self.t0 = self.t = start
        self.prev = 0
        return self._obs(), {"date": self.dates[self.t]}

    def step(self, action):
        a = int(action)
        pnl = self.R[self.t, a]
        cost = self.switch_cost if (a != self.prev and a != 0) else 0.0
        self.prev = a
        date = self.dates[self.t]
        self.t += 1
        truncated = (self.t - self.t0) >= self.episode_len
        return self._obs(), float((pnl - cost) / self.scale), False, truncated, \
            {"date": date, "pnl": pnl - cost, "cost": cost, "action": a}
