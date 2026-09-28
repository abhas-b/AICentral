"""
optrl.market
============
A small, *transparent* market simulator so every notebook runs offline.

It produces daily data for one underlying (think "an index ETF with quarterly
earnings-like events") plus the features a volatility trader would look at:

    vix              implied-vol index level, in vol points (e.g. 13.5 means 13.5%)
    vix3m            3-month implied vol, in vol points
    term_structure   vix / vix3m  (>1 = backwardation = stress)
    iv_rank          where today's vix sits in its 1-year range, 0..100
    rv20             20-day realized vol, in vol points
    rv_iv_spread     rv20 - vix  (positive = options are "cheap")
    ema9, ema20      exponential moving averages of price
    ema_trend        (ema9 - ema20) / close * 100   (percent; <0 = downtrend)
    ret_5d           last-week return, percent
    days_to_earnings trading days until the next earnings event
    dow              day of week, 0=Mon .. 4=Fri

Hidden (the agent never sees it): the *regime* that drives everything.

    0 calm_bull   drifts up, low realized vol, options are rich (IV > RV)
    1 range       no drift, low vol, options are rich
    2 bear_trend  drifts down, medium vol, options fairly priced
    3 vol_shock   sharp selloff, realized vol explodes above IV

The whole point of the tutorial: the agent has to *infer* the regime from the
features and pick the strategy that pays in that regime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

REGIMES = ["calm_bull", "range", "bear_trend", "vol_shock"]

# annual drift, annual realized vol, IV/RV premium, target term structure
REGIME_PARAMS = {
    "calm_bull":  dict(mu=0.35,  rv=0.10, iv_prem=1.35, ts=0.88),
    "range":      dict(mu=0.05,  rv=0.11, iv_prem=1.35, ts=0.91),
    "bear_trend": dict(mu=-0.40, rv=0.19, iv_prem=1.05, ts=0.99),
    "vol_shock":  dict(mu=-0.60, rv=0.45, iv_prem=0.62, ts=1.12),
}

# daily regime transition matrix (rows = from, cols = to). Regimes are sticky:
# expected durations ~ 1/(1-p_stay) = 60, 40, 30, 12 trading days.
TRANSITIONS = np.array([
    [0.9833, 0.0100, 0.0050, 0.0017],
    [0.0125, 0.9750, 0.0100, 0.0025],
    [0.0067, 0.0133, 0.9667, 0.0133],
    [0.0100, 0.0150, 0.0583, 0.9167],
])

TRADING_DAYS = 252
EARNINGS_EVERY = 63          # quarterly
EARNINGS_MOVE = 0.035        # true std-dev of the earnings-day jump
EARNINGS_IMPLIED = 0.032     # what the options market prices (slightly cheap)


@dataclass
class MarketData:
    """Daily market data + hidden regimes. `df` is indexed by trading day."""
    df: pd.DataFrame
    seed: int | None = None
    meta: dict = field(default_factory=dict)

    FEATURES = ["vix", "iv_rank", "term_structure", "ema_trend",
                "rv_iv_spread", "days_to_earnings", "ret_5d"]

    def __len__(self):
        return len(self.df)

    @property
    def close(self):
        return self.df["close"].to_numpy()

    def features(self, cols=None) -> pd.DataFrame:
        return self.df[cols or self.FEATURES]

    def split(self, frac: float = 0.7):
        """Chronological train/test split (never shuffle time series!)."""
        cut = int(len(self.df) * frac)
        return (MarketData(self.df.iloc[:cut].copy(), self.seed, self.meta),
                MarketData(self.df.iloc[cut:].copy(), self.seed, self.meta))


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def simulate_market(n_days: int = TRADING_DAYS * 20, seed: int = 0,
                    s0: float = 100.0, regime_params: dict | None = None,
                    transitions: np.ndarray | None = None,
                    warmup: int = 260) -> MarketData:
    """Simulate `n_days` of market data (after a warm-up used for indicators).

    Returns a MarketData whose DataFrame has one row per trading day.
    """
    rng = np.random.default_rng(seed)
    rp = regime_params or REGIME_PARAMS
    P = transitions if transitions is not None else TRANSITIONS
    N = n_days + warmup
    names = list(rp.keys())

    # 1) hidden regime path (a Markov chain)
    reg = np.zeros(N, dtype=int)
    reg[0] = 0
    for t in range(1, N):
        reg[t] = rng.choice(len(names), p=P[reg[t - 1]])

    mu = np.array([rp[n]["mu"] for n in names])[reg]
    rv = np.array([rp[n]["rv"] for n in names])[reg]
    prem = np.array([rp[n]["iv_prem"] for n in names])[reg]
    ts_target = np.array([rp[n]["ts"] for n in names])[reg]

    # 2) earnings calendar: every 63 days, with a random phase
    phase = rng.integers(0, EARNINGS_EVERY)
    is_earn = ((np.arange(N) - phase) % EARNINGS_EVERY) == 0
    # days until next earnings (0 = earnings today)
    dte_earn = np.zeros(N, dtype=int)
    nxt = None
    for t in range(N - 1, -1, -1):
        if is_earn[t]:
            nxt = t
        dte_earn[t] = (nxt - t) if nxt is not None else EARNINGS_EVERY

    # 3) returns: student-t shocks (fat tails) scaled to the regime's vol
    dt = 1.0 / TRADING_DAYS
    z = rng.standard_t(df=5, size=N) / np.sqrt(5 / 3)       # unit variance
    ret = (mu - 0.5 * rv ** 2) * dt + rv * np.sqrt(dt) * z
    ret += is_earn * rng.normal(0, EARNINGS_MOVE, size=N)
    close = s0 * np.exp(np.cumsum(ret))

    # 4) implied vol: mean-reverts toward regime target (with lag) and jumps
    #    up after down days (the "leverage effect")
    iv = np.zeros(N)
    iv[0] = rv[0] * prem[0]
    for t in range(1, N):
        target = rv[t] * prem[t]
        shock = -1.0 * min(ret[t], 0.0)                     # down day -> IV up
        iv[t] = iv[t - 1] + 0.25 * (target - iv[t - 1]) + shock \
            + 0.006 * rng.standard_normal()
        iv[t] = max(iv[t], 0.07)

    ts = np.zeros(N)
    ts[0] = ts_target[0]
    for t in range(1, N):
        ts[t] = ts[t - 1] + 0.2 * (ts_target[t] - ts[t - 1]) + 0.01 * rng.standard_normal()

    vix = iv * 100
    vix3m = vix / ts
    rv20 = pd.Series(ret).rolling(20).std().to_numpy() * np.sqrt(TRADING_DAYS) * 100
    ema9, ema20 = _ema(close, 9), _ema(close, 20)
    roll = pd.Series(vix).rolling(252, min_periods=60)
    lo, hi = roll.min().to_numpy(), roll.max().to_numpy()
    iv_rank = np.clip((vix - lo) / np.maximum(hi - lo, 1e-9) * 100, 0, 100)
    ret_5d = (close / np.roll(close, 5) - 1) * 100

    df = pd.DataFrame({
        "close": close,
        "ret": ret,
        "vix": vix,
        "vix3m": vix3m,
        "term_structure": ts,
        "iv_rank": iv_rank,
        "rv20": rv20,
        "rv_iv_spread": rv20 - vix,
        "ema9": ema9,
        "ema20": ema20,
        "ema_trend": (ema9 - ema20) / close * 100,
        "ret_5d": ret_5d,
        "days_to_earnings": dte_earn,
        "is_earnings": is_earn,
        "dow": np.arange(N) % 5,
        "regime": reg,
        "regime_name": np.array(names)[reg],
    }).iloc[warmup:].reset_index(drop=True)

    # a business-day calendar just for nicer plots
    df.index = pd.bdate_range("2006-01-02", periods=len(df))
    df.index.name = "date"
    df["dow"] = df.index.dayofweek
    return MarketData(df=df, seed=seed, meta=dict(regimes=names))
