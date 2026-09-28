"""
optrl.strategies
================
The fixed basket of 7 option strategies the agent chooses between, plus a
Black-Scholes pricer to turn market data into strategy P&L.

Every strategy is a list of legs  (quantity, 'C' or 'P', strike_offset)
where strike_offset is measured in *expected moves* (em = S * sigma * sqrt(T)),
so "+1" means "one expected move above spot". Negative quantity = sold.

Position sizing: every trade is sized to risk RISK_BUDGET dollars
(max loss for defined-risk trades, a margin proxy for the short straddle),
so P&L numbers are comparable across strategies.

>>> Swap in YOUR basket: edit STRATEGIES (or skip this file entirely and feed a
>>> P&L table of your own backtests - see Part 9).
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.special import ndtr

from .market import MarketData, TRADING_DAYS, EARNINGS_IMPLIED

STRATEGIES: dict[str, list[tuple[int, str, float]]] = {
    "short_straddle":   [(-1, "C", 0.0), (-1, "P", 0.0)],
    "iron_condor":      [(-1, "C", 1.0), (+1, "C", 2.0), (-1, "P", -1.0), (+1, "P", -2.0)],
    "bull_put_spread":  [(-1, "P", -0.5), (+1, "P", -1.5)],
    "bear_call_spread": [(-1, "C", 0.5), (+1, "C", 1.5)],
    "long_straddle":    [(+1, "C", 0.0), (+1, "P", 0.0)],
    "bull_call_spread": [(+1, "C", 0.0), (-1, "C", 1.0)],
    "bear_put_spread":  [(+1, "P", 0.0), (-1, "P", -1.0)],
}
STRATEGY_NAMES = list(STRATEGIES)
ACTION_NAMES = ["no_trade"] + STRATEGY_NAMES          # action 0 = stay in cash
N_ACTIONS = len(ACTION_NAMES)                          # 8

RISK_BUDGET = 1_000.0      # dollars at risk per trade
MULT = 100                 # contract multiplier
COMMISSION = 0.65          # $ per contract per leg
SLIPPAGE_PCT = 0.03        # half bid-ask as a fraction of option price
SLIPPAGE_MIN = 0.01        # ... but at least 1 cent
STRIKE_STEP = 0.5


# ----------------------------------------------------------------- pricing
def bs_price(S, K, T, sigma, kind):
    """Black-Scholes price (r = 0, no dividends). Works on scalars or arrays."""
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    T = np.maximum(T, 1e-12)
    sd = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sd ** 2) / sd
    d2 = d1 - sd
    call = S * ndtr(d1) - K * ndtr(d2)
    if kind == "C":
        return call
    return call - S + K                                # put-call parity (r=0)


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price_scalar(S, K, T, sigma, kind):
    """Fast scalar version used inside the step() of the environment."""
    if T <= 1e-9:
        return max(S - K, 0.0) if kind == "C" else max(K - S, 0.0)
    sd = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sd * sd) / sd
    call = S * _ncdf(d1) - K * _ncdf(d1 - sd)
    return call if kind == "C" else call - S + K


def effective_iv(vix, days_to_expiry, days_to_earnings):
    """IV (decimal) for an option expiring in `days_to_expiry` trading days.
    If an earnings event happens before expiry, the market adds its implied
    move to the total variance - that's why IV jumps before earnings."""
    base = np.asarray(vix) / 100.0
    T = np.maximum(np.asarray(days_to_expiry), 1e-9) / TRADING_DAYS
    earn_inside = (np.asarray(days_to_earnings) >= 1) & \
                  (np.asarray(days_to_earnings) <= np.asarray(days_to_expiry))
    var = base ** 2 * T + earn_inside * EARNINGS_IMPLIED ** 2
    return np.sqrt(var / T)


def _round_strike(x, S):
    """Round to a strike grid of STRIKE_STEP per $100 of underlying price."""
    step = STRIKE_STEP * np.asarray(S) / 100.0
    return np.round(x / step) * step


def build_legs(name, S, sigma, T):
    """Strikes for strategy `name`. Returns list of (qty, kind, K)."""
    em = S * sigma * np.sqrt(T)
    return [(qty, kind, _round_strike(S + off * em, S))
            for qty, kind, off in STRATEGIES[name]]


def risk_per_contract(name, legs, premiums, S, sigma, T):
    """Dollars at risk for ONE contract of the strategy."""
    net_debit = sum(q * p for (q, _, _), p in zip(legs, premiums))   # >0 debit
    if name == "short_straddle":           # undefined risk -> margin proxy
        return 2 * S * sigma * np.sqrt(T) * MULT
    if name == "long_straddle" or name in ("bull_call_spread", "bear_put_spread"):
        return np.maximum(net_debit, 0.01) * MULT
    # credit spreads / condor: widest wing minus credit
    Ks = {}
    for q, k, K in legs:
        Ks.setdefault(k, []).append(K)
    width = np.maximum.reduce([np.abs(v[0] - v[1]) for v in Ks.values() if len(v) == 2])
    return np.maximum(width + net_debit, 0.01) * MULT          # net_debit<0 = credit


def entry_cost_per_contract(premiums):
    """Commission + half-spread paid when opening (or closing) one contract."""
    c = 0.0
    for p in premiums:
        c = c + COMMISSION + np.maximum(SLIPPAGE_MIN, SLIPPAGE_PCT * p) * MULT
    return c


# ------------------------------------------------------- weekly P&L table
def weekly_pnl_table(market: MarketData, hold_days: int = 5,
                     decision_dow: int = 0) -> pd.DataFrame:
    """For every decision day (default: Mondays), the $ P&L each strategy would
    have made if opened at that close and held to expiry `hold_days` later.

    This is the "counterfactual" table a backtest gives you: one row per week,
    one column per action (no_trade = 0). It powers Parts 2-3 and 8.
    """
    df = market.df
    S_all = df["close"].to_numpy()
    idx = np.where(df["dow"].to_numpy() == decision_dow)[0]
    idx = idx[idx + hold_days < len(df)]
    S0, S1 = S_all[idx], S_all[idx + hold_days]
    T = hold_days / TRADING_DAYS
    sigma = effective_iv(df["vix"].to_numpy()[idx], hold_days,
                         df["days_to_earnings"].to_numpy()[idx])

    out = {"no_trade": np.zeros(len(idx))}
    for name in STRATEGY_NAMES:
        legs = build_legs(name, S0, sigma, T)
        prem = [bs_price(S0, K, T, sigma, k) for (_, k, K) in legs]
        payoff = [np.maximum(S1 - K, 0) if k == "C" else np.maximum(K - S1, 0)
                  for (_, k, K) in legs]
        pnl_1 = sum(q * (pay - p) for (q, _, _), pay, p in zip(legs, payoff, prem)) * MULT
        cost_1 = entry_cost_per_contract(prem)             # expire -> no exit cost
        risk_1 = risk_per_contract(name, legs, prem, S0, sigma, T)
        contracts = RISK_BUDGET / risk_1
        out[name] = contracts * (pnl_1 - cost_1)

    table = pd.DataFrame(out, index=df.index[idx])
    table.index.name = "date"
    return table


def decision_features(market: MarketData, table: pd.DataFrame, cols=None):
    """Features known at the decision close, aligned with `table` rows."""
    cols = cols or MarketData.FEATURES
    return market.df.loc[table.index, cols]
