"""optrl - helper package for the 'RL for options strategy selection' tutorial."""
from .market import simulate_market, MarketData, REGIMES, REGIME_PARAMS
from .strategies import (STRATEGIES, STRATEGY_NAMES, ACTION_NAMES, N_ACTIONS, RISK_BUDGET,
                         weekly_pnl_table, decision_features, bs_price)
from .envs import StrategyBandit, ContextualBandit, OptionsSelectionEnv, DiscretizeObs, PnLTableEnv
from . import metrics

DEFAULT_SEED = 1


def load_default_market(years: int = 20, seed: int = DEFAULT_SEED):
    """The simulated 'history' used across the tutorial (20 years of daily data)."""
    return simulate_market(n_days=252 * years, seed=seed)
