"""
optrl.metrics - performance statistics, rollouts and the tutorial's plots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .strategies import ACTION_NAMES

# ---------------------------------------------------------------- styling
# A colour-blind-checked categorical palette; one fixed colour per action so a
# strategy is always the same colour in every chart of the tutorial.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#8a8983"
ACTION_COLORS = {"no_trade": NEUTRAL, **{n: PALETTE[i] for i, n in enumerate(ACTION_NAMES[1:])}}
DIVERGING = LinearSegmentedColormap.from_list(
    "rb", ["#e34948", "#f3a3a0", "#f0efec", "#86b6ef", "#1c5cab"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#f0efec", "#86b6ef", "#2a78d6", "#0d366b"])


def set_style():
    plt.rcParams.update({
        "figure.figsize": (9, 4), "figure.dpi": 100, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8, "axes.edgecolor": "#8a8983", "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titleweight": "bold",
        "axes.titlesize": 12, "lines.linewidth": 2, "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE), "axes.axisbelow": True,
    })


# ------------------------------------------------------------ statistics
def sharpe(pnl, periods_per_year: int = 52) -> float:
    """Annualised Sharpe ratio of a P&L series (no risk-free rate)."""
    pnl = np.asarray(pnl, float)
    if pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(pnl) -> float:
    """Largest peak-to-trough fall of cumulative P&L (a positive $ number)."""
    eq = np.cumsum(np.asarray(pnl, float))
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float((peak - eq).max())


def cvar(pnl, alpha: float = 0.05) -> float:
    """Average of the worst `alpha` fraction of outcomes (expected shortfall)."""
    x = np.sort(np.asarray(pnl, float))
    n = max(1, int(np.ceil(alpha * len(x))))
    return float(x[:n].mean())


def summarize(pnl, periods_per_year: int = 52, name: str = "") -> pd.Series:
    pnl = np.asarray(pnl, float)
    traded = pnl[pnl != 0]
    return pd.Series({
        "total_pnl": pnl.sum(),
        "mean_per_period": pnl.mean(),
        "sharpe": sharpe(pnl, periods_per_year),
        "max_drawdown": max_drawdown(pnl),
        "cvar_5%": cvar(pnl),
        "win_rate": (traded > 0).mean() if len(traded) else np.nan,
    }, name=name)


# -------------------------------------------------------------- rollouts
def run_policy(env, policy, n_episodes: int = 1, starts=None, seed: int = 0) -> pd.DataFrame:
    """Roll out `policy(obs, env) -> action` and log every step."""
    rows = []
    for ep in range(n_episodes):
        opts = {"start": int(starts[ep])} if starts is not None else None
        obs, info = env.reset(seed=seed + ep, options=opts)
        done = False
        while not done:
            a = int(policy(obs, env))
            obs, r, term, trunc, info = env.step(a)
            rows.append({"episode": ep, "date": info["date"], "action": ACTION_NAMES[a],
                         "reward": r, "pnl": info["pnl"], "cost": info["cost"],
                         "regime": info["regime"], "position": info["position"]})
            done = term or trunc
    return pd.DataFrame(rows)


def full_period_rollout(env_cls, market, policy, **env_kwargs) -> pd.DataFrame:
    """One long episode that walks through the WHOLE market period once."""
    n_steps = (len(market.df) - env_kwargs.get("dte", 10) - 2) // env_kwargs.get("step_days", 5) - 1
    env = env_cls(market, episode_len=n_steps, random_start=False, **env_kwargs)
    return run_policy(env, policy, n_episodes=1)


# ----------------------------------------------------------------- plots
def smooth(x, w: int = 50):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_equity(curves: dict, title: str = "Equity curve", ax=None, ylabel="Cumulative P&L ($)"):
    """curves: {label: pd.Series of per-period P&L (index = dates)}"""
    ax = ax or plt.subplots(figsize=(9, 4))[1]
    for i, (lab, s) in enumerate(curves.items()):
        s = pd.Series(s)
        c = ACTION_COLORS.get(lab, PALETTE[i % len(PALETTE)])
        ax.plot(s.index, s.cumsum().values, label=lab, color=c, lw=2)
    ax.axhline(0, color="#8a8983", lw=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=9)
    return ax


def plot_selection_heatmap(df: pd.DataFrame, row: str, col: str = "action", title=None,
                           normalize: str = "index", ax=None, fmt="{:.0%}"):
    """Heatmap of how often each action was chosen, broken down by `row`
    (e.g. the hidden regime, or an IV-rank bucket)."""
    ct = pd.crosstab(df[row], df[col], normalize=normalize)
    return plot_share_matrix(ct, title or f"How often each action is chosen, by {row}", ax=ax, fmt=fmt)


def plot_share_matrix(mat: pd.DataFrame, title: str, ax=None, fmt="{:.0%}"):
    """Sequential heatmap of a rows x actions matrix of shares (0..1)."""
    if set(mat.columns) <= set(ACTION_NAMES):          # always show all 8 actions, same order
        mat = mat.reindex(columns=ACTION_NAMES, fill_value=0.0)
    ax = ax or plt.subplots(figsize=(1.1 * mat.shape[1] + 2, 0.55 * mat.shape[0] + 1.6))[1]
    vmax = max(float(np.nanmax(mat.values)), 1e-9)
    ax.imshow(mat.values, cmap=SEQUENTIAL, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=35, ha="right")
    ax.set_yticks(range(mat.shape[0]), mat.index)
    ax.grid(False)
    for (r, c), v in np.ndenumerate(mat.values):
        if v > 0.005:
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 * vmax else "#0b0b0b")
    ax.set_title(title)
    return ax


def plot_value_heatmap(mat: pd.DataFrame, title: str, ax=None, fmt="{:.0f}", center=0.0):
    """Diverging heatmap (e.g. mean P&L of each strategy in each regime)."""
    ax = ax or plt.subplots(figsize=(1.1 * mat.shape[1] + 2, 0.55 * mat.shape[0] + 1.6))[1]
    lim = np.nanmax(np.abs(mat.values - center))
    ax.imshow(mat.values, cmap=DIVERGING, vmin=center - lim, vmax=center + lim, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=35, ha="right")
    ax.set_yticks(range(mat.shape[0]), mat.index)
    ax.grid(False)
    for (r, c), v in np.ndenumerate(mat.values):
        ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=8,
                color="white" if abs(v - center) > 0.6 * lim else "#0b0b0b")
    ax.set_title(title)
    return ax


def plot_learning_curve(rewards, w: int = 50, title="Learning curve", label=None, ax=None, color=None):
    new = ax is None
    ax = ax or plt.subplots(figsize=(9, 3.5))[1]
    ax.plot(smooth(rewards, w), label=label, color=color)
    if new:
        ax.set_title(title)
        ax.set_xlabel("episode")
        ax.set_ylabel(f"reward ({w}-pt moving avg)")
    return ax
