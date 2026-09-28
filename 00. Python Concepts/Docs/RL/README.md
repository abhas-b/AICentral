# Reinforcement Learning for Options Strategy Selection: a course from zero

This course teaches reinforcement learning (RL) from scratch through one project: an agent that,
at each decision point, **chooses which of your 7 option strategies to deploy (or none)** from
market conditions. It doesn't design strategies. It selects among the ones you already have.

By the end you'll be able to build, evaluate and iterate on an RL project like this on your own.

## What's inside

| Notebook | Part | You build | Runtime* |
|---|---|---|---|
| `00_orientation.ipynb` | RL vs supervised learning, why strategy selection suits RL, roadmap, setup | explore the market and the strategy P&L table | <1 min |
| `01_core_concepts.ipynb` | agent, environment, state, action, reward, policy, value, return, γ, exploration | a hand-written agent in the RL loop | <1 min |
| `02_bandits.ipynb` | multi-armed bandits: ε-greedy, UCB, Thompson sampling | strategy picking on realistic P&L distributions | ~1 min |
| `03_contextual_bandits.ipynb` | features, ridge models, LinUCB, shadow P&L | **your first regime-aware selector** | <1 min |
| `04_mdps_bellman.ipynb` | Markov property, Bellman equations, value iteration | a solved "hold / switch / sit out" MDP | <1 min |
| `05_tabular_q_sarsa.ipynb` | TD learning, Q-learning, SARSA | a tabular agent on the discretised trading MDP | ~2.5 min |
| `06_building_the_env.ipynb` | Gymnasium env design: state, actions, positions, reward, episodes, tests | `MyOptionsEnv` from scratch + a test suite | <1 min |
| `07a_deep_rl_dqn.ipynb` | just-enough neural nets, **DQN from scratch** | a DQN agent in PyTorch | ~2 min |
| `07b_policy_gradients_ppo.ipynb` | REINFORCE, PPO, Stable-Baselines3, validation checkpointing | PPO and DQN agents with SB3 | ~4 min |
| `08_evaluation_backtesting.ipynb` | baselines, walk-forward, bootstrap, best-of-N, many worlds, stress tests | an honest verdict: skill or luck? | ~4 min |
| `09_your_data_and_iteration.ipynb` | plugging in **your** data and `strategy_1..7`, pitfalls, the experiment loop, deployment | an experiment log and a one-time final test | ~2 min |
| `10_capstone.ipynb` | project brief, milestones, rubric, reference pipeline | the whole thing end to end | ~1 min |

\*On a 2-core laptop CPU. No GPU needed.

Every concept comes with **(1)** an intuitive analogy, **(2)** an options-trading scenario
(including the recurring *"Monday, VIX 13–14, EMA9 < EMA20: what should the agent pick?"*), and
**(3)** a hands-on exercise with code, the expected output and a solution. Exercise cells are
marked 🛠 and contain a `# TODO`. The solution follows in the next cell. Try it before you scroll.

The `html/` folder contains read-only copies of every notebook with outputs, for reading on
any device.

### One change to the suggested order
**The environment (Part 6) comes before deep RL (Part 7).** Deep RL agents are black boxes, and
most "RL doesn't work" problems in trading are environment bugs: look-ahead, missing costs,
wrong episode handling. Building and testing the environment first means you know exactly what
the network sees and why it gets paid. Tabular methods (Part 5) already use the environment as
a black box, so you've seen the interface before you open it up.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                          # open 00_orientation.ipynb
```

## The `optrl/` helper package

So that everything runs offline, the course ships a small, readable package:

* `market.py`: a regime-switching market simulator (price, VIX, VIX3M, term structure, trend,
  realised vol, earnings). The *hidden regime* is used only for grading what the agent learned.
* `strategies.py`: the 7 stand-in strategies (short straddle, iron condor, bull put, bear call,
  long straddle, bull call, bear put), a Black-Scholes pricer, sizing to \$1,000 risk, and costs.
  `weekly_pnl_table()` produces the counterfactual "what would each strategy have made" table.
* `envs.py`: `StrategyBandit`, `ContextualBandit`, `OptionsSelectionEnv` (the full Gymnasium MDP
  with carry-over and costs), `DiscretizeObs` (for tabular RL) and `PnLTableEnv` (an env built
  from *your* backtest table).
* `metrics.py`: Sharpe, drawdown, CVaR, rollouts and all the plots.
* `data.py`: loaders, point-in-time feature building and validation for **your own data**.

## Using your own strategies

Part 9 walks through it. In short, you provide:

1. `data/market_daily.csv` with `date, close, vix, vix3m` (plus optional `data/earnings_dates.csv`)
2. `data/strategy_pnl.csv` with `date, strategy_1, …, strategy_7`: the P&L of each strategy if
   opened at that date, managed by your rules, net of costs, sized to the same risk unit

Then rerun Parts 9 and 10.

## Honest expectations

The simulator is built so that strategy selection *can* add value. Real markets are noisier and
change over time. The course deliberately shows agents that overfit, results that depend on the
random seed, and cases where a simple rule or a linear contextual bandit beats deep RL, because
seeing these is the most useful skill in applied RL for trading. Nothing here is financial advice.
