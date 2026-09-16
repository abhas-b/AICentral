# Becoming a Python Expert — A Learning Roadmap for Quant + Agentic AI

You have two destinations (systematic trading **and** agentic AI). They look different but
share most of their foundation. This plan builds **one trunk** of deep Python +
engineering skill, then **branches** into each domain, then **reconverges** at production
deployment. Where your existing experience lets you skip ahead, it's flagged.

> **You already have:** working Python, OOP + decorators (just covered), pandas, SQLite,
> KiteConnect, Streamlit/Dash, Pine Script, and real domain knowledge in markets. That
> puts you past "beginner" — this roadmap is about depth and breadth toward *expert*.

> **A note on the agentic-AI tooling:** framework names below (LangGraph, Pydantic AI,
> MCP, specific vector DBs, etc.) are the landscape as of early 2026 and this space moves
> fast. Treat the **concepts** as durable and the **specific tools** as swappable. I can
> pull the current tooling landscape before you start any Track-B phase if you want the
> freshest picks.

---

## The map at a glance

```
                    ┌─────────────────────────────────────────┐
                    │  TRUNK — shared foundation (do first)    │
                    │  1 Language mastery                      │
                    │  2 Concurrency & async (asyncio)         │
                    │  3 Type system & Pydantic                │
                    │  4 Engineering craft (test/tool/package) │
                    │  5 Data & storage (NumPy/pandas/SQL)     │
                    └───────────────┬─────────────────────────┘
                                    │  branch
              ┌─────────────────────┴──────────────────────┐
              ▼                                             ▼
   ┌──────────────────────┐                    ┌──────────────────────────┐
   │  TRACK A — QUANT      │                    │  TRACK B — AGENTIC AI    │
   │  A1 Sci stack (SciPy) │                    │  B1 LLM fundamentals+APIs│
   │  A2 Time series       │                    │  B2 Structured out/tools │
   │  A3 Stats/ML finance  │                    │  B3 RAG + vectors        │
   │  A4 Backtest/event    │                    │  B4 Agent orchestration  │
   │  A5 Perf/execution    │                    │  B5 MCP + tool ecosystems│
   └──────────┬───────────┘                    │  B6 Eval/observability   │
              │                                 │  B7 async scale/streaming│
              │                                 └───────────┬──────────────┘
              └─────────────────────┬───────────────────────┘  reconverge
                                    ▼
                    ┌─────────────────────────────────────────┐
                    │  PRODUCTION — shared                     │
                    │  6 Serving & deploy (FastAPI/Docker/CI)  │
                    │  7 System design & architecture          │
                    │  8 Capstones                             │
                    └─────────────────────────────────────────┘
```

**Convergence points (learn once, use in both):** `asyncio`, Pydantic, FastAPI, SQL /
databases, testing, packaging & tooling, logging/observability, Docker. These are where
the two goals overlap the most — prioritize them.

---

## How to sequence it (recommended)

Don't do all of Track A then all of Track B. **Finish the trunk, then interleave** the two
branches so momentum in one reinforces the other. A realistic order:

1. Trunk phases 1–5 (this is the biggest investment and pays off everywhere).
2. Track B, phases B1–B2 (LLM APIs + structured output/tool calling) — fastest path to
   something impressive, and tool-calling ideas mirror your semi-automated execution style.
3. Track A, phases A1–A2 (SciPy + time series) — sharpens your existing quant work.
4. Alternate B3→A3→B4→A4 … as interest pulls you.
5. Production phases 6–8 once you have real projects to deploy.

Rough total to genuine expert-level breadth: think in terms of **12–18 months** of
consistent effort, not weeks. "Expert" is a direction, not a finish line.

---

# TRUNK — the shared foundation

## Phase 1 — Advanced Python language mastery

Go from "I can write Python" to "I understand the data model." This is the single highest-
leverage phase; everything else compounds on it.

**Topics**

- **The data model / dunder protocols** (extends what you just learned): iteration
  protocol, `__iter__`/`__next__`, sequence/mapping protocols, numeric protocols.
- **Iterators & generators**: `yield`, generator expressions, lazy evaluation, `yield from`,
  infinite streams, pipelines. *(Directly useful for streaming bars/ticks and for
  streaming LLM tokens.)*
- **Comprehensions** at fluency: list/dict/set/generator, nested, conditional.
- **Closures & scope**: LEGB rule, `nonlocal`, `global`, late-binding gotchas.
- **`functools` / `itertools` / `operator`**: `partial`, `reduce`, `cache`, `chain`,
  `groupby`, `islice`, `accumulate`, `product`, `permutations`. Master `itertools` — it's
  the standard-library superpower.
- **Exceptions done well**: exception hierarchy, custom exceptions, `else`/`finally`,
  chaining (`raise ... from`), exception groups & `except*` (3.11+), context.
- **Context managers** in depth: `contextlib` (`contextmanager`, `ExitStack`,
  `suppress`, `closing`).
- **Descriptors, `__init_subclass__`, `__set_name__`, metaclasses** — the deep machinery.
  Learn descriptors well; treat metaclasses as "recognize, rarely write."
- **Memory & object model**: references vs copies, mutability, identity vs equality,
  `is` vs `==`, `id()`, garbage collection, `weakref`, interning.
- **Standard library fluency**: `collections` (`defaultdict`, `Counter`, `deque`,
  `namedtuple`), `dataclasses`, `enum`, `pathlib`, `datetime`/`zoneinfo` (IST handling!),
  `json`, `logging`, `argparse`, `subprocess`, `re`, `decimal` (money math).
- **Pythonic idioms**: EAFP vs LBYL, truthiness, unpacking, `*`/`**`, walrus `:=`,
  structural pattern matching (`match`/`case`, 3.10+).

**How to practice:** re-implement 3–4 stdlib pieces yourself (an `LRU` cache, a `Counter`,
a simple `@dataclass`-like decorator). Read the CPython docs' *Data Model* page end to end.

**Milestone:** build a lazy, generator-based data pipeline that reads your SQLite bars and
streams indicator calculations without loading everything into memory.

---

## Phase 2 — Concurrency & async (the bridge phase)

This is the most important convergence skill you don't fully have yet. Agents make many
concurrent API/tool calls; data systems juggle many I/O streams. `asyncio` is central to
modern agent frameworks and to responsive data feeds.

**Topics**

- **The three models & when to use each**: threading (I/O-bound, limited by the **GIL**),
  multiprocessing (CPU-bound, true parallelism), asyncio (high-concurrency I/O).
- **The GIL**: what it actually locks, why threads don't speed up CPU work, and the
  no-GIL / free-threaded work landing in recent CPython.
- **`concurrent.futures`**: `ThreadPoolExecutor`, `ProcessPoolExecutor`, `as_completed`.
- **`asyncio` deeply**: `async`/`await`, event loop, coroutines vs tasks, `gather`,
  `create_task`, `TaskGroup` (3.11+), timeouts, cancellation, `Semaphore` for rate
  limiting, async context managers & iterators, `async for`/`async with`.
- **Async libraries**: `httpx`/`aiohttp` (async HTTP — you'll hammer LLM and broker APIs),
  `anyio`.
- **Pitfalls**: blocking the event loop, mixing sync/async, `run_in_executor`.

**Why it matters for both:** an agent calling 5 tools concurrently, or a backtester
fetching many symbols' data in parallel, are the *same* async problem.

**Milestone:** rewrite one KiteConnect data-fetch routine to pull N symbols concurrently
with `asyncio` + `httpx` + a `Semaphore` for rate limiting; measure the speedup.

---

## Phase 3 — Type system & data validation (Pydantic)

Type hints + Pydantic are load-bearing in *both* modern quant codebases and every agentic
framework (structured LLM outputs are literally Pydantic models).

**Topics**

- **`typing` in depth**: `Optional`, `Union`/`|`, `Literal`, `TypedDict`, `Protocol`
  (structural typing — often better than ABCs), `Generic`, `TypeVar`, `ParamSpec`,
  `Callable`, `overload`, `Annotated`, `Self`, `TypeAlias`.
- **Static checking**: `mypy` or `pyright`/`pylance`, gradual typing, `# type: ignore`
  discipline, strict mode.
- **Pydantic v2**: models, validation, `field_validator`/`model_validator`, `Field`
  constraints, serialization, `Settings` for config, custom types, JSON schema
  generation. **This is the single most transferable library across your two goals.**

**Milestone:** define Pydantic models for your `Order`/`Trade`/`Signal` domain with
validation (positive qty, valid symbols, price bounds) and use the same modeling style
later for LLM structured outputs.

---

## Phase 4 — Engineering craft

The difference between a scripter and an engineer. Non-negotiable for "expert."

**Topics**

- **Testing**: `pytest` (fixtures, parametrize, markers, conftest), mocking
  (`unittest.mock`, `monkeypatch`), coverage, **property-based testing with `hypothesis`**
  (excellent for numeric/financial code), testing async code.
- **Tooling & packaging**: virtual envs, **`uv`** (fast modern installer/resolver) or
  Poetry, `pyproject.toml`, `ruff` (lint + format, replaces black/flake8/isort), building
  and publishing a package, entry points, `pipx`.
- **Version control at depth**: Git branching, rebasing, bisect, hooks, `pre-commit`.
- **Debugging & profiling**: `pdb`/`breakpoint()`, `logging` done properly (not `print`),
  `cProfile` + `snakeviz`, `line_profiler`, `memory_profiler`, `py-spy` for live processes.
- **Performance optimization**: algorithmic thinking, vectorization vs loops, caching,
  when to reach for `numba`/`Cython`/native extensions.
- **Design patterns in Python** (Pythonic versions — not Java-style): strategy, factory,
  observer, dependency injection, and knowing when *not* to use a pattern.

**Milestone:** take one of your existing tools (say the scanner), add a `pytest` suite with
`hypothesis`, wire up `ruff` + `pre-commit`, package it with `uv`, and profile the hot path.

---

## Phase 5 — Data & storage foundation

Shared substrate for both tracks: numerical arrays, dataframes, and databases.

**Topics**

- **NumPy** properly: ndarrays, dtypes, broadcasting, vectorization, fancy indexing,
  views vs copies, `einsum`, random generators. *(Underpins pandas, ML, and embeddings.)*
- **pandas** beyond the basics (you use it — go deeper): MultiIndex, `groupby` internals,
  `merge`/`join`, `resample`, `rolling`/`ewm`, categorical dtypes, method chaining,
  performance (avoid `apply` where vectorization works), `datetime`/timezone handling.
- **Polars**: the fast, expression-based dataframe library — increasingly the default for
  performance-sensitive data work. Learn the lazy API.
- **SQL mastery**: joins, window functions, CTEs, indexing, query planning, transactions.
  You use SQLite; also learn Postgres concepts. Look at **DuckDB** for fast analytical
  queries over your local data.
- **Serialization/formats**: Parquet, Arrow, JSON, msgpack.

**Milestone:** benchmark the same indicator computation in pandas vs Polars vs DuckDB over
your bar database; understand *why* the fast one is fast.

---

# TRACK A — Quantitative / systematic trading

You have domain knowledge; this track adds rigor and computational depth.

## A1 — Scientific computing (SciPy)
Optimization, interpolation, linear algebra, signal processing, statistical functions.
Useful for parameter optimization, curve fitting, and filter-based indicators.

## A2 — Time-series analysis
Stationarity, differencing, autocorrelation, ARIMA/SARIMA, GARCH (volatility),
cointegration (pairs trading), `statsmodels`. Proper train/test discipline for time
series (walk-forward, no look-ahead — you already know survivorship bias, extend that
rigor). Feature engineering for financial series.

## A3 — Statistics, probability & ML for finance
Probability distributions, hypothesis testing, Bayesian basics, Monte Carlo (you've built
simulators — formalize the theory). `scikit-learn` pipeline discipline, cross-validation
adapted for time series, overfitting/regularization, and an honest view of ML's limits in
markets. Optionally: gradient boosting (XGBoost/LightGBM), and the basics of RL for
execution/strategy (careful — heavily overhyped in retail trading).

## A4 — Backtesting & event-driven architecture
Event-driven vs vectorized backtests, avoiding look-ahead and survivorship bias, realistic
cost/slippage modeling, position sizing and risk (you know Kelly — formalize portfolio
risk). Study frameworks (`vectorbt`, `backtrader`, `zipline`-style designs) then build
your own event loop — it's a great systems exercise and matches your existing
`options_tester` work.

## A5 — Performance & execution
Latency thinking, efficient data structures for order books, numba/Cython for hot loops,
handling real-time streams (websockets), and the reliability/idempotency concerns of
semi-automated execution (your preferred style: signals + manual approval).

**Track A capstone:** a properly-engineered, tested, event-driven backtester with realistic
costs, walk-forward validation, and a clean plugin interface for strategies (reuse the
ABC + registry-decorator pattern from your last tutorial).

---

# TRACK B — Agentic AI

This is the newer domain for you. Build it on the trunk — especially async, Pydantic, and
FastAPI. Order matters here; each phase builds on the last.

## B1 — LLM fundamentals & the model APIs
- **Conceptual**: what a transformer/LLM does at a high level, tokenization, context
  windows, temperature/sampling, embeddings, why models hallucinate, cost/latency tradeoffs.
- **The APIs**: Anthropic and OpenAI messages APIs, system vs user vs assistant roles,
  streaming responses, token accounting, retries/backoff. Prompt engineering as an
  engineering discipline (not folklore): clear instructions, few-shot, decomposition.
- Practice: call the API directly with `httpx` before touching any framework, so you
  understand what frameworks abstract.

## B2 — Structured outputs & tool calling (the heart of agents)
- **Structured outputs**: force models to return validated JSON → **Pydantic** models
  (this is why Phase 3 mattered). Libraries: `instructor`, Pydantic AI, native
  structured-output/JSON-schema features.
- **Function/tool calling**: define tools, let the model choose and fill arguments, execute,
  feed results back. This is *exactly* your "signal generation with manual approval"
  philosophy applied to AI — the model proposes, your code (or you) disposes.
- Practice: build a small "assistant" that calls 2–3 real Python functions (e.g., fetch a
  quote, compute an indicator) via tool calling.

## B3 — RAG (Retrieval-Augmented Generation) & vectors
- **Embeddings**: what they are, cosine similarity, embedding models.
- **Vector databases**: Chroma, Qdrant, `pgvector`, FAISS, Pinecone — indexing, metadata
  filtering, hybrid search.
- **RAG pipeline**: chunking strategies, retrieval, reranking, context assembly, evaluation
  of retrieval quality. Know when RAG helps and when it's the wrong tool.

## B4 — Agent orchestration
- **Agent patterns**: ReAct (reason+act loops), planning, reflection/self-critique,
  tool-use loops, and multi-agent designs (supervisor/worker, debate).
- **Frameworks**: **LangGraph** (graph/state-machine orchestration — currently the most
  robust for complex agents), **Pydantic AI** (type-safe, lightweight), plus awareness of
  LlamaIndex (RAG-heavy), CrewAI/AutoGen (multi-agent). Learn the *concepts* via one, then
  you can read them all. State management, memory, and control flow are the real skills.
- Practice: build a multi-step agent with a graph — plan → retrieve → act → reflect — with
  explicit state.

## B5 — MCP & tool ecosystems
- **Model Context Protocol (MCP)**: the open standard for connecting models to tools/data
  sources via servers — increasingly the interoperability layer for agents. Learn to both
  *use* MCP servers and *write* one exposing your own tools (e.g., your market-data DB).

## B6 — Evaluation, observability & guardrails
- **Evaluation**: LLM apps are non-deterministic — you need evals (golden datasets, LLM-as-
  judge, regression tests). Tools like LangSmith / tracing platforms, and structured
  logging of prompts, tokens, latencies, costs.
- **Guardrails & safety**: input/output validation, prompt-injection defense (critical when
  agents touch tools/data — an injected instruction must never trigger a real action),
  content filtering, and human-in-the-loop gating for consequential actions.
- **Cost/latency engineering**: caching, batching, model routing (cheap model for easy
  steps), streaming for UX.

## B7 — Async at scale & streaming
Bring Phase 2 to bear: concurrent tool calls, streaming token handlers, backpressure,
timeouts, and graceful degradation. This is where junior and senior agent engineers diverge.

**Track B capstone:** a production-shaped agent — FastAPI backend, tool calling into real
functions, RAG over a document set, LangGraph (or Pydantic AI) orchestration with explicit
state, streaming responses, Pydantic-validated I/O, evals, tracing, and human-in-the-loop
approval before any side-effectful action.

---

# RECONVERGENCE — production & architecture (shared)

## Phase 6 — Serving & deployment
- **FastAPI**: async endpoints, Pydantic request/response models, dependency injection,
  background tasks, websockets, streaming responses. (Serves *both* a trading dashboard/API
  and an agent backend.)
- **Containerization**: Docker, multi-stage builds, `docker compose`.
- **CI/CD**: GitHub Actions (tests, lint, build on push).
- **Config & secrets**: env-based settings (Pydantic Settings), never hardcode API keys.
- **Cloud basics**: deploying a container, managed databases, object storage, serverless.

## Phase 7 — System design & architecture
Reliability, idempotency, retries, queues/task systems (Celery/Redis, or lighter),
caching layers, observability (metrics/logs/traces), and designing for failure. For
trading: correctness and auditability. For agents: cost control and safe autonomy.

## Phase 8 — Capstones
Ship the two capstones above, deployed and observable. Then, the ambitious fusion project:
**an agentic research assistant for your own trading** — an agent that queries your bar
database (via an MCP server you wrote), runs your backtester as a tool, retrieves your
Wyckoff/VPA notes via RAG, and proposes (never auto-executes) analyses for your approval.
That single project exercises nearly everything on this map.

---

# Prioritization — if you have limited time

**Do these first, in this order — they unlock the most:**
1. Phase 1 (language mastery) — everything compounds on it.
2. Phase 2 (asyncio) — the biggest gap for both goals.
3. Phase 3 (typing + Pydantic) — load-bearing everywhere.
4. Phase 4 (testing/tooling) — turns scripts into engineering.
5. Then pick the branch that excites you *this month* and go deep (B1–B2 is the fastest
   "wow"; A1–A2 sharpens what you already do).

**Skim/skip for now:** metaclasses (recognize, don't write), fine-tuning LLMs (API + RAG +
tools covers ~90% of agentic use cases), heavy multi-agent frameworks until you've built a
single agent well.

---

# Habits that separate experts

- **Read source code** of libraries you use (pandas, Pydantic, an agent framework).
- **Build the thing before reaching for the framework** — call the raw API, write the raw
  event loop — so you understand what's being abstracted.
- **Test the non-deterministic stuff too** (property-based tests for numerics; evals for
  agents).
- **Profile before optimizing**; measure, don't guess.
- **Keep a "why" journal** — for every design decision, one line on the tradeoff. This is
  how tacit expertise accumulates.

---

*Want me to turn any single phase into a full hands-on tutorial like the classes/decorators
one? Phase 2 (asyncio) or Track B's B1–B2 (LLM APIs + tool calling) would be the highest-
leverage next builds. I can also pull the current agentic-AI tooling landscape first so the
framework picks are up to date before we dive in.*
