# Deep Python Mastery — A Learning Plan

A focused, exhaustive roadmap to mastering **Python the language** and the engineering
craft around it. No web frameworks, no AI tooling — just the language, its runtime, its
standard library, and the practices that separate a scripter from an expert.

> **Your baseline:** dicts, lists, `for` loops, list comprehensions, plus classes &
> decorators (recently covered). This plan starts just above that and climbs to the
> internals. Every phase lists topics, why they matter, how to practice, and a milestone.

> **Version note:** examples target modern Python (3.10–3.13). A few cutting-edge features
> (free-threading, PEP 695 type syntax, subinterpreters) are noted by version. The language
> keeps moving; I can check the very latest release's additions if you want before you start.

---

## The shape of the journey

```
Phase 0  From intermediate to fluent      (fill the gap above fundamentals)
Phase 1  The object & data model          (dunders, equality, containers)
Phase 2  Functions in depth               (closures, functools, dispatch)
Phase 3  Iteration, generators, laziness  (the iterator protocol, itertools)
Phase 4  The type system & static analysis(typing, mypy, protocols, generics)
Phase 5  Metaprogramming                  (descriptors, metaclasses, reflection)
Phase 6  Concurrency & parallelism        (threads, GIL, multiprocessing, asyncio)
Phase 7  CPython internals & performance  (bytecode, memory, GC, profiling)
Phase 8  Standard-library mastery         (the batteries, properly)
Phase 9  Errors, robustness & logging     (exceptions, warnings, defensive design)
Phase 10 Engineering craft                (testing, tooling, packaging, design)
```

Roughly linear, but Phases 8–10 can run in parallel with everything else — dip into the
stdlib and write tests continuously rather than saving them for the end.

---

## Phase 0 — From intermediate to fluent

The layer just above your stated baseline. Quick to absorb, and everything later assumes it.

**Topics**
- **Sequence tricks**: slicing (incl. step and negative), `enumerate`, `zip`/`zip(strict=)`,
  `reversed`, `sorted` with `key`, `any`/`all`, unpacking (`a, *rest, b`), swapping.
- **Dict/set fluency**: dict & set comprehensions, `dict.get`/`setdefault`, merge operators
  (`|`, `|=`), `dict` ordering guarantees, set algebra.
- **Truthiness & idioms**: what's falsy, short-circuiting, the ternary expression, **EAFP vs
  LBYL**, duck typing.
- **Function args**: `*args`/`**kwargs`, default-argument mutable trap, keyword-only (`*`)
  and positional-only (`/`) parameters, argument unpacking at call sites.
- **Identity vs equality**: `is` vs `==`, `None` checks, mutability, interning surprises.
- **f-strings in full**: the format spec mini-language (`:>10,.2f`), `{x=}` debugging,
  `!r`/`!s` conversions, nested fields.
- **The walrus operator** `:=` and where it genuinely helps.
- **Structural pattern matching** (`match`/`case`): literal, capture, sequence, mapping,
  class patterns, OR patterns, guards, wildcard `_`.

**Milestone:** rewrite a chunk of old code of yours to be idiomatic — replace index loops
with `enumerate`/`zip`, manual dict-building with comprehensions, and nested `if` chains
with `match` where it reads better.

---

## Phase 1 — The object & data model

Python's power comes from the data model: your objects can plug into every language
operator and built-in by implementing the right dunder methods. You've met these with
classes; now learn the *full* protocol set.

**Topics**
- **Representation**: `__repr__` vs `__str__`, `__format__` (hook into f-strings),
  `__bytes__`.
- **Equality, hashing, ordering**: `__eq__`, `__hash__` (the equal-objects-equal-hashes
  contract), `__lt__` and friends, `functools.total_ordering`, unhashable-after-`__eq__`
  rule.
- **Containers**: `__len__`, `__getitem__`, `__setitem__`, `__delitem__`, `__contains__`,
  `__iter__`, `__reversed__`; slicing support via `slice` objects; building correct custom
  containers by subclassing `collections.abc` (`Sequence`, `Mapping`, `MutableMapping`).
- **Callables**: `__call__` (objects that act like functions).
- **Attribute access**: `__getattr__`, `__getattribute__`, `__setattr__`, `__delattr__`,
  `__dir__` — and the difference between them.
- **Context managers**: `__enter__`/`__exit__`, reentrancy, suppressing exceptions.
- **Numeric protocol**: `__add__`/`__radd__`/`__iadd__` (and the whole arithmetic family),
  `__neg__`, `__abs__`, `__round__`, `__index__`, `__bool__`.
- **Copying & pickling hooks**: `__copy__`, `__deepcopy__`, `__getstate__`, `__setstate__`.
- **Class creation hooks**: `__init_subclass__`, `__set_name__`, `__class_getitem__` (what
  makes `list[int]` work).
- **Descriptors (intro)**: `__get__`/`__set__`/`__delete__`; data vs non-data descriptors;
  the realization that `property`, methods, `classmethod`, `staticmethod` are *all*
  descriptors. (Deep dive in Phase 5.)
- **`__slots__`**: memory savings, tradeoffs, inheritance interactions.

**Milestone:** implement a fully-featured custom container (say a `TimeSeries` or
`OrderedSet`) that supports `len`, indexing, slicing, iteration, `in`, equality, a good
`repr`, and correct hashing — verified against `collections.abc`.

---

## Phase 2 — Functions in depth

**Topics**
- **First-class functions & closures**: LEGB scope resolution, `nonlocal`/`global`,
  late-binding closures (the loop-variable gotcha), `__closure__`/cell objects.
- **Decorators mastery** (you have this — cement it): stacking, arguments (3-layer),
  class-based, decorating methods/classes, `functools.wraps`, preserving signatures with
  `inspect`.
- **`functools`**: `partial`, `partialmethod`, `reduce`, `cache`/`lru_cache`,
  `cached_property`, `singledispatch`/`singledispatchmethod`, `total_ordering`,
  `cmp_to_key`, `wraps`/`update_wrapper`.
- **Functional style**: `map`/`filter`/`sorted`/`min`/`max` with `key`, the `operator`
  module (`itemgetter`, `attrgetter`, `methodcaller`), composition, currying via `partial`,
  when *not* to go functional (readability).
- **Callable introspection**: `inspect.signature`, binding arguments, default values,
  annotations at runtime.

**Milestone:** build a small, reusable decorator library — `@timed`, `@retry(n)`,
`@memoize`, `@validate_types`, `@log_calls` — all signature-preserving and stackable.

---

## Phase 3 — Iteration, generators & laziness

The heart of Pythonic data processing and memory efficiency.

**Topics**
- **The iterator protocol**: `__iter__`/`__next__`, `StopIteration`, iterables vs
  iterators, why `for` works, manual iteration with `iter()`/`next()` (incl. the
  two-argument sentinel form).
- **Generators**: `yield`, generator functions vs expressions, lazy evaluation, statefulness,
  memory wins over lists.
- **Generator delegation**: `yield from` for composing generators.
- **Generators as coroutines** (the classic mechanism): `.send()`, `.throw()`, `.close()` —
  worth understanding conceptually as the root of `async`.
- **`itertools` in full**: `chain`, `islice`, `tee`, `cycle`, `repeat`, `count`,
  `accumulate`, `groupby`, `product`, `permutations`, `combinations`, `zip_longest`,
  `starmap`, `takewhile`/`dropwhile`, `filterfalse`, `pairwise`, `batched` (3.12).
- **Lazy pipelines**: chaining generators into memory-efficient data flows; when laziness
  bites (single-pass exhaustion, debugging).

**Milestone:** build a lazy pipeline that reads a large data source row-by-row and threads
it through several generator stages (parse → filter → transform → aggregate) with constant
memory — then prove the memory profile with `tracemalloc`.

---

## Phase 4 — The type system & static analysis

Type hints are now central to professional Python. This phase pays off in every codebase
you touch and directly powers data-validation libraries.

**Topics**
- **Core annotations**: variables, parameters, returns; `Optional`/`X | None`, `Union`/`|`,
  `Any`, `Final`, `ClassVar`.
- **Precise types**: `Literal`, `TypedDict` (with `total`, `Required`/`NotRequired`),
  `NewType`, `Annotated`, `cast`, `assert_type`, `reveal_type`.
- **Generics**: `TypeVar` (bound, constrained, variance — covariant/contravariant),
  `Generic[T]`, generic functions and classes; new **PEP 695** syntax (3.12):
  `def first[T](x: list[T]) -> T` and `type Vector = list[float]`.
- **Callables & decorators typed correctly**: `Callable`, `ParamSpec`, `Concatenate` (so
  your decorators keep their signatures under type checking).
- **Structural typing**: `Protocol`, `runtime_checkable` — duck typing the type checker can
  verify; usually preferable to ABCs for interfaces.
- **Narrowing**: `isinstance`, `TypeGuard`/`TypeIs`, `Never`, exhaustiveness checking with
  `match`.
- **Special forms**: `Self` (3.11), `@override` (3.12), `@overload` for multiple signatures.
- **Tooling**: `mypy` and/or `pyright`, strict mode, gradual typing, `# type: ignore`
  discipline, stub files (`.pyi`), `typing.TYPE_CHECKING` for import cycles.

**Milestone:** take one of your modules, add complete type hints, run it under `mypy
--strict` until clean, and type a generic decorator correctly with `ParamSpec`.

---

## Phase 5 — Metaprogramming

The "code that manipulates code" layer. Descriptors + decorators + metaclasses are the
trinity. You'll rarely *write* metaclasses, but understanding them demystifies frameworks.

**Topics**
- **Descriptors, deep**: data vs non-data precedence, building reusable validators/typed
  attributes, exactly how `property`/`classmethod`/`staticmethod`/methods are implemented as
  descriptors, `__set_name__` for self-naming descriptors.
- **Dynamic attributes**: `__getattr__` vs `__getattribute__`, `__setattr__`, computed and
  proxied attributes, `__slots__` interactions.
- **Class creation**: `__init_subclass__` (the modern, lightweight alternative to
  metaclasses), metaclasses via `type(name, bases, ns)` and `class Meta(type)`, real
  metaclass use cases (registries, enforcement, ABCs), `abc.ABCMeta`, `__subclasshook__`,
  virtual subclasses and `ABC.register`.
- **Reflection & introspection**: `getattr`/`setattr`/`hasattr`/`delattr`, `vars`, `dir`,
  `type`, `isinstance`/`issubclass`, the `inspect` module (signatures, source, members,
  stack frames).
- **Code as data**: `compile`, `exec`, `eval` (and why to avoid them), the `ast` module
  (parsing/transforming source), `dis` for reading bytecode.

**Milestone:** build a small declarative system — e.g., a schema/ORM-lite where class
attributes are typed descriptor fields, subclasses auto-register via `__init_subclass__`,
and instances validate on assignment. (This is how real ORMs and Pydantic-likes work.)

---

## Phase 6 — Concurrency & parallelism

The biggest genuinely-new area for most intermediate Pythonistas, and essential for I/O-
heavy work.

**Topics**
- **The models & when to use each**: threads (I/O-bound), processes (CPU-bound), async
  (high-concurrency I/O) — and the reasoning to pick correctly.
- **The GIL**: what it protects, why threads don't parallelize CPU work, and the
  **free-threaded / no-GIL** builds (experimental in 3.13) plus per-interpreter GIL work.
- **Threading**: `threading`, locks/RLock/Condition/Event/Semaphore, thread safety, race
  conditions, deadlocks, `queue.Queue` for safe hand-off, `thread_local`.
- **Multiprocessing**: `multiprocessing`, `Pool`, pipes/queues, shared memory
  (`shared_memory`), pickling constraints, `if __name__ == "__main__"` guard.
- **`concurrent.futures`**: `ThreadPoolExecutor`/`ProcessPoolExecutor`, `submit`/`map`,
  `as_completed`, `Future` objects — the high-level API to reach for first.
- **`asyncio`, deeply**: the event loop, coroutines vs tasks, `await`, `gather`,
  `create_task`, `TaskGroup` (3.11), cancellation & `CancelledError`, timeouts
  (`asyncio.timeout`), async sync primitives, `run_in_executor` to offload blocking code,
  async generators (`async for`), async context managers (`async with`, `AsyncExitStack`),
  and **`contextvars`** for per-task state.
- **Ecosystem awareness**: `anyio`/`trio` (structured concurrency), `asyncio` pitfalls
  (blocking the loop, sync/async mixing).
- **Subinterpreters** (3.12 C-API / 3.13+): the emerging isolation model.

**Milestone:** implement the same concurrent I/O task three ways — `ThreadPoolExecutor`,
`asyncio` + an async HTTP client, and a process pool for a CPU-bound variant — and measure
which wins and why. Add a `Semaphore` to bound concurrency.

---

## Phase 7 — CPython internals & performance

Understanding the runtime turns "it's slow" into "here's exactly why."

**Topics**
- **Execution model**: source → bytecode → the evaluation loop; code objects, frames, the
  `dis` module; what `.pyc` files are.
- **Object & memory model**: everything-is-an-object, reference counting, the cyclic
  **garbage collector** (`gc` module, generations, `__del__` pitfalls, reference cycles),
  object layout, `sys.getsizeof`, small-int and string **interning**, `__slots__` savings,
  `weakref` and finalizers.
- **The import system**: how imports resolve, `sys.modules`, `importlib`, finders/loaders,
  packages vs modules, `__all__`, relative imports, **namespace packages**, breaking
  circular imports.
- **Profiling**: `cProfile` + `snakeviz`/`pstats`, `timeit` for micro-benchmarks,
  `line_profiler`, `memory_profiler`, `tracemalloc`, `py-spy`/`scalene` for live/sampling
  profiling. **Profile before optimizing — always.**
- **Optimization**: algorithmic/data-structure choices first, vectorization, caching, string
  `join` vs concatenation, local-variable lookups, avoiding needless work; reading `dis`
  output to understand cost.
- **Going native when needed**: `ctypes`/`cffi` (call C), `Cython`, `numba` (JIT), writing
  C extensions conceptually, and awareness of PyPy.

**Milestone:** take a slow function, profile it, form a hypothesis from the `dis`/profile
output, optimize (algorithm → vectorize → cache → native as needed), and document the
before/after with numbers.

---

## Phase 8 — Standard-library mastery

"Batteries included" is Python's superpower — experts reach for the stdlib before pip.
Tour these until you know what exists (you can look up details later).

**Topics**
- **Data structures**: `collections` (`defaultdict`, `Counter`, `deque`, `OrderedDict`,
  `namedtuple`, `ChainMap`), `heapq`, `bisect`, `array`, `collections.abc`.
- **Records & constants**: `dataclasses` in full (`field`, `default_factory`, `InitVar`,
  `__post_init__`, `frozen`, `slots=True` (3.10), `kw_only`, `ClassVar`), `enum` (`Enum`,
  `IntEnum`, `Flag`, `IntFlag`, `StrEnum` (3.11), `auto`, `@unique`, functional API),
  `typing.NamedTuple`.
- **Numbers & time**: `math`, `statistics`, `decimal` (correct money math), `fractions`,
  `random`/`secrets`, `datetime` + **`zoneinfo`** (proper timezone handling), `time`,
  `calendar`.
- **Text & data formats**: `re` (regex mastery — groups, lookahead/behind, flags,
  compiled patterns), `string`, `textwrap`, `json`, `csv`, `pickle` (+ its security
  caveats), `sqlite3`, `shelve`, `struct`, `base64`.
- **Files & OS**: `pathlib` (use it over `os.path`), `os`, `sys`, `shutil`, `tempfile`,
  `glob`, `io` (`StringIO`/`BytesIO`, buffering), `subprocess` (running processes safely).
- **Program plumbing**: `logging` (hierarchical loggers, handlers, formatters, filters —
  not `print`), `argparse`, `contextlib` (`contextmanager`, `ExitStack`, `suppress`,
  `closing`, `redirect_stdout`), `functools`, `itertools`, `operator`, `warnings`,
  `traceback`, `atexit`, `abc`, `copy`, `weakref`.

**Milestone:** replace ad-hoc code in a project with the right stdlib tool in three places
(e.g., `Counter` for tallies, `pathlib` for paths, `logging` configured properly with
levels and a file handler).

---

## Phase 9 — Errors, robustness & logging

Expert code fails clearly and recovers gracefully.

**Topics**
- **Exception model**: the built-in hierarchy, catching specifically (never bare `except`),
  `else`/`finally`, re-raising, **exception chaining** (`raise ... from`), the difference
  between `raise X` and `raise X from None`.
- **Custom exceptions**: designing an exception hierarchy for your library/app.
- **Exception groups** (3.11): `ExceptionGroup` and `except*` for concurrent/aggregated
  errors.
- **Cleanup & resources**: context managers for guaranteed release, `ExitStack` for dynamic
  resource sets, `try/finally` patterns.
- **Warnings**: the `warnings` module, deprecation strategy.
- **Defensive design**: validation at boundaries, fail-fast, `assert` (and why it's not for
  production validation), idempotency, retries with backoff.
- **Debugging**: `pdb`/`breakpoint()`, post-mortem debugging, `faulthandler`, reading
  tracebacks fluently.

**Milestone:** add a coherent custom exception hierarchy and structured logging to a
project, so every failure is catchable by type and every run is traceable through logs.

---

## Phase 10 — Engineering craft

The practices that make code trustworthy and maintainable — the real dividing line at
"expert."

**Topics**
- **Testing**: `pytest` (fixtures, `parametrize`, markers, `conftest`, plugins), mocking
  (`unittest.mock`, `monkeypatch`, patching correctly), coverage, `doctest`,
  **property-based testing with `hypothesis`** (superb for numeric/edge-case code), testing
  async and concurrency, `tox`/`nox` for multi-env.
- **Static quality**: `ruff` (lint + format — replaces black/flake8/isort), `mypy`/`pyright`,
  `pre-commit` hooks, complexity/coverage gates.
- **Environments & packaging**: `venv`, **`uv`** (fast, modern) or Poetry/PDM,
  `pyproject.toml` (PEP 517/518/621), dependency resolution & lock files, building
  (`build`) and publishing (`twine`) a package, entry points/console scripts, editable
  installs, semantic versioning.
- **Version control**: Git fluency (branching, rebase, bisect, hooks), meaningful commits.
- **Design & architecture**: Pythonic design patterns (strategy, factory, observer,
  adapter, dependency injection), **SOLID** applied to Python, composition over inheritance,
  `Protocol`-based interfaces, module/package structure, separating I/O from logic for
  testability, clean-architecture ideas.

**Milestone:** turn one project into a *proper package*: `pyproject.toml`, a `pytest` +
`hypothesis` suite with coverage, `ruff` + `mypy` + `pre-commit` wired up, installable with
`uv`, and a README. This single exercise exercises most of the phase.

---

# The exhaustive mastery checklist

Track yourself against this. If you can *explain and use* each item, you're at expert
breadth. (Grouped, not strictly ordered.)

**Language core**
- [ ] Slicing, unpacking, star-expressions, positional-only `/` & keyword-only `*` params
- [ ] Mutable-default trap; identity vs equality; interning
- [ ] Truthiness, EAFP vs LBYL, duck typing
- [ ] f-string format mini-language; `{x=}`; `!r`/`!s`
- [ ] Walrus `:=`; structural pattern matching (all pattern kinds + guards)

**Object & data model**
- [ ] `__repr__`/`__str__`/`__format__`/`__bytes__`
- [ ] `__eq__`/`__hash__` contract; ordering + `total_ordering`
- [ ] Container protocol; slicing via `slice`; `collections.abc` subclassing
- [ ] `__call__`; attribute-access dunders; context-manager dunders
- [ ] Full arithmetic/`__r*__`/`__i*__`; `__bool__`/`__index__`
- [ ] `__init_subclass__`, `__set_name__`, `__class_getitem__`; `__slots__`

**Functions**
- [ ] Closures, LEGB, `nonlocal`/`global`, late-binding gotcha
- [ ] Decorators: args, class-based, stacking, signature-preserving
- [ ] `functools` (partial, cache, singledispatch, total_ordering, wraps)
- [ ] `operator` module; functional composition; `inspect.signature`

**Iteration**
- [ ] Iterator protocol; iterables vs iterators; two-arg `iter`
- [ ] Generators; `yield from`; `.send`/`.throw`/`.close`
- [ ] `itertools` (chain, groupby, islice, accumulate, product, pairwise, batched)
- [ ] Lazy pipelines; single-pass pitfalls

**Type system**
- [ ] `Optional`/`Union`/`Literal`/`Final`/`ClassVar`/`Annotated`/`TypedDict`/`NewType`
- [ ] Generics: `TypeVar` (bound/constrained/variance), `Generic`, PEP 695 syntax
- [ ] `ParamSpec`/`Concatenate` for decorators; `Callable`
- [ ] `Protocol`/`runtime_checkable`; `Self`/`@override`/`@overload`
- [ ] Narrowing, `TypeGuard`/`TypeIs`, `Never`, exhaustiveness
- [ ] `mypy`/`pyright` strict; stubs; `TYPE_CHECKING`

**Metaprogramming**
- [ ] Descriptors (data vs non-data); how property/methods are built
- [ ] `__getattr__` vs `__getattribute__`; dynamic/proxied attributes
- [ ] Metaclasses & `type()`; `__init_subclass__`; ABCs, `register`, `__subclasshook__`
- [ ] `inspect`; `ast`; `dis`; `compile`/`exec`/`eval` (and their dangers)

**Concurrency**
- [ ] Thread vs process vs async decision model
- [ ] GIL internals; free-threaded/no-GIL awareness
- [ ] `threading` primitives; `queue`; thread safety
- [ ] `multiprocessing`, pools, shared memory, pickling limits
- [ ] `concurrent.futures`
- [ ] `asyncio`: loop, tasks, `TaskGroup`, cancellation, timeouts, async gens/`with`
- [ ] `contextvars`; `run_in_executor`; anyio/trio awareness

**Internals & performance**
- [ ] Bytecode/`dis`; code objects & frames; `.pyc`
- [ ] Refcounting + cyclic GC; `__del__` pitfalls; `weakref`
- [ ] Memory sizing; interning; `__slots__` savings; `tracemalloc`
- [ ] Import system; namespace packages; circular-import fixes
- [ ] Profiling (`cProfile`, `timeit`, line/memory/py-spy)
- [ ] Optimization strategy; `ctypes`/`Cython`/`numba` awareness

**Standard library**
- [ ] `collections`, `heapq`, `bisect`, `array`
- [ ] `dataclasses` (all options), `enum` (all variants), `NamedTuple`
- [ ] `decimal`/`fractions`/`statistics`; `datetime`+`zoneinfo`
- [ ] `re` mastery; `json`/`csv`/`pickle`/`sqlite3`/`struct`
- [ ] `pathlib`, `io`, `subprocess`, `shutil`, `tempfile`
- [ ] `logging`, `argparse`, `contextlib`, `warnings`, `copy`, `weakref`

**Robustness**
- [ ] Exception hierarchy; specific catching; chaining; `raise ... from`
- [ ] Custom exception design; `ExceptionGroup`/`except*`
- [ ] Context managers & `ExitStack` for cleanup
- [ ] `pdb`/`breakpoint`; reading tracebacks; post-mortem

**Craft**
- [ ] `pytest` (fixtures/parametrize/mock); `hypothesis`; coverage; `doctest`
- [ ] `ruff`, `mypy`, `pre-commit`, `tox`/`nox`
- [ ] `uv`/Poetry, `pyproject.toml`, build/publish, entry points
- [ ] Git fluency; design patterns; SOLID; composition; `Protocol` interfaces

---

## Prioritization — the highest-leverage order

If you want the steepest capability gain fastest:

1. **Phase 1 (data model)** and **Phase 3 (generators/iteration)** — these two make you
   *think* in Python and unlock most idioms.
2. **Phase 6 (concurrency/asyncio)** — the biggest genuinely-new skill; huge for any I/O
   work (broker APIs, data feeds).
3. **Phase 4 (typing)** and **Phase 10 (testing/tooling)** — turn working code into
   professional code; compound forever.
4. **Phase 7 (internals/perf)** and **Phase 5 (metaprogramming)** — the depth that
   distinguishes experts and demystifies every library you'll read.
5. **Phases 8–9 (stdlib, errors)** — absorb continuously, alongside the rest.

Skim, don't grind: metaclasses (recognize > write), C extensions (awareness first), PyPy.

---

## Reference shelf (durable classics)

- **Fluent Python** (Luciano Ramalho, latest ed.) — the definitive advanced-Python book;
  maps almost 1:1 onto Phases 1–5.
- **Python Cookbook** (Beazley & Jones) — recipe-driven depth across the stdlib and idioms.
- **Effective Python** (Brett Slatkin) — 90+ concrete best-practice items.
- **CPython Internals** (Anthony Shaw) — for Phase 7.
- **High Performance Python** (Gorelick & Ozsvald) — profiling & optimization.
- **Architecture Patterns with Python** (Percival & Gregory) — design at scale.
- **Official docs**: the *Language Reference* (esp. the *Data Model* chapter) and
  *PyMOTW* (Doug Hellmann) for the stdlib.
- **Talks**: David Beazley (generators, `asyncio`, concurrency) and Raymond Hettinger
  (idioms, descriptors, dataclasses) — some of the best advanced Python teaching anywhere.

---

*Want me to build a full hands-on tutorial (like the classes/decorators one) for the next
phase? The two highest-leverage builds are **Phase 1 (the data model)** — deepest return
on how you think in Python — or **Phase 6 (asyncio & concurrency)** — the biggest new
capability. Say which and I'll write it end-to-end with examples and exercises.*
