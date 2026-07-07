# LLM Council Transcript - Pylance/Pyright Remediation

Generated: 2026-06-25 17:04:55 Asia/Seoul

## Original Question

Analyze my repo and find all Pylance issues and propose a plan of fixes `$llm-council`.

## Framed Question

Core request: Analyze the local repo at `/Users/danil/Public/imperial`, find all Pylance/Pyright issues, and propose a fix plan. Do not implement yet.

Repo context: Python 3.12+ private/local RAG project. Package code is under `src/imperial_rag`; scripts are under `scripts`; tests are under `tests`. The repo has no `pyrightconfig.json` and no `.vscode/settings.json`. `pyproject.toml` has pytest `pythonpath = ["src"]`, Ruff config, and mypy config with `python_version = "3.12"`, `ignore_missing_imports = true`, and `follow_imports = "silent"`. The existing quality gate is `scripts/check.sh`: Ruff, mypy on `src/imperial_rag`, pytest with coverage, and a diff check. The current `uv` `.venv` resolves to Python 3.14.3 even though the project target is Python 3.12+.

Observed Pylance/Pyright baseline: `npx --yes pyright@latest --outputjson --pythonversion 3.12 --pythonpath .venv/bin/python src scripts tests` using Pyright 1.1.411 analyzed 99 files and produced 548 errors and 7 warnings. Default Python-version analysis with the same interpreter produced the same counts. A run without `--pythonpath .venv/bin/python` produced extra missing-import noise, so the interpreter-backed 548/7 run is the useful baseline.

Diagnostic groups from the real run:

- `reportAttributeAccessIssue`: 475 errors
- `reportIndexIssue`: 41 errors
- `reportTypedDictNotRequiredAccess`: 9 errors
- `reportOptionalMemberAccess`: 7 errors
- `reportArgumentType`: 5 errors
- `reportGeneralTypeIssues`: 3 errors
- `reportOperatorIssue`: 3 errors
- `reportIncompatibleMethodOverride`: 2 errors
- `reportFunctionMemberAccess`: 2 errors
- `reportOptionalCall`: 1 error
- `reportUnsupportedDunderAll`: 6 warnings
- `reportAssertAlwaysTrue`: 1 warning

Top files by diagnostics:

- `tests/test_tracing.py`: 166
- `tests/test_providers.py`: 70
- `tests/test_web_app.py`: 57
- `tests/test_scripts.py`: 32
- `tests/test_ragas_eval.py`: 29
- `tests/test_evals.py`: 19
- `tests/test_module_structure.py`: 17
- `tests/test_pipeline.py`: 17
- `scripts/run_ragas_eval.py`: 15
- `scripts/run_phoenix_eval.py`: 13
- `src/imperial_rag/ingestion/pipeline.py`: 11
- `tests/test_elasticsearch_keyword.py`: 11
- `tests/test_pipeline_integration.py`: 11
- `src/imperial_rag/answering/workflow.py`: 9

Root-cause observations from code inspection:

1. Several package `__init__.py` files use star imports plus dynamic `__all__ = [name for name in globals() if not name.startswith("_")]`; some also subclass `ModuleType` to forward test monkeypatch assignments to underlying implementation modules. Pyright cannot infer these exports and emits unknown import symbols plus `reportUnsupportedDunderAll` warnings. Explicit `ingestion` and `app` `__init__` files do not have this problem.
2. Many tests monkeypatch fake `ModuleType` objects or imported package facades, so Pylance reports unknown module attributes on assignments like fake `Settings`, fake `configure_phoenix_tracing`, and generated fake OpenTelemetry objects.
3. Runtime source issues are smaller but real: `object`-typed workflow/retriever fields cause unknown `.invoke()` / `.as_retriever()` access; TypedDict fields in query/ingestion workflows are marked optional and then indexed as required; some `Mapping` values are inferred as `object` and indexed; optional cache wrappers/callables are called without narrowing.
4. A few true API/type mismatches exist: `Document` passed to `enumerate` in ingestion extraction, `ChatQwen(api_key=str)` conflicts with current type stubs expecting `SecretStr | callable | None`, metric parser list type excludes literal `"none"` while code appends it, Qdrant factory receives `object` where `Embeddings | None` is expected, XLSX workbook save receives `Path` where stubs expect `str | IO`, fake Elasticsearch subclass overrides are incompatible, and function attributes `.calls` need a typed fake class or casts.

Question for council: What is the strongest remediation strategy and fix order for eliminating the Pylance/Pyright issues in this repo without damaging runtime behavior or tests? Should the plan prioritize config/baseline, explicit exports, source type tightening, test helper typing, or suppressions? What should the user do first?

Stakes: The user wants an accurate plan before implementation. A bad plan could waste effort fixing hundreds of symptoms, weaken useful tests/monkeypatching, hide real type bugs with broad ignores, or break the private RAG runtime.

## Advisor Responses

### The Contrarian

The trap is treating 548 Pyright errors as 548 bugs. They are not. If you optimize for "green count" first, you will mutilate the monkeypatch-heavy tests and compatibility facades that were intentionally added to preserve old imports.

Fix order should be:

1. Lock the Pyright contract first: `pyrightconfig.json` with Python 3.12, `.venv`, `extraPaths: ["src"]`, and explicit includes for `src`, `scripts`, `tests`. No broad suppressions. This makes the target stable.
2. Fix real source typing next: `object` retriever/workflow fields, optional call/cache narrowing, TypedDict required fields, mapping value casts, `ChatQwen` key typing, Qdrant embeddings, XLSX `Path` conversion, metric literal `"none"`. These are the only diagnostics likely to hide runtime bugs.
3. Then fix package exports. Dynamic `__all__` and star facades are poison for Pyright. Replace them with explicit exports, but preserve alias/module-forwarding behavior where tests depend on monkeypatch propagation.
4. Only after that touch tests. Build typed fake helper classes/modules in shared test utilities and cast deliberately around generated fake OpenTelemetry/Elasticsearch objects. Do not rewrite tests to satisfy the checker if that weakens behavior coverage.
5. Suppress last, line-local only, for bad third-party stubs or deliberate dynamic module tricks.

What to do first: add the stable Pyright config and rerun the same command, then split the report into `src`, `scripts`, `tests`, and `facade-export` buckets. Do not start by fixing `tests/test_tracing.py`; that is symptom country.

### The First Principles Thinker

The real problem is not 548 Pyright errors. The real problem is that the repo has no explicit static-type contract, so Pyright is being asked to infer a dynamic compatibility layer and monkeypatch-heavy tests as if they were ordinary production APIs.

Do first: create the type-checking contract. Add `pyrightconfig.json` that pins `pythonVersion: "3.12"`, `venvPath`, `venv`, `extraPaths: ["src"]`, and scoped include/exclude rules. Do not tune diagnostics down yet. This establishes the measurement instrument.

Then fix in this order:

1. Source types that represent real runtime contracts: retrievers, workflows, TypedDict required fields, optional callables, provider factory return types. These are the only issues likely to hide production bugs.
2. Explicit package exports. Replace dynamic `__all__` in package facades with explicit re-exports, but preserve monkeypatch forwarding where tests depend on it. The goal is not elegance; it is making compatibility intentional.
3. Test typing helpers. Add typed fake classes/protocols/builders for fake modules, OpenTelemetry objects, callable counters, and fake Elasticsearch clients. Do not rewrite tests away from monkeypatching just to appease Pyright.
4. Narrow suppressions only after the above. Suppress facade magic or intentionally dynamic test seams locally, never globally.

The wrong first move is chasing `reportAttributeAccessIssue` one line at a time. Most of those are artifacts of untyped dynamic boundaries. Stabilize the boundaries first, then the count will collapse for the right reasons.

### The Expansionist

Turn this from cleanup into a typed contract project. First create `pyrightconfig.json` and `.vscode/settings.json` so everyone sees the same 548/7 baseline: `pythonVersion: "3.12"`, `.venv/bin/python`, `include: ["src", "scripts", "tests"]`, and no broad `ignore`. Add a JSON diagnostic summary command beside `scripts/check.sh`, initially non-blocking.

Then take the highest-upside collapse first: replace dynamic facade exports with explicit `__all__` and typed re-exports. That likely removes hundreds of false unknowns while improving package API discoverability.

Next tighten source types where Pyright found real ambiguity: protocols for retrievers/workflows, required TypedDict keys or `.get()` narrowing, concrete cache/callable narrowing, and real fixes for `ChatQwen`, `Document` enumeration, Qdrant embeddings, XLSX save, metric literals. These are runtime hardening wins, not type cosmetics.

Only after source is clean should tests get a typed fake layer: helper classes/protocols for monkeypatched modules, OpenTelemetry fakes, callable fakes with `.calls`, and Elasticsearch subclasses. Preserve monkeypatch power; just stop making every fake an untyped `ModuleType`.

Suppressions come last and should be tiny, local, and documented. The first concrete move: land config plus a repeatable diagnostic grouping report, then fix explicit exports before touching individual test symptoms.

### The Outsider

Your first problem is not "548 errors." It is an undefined contract. Pyright is analyzing a codebase full of dynamic compatibility tricks without being told what the stable public surface is.

Do this order:

1. Add `pyrightconfig.json` first, but not as a dumping ground for ignores. Lock `pythonVersion: "3.12"`, point at `.venv`, include `src`, `scripts`, `tests`, and make the baseline reproducible. This prevents every future discussion from arguing over tool invocation.
2. Fix package exports next. Dynamic `__all__` and facade modules are poison for Pylance. Replace them with explicit exports or typed compatibility wrappers. This should come before touching hundreds of symptoms, because many "unknown attribute" errors are downstream of unclear module surfaces.
3. Tighten real source types. Replace `object` workflow fields with small `Protocol`s for `.invoke()`, `.as_retriever()`, embeddings, caches, and callables. Fix TypedDict requiredness, optional narrowing, and the concrete API mismatches you already found.
4. Only then clean tests. Do not rewrite monkeypatch-heavy tests to satisfy aesthetics. Add typed fake classes, `SimpleNamespace` only where appropriate, and narrow `cast()` at dynamic test seams.
5. Suppress last. Local `# pyright: ignore[...]` is acceptable for deliberate dynamic compatibility. File-level disables or blanket "test ignore" settings would throw away the value of Pylance.

First concrete step: add the minimal Pyright config and rerun the exact JSON baseline. Then attack explicit exports before any individual diagnostic cleanup.

### The Executor

Start with config, but do not use config to hide the problem. Commit a `pyrightconfig.json` that reproduces the real target: `pythonVersion: 3.12`, include `src`, `scripts`, `tests`, use `.venv`, and add `extraPaths: ["src"]`. That makes Pylance and CI speak the same language.

Then fix in this order:

1. Explicit package exports. Replace dynamic `__all__` in package facades with explicit exports. Keep monkeypatch forwarding only where needed, but make the public surface statically visible. This removes the biggest false-positive multiplier.
2. Source type tightening. Fix the real runtime-adjacent issues next: protocols for retriever/workflow objects, required vs optional `TypedDict` fields, optional callable narrowing, `ChatQwen` API key typing, Qdrant embedding typing, workbook save path conversion, parser literal `"none"`.
3. Test helper typing. Do not rewrite test behavior. Add typed fake classes/helpers and use `cast(Any, ...)` only at monkeypatch seams. Most test errors are Pyright not understanding dynamic fakes, not broken tests.
4. Third-party/API edge suppressions. Use narrow `# pyright: ignore[rule]` only after proving the code is correct and the stub is the problem.

Do not disable `reportAttributeAccessIssue` globally. That would erase the signal. First Monday action: add the minimal Pyright config and regenerate the JSON baseline so every later fix has a stable before/after count.

## Anonymization Mapping

- Response A: The Expansionist
- Response B: The Contrarian
- Response C: The Outsider
- Response D: The Executor
- Response E: The First Principles Thinker

## Peer Reviews

### Peer Review 1

1. Strongest: A. It has the best fix order: stabilize the contract, generate a repeatable diagnostic report, collapse dynamic facade/export noise first, then fix real source contracts, then tests, then suppressions. The non-blocking summary command is a useful ratchet.
2. Biggest blind spot: E. It puts source typing before explicit exports. With 475 attribute-access errors and dynamic facades called out as a major root cause, that risks chasing downstream symptoms before removing the largest false-positive multiplier.
3. All five missed: first inventory the compatibility surface before changing exports. Run `rg` for imports from facade modules, identify monkeypatch-forwarding dependencies, and add import/alias smoke tests before replacing dynamic `__all__`. Otherwise "explicit exports" can silently break old import paths while making Pyright green. Also decide the enforcement rollout: advisory baseline first, then ratchet `src`, then `scripts/tests`, rather than making all 548 errors a single gate.

### Peer Review 2

1. Strongest: A. It handles both the editor/runtime mismatch and the remediation loop: `pyrightconfig` plus `.vscode` settings, machine-readable diagnostics, then the real high-volume causes in the right order: facade exports, source contracts, tests, suppressions last. I would borrow D's explicit warning not to disable `reportAttributeAccessIssue` globally.
2. Biggest blind spot: B. Its "source typing before package exports" order risks chasing noisy false positives caused by the dynamic package facades. With star exports and dynamic `__all__`, Pyright may misread the public surface, so source errors are harder to distinguish from import/export artifacts.
3. All five miss a ratchet strategy: capture the JSON baseline, bucket by owner/category, add a no-new-errors gate, then lower the baseline in small commits. Without that, the cleanup is a one-time campaign instead of a durable typing workflow.

### Peer Review 3

1. Strongest: A. It combines the right order with measurement: config/editor parity, diagnostic summary, facade cleanup, real source contract fixes, typed tests, then narrow suppressions. That avoids wasting time on cascaded false positives.
2. Biggest blind spot: E. Starting with "source runtime contracts" before explicit exports risks chasing errors amplified by dynamic facades. In this repo, package surfaces are one of the main noise multipliers.
3. All five missed: a ratchet plan. Preserve the 548-error baseline, classify counts by diagnostic code/path, fix one bucket at a time, and add a non-blocking CI or script gate that prevents new Pyright errors while the backlog burns down.

### Peer Review 4

1. Strongest: A. It handles both remediation and feedback loop: config/Pylance settings first, diagnostic summary, then high-noise facade exports, real source contracts, typed tests, and local suppressions last. I would borrow B's `src/scripts/tests/facade` bucketing and D's warning not to disable `reportAttributeAccessIssue` globally.
2. Biggest blind spot: E. "Source runtime contracts" before explicit exports risks working under polluted diagnostics from facade/import noise, and it lacks a clear reporting loop to prove each phase reduced the right errors.
3. All five miss an explicit ratchet: pin the canonical Pyright command/env, save the baseline, track counts by bucket/rule, gate `src` before `tests`, and add CI or a non-blocking check that prevents new diagnostics while legacy issues are burned down. Also run runtime tests after type-only fixes.

### Peer Review 5

1. Strongest: A. It has the best fix order: establish config/editor parity, add a diagnostic artifact, remove high-fanout facade noise, then fix real source contracts, tests, and suppressions last.
2. Biggest blind spot: E. It starts source contracts before exports and lacks reporting/bucketing. In this repo, dynamic facades can dominate Pylance noise, so E risks chasing downstream symptoms.
3. All five miss the repo-specific runtime identity risk: making facades "explicit" must not turn alias-style compatibility modules into wrappers that break monkeypatch tests. The plan should allow `.pyi` stubs, `TYPE_CHECKING` exports, or alias-preserving modules, plus a Pyright JSON ratchet so the 548-error baseline only moves down.

## Chairman Synthesis

### Where the Council Agrees

The council agrees that the 548-error count is primarily a static-analysis contract problem plus a small set of real type-contract issues, not 548 unrelated runtime bugs. Every advisor starts with a stable Pyright/Pylance configuration and rejects broad ignores. Every advisor also preserves the monkeypatch-heavy tests and dynamic compatibility behavior until there is evidence that changing it is safe.

High-confidence points:

- Add a reproducible Pyright contract first.
- Do not globally disable `reportAttributeAccessIssue`.
- Do not fix diagnostics one line at a time before removing high-fanout causes.
- Keep compatibility and monkeypatch behavior intact.
- Use local suppressions only after source contracts, exports, and typed fakes have been addressed.

### Where the Council Clashes

The main clash is the order of source contracts versus package exports. The Contrarian and First Principles Thinker are more willing to fix real source typing before facades. The Expansionist, Outsider, Executor, and peer reviewers argue that dynamic facades should come earlier because they produce a large share of the attribute-access noise.

The resolution: add config first, then inventory compatibility surfaces, then make package export surfaces statically visible before doing broad source/test cleanup. Do not blindly rewrite facades; preserve alias identity and monkeypatch forwarding where it is part of the test/runtime contract.

### Blind Spots the Council Caught

The peer-review round caught two missing elements:

1. Before changing exports, inventory all compatibility imports and monkeypatch-forwarding dependencies. Add smoke tests for import/alias behavior that must survive.
2. Use a ratchet strategy: save the JSON baseline, group by path/rule, block new diagnostics, and reduce the baseline in small categories rather than creating one giant all-or-nothing gate.

### The Recommendation

Use this fix order:

1. Add a minimal `pyrightconfig.json` and optional VS Code workspace settings that define the same interpreter, Python version, includes, and `extraPaths` for both CLI and Pylance.
2. Add a diagnostic summary command that reads Pyright JSON and groups diagnostics by top-level path, rule, and high-volume message family. Keep it non-blocking at first.
3. Inventory compatibility facades and monkeypatch-forwarding paths using `rg`, then add smoke tests for import paths that must remain valid.
4. Replace dynamic facade exports with explicit static surfaces, `TYPE_CHECKING` exports, or `.pyi` stubs where runtime alias identity must stay dynamic.
5. Tighten production type contracts: workflow/retriever protocols, TypedDict requiredness, optional callable narrowing, and concrete argument mismatches.
6. Add typed test fake helpers for ModuleType fakes, OpenTelemetry/Phoenix clients, callable counters, and fake Elasticsearch clients.
7. Use local suppressions only for deliberate dynamic seams or bad third-party stubs, and document each one.
8. Ratchet enforcement: advisory full-repo report first, then clean `src`, then scripts, then tests.

### The One Thing to Do First

Create the minimal Pyright/Pylance configuration and regenerate the JSON baseline. Do not start by editing `tests/test_tracing.py` or adding broad ignores.

Suggested first config shape:

```json
{
  "include": ["src", "scripts", "tests"],
  "exclude": [".venv", ".imperial_rag", "documents", "docs"],
  "extraPaths": ["src"],
  "pythonVersion": "3.12",
  "pythonPlatform": "Darwin",
  "venvPath": ".",
  "venv": ".venv"
}
```

Then rerun:

```bash
npx --yes pyright@latest --outputjson --pythonversion 3.12 --pythonpath .venv/bin/python src scripts tests
```
