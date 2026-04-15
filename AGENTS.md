# AGENT GUIDE: mjlab

This document is a practical handbook for human contributors and coding agents working in this repository.
It focuses on fast orientation, safe change patterns, and reproducible validation.

---

## 1) Project Identity

**mjlab** is a robotics RL framework that combines an Isaac Lab-like API with MuJoCo Warp physics.

Primary goals:
- Modular environment/task construction.
- High-throughput RL training on NVIDIA GPUs.
- Practical sim-to-real research workflows.

Important constraints:
- Training is GPU-first (MuJoCo Warp + CUDA stack).
- CPU or macOS usage is mostly for lightweight evaluation/dev, not high-throughput training.

REPPO integration note:
- `rsl_rl` stays REPPO-only.
- Any compatibility with older `rsl_rl` calling conventions, checkpoint shapes, or runner expectations belongs in `src/mjlab/rl/runner.py` and task config wrappers here in `mjlab2`.
- Keep the `rsl_rl` checkout clean; do not patch it for `mjlab2`-specific legacy behavior.
- FastTD3 follows the same boundary: the Yam FastTD3 task is registered in `src/mjlab/tasks/manipulation/config/yam/` and uses `MjlabFastTD3Runner` for all compatibility translation.
- FastTD3 checkpoints are native only; do not maintain a legacy `model_state_dict` resume path in this repository.
- The Yam FastTD3 config mirrors the reference distributional-critic path, including per-env exploration noise and clipped double Q selection.
- The current parity pass also aligns the Yam FastTD3 hidden-layer defaults with the reference architecture and keeps reward-normalization support configurable from the mjlab2 side.
- REPPO defaults in `rsl_rl` were checked against the reference TorchRL config; keep task-specific overrides in `mjlab2` rather than broadening the core library.
- For read-only Slurm overlay runs, prefer `scripts/slurm_overlay_env.sh` and keep writable cache/checkpoint paths outside the overlay. `mjlab.utils.wandb` now honors `MJLAB_WANDB_CACHE_DIR`, then `MJLAB_CACHE_DIR`, then `XDG_CACHE_HOME` before falling back to `~/.cache`.
- On the `torch` cluster, the canonical launch order is: `singularity exec --nv ...` -> `cd` into the repo -> `source scripts/slurm_overlay_env.sh ...` -> `uv run --no-sync --locked ...`. Do not rely on the deprecated `/ext3/env.sh` path for repo environment management.
- On the `torch` cluster, node-allocation notification belongs in the home-layer Slurm helpers such as `get-cpu` / `get-gpu*`, not inside `mjlab2`.
- For W&B tracking runs, keep the default project mapping stable: single-motion tracking logs to `single_track`, multi-motion tracking logs to `multi_track`. Only override `agent.wandb_project` when you intentionally want a different project.
- This project mapping was verified against real `torch` runs: single-track logs cleanly to W&B and active multi-track runs save under `g1_multitracking` with `wandb_project: multi_track`.

---

## 2) Repository Map (High Signal)

```text
.
├── src/mjlab/
│   ├── actuator/        # Actuator abstractions and implementations
│   ├── entity/          # Entity data models and wrappers
│   ├── envs/            # Environment and MDP-level logic
│   ├── managers/        # Manager pattern (actions/obs/rewards/events/...)
│   ├── scene/           # Scene composition and runtime integration
│   ├── scripts/         # CLI entry scripts (train/play/demo/tools)
│   ├── tasks/           # Task definitions (tracking, velocity, etc.)
│   └── utils/           # Shared utilities (config/logging/buffers/noise/...)
├── tests/               # Unit and integration-style tests
├── docs/                # Sphinx documentation source
├── pyproject.toml       # Build/deps/tooling configuration
├── Makefile             # Common developer commands
└── README.md            # User-facing overview and quick start
```

When starting work, locate the change in `src/mjlab/*`, then immediately open nearby tests in `tests/*`.

---

## 3) Architecture Notes for Agents

### 3.1 Manager-centric runtime design
Many runtime behaviors are delegated to manager classes (action/observation/reward/event/termination).

Agent implications:
- Behavior changes often require manager config + manager logic updates.
- Keep data contracts explicit (tensor shape, dtype, device, batched env dimension).

### 3.2 Task and MDP composition
Task definitions coordinate scene, command generation, rewards, terminations, and randomization.

Agent implications:
- Small reward/termination edits can destabilize training.
- Always verify assumptions with at least a targeted test path or smoke scenario.

### 3.3 Actuator/sensor boundaries
Actuators and sensors are decoupled components with task-level wiring.

Agent implications:
- Preserve interface stability when modifying class signatures.
- Add compatibility shims only when necessary and document why.

### 3.4 Compatibility/forked utilities
`src/mjlab/utils/lab_api/` includes compatibility-oriented code inspired by Isaac Lab.

Agent implications:
- Treat this area as high-sensitivity.
- Avoid broad refactors unless the change objective explicitly targets compatibility layers.

---

## 4) Development Workflow (Preferred)

### 4.1 Environment bootstrap
```bash
uv sync --all-extras --all-packages --group dev
```

### 4.2 Iteration loop
```bash
make format
make type
make test-fast
```

### 4.3 Full verification (before risky merges)
```bash
make test
```

### 4.4 Packaging sanity checks
```bash
make build
```

### 4.5 Docs build (when touching docs/API narrative)
```bash
make docs
```

---

## 5) Coding Rules for Agents

1. Keep patches minimal and local.
2. Prefer extending existing patterns over introducing new abstractions.
3. Do not silently change CLI/config semantics.
4. For public behavior changes:
   - update tests,
   - update docs/examples if user-facing,
   - note migration impact in PR body.
5. Never add broad try/except wrappers that hide actionable errors.
6. Preserve deterministic behavior in tests whenever possible.

---

## 6) Change Playbooks

### 6.1 Add a new reward term
1. Add implementation in the correct MDP/reward module.
2. Register/configure term in task config.
3. Add or update tests for:
   - expected sign/range,
   - edge conditions,
   - batched env behavior.
4. Run `make test-fast` at minimum.

### 6.2 Modify actuator behavior
1. Confirm actuator interface and downstream manager expectations.
2. Validate units/scaling/clipping assumptions.
3. Update tests for delayed/builtin/xml/learned paths as needed.
4. Prefer incremental changes over full rewrites.

### 6.3 Add a script/CLI command
1. Implement under `src/mjlab/scripts/`.
2. Register entry point in `pyproject.toml` `[project.scripts]` if needed.
3. Add usage notes in README/docs if public.
4. Add at least one smoke-level validation path.

### 6.4 Touch configuration loading
1. Preserve backward compatibility where practical.
2. Fail with clear error messages for invalid config.
3. Keep precedence explicit: task defaults -> `config_path` / `config_source` -> CLI.
4. Add tests for missing/invalid/override cases.

---

## 7) Testing Strategy

Recommended levels:
- **Level A (required for most PRs):**
  - `make format`
  - `make type`
  - `make test-fast`
- **Level B (for behavior-sensitive changes):**
  - targeted test modules + `make test`
- **Level C (release confidence):**
  - `make build`
  - docs build if relevant

If GPU-dependent tests cannot run in the current environment, explicitly report the gap and provide the exact command to run later.

---

## 8) Performance and Stability Guardrails

- Watch for hidden CPU/GPU transfers and device mismatch.
- Keep tensor operations vectorized across environments.
- Avoid Python loops in hot simulation/training paths unless measured.
- Be careful with randomization ranges; unstable distributions can break training quickly.
- For NaN/inf issues, inspect observation scaling, reward terms, actuator outputs, and physics parameters first.

---

## 9) Troubleshooting Checklist

### Install/runtime
- Python version in supported range (`>=3.10,<3.14`).
- Dependency sync completed successfully.
- CUDA/driver compatibility for Warp/Torch stack.
- Correct MuJoCo/MUJOCO_GL runtime setup for the host.

### Training instability
- Reduce env count and learning rate.
- Run with deterministic or simplified agent (`random` / `zero`) to isolate environment issues.
- Enable diagnostics around NaN guard and intermediate tensors.

### Reproducibility
- Record seed, commit hash, task ID, and command line.
- Keep config overrides in PR description or experiment logs.

---

## 10) PR Quality Checklist (Agent Self-Review)

Before finalizing, verify all items:

- [ ] Scope is clear and limited.
- [ ] Code follows existing module patterns.
- [ ] Tests added/updated for behavior changes.
- [ ] Local commands and outcomes are documented.
- [ ] User-facing docs updated when needed.
- [ ] Risks and follow-ups are called out explicitly.

Suggested commit prefixes:
- `feat:` new capability
- `fix:` bug fix
- `refactor:` structural cleanup without behavior change
- `test:` tests only
- `docs:` documentation only
- `chore:` maintenance

---

## 11) Quick Command Reference

```bash
# format + lint fixes
make format

# static typing
make type

# full test set
make test

# faster test set (skip slow)
make test-fast

# cpu-only test mode
make test-cpu-fast

# package build + smoke import check
make build

# docs build
make docs
```

---

## 12) Notes for Future Updates of This File

When updating this guide, prefer additive changes and keep examples executable.
If repository conventions evolve (new tooling, new task layout, CI policy changes), update this file in the same PR.


---

## 13) Config Tooling Deep Dive (Must-Read)

This repository has two different "config" concepts. Do not mix them:

1. **Runtime/training dataclass configs** via the `tyro + OmegaConf` helpers in `src/mjlab/utils/config_loader.py` and `src/mjlab/utils/omegaconf_cli.py`.
2. **MuJoCo spec editing configs** via `SpecCfg` subclasses in `src/mjlab/utils/spec_config.py`.

### 13.1 Runtime config helpers (external source -> dataclass)

Use this stack for experiment/training/task dataclasses loaded from YAML/JSON/local paths/HTTP(S)/W&B artifact URIs, then overridden from the CLI.

Current boundary:
- Dataclasses define schema and defaults.
- `OmegaConf` handles source loading, merging, and nested override application.
- `tyro` handles CLI parsing and help generation.

Core load/save APIs:
- `load_dataclass_from_yaml(cls, path)`
- `load_dataclass_from_json(cls, path)`
- `load_dataclass_from_dict(cls, data)`
- `load_config_path(path)`
- `load_config_source(source)`
- `save_dataclass_to_yaml(cfg, path)`
- `save_dataclass_to_json(cfg, path)`
- `override_dataclass_from_source(cfg, source)`

Companion helpers:
- `apply_config_overrides(cfg, overrides, recursive=True)`
- `merge_configs(base_cfg, override_cfg, recursive=True)`

CLI helpers:
- `parse_dataclass_cli(config_type, args=None, config_path_arg="config_path", config_source_arg="config_source")`
- `parse_choice_and_dataclass(...)`
- `run_cli_function(...)`

Recommended usage pattern:
```python
from dataclasses import dataclass, field
from mjlab.utils import (
  apply_config_overrides,
  load_dataclass_from_yaml,
)

@dataclass(kw_only=True)
class OptimCfg:
  lr: float = 3e-4
  epochs: int = 100

@dataclass(kw_only=True)
class TrainCfg:
  seed: int = 0
  optim: OptimCfg = field(default_factory=OptimCfg)

cfg = load_dataclass_from_yaml(TrainCfg, "conf/train.yaml")
apply_config_overrides(cfg, {"optim": {"lr": 1e-4}}, recursive=True)
```

Agent rules for runtime config helpers:
- Always use plain `@dataclass(kw_only=True)` config objects.
- Keep defaults meaningful; partial YAML should still instantiate safely.
- Validate external paths and sources early and fail loudly with clear error messages.
- Prefer `apply_config_overrides` for patch-style overrides inside scripts.
- Keep nested override dictionaries shape-compatible with target dataclasses.
- Preserve the documented precedence order: task defaults -> `config_path` / `config_source` -> CLI overrides.
- When task-specific config is selected dynamically, verify whether a field is runtime-consumed or only used during task-factory construction before claiming it is safely overridable.

Known caveat:
- `merge_configs(...)` uses field-default comparison heuristics; if defaults are dynamic or mutable, verify expected merge behavior with a focused test.
- Some tracking-task fields are structurally decided in the task factory before late CLI mutation. Help-visible fields are not automatically guaranteed to be late-overridable.

### 13.2 `SpecCfg` (dataclass -> mutate `mujoco.MjSpec`)

Use this family when the goal is physics/spec mutation (textures, materials, collision, lights, etc.), not experiment hyperparameter loading.

Typical flow:
1. Build/obtain `mujoco.MjSpec`.
2. Instantiate one or multiple `SpecCfg` subclasses.
3. Call each config's `edit_spec(spec)`.
4. Compile/run and validate behavior.

Agent rules for `SpecCfg` edits:
- Keep validation in each cfg (`validate()` methods) strict and explicit.
- Preserve regex-based targeting semantics for collision configuration.
- When changing defaults, check downstream effects on contact, friction, and solver stability.

### 13.3 Which config tool should I use?

- Need to load/save trainer/env/task options from files, URLs, W&B, or the CLI? -> **runtime config helper stack**.
- Need to modify MuJoCo model/spec internals programmatically? -> **`SpecCfg` stack**.
- Need both? -> Load dataclass config first, then transform into `SpecCfg` instances explicitly.

### 13.4 Config-change checklist for PRs

When a PR touches config tooling, include:
- [ ] A before/after example command or YAML snippet.
- [ ] Error behavior for invalid fields/types.
- [ ] Precedence behavior: task defaults -> config source -> CLI.
- [ ] At least one targeted test for the modified path.
- [ ] Note whether the change affects runtime config helpers, `SpecCfg`, or both.
