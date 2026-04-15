![Project banner](docs/source/_static/mjlab-banner.jpg)

# mjlab

<p align="left">
  <img alt="tests" src="https://github.com/mujocolab/mjlab/actions/workflows/ci.yml/badge.svg" />
  <a href="https://mujocolab.github.io/mjlab/"><img alt="docs" src="https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg" /></a>
  <a href="https://mujocolab.github.io/mjlab/nightly/"><img alt="benchmarks" src="https://img.shields.io/badge/nightly-blue" /></a>
</p>

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API
with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp)
physics to provide lightweight, modular abstractions for RL robotics research
and sim-to-real deployment.

---

## Quick Start

mjlab requires an **NVIDIA GPU** for training (via MuJoCo Warp).
macOS is supported only for evaluation, which is significantly slower.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Run the demo (no installation needed):

```bash
uvx --from mjlab --with "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@7c20a44bfed722e6415235792a1b247ea6b6a6d3" demo
```

This launches an interactive viewer with a pre-trained Unitree G1 agent tracking a reference dance motion in MuJoCo Warp.

> ❓ Having issues? See the [FAQ](https://mujocolab.github.io/mjlab/source/faq.html).

**Try in Google Colab (no local setup required):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb)

Launch the demo directly in your browser with an interactive Viser viewer.

---

## Installation

**From source:**

```bash
git clone https://github.com/mujocolab/mjlab.git
cd mjlab
uv sync --all-extras --all-packages --group dev
uv run demo
```

`uv` manages the project environment in `.venv`. Ensure any required system
runtime libraries for CUDA, MuJoCo, or Warp are available on the host before
running `uv sync` or `uv run`.

**From PyPI:**

```bash
uv add mjlab "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@7c20a44bfed722e6415235792a1b247ea6b6a6d3"
```

A Dockerfile is also provided.

For full setup instructions, see the [Installation Guide](https://mujocolab.github.io/mjlab/source/installation.html).

---

## Slurm / Read-Only Overlay

On the `torch` cluster, use the repo-owned environment/bootstrap only.

1. Enter the container with `singularity exec --nv ...`
2. `cd` into the repo
3. Source [`scripts/slurm_overlay_env.sh`](./scripts/slurm_overlay_env.sh)
4. Run `uv run --no-sync --locked ...`

Use only the `mjlab2` environment management in jobs. Do not rely on the old
`/ext3/env.sh` path.

[`scripts/slurm_overlay_env.sh`](./scripts/slurm_overlay_env.sh) redirects shared
cache, log, and artifact paths such as `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`,
`WARP_CACHE_PATH`, `TORCH_HOME`, `WANDB_*`, `XDG_*`, and
`MJLAB_WANDB_CACHE_DIR` into the writable runtime tree. It preserves the
container/cluster `HOME` and `TMPDIR` by default; override `MJLAB_HOME_DIR` or
`MJLAB_TMP_DIR` only when you explicitly need different values.

```bash
source scripts/slurm_overlay_env.sh /scratch/$USER/env_data/mjlab "$SLURM_JOB_ID"
uv run --no-sync --locked train Mjlab-Velocity-Flat-Unitree-G1
```

By default the script follows the shared runtime tree already used on `torch`:

```text
/scratch/$USER/env_data/mjlab/
  cache/
  logs/
  wandb/
```

Default split:
- shared across runs: `cache/*`, `logs/*`, `wandb/*`
- preserved from the host/container unless explicitly overridden: `HOME`, `TMPDIR`

`train.py` will automatically use `MJLAB_OUTPUT_DIR` for logs when it is set, and
the script now defaults that to `/scratch/$USER/env_data/mjlab/logs`. Override
`MJLAB_CACHE_DIR`, `MJLAB_WANDB_DIR`, `MJLAB_WANDB_ARTIFACT_DIR`,
`MJLAB_HOME_DIR`, `MJLAB_TMP_DIR`, or `MJLAB_OUTPUT_DIR` before sourcing the
script only when a cluster needs a different layout.

You can also use a manual suffix outside Slurm:

```bash
source scripts/slurm_overlay_env.sh /scratch/$USER/env_data/mjlab
```

Canonical `torch` batch workflow:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mjlab-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
cd /ext3/mjlab2

srun singularity exec --nv \
  --overlay /scratch/$USER/workdir/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash -lc '
    set -euo pipefail
    cd /ext3/mjlab2
    source scripts/slurm_overlay_env.sh /scratch/$USER/env_data/mjlab
    uv run --no-sync --locked train Mjlab-Velocity-Flat-Unitree-G1 env.scene.num_envs=4096
  '
```

Notes:

- `--overlay` is mounted read-only as `:ro` for training.
- `scripts/slurm_overlay_env.sh` is the runtime layout layer.
- `uv run --no-sync --locked` avoids mutating the environment inside the job.
- This keeps `TORCH_HOME`, `WANDB_CONFIG_DIR`, `XDG_*`, and `MJLAB_OUTPUT_DIR` under repo-controlled setup while preserving the cluster default `HOME`/`TMPDIR` unless you override them.

Interactive debugging on `torch` should follow the same pattern. Enter the
container first, apply the overlay env, then run a tiny command inside the live
shell before touching `sbatch` scripts.

Recommended helpers:

```bash
load_mjlab_env() {
  singularity exec --nv --fakeroot \
    --overlay /scratch/hz3862/workdir/overlay-15GB-500K.ext3:rw \
    /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
    /bin/bash -lc '
      cd /ext3/mjlab2
      source scripts/slurm_overlay_env.sh /scratch/$USER/env_data/mjlab
      exec bash -l
    '
}

load_mjlab_train_env() {
  singularity exec --nv \
    --overlay /scratch/hz3862/workdir/overlay-15GB-500K.ext3:ro \
    /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
    /bin/bash -lc '
      cd /ext3/mjlab2
      source scripts/slurm_overlay_env.sh /scratch/$USER/env_data/mjlab
      exec bash -l
    '
}
```

Example interactive smoke test:

```bash
uv run --no-sync --locked train Mjlab-MultiTracking-Flat-Unitree-G1 \
  --motion-dir artifacts \
  --agent.max-iterations 1 \
  --agent.num-steps-per-env 4 \
  --env.scene.num-envs 1
```

Use [`scripts/slurm_exec.sh`](./scripts/slurm_exec.sh) only when you are
already inside the container or when you are not using the `torch`
Singularity-based launch path. It still provides:

- `scripts/slurm_overlay_env.sh`
- `PYTHONPATH=$REPO_ROOT/src`
- optional `uv sync --frozen --group dev` when `MJLAB_UV_SYNC=1`

---

## Training Examples

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 env.scene.num_envs=4096
```

**Multi-GPU Training:** Scale to multiple GPUs using `gpu_ids=[...]`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  gpu_ids=[0,1] \
  env.scene.num_envs=4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/source/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 wandb_run_path=your-org/mjlab/run-id
```

To refresh the locked `rsl-rl-lib` Git commit and resync the local environment:

```bash
scripts/update_rsl_rl_lib.sh
```

To update `uv.lock` only without syncing:

```bash
scripts/update_rsl_rl_lib.sh --no-sync
```

---

### 2. Motion Imitation

Train a Unitree G1 to mimic reference motions. mjlab uses
[WandB](https://wandb.ai) to manage reference motion datasets:

1. **Create a registry collection** in your WandB workspace named `Motions`

2. **Set your WandB entity**:
   ```bash
   export WANDB_ENTITY=your-organization-name
   ```

3. **Process and upload motion files**:
   ```bash
   MUJOCO_GL=egl uv run src/mjlab/scripts/csv_to_npz.py \
     --input-file /path/to/motion.csv \
     --output-name motion_name \
     --input-fps 30 \
     --output-fps 50 \
     --render  # Optional: generates preview video
   ```

> [!NOTE]
> For detailed motion preprocessing instructions, see the
> [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).


#### Train and Play
```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  registry_name=your-org/motions/motion-name \
  env.scene.num_envs=4096

MJLAB_MOTION_HORIZON=1 uv run train Mjlab-MultiTracking-Flat-Unitree-G1 \
  agent.wandb_project=multitrack \
  agent.run_name=H_1_10s_bro \
  registry_name=wandb-registry-motions/lafan_dataset \
  agent.max_iterations=30000 \
  env.scene.num_envs=4096
```
---

### 3. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP **before** training.

```bash
uv run play Mjlab-Your-Task-Id agent=zero  # Sends zero actions.
uv run play Mjlab-Your-Task-Id agent=random  # Sends uniform random actions.
```

> [!NOTE]
> When running motion-tracking tasks, add
> `registry_name=your-org/motions/motion-name` to the command.

---

## Training with a YAML Config File

The train and play CLIs use:

- `tyro` for CLI parsing and `--help`
- `OmegaConf` for config-file loading and merge behavior

The merge order is:

1. task defaults
2. `config_path` or `config_source`
3. CLI overrides

CLI help is available from both top-level commands and task-specific commands:

```bash
uv run train --help
uv run train Mjlab-Velocity-Flat-Unitree-G1 --help
uv run play --help
uv run play Mjlab-Velocity-Flat-Unitree-G1 --help
```

If you prefer scheduling runs from a YAML file, use either:

- `config_path=...` for local YAML files
- `config_source=...` for local path, HTTP(S), or W&B artifact URI

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 config_path=conf/train.yaml
```

Example `conf/train.yaml`:

```yaml
env:
  scene:
    num_envs: 4096
agent:
  max_iterations: 30000
video: false
enable_nan_guard: true
```

`config_path=...` supports partial overrides:

- Fields present in YAML override defaults loaded from task registry.
- Missing fields keep task defaults.

CLI overrides can be passed either as dotlist assignments or standard flags:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  config_path=conf/train.yaml \
  env.scene.num_envs=8192
```

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --config-path conf/train.yaml \
  --env.scene.num-envs 8192
```

CLI overrides still win over YAML:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  config_path=conf/train.yaml \
  env.scene.num_envs=8192
```

Remote YAML via HTTP(S):

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  config_source=https://example.com/configs/train.yaml
```

YAML from W&B artifact:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  config_source=wandb://your-entity/your-project/train-configs:latest/config.yaml
```

---

## Documentation

Full documentation is available at **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**.

---

## Development

Run tests:

```bash
make test          # Run all tests
make test-fast     # Skip slow integration tests
```

Format code:

```bash
uvx pre-commit install
make format
```

Compile documentation locally:

```bash
uv pip install -r docs/requirements.txt
make docs
```

---

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

---

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.
