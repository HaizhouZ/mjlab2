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

If you run `mjlab` inside a read-only container or Slurm overlay, source
[`scripts/slurm_overlay_env.sh`](./scripts/slurm_overlay_env.sh) before
launching training. It redirects writable state such as `HOME`, `TMPDIR`,
`UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `WARP_CACHE_PATH`, `TORCH_HOME`,
`WANDB_*`, `XDG_*`, and `MJLAB_WANDB_CACHE_DIR` into a writable runtime tree.
You can override the top-level writable roots directly with
`MJLAB_HOME_DIR`, `MJLAB_CACHE_DIR`, `MJLAB_TMP_DIR`, and `MJLAB_OUTPUT_DIR`
before sourcing the script.

```bash
source scripts/slurm_overlay_env.sh /scratch/$USER/mjlab "$SLURM_JOB_ID"
uv run train Mjlab-Velocity-Flat-Unitree-G1
```

This creates a per-run layout like:

```text
/scratch/$USER/mjlab/$SLURM_JOB_ID/
  home/
  cache/
  tmp/
  outputs/
```

`train.py` will automatically use `MJLAB_OUTPUT_DIR` for logs when it is set, so
checkpoints and run outputs land under the writable `outputs/` directory by
default.

If you want the same runtime layout but a different cache location, set
`MJLAB_CACHE_DIR` before sourcing the script.

You can also use a manual suffix outside Slurm:

```bash
source scripts/slurm_overlay_env.sh /scratch/$USER/mjlab exp_yam_td3
```

For batch jobs, [`scripts/slurm_exec.sh`](./scripts/slurm_exec.sh) is the more
convenient entrypoint. It will:

- source `scripts/slurm_overlay_env.sh`
- export `PYTHONPATH=$REPO_ROOT/src`
- optionally run `uv sync --frozen --group dev` when `MJLAB_UV_SYNC=1`
- optionally send an email when the job starts on a node when `MJLAB_SLURM_NOTIFY_TO` is set
- execute the command you pass after `--`

Example `sbatch` payload:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mjlab-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
cd /path/to/mjlab2

scripts/slurm_exec.sh /scratch/$USER/mjlab "$SLURM_JOB_ID" -- \
  uv run train Mjlab-Velocity-Flat-Unitree-G1 env.scene.num_envs=4096
```

Optional email notification on allocation:

```bash
export MJLAB_SLURM_NOTIFY_TO="you@example.com"
export MJLAB_SLURM_NOTIFY_FROM="mjlab@example.com"
export MJLAB_SMTP_HOST="smtp.example.com"
export MJLAB_SMTP_PORT=587
export MJLAB_SMTP_USERNAME="smtp-user"
export MJLAB_SMTP_PASSWORD="smtp-password"
export MJLAB_SMTP_STARTTLS=1
```

When these are set, `scripts/slurm_exec.sh` sends one email after the Slurm job
starts on a node and before the training command is executed. The message
includes the job ID, job name, assigned node, runtime root, and full command.

Dry-run preview without sending:

```bash
export MJLAB_SLURM_NOTIFY_TO="you@example.com"
export MJLAB_SLURM_NOTIFY_FROM="mjlab@example.com"
export MJLAB_SLURM_NOTIFY_DRY_RUN=1

scripts/slurm_exec.sh /scratch/$USER/mjlab "$SLURM_JOB_ID" -- \
  uv run train Mjlab-Velocity-Flat-Unitree-G1 env.scene.num_envs=4096
```

In dry-run mode, the notifier prints the email subject and body to stdout and
does not require `MJLAB_SMTP_HOST`.

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
