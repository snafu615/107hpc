# hpc107-cli

`hpc107-cli` is a deterministic, two-part command-line workflow for computing
tasks on the USTC 107 Slurm cluster. It preserves the platform conventions of
the known-working `submit107` project while adding strict validation,
structured run records, resumable monitoring, and explicit artifact plans.

There is no LLM in the execution path:

- `hpc107-local` runs on the local workstation. It inspects or scaffolds a
  project, renders a reviewed Slurm script, pushes source with Git, and prints
  safe Pan transfer commands.
- `hpc107` runs on the 107 login node. It validates the project and live Slurm
  environment, submits the recorded script, monitors the job, diagnoses common
  failures, and inventories artifacts.

Both commands are installed from the same small package so their config and
validation rules cannot drift.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

Install the package on both the local workstation and the 107 login node. The
cluster-side computing environment is separate from the CLI installation:
for `environment.manager: uv`, `hpc107 submit` creates the project `.venv`
with `uv sync` when needed.

## Computing-task contract

The canonical project shape is the structure documented and shipped by
`submit107`:

```text
my-task/
  src/
    __init__.py
    data.py
    model.py
    train.py          # canonical entry
  scripts/
    train.sbatch      # deterministic generated script
  logs/               # Slurm stdout/stderr; excluded from Git
  outputs/            # generated results; excluded from Git
  checkpoints/        # model state; excluded from Git
  pyproject.toml       # or requirements.txt / environment.yml
  hpc107.yaml
  .gitignore
```

`src/train.py` is preferred. Root-level `train.py` and `main.py` are accepted
as compatibility layouts. A dependency manifest is mandatory. The CLI rejects
missing entry points, path traversal, unknown native config fields, and unsafe
values instead of guessing through an invalid project.

Create a new task or inspect an existing one:

```bash
hpc107-local template my-task
hpc107-local inspect ./my-task
hpc107-local prepare ./my-task
```

`prepare` creates `hpc107.yaml` and `scripts/train.sbatch`. If a compatible
script already exists, it is preserved. Replacement requires
`--force-script`, and the previous script is backed up.

## End-to-end workflow

### 1. Prepare and upload source locally

```bash
cd my-task
hpc107-local prepare .
hpc107-local push . --remote <git-url> --commit
hpc107-local pan-plan . --remote <pan-remote>
hpc107-local handoff .
```

Git carries source, config, and the generated Slurm script. Pan carries large
`data/` or `datasets/` directories. Pan commands are printed for review and
are never executed automatically. They use `rclone copy`, never destructive
`sync`, `delete`, or `purge` operations.

### 2. Clone, validate, and run on 107

Follow the commands printed by `handoff`, then:

```bash
hpc107 check .
hpc107 submit . --yes --watch
```

`submit` records the exact script SHA-256 before calling Slurm. It checks the
project and environment, checks `Students` with `sinfo`, runs `bash -n`, uses
`sbatch --test-only` when the installed Slurm exposes it, and submits with
`sbatch --parsable`. The returned run ID is durable.

If the shell disconnects:

```bash
hpc107 status <run-id> --project .
hpc107 watch <run-id> --project .
hpc107 diagnose <run-id> --project .
```

State is stored atomically under `.hpc107/runs/<run-id>/state.json`. Terminal
state is resolved from `sacct` after the job leaves `squeue`.

### 3. Inventory and download results

On 107:

```bash
hpc107 artifacts <run-id> --project .
```

This writes an artifact inventory and prints reviewed Pan uploads for
`outputs/`, `checkpoints/`, and `logs/`. Run the desired commands manually.
Then, on the local workstation:

```bash
hpc107-local fetch-plan . --remote <pan-remote>
```

Run the printed `rclone copy` commands to download results. For small outputs,
`rsync` is also a reasonable manual transport.

## Configuration

Copy `hpc107.example.yaml` to the computing project as `hpc107.yaml`. The
effective precedence is:

```text
built-in defaults
  < .submit107.yaml
  < hpc107.yaml
  < SUBMIT107_* environment variables
  < HPC107_* environment variables
```

The one-way `.submit107.yaml` reader eases migration; new configuration is
written only as `hpc107.yaml`. Platform defaults remain:

```yaml
resources:
  partition: Students
  qos: qos_stu_default
  cpus: 4
  memory: 16G
  gpus: 0
  walltime: "1:00:00"
```

When no project config exists, local preparation uses the same conservative
binary inference as `submit107`: a recognized GPU framework requests one GPU,
4 CPUs, 16G, and two hours; other projects request no GPU, 2 CPUs, 4G, and one
hour. Configuration always wins over inference.

## Deterministic Slurm contract

The owned Jinja template preserves the proven meanings:

- `Students` partition and `qos_stu_default` QOS;
- `--cpus-per-task`, `--mem`, optional generic `--gres=gpu:N`, and `--time`;
- separate `logs/%x_%j.out` and `.err` files;
- `set -euo pipefail` and `cd "$SLURM_SUBMIT_DIR"`;
- `.venv/bin/python` for uv projects, or the declared conda/system command.

The output is semantic-compatible with `submit107`, not copied byte-for-byte.
See [docs/architecture.md](docs/architecture.md) and
[docs/live-acceptance.md](docs/live-acceptance.md).

## Future LLM adapter

`hpc107.common.normalizer.ProjectNormalizer` is a deliberately narrow future
extension point. An LLM may eventually propose a manifest for a nonconforming
project, but it cannot submit jobs or bypass validation. Any proposal must be
shown to the user and pass the same deterministic config, project, script, and
Slurm gates before execution. Conforming projects never need the LLM path.
