# Architecture and invariants

## Design

The package deliberately separates location-specific orchestration while
sharing policy:

```text
local workstation                         107 login node
hpc107-local                              hpc107
  inspect/scaffold                          validate project + environment
  render reviewed script       Git/Pan      verify Slurm + script syntax
  commit/push source          -------->      prepare project environment
  print data copy plans                      submit and persist job ID
                                              poll squeue, resolve with sacct
  print result fetch plans    <--------      inventory + result copy plans

                  hpc107.common
        typed config, project contract, owned template,
        path/field validation, transfer safety, LLM boundary
```

The split is operational, not duplicated: both entry points live in one Python
distribution and import the same deterministic rules.

## Submission state machine

```text
VALIDATED -> PREFLIGHTED -> SUBMITTED -> PENDING/RUNNING -> terminal state
```

Every update replaces `state.json` atomically. A run records its project root,
script path, script digest, Slurm job ID, timestamps, accounting fields, and a
bounded deterministic diagnosis. The script digest is checked again immediately
before submission to close the validate/execute gap.

## Safety invariants

1. Training is submitted through Slurm; the CLI never runs the task directly
   on the login node.
2. The script is generated from an owned, strict template and validated against
   the effective configuration.
3. Partition, QOS, CPUs, memory, GPU count, time, stdout, and stderr are
   explicit and injection-sensitive fields are rejected.
4. All configured project paths are contained relative paths.
5. Existing scripts require explicit replacement and are backed up.
6. Live partition failure blocks submission unless the user explicitly uses
   the compatibility bypass `--skip-check`.
7. Pan integration prints `rclone copy` commands only. It does not execute
   transfers or generate destructive mirroring commands.
8. Data, artifacts, virtual environments, and model weights are excluded from
   Git by the scaffold.
9. The job ID is parsed from machine-readable `sbatch --parsable` output and
   persisted before monitoring begins.
10. An optional future normalizer may propose files but can neither execute nor
    relax deterministic gates.

## submit107 alignment

The compatibility baseline is `submit107` commit
`fe1542bc04066a86aad91d48515818df4f989ac6`. Platform-significant behavior is
preserved: two runtime locations, Git plus Pan handoff, Students/QOS defaults,
generic GPU GRES, `%x_%j` logs, the Slurm submit directory, strict Bash mode,
and project-local uv execution.

Intentional improvements are:

- canonical `src/train.py` with root-entry compatibility;
- strict typed native configuration plus a one-way legacy reader;
- rejection instead of a nonexistent-entry fallback;
- safe interpolation and relative-path containment;
- log/output directory creation;
- `bash -n` and optional `sbatch --test-only`;
- parsable job IDs, atomic JSON run state, resumable live monitoring, and
  `sacct` terminal resolution;
- shared Pan safety checks and explicit local result-fetch plans.

Notebook conversion, direct SSH/SCOW automation, automatic transfer execution,
and LLM normalization are intentionally outside the deterministic core.

## Change rule

A change to cluster-facing behavior must preserve the invariants above, add a
fixture/unit test, and pass the bounded live acceptance procedure. If the
reference project changes, record its new commit and compare source/tests before
changing compatibility claims.
