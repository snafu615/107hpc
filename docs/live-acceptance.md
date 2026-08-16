# Bounded 107 live acceptance

Local tests prove deterministic behavior; only a real cluster submission can
prove the current platform accepts it. Use a disposable, CPU-only smoke task
that writes a small text file and contains no secrets or valuable data.

## Evidence to save

Record the date, package commit, sanitized script, script SHA-256, run ID, Slurm
job ID, and terminal `state.json`. Do not record credentials, tokens, cookies,
private keys, or unredacted personal identifiers.

## Procedure

1. Create the project with `hpc107-local template hpc107-smoke`.
2. Replace its placeholder `src/train.py` with a bounded CPU-only program that
   creates `outputs/smoke.txt`, and remove GPU dependencies from its manifest.
3. Run `hpc107-local prepare hpc107-smoke --force-script` and confirm the
   generated config has `gpus: 0`.
4. Push the disposable project and clone it on the 107 login node.
5. Install this CLI on the login node and run `hpc107 check .`. Save sanitized
   `sinfo` output and whether `sbatch --test-only` was available.
6. Run `hpc107 submit . --yes --watch`. Do not bypass the live precheck for the
   acceptance run.
7. Confirm the state becomes `COMPLETED`, the exit code is `0:0`, and the
   expected `%x_%j` stdout/stderr files exist.
8. Run `hpc107 artifacts <run-id> --project .` and confirm the inventory lists
   the small output.
9. If Pan is configured, run the printed `rclone copy` upload, then locally run
   a command printed by `hpc107-local fetch-plan` and compare the file hash.
10. Remove only the disposable task and remote test artifacts after separately
    verifying their exact paths.

Any failure should be captured as sanitized command output and treated as a
compatibility defect or dated platform change, not bypassed silently.
