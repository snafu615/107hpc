"""Domain-specific errors with user-facing messages."""


class HPC107Error(RuntimeError):
    """Base error for expected workflow failures."""


class ConfigurationError(HPC107Error):
    """Configuration could not be parsed or validated."""


class ProjectValidationError(HPC107Error):
    """Project structure or content does not satisfy the deterministic contract."""


class SlurmError(HPC107Error):
    """A Slurm command failed or produced unexpected output."""
