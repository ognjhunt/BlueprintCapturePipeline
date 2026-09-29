"""Dependency-free screened errors shared with CPU-only publisher admission."""


class OwnerTargetVersionError(ValueError):
    """Fixed public code; no path, policy, OS exception text or secret."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)
