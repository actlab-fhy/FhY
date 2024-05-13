"""Option Definitions for Compilation of FhY Source Code."""

from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class CompilationOptions(object):
    """Supported FhY Compilation Options."""

    verbose: bool = field(default=False)
