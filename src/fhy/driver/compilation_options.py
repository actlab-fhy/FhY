"""Option definitions for compilation of FhY source code."""

from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class CompilationOptions(object):
    """Supported FhY compilation options.

    Args:
        verbose (bool): provide more debugging logs if true.

    """

    verbose: bool = field(default=False)
