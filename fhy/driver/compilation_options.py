from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class CompilationOptions(object):
    verbose: bool = field(default=False)
