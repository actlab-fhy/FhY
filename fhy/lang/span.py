# TODO Jason: Add docstring
from typing import Optional


class _Span:
    start: int
    end: int

    def __init__(self, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop

    def __repr__(self) -> str:
        return f"{self.start:,d}:{self.stop:,d}"


# TODO: Jason: Create Source object that can track the source file
class Source(object):
    namespace: str

    def __init__(self, namespace) -> None:
        self.namespace = namespace

    def __repr__(self) -> str:
        return self.namespace


class Span(object):
    # TODO Jason: Add docstring
    source: Source
    line: _Span
    column: _Span

    def __init__(self,
                 line: _Span,
                 column: _Span,
                 source: Optional[Source] = None,
                 ) -> None:
        self.source = source
        self.line = line
        self.column = column

    def __repr__(self) -> str:
        text = ""
        if self.source is not None:
            text += f"{self.source} "
        text += f"Lines {self.line} Columns: {self.column}"
        return text

    # TODO Jason: Implement the functionality of this class
