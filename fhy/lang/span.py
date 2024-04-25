"""Container Classes to define source positions of an object.

Classes:
    Slice: start and stop Positions
    Source: source file or namespace definition
    Span: full context of an object (Source + Slice)

"""


class Slice:
    """General definition of start and stop positions.

    Args:
        start (int): start position
        stop (int): stop position

    """

    start: int
    end: int

    def __init__(self, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Slice)
            and self.start == value.start
            and self.stop == value.stop
        )

    def __repr__(self) -> str:
        return f"{self.start:,d}:{self.stop:,d}"


# TODO: Jason: Create Source object that can track the source file
class Source(object):
    """Defines source file or namespace.

    Args:
        namespace (str): valid filepath or namespace

    """

    namespace: str

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Source) and self.namespace == value.namespace

    def __repr__(self) -> str:
        return self.namespace


class Span(object):
    """Context to describe Locations of an object.

    Args:
        start_line (int): line start position
        end_line (int): line end position
        start_column (int): column start position
        end_column (int): column end position
        source (Source): source object defining file or namespace

    Attributes:
        source (Source): file or namespace source
        line (_Span): start and stop line positions
        column(_Span):start and stop column positions

    """

    source: Source
    line: Slice
    column: Slice

    def __init__(
        self,
        start_line: int,
        end_line: int,
        start_column: int,
        end_column: int,
        source: Source = Source("_null"),
    ) -> None:
        self.source = source
        self.line = Slice(start_line, end_line)
        self.column = Slice(start_column, end_column)

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Span)
            and self.source == value.source
            and self.line == value.line
            and self.column == value.column
        )

    def __repr__(self) -> str:
        text = ""
        if self.source is not None:
            text += f"{self.source} "
        text += f"Lines {self.line} Columns: {self.column}"
        return text
