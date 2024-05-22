"""IO File Handlers."""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Tuple, Union


def standard_path(value: Union[str, Path]) -> Path:
    """Standardize and resolve a file path.

    Raises:
        TypeError: Provided type of value is not of type AnyPath (str | pathlib.Path)
        FileExistsError: Provided Path does not exist on file system.

    """
    if isinstance(value, str):
        path = Path(value)

    elif isinstance(value, Path):
        path = value

    else:
        msg = 'Expected "value" argument of type "str" | "pathlib.Path".'
        raise TypeError(f"{msg} Received: {value}")

    if not path.exists():
        raise FileExistsError(f"Filepath Does Not Exist: {value}")

    return path.resolve()


@contextmanager
def open_file(
    file_path: Path, mode: str
) -> Generator[Tuple[Optional[str], Optional[Exception]], None, None]:
    """Overly cautious context manager to open a file with better error reporting.

    Args:
        file_path (Path): _description_
        mode (str): Valid Read File Mode.

    Yields:
        Generator[Tuple[Optional[str], Optional[Exception]], None, None]: Provide
        text from file source and exception if relevant.

    """
    try:
        stream = open(file_path, mode)

    except Exception as e:
        yield None, e

    else:
        try:
            yield stream.read(), None

        finally:
            stream.close()


def read_file(file_path: Path) -> str:
    """Read a file.

    Args:
        file_path (Path): Valid File Path

    Raises:
        FileExistsError: Invalid Filepath or an error occurred during streaming.

    Returns:
        str: Contents of filepath

    """
    with open_file(file_path, "r") as (text, err):
        if err is not None:
            raise FileExistsError(f"Unable to Read provided file: {file_path}") from err

        if text is None:
            raise FileExistsError(f"No Text was read from file: {file_path}")

    return text
