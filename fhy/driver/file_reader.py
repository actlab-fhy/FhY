from pathlib import Path
from contextlib import contextmanager


@contextmanager
def open_file(file_path: Path, mode: str):
    """Context manager to Open a File, with Better Error Reporting and Handling."""
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
    with open_file(file_path, "r") as (text, err):
        if err is not None:
            raise FileExistsError(f"Unable to Read provided file: {file_path}") from err
        if text is None:
            raise FileExistsError(f"No Text was read from file: {file_path}")
    return text
