# Copyright (c) 2024 FhY Developers
# Christopher Priebe <cpriebe@ucsd.edu>
# Jason C Del Rio <j3delrio@ucsd.edu>
# Hadi S Esmaeilzadeh <hadi@ucsd.edu>
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be
# used to endorse or promote products derived from this software without specific prior
# written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""IO file handlers."""

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
        raise FileExistsError(f"Filepath does not exist: {value}")

    return path.resolve()


@contextmanager
def open_file(
    file_path: Path, mode: str
) -> Generator[Tuple[Optional[str], Optional[Exception]], None, None]:
    """Overly cautious context manager to open a file with better error reporting.

    Args:
        file_path (Path): valid file path
        mode (str): Valid IO mode literal to read a file.

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
