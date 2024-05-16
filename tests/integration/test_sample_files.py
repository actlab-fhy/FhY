"""Integration Tests - Compiling Source Code from File using Fhy Entry Point."""

import os
import re
from glob import glob

import pytest

from .utils import access_cli, get_diff

HERE = os.path.abspath(os.path.join(__file__, os.pardir))
OUTPUT = os.path.join(HERE, "Samples", "Output")
SAMPLES = os.path.join(HERE, "Samples", "*.fhy")

examples = glob(SAMPLES)


def grab_expected_output_file(filepath: str) -> str:
    """Grab Expected Output File from an Input Text Filepath."""
    basename: str = os.path.basename(filepath).split(".")[0]
    name = f"{basename}_output.fhy"
    path_out = os.path.join(OUTPUT, name)
    if not os.path.exists(path_out):
        raise FileNotFoundError(f"Expected Output File Does Not Exist: {basename}")

    return path_out


def iter_lines(text: str):
    r"""Iterate through lines of text, without newline character(s) present at line end.

    Note:
        We are grouping together multiple new line characters here

    Example:
        .. code-block:: python

            text = "test\n\r\n\n\n\nstring\n\n\n"
            assert list(iter_lines(text)) == ["test", "string", ""]
            assert "\n".join(iter_lines(text)) == "test\nstring\n

    """
    yield from re.split("[\r\n]+", text)


# NOTE: We might change how the FhY Entrypoint Outputs information
def cleanup_pretty_print_output(output: str) -> str:
    """Cleanup output of fhy --pretty option, to remove filename."""
    generator = iter_lines(output)
    for line in generator:
        if line.startswith("=") and line.endswith("="):
            break

    # Return the remaining output, removing newline character at the end
    return "\n".join(generator).strip()


@pytest.mark.parametrize("file", examples)
def test_single_file_examples_through_cli_pretty(file):
    """Tests FhY Entry Point using Pretty Print on a collection of Example Files."""
    output = access_cli("-m", file, "serialize", "-f", "pretty")
    result = cleanup_pretty_print_output(output)

    out_path = grab_expected_output_file(file)
    with open(out_path, "r") as st:
        expected = st.read()

    if result != expected:
        get_diff(result, expected)

    assert result == expected, "Unexpected Output from FhY"
