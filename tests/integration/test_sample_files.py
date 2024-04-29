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
    basename: str = os.path.basename(filepath).split(".")[0]
    name = f"{basename}_output.fhy"
    outpath = os.path.join(OUTPUT, name)
    if not os.path.exists(outpath):
        raise FileNotFoundError(f"Expected Output File Does Not Exist: {basename}")
    return outpath


def iter_lines(text: str):
    """Simple Utility to iterate through lines of text, without the newline character
    present at the end of the line.

    """
    yield from re.split("\r\n|\r|\n", text)


# NOTE: We might change how the FhY Entrypoint Outputs information
def cleanup_pretty_print_output(output: str) -> str:
    test: bool = False
    lines = []
    for line in iter_lines(output):
        if test:
            lines.append(line)
        elif line.startswith("="):
            test = True
    value = "\n".join(lines[:-1])
    if value.endswith("\n"):
        return value[:-1]
    return value


@pytest.mark.parametrize("file", examples)
def test_single_file_examples_through_cli_pretty(file):
    """Tests FhY Entry Point using Pretty Print"""
    result = cleanup_pretty_print_output(access_cli(file, "--pretty"))

    pathout = grab_expected_output_file(file)
    with open(pathout, "r") as st:
        expected = st.read()

    if result != expected:
        get_diff(result, expected)

        assert result == expected, "Unexpected Output from FhY"
