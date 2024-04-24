"""
...

"""

import difflib
import os
import subprocess
from glob import glob
from io import StringIO

import pytest

from fhy.lang.printer import pprint_ast

from ..utils import construct_ast, lexer, parser

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


def get_diff(a, b):
    """Helper Utility to print a diff between two strings"""
    for i, s in enumerate(difflib.ndiff(a, b)):
        if s[0] == " ":
            continue
        elif s[0] == "-":
            print('Delete "{}" from position {}'.format(s[-1], i))
        elif s[0] == "+":
            print('Add "{}" to position {}'.format(s[-1], i))


@pytest.mark.parametrize("file", examples)
def test_single_file_examples(parser, file):
    """Tests the Construction Internally First"""
    with open(file, "r") as stream:
        text = stream.read()

    ast = construct_ast(parser, text)
    result: str = pprint_ast(ast, "  ", False)

    pathout = grab_expected_output_file(file)
    with open(pathout, "r") as st:
        expected = st.read()

    print(result)
    assert result == expected, f"Did not Create Expected Output: {file}"


def access_cli(filepath: str, *args) -> str:
    """Access FhY Entry Point using subprocess and return the decoded stdout"""
    result = subprocess.run(["fhy", filepath, *args], stdout=subprocess.PIPE)
    output = result.stdout.decode()
    return output


# NOTE: We might change how the FhY Entrypoint Outputs information
def cleanup_pretty_print_output(output: str) -> str:
    stream = StringIO(output)
    stream.seek(0)
    test: bool = False
    lines = []
    for line in stream.readlines():
        if test:
            lines.append(line)
        elif line.startswith("="):
            test = True
    value = "".join(lines[:-1])
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
