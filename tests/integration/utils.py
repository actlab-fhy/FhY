"""Common Integration Test Utilities."""

import difflib
import subprocess


def get_diff(a, b):
    """Helper Utility to print a diff between two strings"""
    for i, s in enumerate(difflib.ndiff(a, b)):
        if s[0] == " ":
            continue
        elif s[0] == "-":
            print('Delete "{}" from position {}'.format(s[-1], i))
        elif s[0] == "+":
            print('Add "{}" to position {}'.format(s[-1], i))


def access_cli(filepath: str, *args) -> str:
    """Access FhY Entry Point using subprocess and return the decoded stdout"""
    result = subprocess.run(["fhy", filepath, *args], stdout=subprocess.PIPE)
    output = result.stdout.decode()
    return output
