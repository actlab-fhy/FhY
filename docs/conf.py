# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FhY"
copyright = "2024, Christopher Priebe, Jason Del Rio"
author = "Christopher Priebe, Jason Del Rio"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys

# https://sphinx-rtd-tutorial.readthedocs.io/en/latest/sphinx-config.html#autodoc-configuration
sys.path.insert(0, os.path.abspath(f"../{project}/"))  # Required for autosummary + autodoc

extensions = [
    "sphinx.ext.duration",  # Benchmarks Sphinx Build Times
    "sphinx.ext.autodoc",  # Extracts Docstrings from Code
    'sphinx.ext.autosummary',  # Collects Code to Summarize Automatically with Autodoc
    "sphinx.ext.coverage",  # Measures Package Documentation Coverage
    "sphinx.ext.napoleon",  # Supports Numpy and Google Style Docstring Formatting
    # "sphinx.ext.linkcode",  # Add Links to each documented element to original code
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
