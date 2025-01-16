# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "fhy"
copyright = "2024, Christopher Priebe, Jason C Del Rio, Hadi S Esmaeilzadeh"
author = "Christopher Priebe, Jason Del Rio"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys

# https://sphinx-rtd-tutorial.readthedocs.io/en/latest/sphinx-config.html#autodoc-configuration
sys.path.insert(0, os.path.abspath(f"../../src"))  # Required for autosummary + autodoc
sys.path.append(os.path.abspath("./_ext"))  # Required for custom extensions

templates_path = ["_templates"]

extensions = [
    "sphinx.ext.duration",  # Benchmarks Sphinx Build Times
    "sphinx.ext.autodoc",  # Extracts Docstrings from Code
    "sphinx.ext.autosummary",  # Collects Code to Summarize Automatically with Autodoc
    "sphinx.ext.coverage",  # Measures Package Documentation Coverage
    "sphinx.ext.napoleon",  # Supports Numpy and Google Style Docstring Formatting
    # "sphinx.ext.linkcode",  # Add Links to each documented element to original code
    "fhy_pygments_lexer",  # Custom Pygments Lexer for FhY
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_google_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# See https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/index.html
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/theme.css", "css/custom.css"]
html_js_files = []
html_context = {
    "default_mode": "light"
}
html_theme_options = {
    "logo": {
        "text": "FhY",
        "alt_text": "FhY documentation - Home",
        "image_light": "_static/img/fhy_logo.png",
        "image_dark": "_static/img/fhy_logo.png",
        "link": "index",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://www.github.com/actlab-fhy/fhy",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
    ],
    "navbar_end": ["navbar-icon-links"]
    # "navbar_end": ["navbar-icon-links", "theme-switcher"],  # TODO: Add back dark/light mode theme switcher
}
html_additional_pages = {
    "index": "index.html",
}
