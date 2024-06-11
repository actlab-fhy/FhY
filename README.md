<div align=center>
    <img
        src="docs/source/_static/img/fhy_logo.png"
        alt="FhYLogo"
        height=100ex
    >
</div>

<h1>
    FhY
</h1>

*A Language for Modeling Physical Things*

*FhY* is a cross-domain language with mathematical foundations that moves beyond the
current paradigm of domain-specific languages to enable cross-domain multi-acceleration.

<!-- omit in toc -->
## Table of Contents
- [*FhY* Documentation](#fhy-documentation)
  - [*FhY* Design Blog](#fhy-design-blog)
- [Installing *FhY*](#installing-fhy)
  - [Install *FhY* from PyPi](#install-fhy-from-pypi)
  - [Build *FhY* from Source Code](#build-fhy-from-source-code)
- [Using *FhY* from the Command Line](#using-fhy-from-the-command-line)
- [FhY License](#fhy-license)
- [Dependencies](#dependencies)


## *FhY* Documentation

### *FhY* Design Blog
The *FhY developers* provide a transparent "***Glass-House***" design blog to openly
describe current implementation details and software design decisions. We also discuss
various issues encountered during development and their solutions. Visit our
[*Read the Docs* page](https://fhy.readthedocs.io/en/latest/design_blog/index.html) to
gain familiarity with or to become involved with *FhY* language and development.

## Installing *FhY*

If not already available on your system, [Install Java 11 JDK](https://www.azul.com/downloads/?version=java-11-lts&package=jdk#zulu) specific for your host OS and architecture.

### Install *FhY* from PyPi
**Coming Soon**

### Build *FhY* from Source Code

1. Clone the repository from GitHub.

    ```bash
    git clone https://github.com/actlab-fhy/FhY.git
    ```

2. Create and prepare a Python virtual environment.

    ```bash
    cd FhY
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    pip install setuptools wheel
    pip install -r requirements_build.txt
    ```

3. Build generated parser files using the available build script.

    ```bash
    ./build_grammar.sh
    ```

4. Install FhY.

    ```bash
    # Standard Installation
    pip install .

    # For Developers
    pip install ".[dev]"
    ```


## Using *FhY* from the Command Line
After installing *FhY*, an entry point is provided to use *FhY* directly from the
command line. Start with a simple example from our integration tests (if you cloned the
*FhY* repository), and serialize to *FhY* text to stdout.

```bash
fhy --module tests/integration/data/input/matmul.fhy serialize --format pretty
```

Use the help flag option to get more up to date information on usage of the CLI tool.
```bash
fhy --help
```

## FhY License

FhY is distributed under the [BSD-3](LICENSE) license.


## Dependencies

| Software | License | Year(s)   | Copyright Holder(s)                                 |
|:--------:|:-------:|:---------:|:----------------------------------------------------|
| [ANTLR](https://github.com/antlr/antlr4) | [BSD-3](https://www.antlr.org/license.html) | 2012 | Terence Parr and Sam Harwell |
| [NetworkX](https://github.com/networkx/networkx) | [BSD-3](https://networkx.org/documentation/stable/#license) | 2004-2024 | NetworkX Developers<br>Aric Hagberg <hagberg@lanl.gov><br>Dan Schult <dschult@colgate.edu><br>Pieter Swart <swart@lanl.gov> |
| [Typer](https://typer.tiangolo.com/) | [MIT](https://github.com/tiangolo/typer/blob/master/LICENSE) | 2019 | Sebastián Ramírez |


**DISCLAIMER**: the table above does **NOT** represent endorsement of *FhY* software, *FhY* copyright holders, or *FhY* contributors by any of the listed copyright holders, contributors, and respective softwares or organizations.
