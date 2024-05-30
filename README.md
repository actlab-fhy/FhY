
<h1>
    <img
        src="docs/source/_static/img/fhy_logo.png"
        alt="FhYLogo"
        height=30ex
        padding-right=50px
    >
    FhY Frontend
</h1>

*A Language for Modeling Physical Things*

*FhY* is a cross-domain language with mathematical foundations that moves beyond the
current paradigm of domain-specific languages to enable cross-domain multi-acceleration.

## Table of Contents
- [Table of Contents](#table-of-contents)
- [FhY Developers Blog](#fhy-developers-blog)
- [Installing FhY](#installing-fhy)
  - [FhY from pip](#fhy-from-pip)
  - [FhY from Source Code](#fhy-from-source-code)
- [Using FhY from the Command Line](#using-fhy-from-the-command-line)
- [FhY License](#fhy-license)
- [Dependencies](#dependencies)


## FhY Developers Blog

The *FhY developers* provide a transparent "***Glass House***" software design blog
regarding current implementation details to describe design decisions and discuss
various development issues and solutions on our readthedocs page,
found [here](https://fhy.readthedocs.io/en/latest/design_blog/index.html). We encourage
new contributors and users to review the blog for more information regarding *FhY* and
get involved.

## Installing FhY

If not already available on your system, [Install Java 11 JDK](https://www.azul.com/downloads/?version=java-11-lts&package=jdk#zulu), specific for your OS and architecture.

### FhY from pip
**Coming Soon**

### FhY from Source Code

1. Clone the repository from GitHub

    ```bash
    git clone https://github.com/actlab-fhy/FhY.git
    ```

2. Create and prepare a Python virtual environment

    ```bash
    cd FhY
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    pip install setuptools wheel
    pip install -r requirements_build.txt
    ```

3. Build generated Parser Files using the available build script.

    ```bash
    ./build_grammar.sh
    ```

4. Install FhY

    For standard installation:
    ```bash
    pip install .
    ```

    Developers may like to install FhY with extra dependencies:
    ```bash
    pip install ".[dev]"
    ```

## Using FhY from the Command Line
After installing *FhY* an entry point is provided to use *FhY* directly from the
command line. Start with a simple example from our integration tests (if you cloned the
*FhY* repository), and serialize to FhY text to stdout.

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
| [ANTLR](https://www.antlr.org/license.html) | BSD-3 | 2012 | Terence Parr and Sam Harwell |
| [NetworkX](https://networkx.org/documentation/stable/#license) | BSD-3 | 2004-2024 | NetworkX Developers<br>Aric Hagberg <hagberg@lanl.gov><br>Dan Schult <dschult@colgate.edu><br>Pieter Swart <swart@lanl.gov> |