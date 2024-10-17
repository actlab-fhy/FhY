# Contributing to FhY

Pull requests are always welcome, and the FhY community appreciates any help you give.

## Working with FhY - For Developers

1. Download the development branch of FhY source code.

```bash
git clone https://github.com/actlab-fhy/FhY.git -b dev
cd FhY
```

2. Create a new virtual environment (instructions to create and activate a virutal
   environment may differ here on different OS and shells)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install setuptools wheel
pip install -r requirements_build.txt
```

2. Build files using the available build script.
```bash
./build_grammar.sh
```

3. Install FhY in your virtual environment
```bash
pip install -e ".[dev]"
```

4. Initialize pre-commit
```bash
pre-commit install
pre-commit run --all-files
```

5. Run Unit tests against current development using tox. You may need to install several
   python versions to fully test FhY.
```bash
tox -p
```

And now you are ready to start hacking with FhY.

## Creating a new Pull Request
When submitting a pull request, we ask you to check the following:

1. First create an issue on FhY to reference before starting a pull request and discuss
   possible implementation details, or nuances.

2. Unit tests, documentation, and code style are in order.
   1. It's also OK to submit work in progress if you're unsure of what this exactly
      means, in which case you'll likely be asked to make some further changes.

3. The contributed code will be licensed under FhY's
   [license](https://github.com/actlab-fhy/FhY/blob/main/LICENSE). If you did not write
   the code yourself, you ensure the existing license is compatible and include the
   license information in the contributed files, or obtain permission from the original
   author to relicense the contributed code.


## Coding style

Most of our code is automatically linted and formatted using the developer tools ruff,
pylint, and mypy (for static typing). For reference, we also take inspiration from
[Google's style guide](https://google.github.io/styleguide/pyguide.html).

### Doctstrings

We are slightly picky about docstrings. We use google style docstrings in active voice.
The first line should succintly summarize the function or class, ending in a period.
Further explanation may be provided on other lines after a break. `Arguments`, `Returns`
, and `Raises` should be documented in public functions. Other sections are optional,
and should be provided as seen fit, for example a `Usage` or `Notes` section may be
helpful.
