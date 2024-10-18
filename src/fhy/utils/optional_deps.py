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

"""Optional dependency utilities.

Based on `pandas.compat._optional` module.
"""

import enum
import importlib
import sys
from types import ModuleType

from packaging.version import Version

VERSIONS = {"torch": "2.3.0"}


INSTALL_MAPPING = {}


def get_version(module: ModuleType) -> str:
    """Get the version of a module.

    Args:
        module: The module to get the version of.

    Returns:
        The version of the module.

    Raises:
        ImportError: If the version cannot be determined.

    """
    version = getattr(module, "__version__", None)
    if version is None:
        version = getattr(module, "__VERSION__", None)

    if version is None:
        raise ImportError(f"Cannot determine version for {module.__name__}")
    return version


class ErrorLevel(enum.Enum):
    """Error level handling options for optional dependencies."""

    RAISE = "raise"
    # WARN = "warn"
    IGNORE = "ignore"


def import_optional_dependency(
    name: str,
    extra: str = "",
    error_level: ErrorLevel = ErrorLevel.RAISE,
    min_version: str | None = None,
):
    """Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice
    message will be raised. If a dependency is present, but too old,
    we raise.

    Args:
        name: Module name.
        extra: Additional text to include in the ImportError message.
        error_level: What to do when a dependency is not found or its version is
            too old (e.g., raise, warn, or ignore). If it is `RAISE`, an
            ImportError will be raised. If it is `WARN`, there will be a warning
            that the version is too old and `None` will be returned. If it is
            `IGNORE`, if the module is not installed, `None` will be returned.
            Otherwise, the module will be returned even if the version is too old.
        min_version: Specify a minimum version that is different from the global
            minimum version required. Defaults to `None`.

    Returns:
        The imported module, when found and the version is correct. `None` is
        returned when the package is not found and `ERRORS` is `IGNORE`, or when
        the package is found but the version is too old and `ERRORS` is `WARN`.

    """
    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    msg = (
        f'Missing optional dependency "{install_name}". {extra}\n'
        f"Use pip or conda to install {install_name}."
    )
    try:
        module = importlib.import_module(name)
    except ImportError:
        if error_level == ErrorLevel.RAISE:
            raise ImportError(msg)
        else:
            return None

    parent = name.split(".")[0]
    if parent != name:
        install_name = parent
        module_to_get = sys.modules[install_name]
    else:
        module_to_get = module

    minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
    if minimum_version:
        version = get_version(module_to_get)
        if version and Version(version) < Version(minimum_version):
            msg = (
                f'FhY requires version "{minimum_version}" or newer of "{parent}" '
                f'(version "{version}" currently installed).'
            )
            # TODO: Implement WARN level
            # if error_level == ErrorLevel.WARN:
            #     pass
            #     warnings.warn(
            #         msg,
            #         UserWarning,
            #         stacklevel=find_stack_level(inspect.currentframe()),
            #     )
            #     return None
            if error_level == ErrorLevel.RAISE:
                raise ImportError(msg)

    return module
