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

"""Custom FhY exceptions and FhY exception registry.

All custom FhY exceptions can be viewed at 'Fhy.utils.errors.FHY_ERRORS'

"""

FHY_ERRORS: dict[type[Exception], str] = {}


def register_fhy_error(error: type[Exception]) -> type[Exception]:
    """Register custom FhY exceptions."""
    FHY_ERRORS[error] = error.__doc__ or error.__name__

    return error


def _initialize_builtins() -> None:
    """Initialize registration of builtin python exceptions."""
    if len(FHY_ERRORS) == 0:
        for _exc in [
            AssertionError,
            AttributeError,
            FileExistsError,
            FileNotFoundError,
            IndexError,
            KeyError,
            RuntimeError,
            SyntaxError,
            TypeError,
            ValueError,
        ]:
            register_fhy_error(_exc)


_initialize_builtins()


@register_fhy_error
class UsageError(Exception):
    """User induced error."""


@register_fhy_error
class FhYASTBuildError(RuntimeError):
    """Failed to build FhY AST nodes from source."""


@register_fhy_error
class FhYSyntaxError(SyntaxError):
    """Syntax error in FhY source code."""


@register_fhy_error
class FhYSemanticsError(Exception):
    """Error in FhY program semantics."""


@register_fhy_error
class FhYImportError(ImportError):
    """Problematic import statement detected from FhY source code."""


@register_fhy_error
class UnregisteredASTNode(KeyError):
    """ASTNode information has not been registered with FhY."""


@register_fhy_error
class FieldAttributeError(Exception):
    """Attempted to assign a value to an unsupported attribute of the object."""
