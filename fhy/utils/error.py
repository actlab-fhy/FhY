"""Module Containing all Custom FhY Exceptions.

All custom FhY Exceptions can be viewed at `Fhy.utils.errors.FHY_ERRORS`

"""

from typing import Dict, Type

FHY_ERRORS: Dict[Type[Exception], str] = {}


def register_fhy_error(error: Type[Exception]) -> Type[Exception]:
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
    """User Induced Error."""

    ...


@register_fhy_error
class FhYASTBuildError(RuntimeError):
    """Failed to Build FhY AST nodes from Source."""


@register_fhy_error
class FhYSyntaxError(SyntaxError):
    """Syntax Error in FhY Source Code."""


@register_fhy_error
class FhYSemanticsError(Exception):
    """Error in FhY Program Semantics."""


@register_fhy_error
class FhYImportError(ImportError):
    """Problematic Import Statement Detected from FhY Source Code."""

    ...


@register_fhy_error
class UnregisteredASTNode(KeyError):
    """ASTNode information has not been registered with FhY."""


@register_fhy_error
class FieldAttributeError(Exception):
    """Attempted to assign a value to an unsupported attribute of the object."""
