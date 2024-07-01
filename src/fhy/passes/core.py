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

"""core pass infrastructure and registry.

This basic setup is to define a consistent architecture for Passes to be performed on
the fDFG data structure. Including a registry to collect all passes of defined
optimization levels

"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)


# TODO: May want to define explicit Optimization flags
@dataclass
class PassInfo:
    """Compile pass information.

    Args:
        level (int): optimization level flag.
        function (type["PassCore"]): callable function to perform pass
        name (str): name of given pass
        required (list[str | type["PassCore"]]): Passes required to be performed before
            current pass is attempted.
        sized: (bool): If true, pass is compatible to optimize binary by size

    """

    level: int
    function: type["PassCore"]
    name: str
    required: list[str | type["PassCore"]] = field(default_factory=list)
    sized: bool = field(default=False)


# TODO: We may want to further refine Pass Types, such as passes performed on moddule
#       level, function level, etc...
class PassCore(ABC):
    """Pass Class."""

    # TODO: We may want to accept a graph here, or other kwargs for each phase
    #       pre, proceed, post methods.
    def __init__(self, level) -> None:
        self.level = level

    def __init_subclass__(
        cls,
        level: int,
        required: list[str | type["PassCore"]],
        name: str | None = None,
        size: bool = False,
    ) -> None:
        register_pass(level, name, required, size=size)(cls)

        return super().__init_subclass__()

    def pre(self, *args, **kwargs): ...

    @abstractmethod
    def proceed(self, *args, **kwargs):
        raise NotImplementedError("Must Implement Pass proceed method.")

    def post(self, graph, *args, **kwargs):
        # NOTE: This may need a (general) sanity check on the fDFG
        ...

    # TODO: Requires review on usage once we have the fDFG structure defined
    #       This is a partial placeholder at the moment.
    def run(self, graph, *args, **kwargs):
        self.pre(graph, *args, **kwargs)
        self.proceed(graph, *args, **kwargs)
        self.post(graph, *args, **kwargs)

    def is_performed(self, value: int) -> bool:
        return self.level >= value

    def __call__(self, graph, level: int, *args, **kwargs) -> None:
        if self.is_performed(level):
            return self.run(*args, **kwargs)


# P = TypeVar("P", bound=PassCore)
P = type[PassCore]


class _PassRegistry:
    """Registry of compiler passes."""

    registry: list[PassInfo]
    log: logging.Logger

    def __init__(self, log: logging.Logger = _log) -> None:
        self.log = log
        self.registry = []

    def append(self, item: PassInfo) -> None:
        self.registry.append(item)

    def remove(self, item: str | PassCore):
        """Communicate what is removed."""
        if isinstance(item, str):
            # Retrieve by name
            fi = filter(lambda x: x.name == item, self.registry)

        elif isinstance(item, PassCore):
            # Retrieve by Pass function
            fi = filter(lambda y: y.function == item, self.registry)

        else:
            raise TypeError("Nope Nope Nope.")

        for j in fi:
            self.registry.remove(j)
            self.log.info("Compiler Pass Removed: %s", j.name)


PassRegistry = _PassRegistry()


# TODO: I'm not certain it makes sense to instantiate each pass class?
#       but it does make sense if we allow other functions (not sublasses of PassCore)
#       such that all can be called exactly the same, unless we wrap a function as
#       a subclass PassCore, where the function is dynamically assigned to proceed
#       method...
def register_pass(
    level: int,
    name: str | None = None,
    required: list[str | P] | None = None,
    size: bool = False,
) -> Callable[[P], P]:
    """Register a compiler pass."""

    def inner(f: P) -> P:
        moniker = name or f.__qualname__
        wrap = PassInfo(
            level=level, function=f, name=moniker, required=required or [], sized=size
        )
        PassRegistry.append(wrap)

        return f

    return inner
