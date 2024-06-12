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

"""Program Root Node."""

# from fhy.lang.ast.core import Module as ASTModule
from .table import SymbolTable
from collections import deque
from fhy.fdfg.core import FDFG
from fhy.fdfg.node.fractalized import FractalizedNode, FunctionNode
from fhy.fdfg.converter.from_fhy_ast import _convert_fhy_ast_function_to_fdfg
from fhy.lang.ast.pprint import pformat_ast
from fhy.lang.ast.passes.expression_decomposer import decompose_expressions


class Program(object):
    """Program object."""

    _components: dict  # Dict[Identifier, Union[ASTModule]]
    _symbol_table: SymbolTable

    def __init__(self) -> None:
        self._components = {}
        self._symbol_table = SymbolTable()

    def convert_to_fdfg(self) -> None:
        """Convert the program to an f-DFG program."""
        print(f"\nBefore conversion:\n")
        for component in self._components.values():
            print(pformat_ast(component, is_identifier_id_printed=True))

        self._components = {
            k: decompose_expressions(v, self._symbol_table)
            for k, v in self._components.items()
        }

        print(f"\nAfter conversion:\n")
        for component in self._components.values():
            print(pformat_ast(component, is_identifier_id_printed=True))

        for name, component in self._components.items():
            # Check that the components are AST modules
            pass

        function_fdfgs = {}

        return

        # Convert each function to an f-DFG
        for name, component in self._components.items():
            print(pformat_ast(component, is_identifier_id_printed=True))
            for function in component.statements:
                function_fdfg = _convert_fhy_ast_function_to_fdfg(
                    function, self._symbol_table
                )
                function_fdfgs[function.name] = function_fdfg

        # Link the functions together
        fdfg_queue: deque[FDFG] = deque(function_fdfgs.values())
        while fdfg_queue:
            fdfg = fdfg_queue.popleft()
            for node_name, node_attrs in fdfg.graph.nodes(data=True):
                if "data" not in node_attrs:
                    continue
                node = node_attrs["data"]
                if isinstance(node, FunctionNode) and not node.is_fdfg_set():
                    node.fdfg = function_fdfgs[node.symbol_name]
                if isinstance(node, FractalizedNode):
                    fdfg_queue.append(node.fdfg)

        self._components = function_fdfgs

        from fhy.fdfg.visualize import plot_fdfg

        _, main_component = next(
            filter(lambda x: x[0].name_hint == "main", self._components.items())
        )
        plot_fdfg(main_component)
