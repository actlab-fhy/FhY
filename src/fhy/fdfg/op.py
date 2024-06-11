from fhy.ir.identifier import Identifier


class Op(object):
    _name: Identifier
    # _input_signature: list[Type]
    # _output_signature: list[Type]
    # _template_types: list[Type]

    def __init__(
        self,
        name: Identifier,
        # input_signature: list[Type],
        # output_signature: list[Type],
        # template_types: Optional[list[Type]] = None,
    ) -> None:
        self._name = name
        # self._input_signature = input_signature
        # self._output_signature = output_signature
        # self._template_types = template_types or []

    @property
    def name(self) -> Identifier:
        return self._name

    # @property
    # def input_signature(self) -> list[Type]:
    #     return self._input_signature

    # @property
    # def output_signature(self) -> list[Type]:
    #     return self._output_signature

    # @property
    # def template_types(self) -> list[Type]:
    #     return self._template_types
