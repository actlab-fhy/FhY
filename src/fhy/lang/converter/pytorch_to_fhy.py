import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export
from torch.export.exported_program import ExportedProgram
from torch.fx.node import Node
from typing import Any, Dict, Optional, Union, Tuple
import torch.fx

#--------------------------------------------------------------------------
# PyTorchOpExtractor
#--------------------------------------------------------------------------

class PyTorchOpExtractor:
    def __init__(self, exported_program:ExportedProgram):
        self.exported_program = exported_program

    # Gets the node info for a paricular node such as
    # name of the node, op_type, inputs, outputs...
    def get_op_info(self, node:Node) ->Dict[str, Any]:
        op_info = {
            'name':self.get_op_name(node),
            'op_type': node.op,
            'inputs': [],
            'outputs': [],
            'attributess': {},
            'metadata': node.meta

        }

        print("-----------")
        print(op_info)
        #print(node.meta)
        print(node.meta.get('tensor_meta', None))
        print("-----------")

        # This extracts the inputs form the nodes
        for arg in node.args:
            input_info = self.extract_argument_info(arg)
            print("This is inputs: ", input_info)
            op_info['inputs'].append(input_info)

        # Extarcts the outputs
        output_info = self.extract_tensor_info(node)
        print("This is output_info: ", output_info)
        if output_info:
            op_info['outputs'].append(output_info)


        # Extract the attributes
        for key, value in node.kwargs.items():
            print(key, value , "This is the kye and theeee value")
            op_info['attributes'][key] = value

        return op_info


    # Extracts the name form the nodes and returns it
    def get_op_name(self, node:Node) ->str:
        if isinstance(node.target, str):
            return node.target
        elif hasattr(node.target, '__name__'):
            return node.target.__name__
        else:
            return str(node.target)

    # Extracts the necessary info for the input give arg and return it
    def extract_argument_info(self, arg:Any) -> Any:
        if isinstance(arg, Node):

            print("---------------------")
            print("This is arg : ", arg)
            print("This is Node : ", Node)
            print("---------------------")

            return self.extract_tensor_info(arg)
        elif isinstance(arg, (int, float, bool, str)):
            print("------------------------------")
            print ("This is the arg elif: ", arg)
            print (type(arg).__name__)
            print("------------------------------")
    # Extract tensor information from the node metada such as
    # shape, type, device ....
    def extract_tensor_info(self, node:Node) ->Optional[Dict[str, Any]]:
        tensor_meta = node.meta.get('tensor_meta', None)
        if tensor_meta:
            shape = tensor_meta.shape
            dtype = tensor_meta.dtype
            requires_grad = tensor_meta.requires_grad
            #device = getattr(tensor_meta, 'device', 'unknown')
            #stride = tensor_meta.stride
            # In the future if we support quantized model metadata
            # has is_quatized and qparams that we can leverage

            return{
                'name' : node.name,
                'dtype' : str(dtype),
                'shape' : list(shape),
                'requires_grad' : requires_grad,
                #'device' : str(device)
            }
        elif 'val' in node.meta and isinstance(node.meta['val'], torch.Tensor):
            val = node.meta['val']
            shape = val.shape
            dtype = val.dtype
            requires_grad = val.requires_grad
            #device = val.device

            return{
                'name' : node.name,
                'dtype': str(dtype),
                'shape': list(shape),
                'requires_grad' : requires_grad,
                #'device': str(device)
            }
        else:
            # if no metadata is avialble return unkown
            return{
                'return': node.name,
                'dtype': 'unknown',
                'shape': [],
                'requires_grad': False,
                #'device': unknown
            }
#--------------------------------------------------------------------------
# Converter
#--------------------------------------------------------------------------
class PyTorchToFhyConverter:
    def __init__(self):
        self.op_extractor = None
        self.op_definitions: Dict[str, Dict[str, Any]] = {}
        self.input_tensor_name = None
        self.graph = None

    def convert(self, pytorch_module:torch.nn.Module, inputs:Tuple[Any, ...])->str:
        #Export the model to the exported program graph(basicly torch.fx)
        exported_program = export(pytorch_module, inputs)
        #Do the decomposition to reduce # of aten function
        decomp_exported_program = exported_program.run_decompositions()


        #Extrat the operations
        self.op_extractor = PyTorchOpExtractor(decomp_exported_program)
        self.graph = decomp_exported_program.graph_module.graph
        self.identify_input_tensor()
        fhy_code = self.generate_fhy_code(decomp_exported_program)
        return fhy_code

    def identify_input_tensor(self):
        for node in self.graph.nodes:
            if node.op == 'placeholder':
                self.input_tensor_name = node.name
                break


    def generate_fhy_code(self, exported_program:ExportedProgram) ->str:
        fhy_code = []

        #loop through the node in exported graph constructing the graph
        for node in self.graph.nodes:
            if node.op in['call_function', 'call_method', 'call_module', 'get_attr']:
                print("---------------------------")
                print("Node Target = ", node.target)
                print("---------------------------")
                # generate the op definition
                self.generate_op_definition(node)

        for op_def in self.op_definitions.values():
            fhy_code.append(self.format_op_definition(op_def))

        # line for Readibility(need to think of something else maybe?)
        fhy_code.append("")

        fhy_code.append(self.generate_main_function(exported_program))

        return '\n'.join(fhy_code)


    #get the node from the extractor class
    def generate_op_definition(self, node:Node) -> None:
        op_info = self.op_extractor.get_op_info(node)
        op_name = op_info['name']

        print("Operation Name : ", op_name)
        if op_name not in self.op_definitions:
            self.op_definitions[op_name] = op_info

    # format the code definition for outputing
    def format_op_definition(self, op_info: Dict[str, Any]) -> str:
        inputs = []

        # get the inputs and append it to the list
        for i, inp in enumerate(op_info['inputs']):
            inp_type = self.get_fhy_type(inp)
            inp_name = inp.get('name', f'in_{i}') if isinstance(inp, dict) else f'in_{i}'
            inputs.append(f"input {inp_type} {inp_name}")

        inputs_str = ', '.join(inputs)

        outputs =[]
        # get the outputs and append it to the list
        for i, out in enumerate(op_info['outputs']):
            out_type = self.get_fhy_type(out)
            out_name = out.get('name', f'out_{i}') if isinstance(inp, dict) else f'out_{i}'
            outputs.append(f"output {out_type} {out_name}")

        outputs_str = ', '.join(outputs)

        # splitting the name since it usually has .default or Tensor
        # don't need that.
        # putting the inputs and the outputs together
        # example :  (input float32[] add) -> output float32[] sum
        op_name = op_info['name']
        base_name = op_name.split('.')[0]
        fhy_code = f"op {base_name}({inputs_str}) -> {outputs_str} {{\n"
        fhy_code += f" // Implementation for {op_name}\n"
        fhy_code += "}"

        print("--------------------")
        print(fhy_code)
        print("--------------------")

        return fhy_code


     # generate the main function or forward
    def generate_main_function(self, exported_program: ExportedProgram) -> str:
        # extract the input arguments from the graph
        input_args = []
        parameter_arg = []
        temp_vars = set()
        output_args = []

        for node in self.graph.nodes:
            if node.op in ['placeholder']:
                arg_info = self.op_extractor.extract_tensor_info(node)
                arg_type = self.get_fhy_type(arg_info)
                input_args.append(f"input {arg_type} {node.name}")
            elif node.op in ['get_attr']:
                arg_info = self.op_extractor.extract_tensor_info(node)
                arg_name = f"p_{node.name}"
                arg_type = self.get_fhy_type(arg_info)
                parameter_arg.append(f"param {arg_type} {arg_name}")
            elif node.op == 'output':
                outputs = node.args[0]
                print("-----------------")
                print(outputs)
                print("-----------------")
                if isinstance(outputs, tuple):
                    for out_node in outputs:
                        out_info = self.op_extractor.extract_tensor_info(out_node)
                        out_name = out_node.name
                        out_type = self.get_fhy_type(out_info)
                        output_args.append(f"output {out_type} {out_name}")
                else:
                    out_info = self.op_extractor.extract_tensor_info(out_node)
                    out_name = out_node.name
                    out_type = self.get_fhy_type(out_info)
                    output_args.append(f"output {out_type} {out_name}")

        declared_vars = set([arg.split()[-1] for arg in input_args + parameter_arg +  output_args])
        print("--------------------")
        print("Declared Variables : ", declared_vars)
        print("--------------------")
        for node in self.graph.nodes:
            if node.op not in['output', 'placeholder', 'get_attr'] and node.name not in declared_vars:
                temp_info = self.op_extractor.extract_tensor_info(node)
                temp_type = self.get_fhy_type(temp_info)
                temp_vars.add(f"temp {temp_type} {node.name};")

        all_args = input_args + parameter_arg

        fhy_code = [f"op main({', '.join(all_args)}) -> {', '.join(output_args)} {{"]
        fhy_code.extend(sorted(temp_vars)) # sorting for consistent order
        fhy_code.append("") # blank line
        for node in self.graph.nodes:
            node_code = self.convert_node(node)
            if node_code:
                fhy_code.append( f"     {node_code}")
        fhy_code.append("}")
        print("-------------------------------------------")
        print(fhy_code)
        print("-------------------------------------------")
        return '\n'.join(fhy_code)

    # converting the nodes to code
    def convert_node(self, node:Node) -> Optional[str]:
        if node.op in ['placeholder', 'get_attr']:
            return None
        elif node.op == 'output':
            outputs = node.args[0]
            if isinstance(outputs, tuple):
                return f"return {', '.join([self.get_node_name(out) for out in outputs])};"
            else:
                return f"return {self.get_node_name(outputs)};"
        else:
            op_info = self.op_extractor.get_op_info(node)
            result_name = self.get_node_name(node)
            args = []
            for arg in op_info['inputs']:
                if isinstance(arg, dict) and 'name' in arg:
                    arg_name = arg['name']
                    print("ARG NAME: ", arg_name)
                    if arg_name.startswith('_'):
                        arg_name = f"{arg_name[1:]}"
                    args.append(arg_name)
                elif isinstance(arg, list):
                    arg_tuple = ', '.join([str(a['value'])if isinstance (a, dict) and 'value' in a else str(a) for a in arg])
                    args.append(f"({arg_tuple})")
                elif isinstance(arg, dict) and 'value' in arg:
                    args.append(str(arg['value']))
                else:
                    args.append('unknown')
            return f"{result_name} = {op_info['name']}({', '.join(args)});"


    # get the nodes name from the output purpose
    def get_node_name(self, node_or_name:Union[Node, str]) ->str:
        if isinstance(node_or_name, Node):
            print(node_or_name.name)
            return node_or_name.name
        else:
            print("This is the name string: ",str(node_or_name))
            return str(node_or_name)

    # get the type properly mapped with the shape
    def get_fhy_type(self, type_info: Any) -> str:
        if type_info is None:
            return "unknown_type"
        
        return self.type_infer(type_info)

    def type_infer(self, type_info: Any) -> str:
        if isinstance(type_info, dict) and 'dtype' in type_info and 'shape' in type_info:
            dtype = self.pytorch_to_fhy_dtype(type_info['dtype'])
            shape = type_info['shape']
            shape_str = ', '.join(map(str, shape))
            return f"{dtype}[{shape_str}]"
        elif isinstance(type_info, list):
            if all(isinstance(item, dict) and 'type' in item for item in type_info):
                types = [self.pytorch_to_fhy_dtype(item['type']) for item in type_info]
                return f"tuple[{', '.join(types)}]"
        elif hasattr(type_info, '__origin__') and type_info.__origin__ is tuple:
            types = [self.type_infer(arg) for arg in type_info.__args__]
            return f"tuple[{', '.join(types)}]"
        
        return "unknown_type"

    # pytorch to fhy type converter
    def pytorch_to_fhy_dtype(self, dtype: str) -> str:
        dtype_mapping = {
           "torch.float32"  : "float32",
            "torch.float"   : "float32",
            "torch.float64" : "float64",
            "torch.double"  : "float64",
            "torch.int32"   : "int32",
            "torch.int64"   : "int64",
            "torch.long"    : "int64",
            "int"           : "int32",
            "float"         : "float32",
        }
        return dtype_mapping.get(str(dtype).lower(), "unknown_not_found")


#--------------------------------------------------------------------------
# Testing
#--------------------------------------------------------------------------


class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(y)
        return a+b

example_args = (torch.randn(10,10), torch.rand(10, 10))

exported_program: torch.export.ExportedProgram = export(Mod(), args= example_args)

core_ir_exported = exported_program.run_decompositions()
# print(core_ir_exported)


class SAC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(256, 256)
        self.linear2 = torch.nn.Linear(256, 256)
        self.nu = torch.nn.Linear(256, 1)
        self.prob = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        nu = self.nu(x)
        prob = self.prob(x)
        return nu, prob

example_input = (torch.randn(1, 256),)
exported_program: torch.export.ExportedProgram = export(SAC(), args= example_input)
# print(exported_program)
#--------------------------------------------------------------------------
#  Res Net Example 
#--------------------------------------------------------------------------

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

example_input = (torch.rand(1, 3, 224, 224),)
# example_input = (torch.randn(1, 256),)
exported_program: torch.export.ExportedProgram = export(model, args= example_input)
print(exported_program)
# ---------------------------------------------------------------------------
m = model
gm = torch.fx.symbolic_trace(m)
gm.graph.print_tabular()

converter = PyTorchToFhyConverter()
fhy_code = converter.convert(model, example_input)
print(fhy_code)
