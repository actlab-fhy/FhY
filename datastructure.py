import networkx as nx
from functools import wraps
from networkx.readwrite import json_graph


def describe(description):
    ''' Decorator to add a description to the class attribute'''
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        wrapped.description = description
        return wrapped
    return decorator

class Node:
    ''' Base class for all nodes in the graph '''
    @describe("Base node class")
    def __init__(self, name, subgraph=None):
        self.name = name
        self.subgraph = subgraph

    # adds the node to the graph with the specified label and color
    # uses json_graph.node_link_data to serialize the subgraph 
    def add(self, graph, label_suffix, color):
        node_data = self.serialize_node_data()
        if self.subgraph:
            node_data['subgraph'] = json_graph.node_link_data(self.subgraph)
        graph.add_node(self.name, label=f"{self.name}\n---\n{label_suffix}", color=color, **node_data)

    def serialize_node_data(self):
        ''' Converts the attributes of the node object into a dictionary
            fromat that can be used by NetworkX to when adding nodes 
            to the graph
        '''
        data = vars(self)
        serialized_data = {}
        for key, value in data.items():
            if key == 'subgraph' and isinstance(value, nx.Graph):
                serialized_data[key] = json_graph.node_link_data(value)
            elif isinstance(value, (str, int, float, list, dict)):
                serialized_data[key] = value
            elif isinstance(value, Operation):
                serialized_data[key] = vars(value)
            else:
                serialized_data[key] = str(value)
        return serialized_data

class Operation:
    ''' Represents the operation performed by the compute node'''
    @describe("Type of the operation")
    def __init__(self, type, sizeX, sizeY):
        self.type = type
        self.sizeX = sizeX
        self.sizeY = sizeY

class ComputeNode(Node):
    ''' Represents the compute node in the graph'''
    @describe("The name of the compute node")
    def __init__(self, name, dimOne, dimTwo, ops, outputDest, subgraph=None):
        super().__init__(name, subgraph) # Call the constructor of the base class
        self.dimOne = dimOne
        self.dimTwo = dimTwo
        self.ops = ops
        self.outputDest = outputDest

    # Add the compute node to the graph with the specified label and color
    def add(self, graph):
        super().add(graph, "Compute Node", '#9b59b6')

class MemoryNode(Node):
    ''' Represents the memory node in the graph'''
    @describe("The name of the memory node")
    def __init__(self, name, numBanks, capacity, dataType, location, size, layout, subgraph=None):
        super().__init__(name, subgraph) # Call the constructor of the base class
        self.numBanks = numBanks
        self.capacity = capacity
        self.dataType = dataType
        self.location = location
        self.size = size
        self.layout = layout
    
    def add(self, graph):
        super().add(graph, "Memory Node", '#3498db')

class PrimitiveOperation:
    ''' Represents the primitive operation performed by the compute node'''
    @describe("Type of the operation")
    def __init__(self, type, dest, src, size):
        self.type = type
        self.dest = dest
        self.src = src
        self.size = size

class Interconnect:
    ''' Represents the interconnect between two nodes in the graph'''
    def __init__(self, name):
        self.name = name

    def add(self, graph, source, target, weight):
        graph.add_edge(source, target, weight=weight, name=self.name)

class ArchitectureCovenantGraph:
    ''' Represents the graph of the architecture covenant'''

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_compute_node(self, compute_node):
        compute_node.add(self.graph)

    def add_memory_node(self, memory_node):
        memory_node.add(self.graph)
    # change this function to include weights
    def add_interconnect(self, interconnect, source, target, weight):
        interconnect.add(self.graph, source, target, weight=weight)

    def add_custom_node(self, name, label, color, **attrs):
        self.graph.add_node(name, label=label, color=color, **attrs)

    def find_short_path(self, source, target):
        return nx.shortest_path(self.graph, source, target)
    
    def find_shortest_path(self, source, target):
        return nx.dijkstra_path(self.graph, source, target, weight='weight')

