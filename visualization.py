import dash
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import networkx as nx
from networkx.readwrite import json_graph


from fhy.ir import Identifier
from fhy.fdfg.core import FDFG, SourceNode, SinkNode
from fhy.fdfg.node.fractalized import FractalizedNode, FunctionNode
from fhy.fdfg.node.parametric import LoopNode, ReductionNode
from fhy.fdfg.edge import Edge
from fhy.fdfg.node.primitive import PrimitiveNode
import fhy.fdfg.ops as fdfg_op



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True


NODE_TO_COLOR_MAP = {
    SourceNode: "#d3d3d3",
    SinkNode: "#d3d3d3",
    FunctionNode: "#204f88",
    LoopNode: "#56bcbe",
    ReductionNode: "#FF5733",
    PrimitiveNode: "#8aedf6",
}


def get_element_id(node_name) -> str:
    return str(node_name.id)

# converthe networx graph to cytoscape graph
# def nx_to_cytoscape(G):
#     '''
#     link: https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
#     calculates the positions of the nodes in the graph using the 
#     spring layout algorithm, which positions nodes using a 
#     force-directed approach that simulates a physical system.

#     '''
#     elements = []
#     positions = nx.spring_layout(G)  # Use spring layout for better visualization

#     for node, data in G.nodes(data=True):
#         node_data = data["data"]
#         element = {
#             'data': {'id': get_element_id(node), "color": NODE_TO_COLOR_MAP[type(node_data)]},
#             'position': {'x': positions[node][0]*500, 'y': positions[node][1]*500}
#         }

#         if isinstance(node_data, SourceNode):
#             element["data"]["label"] = "Source"
#         elif isinstance(node_data, SinkNode):
#             element["data"]["label"] = "Sink"
#         elif isinstance(node_data, PrimitiveNode):
#             element["data"]["label"] = node_data.op.name.name_hint
#         elif isinstance(node_data, LoopNode):
#             element["data"]["label"] = "Loop"
#             element["data"]["indices"] = f"[{', '.join([i.name_hint for i in node_data.index_symbol_names])}]"
#             element["data"]["graph"] = nx_to_cytoscape(node_data.fdfg.graph)
#         elif isinstance(node_data, ReductionNode):
#             element["data"]["label"] = node_data.symbol_name.name_hint
#             element["data"]["reduced_indices"] = f"[{', '.join([i.name_hint for i in node_data.index_symbol_names])}]"
#             element["data"]["graph"] = nx_to_cytoscape(node_data.fdfg.graph)
#         elif isinstance(node_data, FunctionNode):
#             element["data"]["label"] = node_data.symbol_name.name_hint
#             element["data"]["graph"] = nx_to_cytoscape(node_data.fdfg.graph)
#         else:
#             raise RuntimeError()

#         elements.append(element)
    
#     for edge in G.edges(data=True):
#         elements.append({
#             'data': {
#                 'source': get_element_id(edge[0]), 
#                 'target': get_element_id(edge[1]),
#                 "label": edge[2]["data"].symbol_name.name_hint
#             }
#         })

#     return elements

# Topoligical Sort of the graph nodes
def nx_to_cytoscape(G):
    '''
    link: https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
    calculates the positions of the nodes in the graph using the 
    spring layout algorithm, which positions nodes using a 
    force-directed approach that simulates a physical system.

    '''
    elements = []
    # positions = nx.spring_layout(G)  # Use spring layout for better visualization
        # Perform a topological sort of the graph nodes
    sorted_nodes = list(nx.topological_sort(G))
    print(sorted_nodes)
    # Create a custom layout for a top-down layout
    positions = {node: (500, i * 250) for i, node in enumerate(sorted_nodes)}

    for node, data in G.nodes(data=True):
        node_data = data["data"]
        element = {
            'data': {'id': get_element_id(node), "color": NODE_TO_COLOR_MAP[type(node_data)]},
            'position': {'x': positions[node][0], 'y': positions[node][1]}
        }

        if isinstance(node_data, SourceNode):
            element["data"]["label"] = "Source"
        elif isinstance(node_data, SinkNode):
            element["data"]["label"] = "Sink"
        elif isinstance(node_data, PrimitiveNode):
            element["data"]["label"] = node_data.op.name.name_hint
        elif isinstance(node_data, LoopNode):
            element["data"]["label"] = "Loop"
            element["data"]["indices"] = f"[{', '.join([i.name_hint for i in node_data.index_symbol_names])}]"
            element["data"]["graph"] = nx_to_cytoscape(node_data.fdfg.graph)
        elif isinstance(node_data, ReductionNode):
            element["data"]["label"] = node_data.symbol_name.name_hint
            element["data"]["reduced_indices"] = f"[{', '.join([i.name_hint for i in node_data.index_symbol_names])}]"
        elif isinstance(node_data, FunctionNode):
            element["data"]["label"] = node_data.symbol_name.name_hint
            element["data"]["graph"] = nx_to_cytoscape(node_data.fdfg.graph)
        else:
            raise RuntimeError()

        elements.append(element)
    
    for edge in G.edges(data=True):
        elements.append({
            'data': {
                'source': get_element_id(edge[0]), 
                'target': get_element_id(edge[1]),
                "label": edge[2]["data"].symbol_name.name_hint
            }
        })

    return elements

# for the properties of the nodes 
def handle_node_properties(node_data):
    node_type = node_data.get('label', '').split('\n')[0]
    class_name = node_data.get('__class__', '')
    
    properties = []
    for key, value in node_data.items():
        if key not in ['id', 'label', 'color', "__class__", "timeStamp", "graph"]:
            prop_id = f"{node_data['id']}-{key}-tooltip"
            
            properties.append(
                html.Div([
                    html.P(f"{key}: {value}", id=prop_id, style={"cursor": "pointer"}),
                ])
            )
    return properties

# Create the layout for the dash
def create_dash_layout(elements):
    mini_panel_content = generate_mini_panel()
    return html.Div([
        html.Div([
            html.H1("f-DFG Viewer", style={'color': 'white', 'padding': '20px'}),
        ], style={'background-color': '#2c3e50', 'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
        dcc.Store(id='node-properties-store'),
        dcc.Store(id='original-elements', data=elements),
        dcc.Store(id='current-elements', data=elements),  # Store current elements
        dcc.Store(id='history-stack', data=[]),  # Store the history stack
        dbc.Row([
            dbc.Col(
                cyto.Cytoscape(
                    id='cytoscape-graph',
                    elements=elements,
                    style={'width': '99vw', 'height': '95vh'},
                    layout={'name': 'preset'},
                    stylesheet=[
                        {
                            'selector': 'node',
                            'style': {
                                'label': 'data(label)',
                                'text-wrap': 'wrap',
                                'text-max-width': '150px',
                                'font-weight': 'bold',
                                'background-color': 'data(color)',
                                'shape': 'roundrectangle',
                                'width': '200px',
                                'height': '100px',
                                'text-valign': 'center',
                                'text-halign': 'center',
                            }
                        },
                        {
                            'selector': 'edge',
                            'style': {
                                'width': 6,
                                'label': 'data(label)',
                                'font-size': "2em",
                                'target-arrow-shape': 'triangle',
                                'line-color': '#9dbaea',
                                'target-arrow-color': '#9dbaea',
                                'curve-style': 'bezier'
                            }
                        }
                    ]
                ), width=8
            ),
            dbc.Col(
                [
                    html.Div(id='node-properties', className='card p-3'),
                    html.Div(mini_panel_content, className='mini-panel card p-3', style={'marginTop': '20px'}),  # Add the mini panel here
                    
                ],
                width=4,
                style={'height': 'calc(100vh - 800px)'}
            )
        ], style={'height': 'calc(100vh - 100px)'}),
        dbc.Button('Back to Main Graph', id="back-to-main-btn", color="primary", className="mt-2", style={'position': 'fixed', 'left': '20px', 'bottom': '20px'}),
        dbc.Button('Back One Step', id="back-one-step-btn", color="primary", className="mt-2", style={'position': 'fixed', 'left': '220px', 'bottom': '20px'})
    ])


@app.callback(
    Output("node-properties-store", "data"),
    [Input("cytoscape-graph", "tapNodeData")]
)
def update_node_properties_store(data):
    if data:
        data['__class__'] = data.get('label', '').split('\n')[0]
        return data
    return {}

@app.callback(
    Output("node-properties", "children"),
    [Input("node-properties-store", "data")]
)
def display_node_properties(data):
    if data:
        properties = handle_node_properties(data)
        return html.Div(properties, className="card p-3")
    return html.Div("Click a node to see its properties", className="card p-3")


@app.callback(
    [Output('cytoscape-graph', 'elements'), Output('history-stack', 'data')],
    [Input('cytoscape-graph', 'tapNodeData'),
     Input('back-to-main-btn', 'n_clicks'),
     Input('back-one-step-btn', 'n_clicks')],
    [State('original-elements', 'data'),
     State('cytoscape-graph', 'elements'),
     State('history-stack', 'data')]
)

# Display the subgraph when a node is clicked
def display_subgraph(tap_node_data, back_to_main_clicks, back_one_step_clicks, original_elements, current_graph_elements, history_stack):
    ctx = dash.callback_context

    if not ctx.triggered:
        return current_graph_elements, history_stack

    triggered_id = ctx.triggered[0]['prop_id']

    if 'back-to-main-btn' in triggered_id:
        return original_elements, []

    if 'back-one-step-btn' in triggered_id:
        if history_stack:
            last_state = history_stack.pop()
            return last_state, history_stack
        return current_graph_elements, history_stack

    if tap_node_data and 'graph' in tap_node_data:
        subgraph_elements = tap_node_data['graph']
        new_history_stack = history_stack + [current_graph_elements]
        return subgraph_elements, new_history_stack

    return current_graph_elements, history_stack


def generate_mini_panel():
    mini_panel_items = []
    for node_type, color in NODE_TO_COLOR_MAP.items():
        mini_panel_items.append(
            html.Div([
                html.Span(style={'backgroundColor': color, 'display': 'inline-block', 'width': '20px', 'height': '20px', 'borderRadius': '50%'}),
                html.Span(f" {node_type.__name__}", style={'marginLeft': '10px'}),
            ], style={'margin': '5px 0', 'display': 'flex', 'alignItems': 'center'})
        )
    return mini_panel_items