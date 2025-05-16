# graph_builder.py

import json
import networkx as nx
from utils.types import Node, Edge

def load_nodes_and_edges(node_path, edge_path):
    with open(node_path, 'r', encoding='utf-8') as f:
        node_data = json.load(f)

    with open(edge_path, 'r', encoding='utf-8') as f:
        edge_data = json.load(f)

    nodes = [Node(**n) for n in node_data]
    edges = [Edge(**e) for e in edge_data]

    return nodes, edges

def build_graph(nodes, edges) -> nx.DiGraph:
    G = nx.DiGraph()

    for node in nodes:
        G.add_node(node.id, **node.__dict__)

    for edge in edges:
        G.add_edge(edge.source, edge.target, relation=edge.relation)

    return G

def save_graph(graph: nx.DiGraph, path: str):
    nx.write_graphml(graph, path)  # Saves as GraphML for inspection in tools like Gephi

# Example usage
if __name__ == "__main__":

    from graph_encoder import encode_graph_with_gnn

    nodes, edges = load_nodes_and_edges(
        "output/psf_requests/data/graph_nodes.json",
        "output/psf_requests/data/graph_edges.json"
    )

    G = build_graph(nodes, edges)
    embedding_dict = encode_graph_with_gnn(G)

    print(f"✅ Embedded {len(embedding_dict)} nodes with GNN")
