# graph_encoder.py

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import networkx as nx

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def encode_graph_with_gnn(graph: nx.DiGraph, dim: int = 384):
    node_list = list(graph.nodes)
    node_idx = {node: i for i, node in enumerate(node_list)}

    # Use identity matrix as dummy features
    x = torch.eye(len(node_list), dtype=torch.float32)

    edge_index = []
    for source, target in graph.edges():
        edge_index.append([node_idx[source], node_idx[target]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).T

    data = Data(x=x, edge_index=edge_index)

    model = GraphSAGE(in_channels=x.size(1), hidden_channels=dim, out_channels=dim)
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    return {
        node_id: embeddings[node_idx[node_id]].numpy()
        for node_id in node_list
    }
