import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn.models import Graphormer
from preprocess import generate_prefix_adder_graph, compute_area, compute_power

# Old preprocessing function removed, using generate_prefix_adder_graph instead.

# --- How to Use ---
if __name__ == '__main__':
    # --- Dummy Data Simulation ---
    # Generate a prefix adder graph
    graphormer_input = generate_prefix_adder_graph(15)
    
    # Compute area and power estimates
    area = compute_area(graphormer_input)
    power = compute_power(graphormer_input)
    
    print("Graph Transformer Model (Graphormer)")
    print("Generated graph with", graphormer_input.num_nodes, "nodes and", graphormer_input.num_edges, "edges")
    print(f"Estimated area: {area}")
    print(f"Estimated power: {power}")
    
    # --- Model Instantiation ---
    model = Graphormer(num_node_types=None, num_edge_types=1)
    
    # --- Forward Pass ---
    with torch.no_grad():
        outputs = model(graphormer_input)

    # The final graph-level representation
    graph_embedding = outputs

    print(f"\nShape of final graph embedding: {graph_embedding.shape}")
    print("This embedding represents the entire circuit and can be used as the state for an RL agent.")