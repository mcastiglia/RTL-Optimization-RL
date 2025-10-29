import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

def generate_prefix_adder_graph(n_bits):
    """
    Generates a graph representation of a prefix adder circuit for n_bits.
    
    Nodes represent bit positions (0 to n_bits-1).
    Node features: [bit_position, fan_in, fan_out, delay_level]
    Edges: Basic prefix connections (linear chain + some prefix links for small n).
    
    Args:
        n_bits (int): Number of bits in the adder.
    
    Returns:
        torch_geometric.data.Data: The graph data object.
    """
    # Node features: bit position
    bit_positions = torch.arange(n_bits, dtype=torch.float)
    
    # Edges: Start with a linear chain (ripple carry)
    edge_list = []
    for i in range(n_bits - 1):
        edge_list.append((i, i + 1))
    
    # Add prefix-specific edges for small n (example for n=4)
    if n_bits >= 4:
        edge_list.extend([(0, 2), (1, 3), (0, 3)])
    # For larger n, you can extend this logic to build a proper prefix tree
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Compute fan-in and fan-out
    fan_in = degree(edge_index[1], num_nodes=n_bits).float()
    fan_out = degree(edge_index[0], num_nodes=n_bits).float()
    
    # Compute delay level (approximate critical path level using BFS)
    delay_levels = compute_delay_levels(edge_index, n_bits)
    
    # Combine features
    x = torch.stack([bit_positions, fan_in, fan_out, delay_levels], dim=1)
    
    data = Data(x=x, edge_index=edge_index)
    return data

def compute_delay_levels(edge_index, num_nodes):
    """
    Computes approximate delay levels (longest path from node 0) assuming unit edge delays.
    Uses a simple BFS to assign levels.
    """
    levels = torch.full((num_nodes,), -1, dtype=torch.float)
    levels[0] = 0  # Start from node 0
    queue = [0]
    visited = set([0])
    
    while queue:
        current = queue.pop(0)
        current_level = levels[current]
        
        # Find neighbors
        neighbors = edge_index[1][edge_index[0] == current]
        for neighbor in neighbors:
            if neighbor.item() not in visited:
                visited.add(neighbor.item())
                levels[neighbor] = current_level + 1
                queue.append(neighbor.item())
    
    # For nodes not reached, set to 0 or handle
    levels = torch.where(levels == -1, torch.tensor(0.0), levels)
    return levels

def compute_area(graph):
    """
    Computes an estimate of the total synthesized area based on the graph structure.
    Simple formula: area = num_nodes + sum(fan_out)  # Proportional to gates and drivers
    """
    fan_out = graph.x[:, 2]  # fan_out is at index 2
    area = graph.num_nodes + fan_out.sum().item()
    return area

def compute_power(graph):
    """
    Computes an estimate of the total power usage based on the graph structure.
    Simple formula: power = sum(fan_in * fan_out)  # Proportional to switching activity
    """
    fan_in = graph.x[:, 1]
    fan_out = graph.x[:, 2]
    power = (fan_in * fan_out).sum().item()
    return power

def generate_rca_graph(n_bits):
    """
    Generates a graph for a Ripple Carry Adder (RCA).
    Nodes: bit positions, Edges: linear chain for carry propagation.
    """
    bit_positions = torch.arange(n_bits, dtype=torch.float)
    edge_list = [(i, i + 1) for i in range(n_bits - 1)]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    fan_in = degree(edge_index[1], num_nodes=n_bits).float()
    fan_out = degree(edge_index[0], num_nodes=n_bits).float()
    delay_levels = compute_delay_levels(edge_index, n_bits)
    
    x = torch.stack([bit_positions, fan_in, fan_out, delay_levels], dim=1)
    data = Data(x=x, edge_index=edge_index)
    return data

def generate_kogge_stone_graph(n_bits):
    """
    Generates a graph for a Kogge-Stone Adder.
    Nodes: bit positions, Edges: all possible prefix connections (dense tree).
    """
    bit_positions = torch.arange(n_bits, dtype=torch.float)
    edge_list = [(i, j) for i in range(n_bits) for j in range(i + 1, n_bits)]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    fan_in = degree(edge_index[1], num_nodes=n_bits).float()
    fan_out = degree(edge_index[0], num_nodes=n_bits).float()
    delay_levels = compute_delay_levels(edge_index, n_bits)
    
    x = torch.stack([bit_positions, fan_in, fan_out, delay_levels], dim=1)
    data = Data(x=x, edge_index=edge_index)
    return data

def generate_sklansky_graph(n_bits):
    """
    Generates a graph for a Sklansky Adder (simplified sparse prefix).
    Nodes: bit positions, Edges: sparse prefix connections.
    """
    bit_positions = torch.arange(n_bits, dtype=torch.float)
    edge_list = []
    for i in range(n_bits):
        for j in range(i + 1, n_bits):
            if (j - i) & (i) == 0:  # Sparse condition
                edge_list.append((i, j))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    fan_in = degree(edge_index[1], num_nodes=n_bits).float()
    fan_out = degree(edge_index[0], num_nodes=n_bits).float()
    delay_levels = compute_delay_levels(edge_index, n_bits)
    
    x = torch.stack([bit_positions, fan_in, fan_out, delay_levels], dim=1)
    data = Data(x=x, edge_index=edge_index)
    return data

def generate_brent_kung_graph(n_bits):
    """
    Generates a graph for a Brent-Kung Adder (another prefix structure).
    Nodes: bit positions, Edges: Brent-Kung specific connections.
    """
    bit_positions = torch.arange(n_bits, dtype=torch.float)
    edge_list = []
    # Simplified Brent-Kung: similar to prefix but with specific merges
    for i in range(n_bits):
        if i % 2 == 0 and i + 1 < n_bits:
            edge_list.append((i, i + 1))
        for j in range(i + 2, n_bits, 2):
            edge_list.append((i, j))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    fan_in = degree(edge_index[1], num_nodes=n_bits).float()
    fan_out = degree(edge_index[0], num_nodes=n_bits).float()
    delay_levels = compute_delay_levels(edge_index, n_bits)
    
    x = torch.stack([bit_positions, fan_in, fan_out, delay_levels], dim=1)
    data = Data(x=x, edge_index=edge_index)
    return data

# def get_real_area_power(graph):
#     """
#     Placeholder function to get real area and power estimates using synthesis tools.
#     For now, returns the analytical estimates.
#     Later, this will call a synthesis engine (e.g., via subprocess or API).
#     """
#     # TODO: Integrate with synthesis tool, e.g., call external script
#     # For example: result = subprocess.run(['synthesis_tool', graph_to_rtl(graph)], capture_output=True)
#     # Parse result for area and power
#     return compute_area(graph), compute_power(graph)

if __name__ == '__main__':
    n = 4  # Example for 4-bit adder
    graph = generate_prefix_adder_graph(n)
    print(f"Generated graph for {n}-bit prefix adder")
    print("Node features [bit_pos, fan_in, fan_out, delay_level]:")
    for i in range(graph.num_nodes):
        print(f"  Node {i}: {graph.x[i].tolist()}")
    print("Edge index:", graph.edge_index.tolist())
    print("Number of nodes:", graph.num_nodes)
    print("Number of edges:", graph.num_edges)

def get_real_area_power(graph, flow_type='fast'):
    import subprocess
    import os
    
    tcl_script = 'full_flow.tcl' if flow_type == 'full' else 'fast_flow.tcl'
    base_dir = os.environ.get('BASE_DIR', '.')
    openroad_dir = f"{base_dir}/OpenROAD"

    # Assume the TCL script is designed to run synthesis and output area/power
    # For now, this is a placeholder - you need to implement the actual synthesis call
    # The TCL script should be modified to accept design parameters or use environment variables

    cmd = f"apptainer exec --bind {openroad_dir}:/workspace ./openroad.sif bash -c \"cd {openroad_dir}/prefix-flow && openroad {tcl_script}\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Parse area and power from the output
    area = None
    power = None
    for line in result.stdout.split('\n'):
        if 'Area:' in line:
            try:
                area = float(line.split('result: design_area = ')[1].strip().split()[0])
            except:
                pass
        if 'Power:' in line:
            try:
                power = float(line.split('result: design_power = ')[1].strip().split()[0])
            except:
                pass
    
    if area is None and power is None:
        # Fallback to analytical computation
        #print(f"Warning: Could not parse area/power from synthesis output, using analytical values")
        area, power = compute_area(graph), compute_power(graph)
    
    return area
