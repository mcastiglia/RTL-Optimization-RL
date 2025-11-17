import csv
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

def parse_arguments():
    args = argparse.ArgumentParser()
    args.add_argument('--w_scalar', type=float, required=True)
    args.add_argument('--c_delay', type=float, default=10.0)
    args.add_argument('--c_area', type=float, default=1e-3)
    args.add_argument('--file_name', type=str, required=True)
    args.add_argument('--input_dir', type=str, required=True)
    args.add_argument('--plot_dir', type=str, required=True)
    args.add_argument('--n64', action='store_true')
    return args.parse_args()

def extract_min_scalarized_graph(file_name: str, w_scalar: float, c_delay: float = 10.0, c_area: float = 1e-3):
    w_delay = w_scalar
    w_area = 1 - w_scalar
    min_score = {}
    min_score['scalar'] = float('inf')
    min_line_num = None
    total_lines_in_file = 0
    
    with open(file_name, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                if (row['verilog_file_name'] == 'verilog_file_name') and (row['delay'] == 'delay'):
                    continue  # Skip header rows within the file
                verilog_file_name = row['verilog_file_name']
                delay = float(row['delay'])
                area = float(row['area'])
                level = int(row['level'])
                size = int(row['size'])
                fanout = int(row['fanout'])
                scalar_score = w_area * (c_area * area) + w_delay * (c_delay * delay)
                
                if scalar_score < min_score['scalar']:
                    min_score['verilog_file_name'] = verilog_file_name
                    min_line_num = reader.line_num
                    min_score['scalar'] = scalar_score
                    min_score['delay'] = delay
                    min_score['area'] = area
                    min_score['level'] = level
                    min_score['size'] = size
                    min_score['fanout'] = fanout
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping row due to error: {e}")
                continue
        total_lines_in_file = reader.line_num
            
    print("Verilog file that minimizes scalar score: ", min_score['verilog_file_name'])
    print("Scalar score: ", min_score['scalar'])
    print("Delay: ", min_score['delay'])
    print("Area: ", min_score['area'])
    print("Level: ", min_score['level'])
    print("Size: ", min_score['size'])
    print("Fanout: ", min_score['fanout'])
    print("CSV line of minimum: ", min_line_num)
    print("Total lines in file: ", total_lines_in_file)
    
    return min_score
    
def load_feature_array(filepath):
    """Load a tab-separated feature list file into a numpy array, handling trailing tabs and empty lines."""
    with open(filepath, 'r') as f:
        lines = []
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                # Split by tab and filter out empty strings from trailing tabs
                values = [int(x) for x in line.split('\t') if x]
                if values:  # Only add non-empty rows
                    lines.append(values)
    
    if lines:
        return np.array(lines, dtype=np.int8)
    else:
        return None
    
def extract_feature_lists(directory_name: str, verilog_file_name: str):
    prefix = verilog_file_name.rsplit("_", 1)[0] + "_"
    hash_value = verilog_file_name.rsplit("_", 1)[1]
    
    feature_arrays = {}
    
    for filename in os.listdir(directory_name):
        if prefix in filename and hash_value in filename and filename.endswith('.log'):
            filepath = os.path.join(directory_name, filename)
            
            # Extract feature name (e.g., 'nodelist', 'levellist', etc.)
            parts = filename.replace('.log', '').split('_')
            if len(parts) >= 5:
                feature_name = parts[-2]  # Second to last part
                
                array = load_feature_array(filepath)
                if array is not None:
                    feature_arrays[feature_name] = array
        
    return feature_arrays
        
# Update levellist based on changes to nodelist (Taken from ArithTreeRL)
def update_levellist(nodelist, levellist):
    n = len(nodelist)
    
    levellist[1:].fill(0)
    levellist[0, 0] = 1
    for m in range(1, n):
        levellist[m, m] = 1
        prev_l = m
        for l in range(m-1, -1, -1):
            if nodelist[m, l] == 1:
                levellist[m, l] = max(levellist[m, prev_l], levellist[prev_l-1, l])+ 1
                prev_l = l
                
    return levellist
    
def plot_prefix_graph(nodelist, minlist, levellist, verilog_file_name, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    n = len(nodelist)
    plot_title = os.path.join(output_dir, "{}.png".format(verilog_file_name))
    plt.clf()
    G = nx.DiGraph()
    
    nodes = []
    for i in range(n):
        for j in range(n):
            if nodelist[i, j] == 1:
                # nodes.append((int(levellist[i,j])-1,j))
                nodes.append((i, int(levellist[i,j])-1))
    
    G.add_nodes_from(nodes)
    
    edges = []
    for x in range(n-1, 0, -1):
        last_y = x
        for y in range(x-1, -1, -1):
            if nodelist[x, y] == 1:
                level = int(levellist[x,y])-1
                # Upper parent
                if nodelist[x, last_y] == 1:
                    last_y_level = int(levellist[x,last_y])-1
                    edges.append(((x, last_y_level), (x, level)))
                    
                # Lower parent
                # if nodelist[last_y-1, y] == 1:
                #     edges.append(((last_y-1, level-1), (x, level)))
                if nodelist[last_y-1, y] == 1:
                    lower_level = int(levellist[last_y-1, y]) - 1
                    edges.append(((last_y-1, lower_level), (x, level)))
                last_y = y
                    
    G.add_edges_from(edges)
    
    pos = {(r, c): (r,c) for r in range(n) for c in range(n)}
        
    nx.draw(
        G, 
        pos,
        with_labels=False,
        node_size=1024 // n,
        edgecolors="black",
        linewidths=1,
        edge_color="gray",
        arrows=False,
        arrowsize=100 // n,
    )
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    
    plt.savefig(plot_title, dpi=300, bbox_inches="tight")        
        
def plot_scalar_bars(
    n64: bool,
    min_score: float,
    w_scalar: float,
    c_delay: float = 10.0,
    c_area: float = 1e-3,
    output_path: str = "scalar_scores.png",
):
    """Plot min_score against scalar scores for RCA, Sklansky, and Brent-Kung."""
    if not n64:
        rca_delay = 1.932497550015271
        rca_area = 256.96
        sklansky_delay = 1.273223118529967
        sklansky_area = 402.72
        brent_kung_delay = 1.2637878323771907
        brent_kung_area = 288.61
    else:
        rca_delay = 3.47527738645499
        rca_area = 486.78
        sklansky_delay = 1.905973210983792
        sklansky_area = 852.26
        brent_kung_delay = 1.7058077655439832
        brent_kung_area = 543.7
    
    rca_score = w_scalar * (c_delay * rca_delay) + (1 - w_scalar) * (c_area * rca_area)
    sklansky_score = w_scalar * (c_delay * sklansky_delay) + (1 - w_scalar) * (c_area * sklansky_area)
    brent_kung_score = w_scalar * (c_delay * brent_kung_delay) + (1 - w_scalar) * (c_area * brent_kung_area)

    labels = ["RL", "rca", "sklansky", "brent_kung"]
    values = [min_score, rca_score, sklansky_score, brent_kung_score]

    plt.clf()
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    ax.set_xlabel("Adder Type")
    ax.set_ylabel("Scalar Score")
    ax.set_title(f"Scalar Scores of Baseline and PrefixRL-Optimized Adders\n(w_scalar={w_scalar:.2f})")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.3g}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "scalar_scores.png"), dpi=300, bbox_inches="tight")
    
def main():
    args = parse_arguments()
    min_score = extract_min_scalarized_graph(args.file_name, args.w_scalar, args.c_delay, args.c_area)
    feature_arrays = extract_feature_lists(args.input_dir, min_score['verilog_file_name'])
    plot_prefix_graph(feature_arrays['nodelist'], feature_arrays['minlist'], feature_arrays['levellist'], min_score['verilog_file_name'], args.plot_dir)
    plot_scalar_bars(args.n64, min_score['scalar'], args.w_scalar, args.c_delay, args.c_area, args.plot_dir)
    
if __name__ == "__main__":
    main()