import csv
import argparse
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import ConnectionPatch
try:
    import imageio.v2 as imageio
except Exception:
    import imageio

def parse_arguments():
    args = argparse.ArgumentParser()
    args.add_argument('--w_scalar', type=float, required=True)
    args.add_argument('--c_delay', type=float, default=1.0)
    args.add_argument('--c_area', type=float, default=1e-2)
    args.add_argument('--file_name', type=str, required=True)
    args.add_argument('--input_dir', type=str, required=True)
    args.add_argument('--verilog_dir', type=str, default=None)
    args.add_argument('--plot_dir', type=str, required=True)
    args.add_argument('--n64', action='store_true')
    args.add_argument('--pareto', action='store_true')
    args.add_argument('--extract_verilog', action='store_true')
    args.add_argument('--w_step', type=float, default=0.1)
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
    
def plot_prefix_graph(nodelist, minlist, levellist, verilog_file_name, output_dir, w_scalar):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    n = len(nodelist)
    plot_title = os.path.join(output_dir, "{}_w{}.png".format(verilog_file_name, w_scalar))
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
        node_size=1024 // (2*n),
        edgecolors="black",
        linewidths=1,
        edge_color="gray",
        arrows=False,
        arrowsize=100 // n,
    )
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    
    plt.savefig(plot_title, dpi=300, bbox_inches="tight")        
        
def draw_prefix_graph_on_axes(nodelist, minlist, levellist, ax):
    n = len(nodelist)
    G = nx.DiGraph()
    nodes = []
    for i in range(n):
        for j in range(n):
            if nodelist[i, j] == 1:
                nodes.append((i, int(levellist[i, j]) - 1))
    G.add_nodes_from(nodes)
    edges = []
    for x in range(n - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if nodelist[x, y] == 1:
                level = int(levellist[x, y]) - 1
                if nodelist[x, last_y] == 1:
                    last_y_level = int(levellist[x, last_y]) - 1
                    edges.append(((x, last_y_level), (x, level)))
                if nodelist[last_y - 1, y] == 1:
                    lower_level = int(levellist[last_y - 1, y]) - 1
                    edges.append(((last_y - 1, lower_level), (x, level)))
                last_y = y
    G.add_edges_from(edges)
    pos = {(r, c): (r, c) for r in range(n) for c in range(n)}
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=max(64, 1024 // max(1, n)),
        edgecolors="black",
        linewidths=1,
        edge_color="gray",
        arrows=False,
        arrowsize=max(8, 100 // max(1, n)),
        ax=ax,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
        
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
    plt.savefig(os.path.join(output_path, "scalar_scores_w{}.png".format(w_scalar)), dpi=300, bbox_inches="tight")
    
def plot_scalar_pareto(n64: bool, min_scores: dict, output_path: str = "scalar_scores.png", c_delay: float = 1.0, c_area: float = 1e-2, w_step: float = 0.1):
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
        
    ws = np.arange(0.0, 1.0, w_step)
    rca_scores = [w * (c_delay * rca_delay) + (1 - w) * (c_area * rca_area) for w in ws]
    sklansky_scores = [w * (c_delay * sklansky_delay) + (1 - w) * (c_area * sklansky_area) for w in ws]
    brent_kung_scores = [w * (c_delay * brent_kung_delay) + (1 - w) * (c_area * brent_kung_area) for w in ws]

    unique_map = {}
    for w, score in min_scores.items():
        area = float(score['area'])
        delay = float(score['delay'])
        key = (round(area, 6), round(delay, 6))
        if key not in unique_map:
            unique_map[key] = {'area': area, 'delay': delay, 'ws': [w]}
        else:
            unique_map[key]['ws'].append(w)

    rl_areas = [v['area'] for v in unique_map.values()]
    rl_delays = [v['delay'] for v in unique_map.values()]
    rl_labels = [f"{min(v['ws']):.2f}" for v in unique_map.values()]
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(sklansky_area, sklansky_delay, label="sklansky")
    ax.scatter(brent_kung_area, brent_kung_delay, label="brent_kung")
    ax.scatter(rca_area, rca_delay, label="rca")
    rl_scatter = ax.plot(rl_areas, rl_delays, label="RL", marker="o", color="red")
    for x, y, text in zip(rl_areas, rl_delays, rl_labels):
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(5, 5), ha="left", fontsize=8)
    ax.legend()
    ax.set_xlabel("Area (um^2)")
    ax.set_ylabel("Delay (ns)")
    ax.set_title("Pareto Frontier for Baseline and RL-Optimized Adders")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.savefig(os.path.join(output_path, "scalar_pareto.png"), dpi=300, bbox_inches="tight")
    
def animate_pareto_with_graph(n64: bool, min_scores: dict, output_path: str, c_delay: float = 1.0, c_area: float = 1e-2, gif_name: str = "scalar_pareto.gif"):
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
    ws_sorted = sorted(min_scores.keys())
    rl_areas_all = [float(min_scores[w]['area']) for w in ws_sorted]
    rl_delays_all = [float(min_scores[w]['delay']) for w in ws_sorted]
    unique_map = {}
    for w in ws_sorted:
        area = float(min_scores[w]['area'])
        delay = float(min_scores[w]['delay'])
        key = (round(area, 6), round(delay, 6))
        fa = min_scores[w].get('feature_arrays', {})
        if key not in unique_map:
            unique_map[key] = {
                'w_list': [w],
                'area': area,
                'delay': delay,
                'feature_arrays': fa
            }
        else:
            unique_map[key]['w_list'].append(w)
    unique_points = list(unique_map.values())
    rl_unique_areas = [p['area'] for p in unique_points]
    rl_unique_delays = [p['delay'] for p in unique_points]
    rl_labels = [f"{min(p['w_list']):.2f}" for p in unique_points]
    # Fix subplot axis ranges across frames for consistent dimensions
    x_candidates = rl_unique_areas + [rca_area, sklansky_area, brent_kung_area]
    y_candidates = rl_unique_delays + [rca_delay, sklansky_delay, brent_kung_delay]
    x_min, x_max = min(x_candidates), max(x_candidates)
    y_min, y_max = min(y_candidates), max(y_candidates)
    x_pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    x_lim = (x_min - x_pad, x_max + x_pad)
    y_lim = (y_min - y_pad, y_max + y_pad)
    frame_paths = []
    for idx, p in enumerate(unique_points):
        w = min(p['w_list'])
        fa = p.get('feature_arrays', {})
        nodelist = fa.get('nodelist')
        minlist = fa.get('minlist')
        levellist = fa.get('levellist')
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1.8, 2.2]})
        ax_left, ax_right = axes
        if nodelist is not None and levellist is not None and minlist is not None:
            draw_prefix_graph_on_axes(nodelist, minlist, levellist, ax_left)
            # Fix left subplot limits to keep layout stable
            n = len(nodelist)
            ax_left.set_xlim(-1, n)
            ax_left.set_ylim(-1, n)
            # Rotate the left graph by 180 degrees (invert both axes)
            ax_left.invert_xaxis()
            ax_left.invert_yaxis()
        else:
            ax_left.text(0.5, 0.5, "Graph unavailable", ha="center", va="center")
            ax_left.set_axis_off()
        ax_right.scatter(sklansky_area, sklansky_delay, label="sklansky")
        ax_right.scatter(brent_kung_area, brent_kung_delay, label="brent_kung")
        ax_right.scatter(rca_area, rca_delay, label="rca")
        ax_right.plot(rl_unique_areas, rl_unique_delays, label="RL", marker="o", color="red", alpha=0.3)
        ax_right.plot(rl_unique_areas[: idx + 1], rl_unique_delays[: idx + 1], marker="o", color="red", label="_nolegend_", alpha=0.8)
        # Highlight current point
        ax_right.scatter([rl_unique_areas[idx]], [rl_unique_delays[idx]], color="red", s=120, zorder=4, edgecolors="black", linewidths=1.5)
        ax_right.set_xlim(x_lim)
        ax_right.set_ylim(y_lim)
        # Add labels for all unique RL points (min w per point)
        for x, y, text in zip(rl_unique_areas, rl_unique_delays, rl_labels):
            ax_right.annotate(text, (x, y), textcoords="offset points", xytext=(5, 5), ha="left", fontsize=8)
        ax_right.legend()
        ax_right.set_xlabel("Area (um^2)")
        ax_right.set_ylabel("Delay (ns)")
        ax_right.set_title(f"Pareto Frontier for Baseline and RL-Optimized Adders")
        ax_right.grid(axis="y", linestyle="--", alpha=0.3)
        # Arrow from left subplot to current point on right subplot
        try:
            conn = ConnectionPatch(
                xyA=(0.98, 0.8), coordsA=ax_left.transAxes,
                xyB=(rl_unique_areas[idx], rl_unique_delays[idx]), coordsB=ax_right.transData,
                arrowstyle="->", mutation_scale=14, color="black", linewidth=1.2
            )
            fig.add_artist(conn)
        except Exception:
            pass
        frame_path = os.path.join(output_path, f"pareto_frame_{idx:03d}.png")
        plt.tight_layout()
        plt.savefig(frame_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)
    gif_path = os.path.join(output_path, gif_name)
    print("Creating GIF...")
    frames = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, frames, duration=len(frames)/2.0, loop=0)
    
        
def extract_verilog(verilog_dir, verilog_file_name, output_dir, w):
     if verilog_dir is None or not os.path.isdir(verilog_dir):
         print(f"Warning: verilog_dir '{verilog_dir}' is not a directory")
         return None
     
     # Common filename candidates
     candidates = [
         os.path.join(verilog_dir, verilog_file_name),
         os.path.join(verilog_dir, f"{verilog_file_name}.sv"),
         os.path.join(verilog_dir, f"{verilog_file_name}.v"),
     ]
     
     src_path = None
     for path in candidates:
         if os.path.isfile(path):
             src_path = path
             break
     
     # If not found, search recursively by basename (without extension)
     if src_path is None:
         target_stem = os.path.splitext(verilog_file_name)[0]
         for root, _, files in os.walk(verilog_dir):
             for fname in files:
                 stem, ext = os.path.splitext(fname)
                 if stem == target_stem and ext.lower() in {'.v', '.sv'}:
                     src_path = os.path.join(root, fname)
                     break
             if src_path is not None:
                 break
     
     if src_path is None:
        print(f"Warning: Verilog file '{verilog_file_name}' not found in '{verilog_dir}'")
        return None
    
     dest_dir = os.path.join(output_dir, "w_optimal_verilog")
     os.makedirs(dest_dir, exist_ok=True)
    
    # Rename copied file to adder_{w}.sv
     try:
        w_str = f"{float(w):.2f}"
     except Exception:
        w_str = str(w)
        
     dest_filename = f"adder_{w_str}.sv"
     dest_path = os.path.join(dest_dir, dest_filename)
     try:
         shutil.copy2(src_path, dest_path)
         print(f"Copied Verilog to: {dest_path}")
         return dest_path
     except Exception as e:
         print(f"Error copying Verilog file: {e}")
         return None
        
def main():
    args = parse_arguments()
    # verilog_dir = os.path.join(args.input_dir)
    min_score = extract_min_scalarized_graph(args.file_name, args.w_scalar, args.c_delay, args.c_area)
    feature_arrays = extract_feature_lists(args.input_dir, min_score['verilog_file_name'])
    plot_prefix_graph(feature_arrays['nodelist'], feature_arrays['minlist'], feature_arrays['levellist'], min_score['verilog_file_name'], args.plot_dir, args.w_scalar)
    plot_scalar_bars(args.n64, min_score['scalar'], args.w_scalar, args.c_delay, args.c_area, args.plot_dir)
    
    if args.pareto:
        min_scores = {}
        seen_verilog = set()
        for w in np.arange(0.0, 1.0, args.w_step):
            min_score = extract_min_scalarized_graph(args.file_name, w, args.c_delay, args.c_area)
            feature_arrays = extract_feature_lists(args.input_dir, min_score['verilog_file_name'])
            min_scores[w] = {'area': min_score['area'], 'delay': min_score['delay'], 'feature_arrays': feature_arrays}
            if args.extract_verilog:
                verilog_name = min_score['verilog_file_name']
                if verilog_name not in seen_verilog:
                    extract_verilog(args.verilog_dir, verilog_name, args.plot_dir, w)
                    seen_verilog.add(verilog_name)
                
        plot_scalar_pareto(args.n64, min_scores, args.plot_dir, args.c_delay, args.c_area, args.w_step)
        animate_pareto_with_graph(args.n64, min_scores, args.plot_dir, args.c_delay, args.c_area)
if __name__ == "__main__":
    main()