import numpy as np
import copy
import os
import global_vars
import math
import hashlib
import subprocess
import shutil
import torch
from typing import List, Tuple
import time
from multiprocessing import Pool, Lock
import multiprocessing
from typing import List
import matplotlib.pyplot as plt
import networkx as nx

_lock = Lock()
step_num = 0

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # start method already set

class Graph_State(object):
    def __init__(self, level, n, size, nodelist, levellist, minlist, level_bound_delta):
        self.n = n
        self.level = level
        self.nodelist = nodelist
        self.levellist = levellist
        self.fanoutlist = np.zeros((self.n, self.n), dtype=np.int8)
        self.minlist = minlist
        self.size = size
        self.delay = None
        self.analytic_delay = None
        self.area = None
        self.level_bound_delta = level_bound_delta
        self.level_bound = int(math.log2(n) + 1 + level_bound_delta)
        
        # Check if the number of nodes is correct
        assert (self.nodelist.sum() - self.n) == self.size
        
    # Get integer representation of the nodelist for hashing (Taken from ArithTreeRL)
    def get_represent_int(self):
        rep_int = 0
        for i in range(1, self.n):
            for j in range(i):
                if self.nodelist[i,j] == 1:
                    rep_int = rep_int * 2 + 1
                else:
                    rep_int *= 2
        self.rep_int = rep_int
        return rep_int
    
    # Add or delete a node from the nodelist - Algorithm 1 (0 for Add, 1 for Delete) from PrefixRL
    def modify_nodelist(self, action_type, x, y):
        next_nodelist = copy.deepcopy(self.nodelist)
        next_minlist = np.zeros((self.n, self.n))
        next_levellist = np.zeros((self.n, self.n))
            
        # Add a node to the nodelist
        if action_type == 0:
            # Node should not be added if it already exists
            assert next_nodelist[x, y] == 0
            next_nodelist[x, y] = 1
            next_nodelist, next_minlist = self.legalize(next_nodelist, next_minlist)
            
        # Delete a node from the nodelist
        elif action_type == 1:
            # TODO: Sometimes this assertion fails
            assert self.minlist[x, y] == 1
            assert self.nodelist[x, y] == 1
            next_nodelist[x, y] = 0
            next_nodelist, next_minlist = self.legalize(next_nodelist, next_minlist)
            
        next_levellist = self.update_levellist(next_nodelist, next_levellist)
        
        return next_nodelist, next_minlist, next_levellist
    
    def legalize(self, cell_map, min_map):
        min_map = copy.deepcopy(cell_map)
        for i in range(self.n):
            min_map[i, 0] = 0
            min_map[i, i] = 0
        for x in range(self.n-1, 0, -1):
            last_y = x
            for y in range(x-1, -1, -1):
                if cell_map[x, y] == 1:
                    cell_map[last_y-1, y] = 1
                    min_map[last_y-1, y] = 0
                    last_y = y
        return cell_map, min_map

    # Update levellist based on changes to nodelist (Taken from ArithTreeRL)
    def update_levellist(self, nodelist, levellist):
        levellist[1:].fill(0)
        levellist[0, 0] = 1
        for m in range(1, self.n):
            levellist[m, m] = 1
            prev_l = m
            for l in range(m-1, -1, -1):
                if nodelist[m, l] == 1:
                    levellist[m, l] = max(levellist[m, prev_l], levellist[prev_l-1, l])+ 1
                    prev_l = l
        return levellist
                    
    # Update fanoutlist based on changes to nodelist (Taken from ArithTreeRL)
    def update_fanoutlist(self):
        self.fanoutlist.fill(0)
        self.fanoutlist[0,0] = 0
        
        for m in range(1, self.n):
            self.fanoutlist[m,m] = 0
            prev_l = m
            
            for l in range(m-1, -1, -1):
                if self.nodelist[m,l] == 1:
                    self.fanoutlist[prev_l-1,l] += 1
                    self.fanoutlist[m, prev_l] += 1
                    prev_l = l
    
    # Get the next state from the action and coordinates (x, y)
    #def evaluate_next_state(self, action_type, x, y, batch_idx: int):
        #start_time = time.time()
        #next_nodelist, next_minlist, next_levellist = self.modify_nodelist(not action_type, x, y)
        #next_level = next_levellist.max()
        #next_size = next_nodelist.sum() - self.n
        
        #next_state = Graph_State(next_level, self.n, next_size, next_nodelist,
            #next_levellist, next_minlist, self.level_bound_delta)
        
        #next_state.update_fanoutlist()
        #fanout = next_state.fanoutlist.max()
        #next_state.level = next_state.levellist.max()
        #next_state.output_verilog()
        
        # Perform synthesis and PnR if not using analytic model
        #if not global_vars.use_analytic_model:
            #next_state.run_yosys()
            #delay, area, power = next_state.run_openroad(batch_idx)

            #next_state.delay = delay
            #next_state.area = area
            #next_state.power = power
        
            #global_vars.synthesis_log.write("{},{:.2f},{:.2f},{:.2f},{:d},{:d},{:d},{:d},{:.2f}\n".format(
                    #next_state.verilog_file_name.split(".")[0], 
                    #next_state.delay, next_state.area, next_state.power, 
                    #int(next_state.level), int(next_state.size), fanout,
                    #global_vars.cache_hit,time.time() - start_time))
            #global_vars.synthesis_log.flush()
        #else:
            #next_state.compute_critical_path_delay()
            #global_vars.synthesis_log.write("{},{:.2f},{:.2f},{:.2f},{:d},{:d},{:d},{:d},{:.2f}\n".format(
                    #next_state.verilog_file_name.split(".")[0], 
                    #next_state.analytic_delay, int(next_state.size), 0, 
                    #int(next_state.level), int(next_state.size), fanout,
                    #global_vars.cache_hit,time.time() - start_time))
            #global_vars.synthesis_log.flush()
            
        #return next_state
    
    
    def evaluate_next_state(self, action_type, x, y, batch_idx: int):
        start_time = time.time()
        next_nodelist, next_minlist, next_levellist = self.modify_nodelist(not action_type, x, y)
        next_level = next_levellist.max()
        next_size = next_nodelist.sum() - self.n

        next_state = Graph_State(
            next_level, self.n, next_size, next_nodelist,
            next_levellist, next_minlist, self.level_bound_delta
        )

        next_state.update_fanoutlist()
        fanout = next_state.fanoutlist.max()
        next_state.level = next_state.levellist.max()
        next_state.output_verilog()

        if not global_vars.use_analytic_model:
            next_state.run_yosys()
            delay, area, power = next_state.run_openroad(batch_idx)
    
            next_state.delay = delay
            next_state.area = area
            next_state.power = power
    
            safe_delay = delay if delay is not None else 1e5
            safe_area  = area  if area  is not None else 1e5
            safe_power = power if power is not None else 1e5
    
            global_vars.synthesis_log.write(
                "{},{:.2f},{:.2f},{:.2f},{:d},{:d},{:d},{:d},{:.2f}\n".format(
                    next_state.verilog_file_name.split(".")[0],
                    safe_delay, safe_area, safe_power,
                    int(next_state.level), int(next_state.size),
                    fanout, global_vars.cache_hit,
                    time.time() - start_time
                )
            )
            global_vars.synthesis_log.flush()
    
        else:
            next_state.compute_critical_path_delay()
    
            safe_delay = next_state.analytic_delay if next_state.analytic_delay is not None else 0.0
    
            global_vars.synthesis_log.write(
                "{},{:.2f},{:.2f},{:.2f},{:d},{:d},{:d},{:d},{:.2f}\n".format(
                    next_state.verilog_file_name.split(".")[0],
                    safe_delay, float(next_state.size), 0.0,
                    int(next_state.level), int(next_state.size),
                    fanout, global_vars.cache_hit,
                    time.time() - start_time
                )
            )
            global_vars.synthesis_log.flush()
    
        return next_state

  
    """
    Compute the critical path delay of the adder
    Start by finding all node coordinates with a maximum number of levels
    Then do a depth-first search of each path
    For each search, keep a running total of the cumulative fanout
    
    delay_total = internal_delay + driving_delay + final_xor_delay, 
    internal_delay = max_levels * (2 * D_I), D_I = 1.0
    driving_delay = D_0 * sum(fanouts of all nodes on the critical path), D_0 = 0.5
    final_xor_delay = 1.0
    
    Example: 32-bit Kogge-Stone adder
    max_levels = log2(32) = 5
    internal_delay = 5 * (2 * 1.0) = 10.0
    driving_delay = 0.5 * (2+2+2+2+1) = 4.5
    final_xor_delay = 1.0
    delay_total = 10.0 + 4.5 + 1.0 = 15.5
    """
    def compute_critical_path_delay(self, D_I: float = 1.0, D_0: float = 0.5):
        max_levels = int(self.levellist.max())
        internal_delay = max_levels * (2 * D_I)
        final_xor_delay = 1.0
        n = self.n
        # Ensure fanoutlist matches current nodelist
        self.update_fanoutlist()
        # Helper to find lower parent of a node (x, y)
        def find_lower_parent(x: int, y: int):
            last_y = x
            for y_above in range(x-1, y, -1):
                if self.nodelist[x, y_above] == 1:
                    last_y = y_above
                    break
            lp_x = last_y - 1
            if lp_x < 0:
                return None
            if self.nodelist[lp_x, y] != 1:
                return None
            return (lp_x, y)
        # Gather all max-level nodes
        sinks = []
        for i in range(n):
            for j in range(n):
                if self.nodelist[i, j] == 1 and int(self.levellist[i, j]) == max_levels:
                    sinks.append((i, j))
        if not sinks:
            return final_xor_delay
        # Traverse lower-parent chains and sum fanouts along the chain
        best_total_fanout = 0
        for sx, sy in sinks:
            total_fanout = 0
            current = (sx, sy)
            # Move down until reaching level 1
            lp = find_lower_parent(current[0], current[1])
            if lp is None:
                break
            lp_level = int(self.levellist[lp[0], lp[1]])
            total_fanout = int(self.fanoutlist[lp[0], lp[1]]) + (max_levels - 2)
            # print(f"Fanout at level {lp_level}: {int(self.fanoutlist[lp[0], lp[1]])}, total fanout: {total_fanout}")
            if total_fanout > best_total_fanout:
                best_total_fanout = total_fanout
                
        # print(f"Best total fanout: {best_total_fanout}")
        driving_delay = D_0 * best_total_fanout
        # print(f"Driving delay: {driving_delay}")
        delay_total = internal_delay + driving_delay + final_xor_delay
        # print(f"Total analytic delay: {delay_total}")
        self.analytic_delay = delay_total
        return delay_total
        
        
    # Output the nodelist as ASCIIart to a file (Taken from ArithTreeRL)
    def output_feature_list(self, feature_name, feature_list):
        featurelist_dir = os.path.join(global_vars.output_dir, "graph_feature_lists")
        if not os.path.exists(featurelist_dir):
            os.mkdir(featurelist_dir)
        fdot_save = open(os.path.join(featurelist_dir, "adder_{}b_{}_{}_{}_{}.log".format(self.n, 
                int(self.levellist.max()), int(self.nodelist.sum()-self.n), feature_name,
                self.hash_value)), 'w')
        for i in range(self.n):
            for j in range(self.n):
                fdot_save.write("{}\t".format(str(int(feature_list[i, j]))))
            fdot_save.write("\n")
        fdot_save.write("\n")
        fdot_save.close()


    def plot_prefix_graph(self):
        if not hasattr(self, 'hash_value'):
            rep_int = self.get_represent_int()
            self.hash_value = hashlib.md5(str(rep_int).encode()).hexdigest()
        
        graph_plots_dir = os.path.join(global_vars.output_dir, "graph_plots")
        if not os.path.exists(graph_plots_dir):
            os.mkdir(graph_plots_dir)
        plot_title = os.path.join(graph_plots_dir, "adder_{}b_{}_{}_graph_{}.png".format(self.n, 
                int(self.levellist.max()), int(self.nodelist.sum()-self.n),
                self.hash_value))
        plt.clf()
        G = nx.DiGraph()
        
        nodes = []
        for i in range(self.n):
            for j in range(self.n):
                if self.nodelist[i, j] == 1:
                    # nodes.append((int(self.levellist[i,j])-1,j))
                    nodes.append((i, int(self.levellist[i,j])-1))
        
        G.add_nodes_from(nodes)
        
        edges = []
        for x in range(self.n-1, 0, -1):
            last_y = x
            for y in range(x-1, -1, -1):
                if self.nodelist[x, y] == 1:
                    level = int(self.levellist[x,y])-1
                    # Upper parent
                    if self.nodelist[x, last_y] == 1:
                        last_y_level = int(self.levellist[x,last_y])-1
                        edges.append(((x, last_y_level), (x, level)))
                        
                    # Lower parent
                    if self.nodelist[last_y-1, y] == 1:
                        edges.append(((last_y-1, level-1), (x, level)))
                    last_y = y
        
        G.add_edges_from(edges)
        
        pos = {(r, c): (r,c) for r in range(self.n) for c in range(self.n)}
            
        nx.draw(
            G, 
            pos,
            with_labels=False,
            node_size=1024 // self.n,
            edgecolors="black",
            linewidths=1,
            edge_color="gray",
            arrows=False,
            arrowsize=100 // self.n,
        )
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        
        plt.savefig(plot_title, dpi=300, bbox_inches="tight")
    
    # Output the nodelist as Verilog code to a file (Taken from ArithTreeRL)
    def output_verilog(self,file_name = None):
        verilog_mid_dir = os.path.join(global_vars.output_dir, "run_verilog_mid")
        if not os.path.exists(verilog_mid_dir):
            os.mkdir(verilog_mid_dir)
            
        # Create a unique hash identifier for each adder state
        rep_int = self.get_represent_int()
        self.hash_value = hashlib.md5(str(rep_int).encode()).hexdigest()
        self.output_feature_list("nodelist", self.nodelist)
        self.output_feature_list("levellist", self.levellist)
        self.output_feature_list("minlist", self.minlist)
        self.output_feature_list("fanoutlist", self.fanoutlist)
        if file_name is None:
            file_name = "run_verilog_mid/adder_{}b_{}_{}_{}.v".format(self.n, 
                int(self.levellist.max()), int(self.nodelist.sum()-self.n),
                self.hash_value)
        self.verilog_file_name = file_name.split("/")[-1]

        verilog_file = open(os.path.join(global_vars.output_dir, file_name), "w")
        verilog_file.write("module adder_top(a,b,s,cout);\n")
        verilog_file.write("input [{}:0] a,b;\n".format(self.n-1))
        verilog_file.write("output [{}:0] s;\n".format(self.n-1))
        verilog_file.write("output cout;\n")
        wires = set()
        for i in range(self.n):
            wires.add("c{}".format(i))
        
        for x in range(self.n-1, 0, -1):
            last_y = x
            for y in range(x-1, -1, -1):
                if self.nodelist[x, y] == 1:
                    assert self.nodelist[last_y-1, y] == 1
                    if y==0:
                        wires.add("g{}_{}".format(x, last_y))
                        wires.add("p{}_{}".format(x, last_y))
                        wires.add("g{}_{}".format(last_y-1, y))
                    else:
                        wires.add("g{}_{}".format(x, last_y))
                        wires.add("p{}_{}".format(x, last_y))
                        wires.add("g{}_{}".format(last_y-1, y))
                        wires.add("p{}_{}".format(last_y-1, y))
                        wires.add("g{}_{}".format(x, y))
                        wires.add("p{}_{}".format(x, y))
                    last_y = y
        
        for i in range(self.n):
            wires.add("p{}_{}".format(i, i))
            wires.add("g{}_{}".format(i, i))
            wires.add("c{}".format(x))
        assert 0 not in wires
        assert "0" not in wires
        verilog_file.write("wire ")
        
        for i, wire in enumerate(wires):
            if i < len(wires) - 1:
                    verilog_file.write("{},".format(wire))
            else:
                verilog_file.write("{};\n".format(wire))
        verilog_file.write("\n")
        
        for i in range(self.n):
            verilog_file.write('assign p{}_{} = a[{}] ^ b[{}];\n'.format(i,i,i,i))
            verilog_file.write('assign g{}_{} = a[{}] & b[{}];\n'.format(i,i,i,i))
        
        for i in range(1, self.n):
            verilog_file.write('assign g{}_0 = c{};\n'.format(i, i))
        
        for x in range(self.n-1, 0, -1):
            last_y = x
            for y in range(x-1, -1, -1):
                if self.nodelist[x, y] == 1:
                    assert self.nodelist[last_y-1, y] == 1
                    if y == 0: # add grey module
                        verilog_file.write('GREY grey{}(g{}_{}, p{}_{}, g{}_{}, c{});\n'.format(
                            x, x, last_y, x, last_y, last_y-1, y, x
                        ))
                    else:
                        verilog_file.write('BLACK black{}_{}(g{}_{}, p{}_{}, g{}_{}, p{}_{}, g{}_{}, p{}_{});\n'.format(
                            x, y, x, last_y, x, last_y, last_y-1, y, last_y-1, y, x, y, x, y 
                        ))
                    last_y = y
        
        verilog_file.write('assign s[0] = a[0] ^ b[0];\n')
        verilog_file.write('assign c0 = g0_0;\n')
        verilog_file.write('assign cout = c{};\n'.format(self.n-1))
        for i in range(1, self.n):
            verilog_file.write('assign s[{}] = p{}_{} ^ c{};\n'.format(i, i, i, i-1))
        verilog_file.write("endmodule")
        verilog_file.write("\n\n")

        verilog_file.write(global_vars.GREY_CELL)
        verilog_file.write("\n")
        verilog_file.write(global_vars.BLACK_CELL)
        verilog_file.write("\n")
        verilog_file.close()

    # Run Yosys to synthesize the Verilog code (Taken from ArithTreeRL)
    def run_yosys(self):
        yosys_mid_dir = os.path.join(global_vars.output_dir, "run_yosys_mid")
        if not os.path.exists(yosys_mid_dir):
            os.mkdir(yosys_mid_dir)
        dst_file_name = os.path.join(yosys_mid_dir, self.verilog_file_name.split(".")[0] + "_yosys.v")
        file_name_prefix = self.verilog_file_name.split(".")[0] + "_yosys"
        if os.path.exists(dst_file_name):
            return
        #     # os.remove(dst_file_name)
        src_file_path = os.path.join(global_vars.output_dir, "run_verilog_mid", self.verilog_file_name)

        yosys_script_dir = os.path.join(global_vars.output_dir, "run_yosys_script")
        if not os.path.exists(yosys_script_dir):
            os.mkdir(yosys_script_dir)
        yosys_script_file_name = os.path.join(yosys_script_dir, 
            "{}.ys".format(file_name_prefix))
        fopen = open(yosys_script_file_name, "w")
        fopen.write(global_vars.yosys_script_format.format(src_file_path, global_vars.openroad_path, dst_file_name))
        # print_stat_command = global_vars.PRINT_STAT_COMMAND.format(global_vars.openroad_path)
        # fopen.write(global_vars.yosys_script_format.format(src_file_path, global_vars.YOSYS_OPTIMIZATION_EFFORT, global_vars.openroad_path, print_stat_command, dst_file_name))
        fopen.close()
        try:
            output = subprocess.check_output(
                ["yosys {}".format(yosys_script_file_name)], 
                shell=True,
                timeout=300,
                stderr=subprocess.STDOUT,
            ).decode("utf-8")
            # print(output, flush=True)
            # for output_line in output.split("\n"):
            #     if "Chip area" in output_line:
            #         print(output_line, flush=True)
            #         break
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Yosys timed out after 300 seconds for {yosys_script_file_name}")
        except subprocess.CalledProcessError as e:
            print("Yosys error: ", e.output, flush=True)
            raise RuntimeError(f"Yosys failed for {yosys_script_file_name}")
        if not global_vars.save_verilog:
            os.remove(src_file_path)
    
    def run_openroad(self, batch_idx: int = 0):

      file_name_prefix = self.verilog_file_name.split(".")[0]
  
      # Check cache
      hash_idx = file_name_prefix.split("_")[-1]
      if hash_idx in global_vars.result_cache:
          delay = global_vars.result_cache[hash_idx]["delay"]
          area = global_vars.result_cache[hash_idx]["area"]
          power = global_vars.result_cache[hash_idx]["power"]
          global_vars.cache_hit += 1
          self.delay = delay
          self.area = area
          self.power = power
          return delay, area, power
  
      # Copy Yosys output to OpenROAD directory
      verilog_file_path = f"{global_vars.openroad_path}adder_tmp_{file_name_prefix}.v"
      yosys_file_name = os.path.join(
          global_vars.output_dir,
          "run_yosys_mid",
          self.verilog_file_name.split(".")[0] + "_yosys.v"
      )
      shutil.copyfile(yosys_file_name, verilog_file_path)
  
      # Write SDC
      sdc_file_path = f"{global_vars.openroad_path}adder_nangate45_{file_name_prefix}.sdc"
      with open(sdc_file_path, "w") as fopen_sdc:
          fopen_sdc.write(global_vars.sdc_format)
  
      # Write TCL
      tcl_path = f"{global_vars.openroad_path}adder_nangate45_{file_name_prefix}.tcl"
      with open(tcl_path, "w") as fopen_tcl:
          fopen_tcl.write(global_vars.openroad_tcl.format(
              f"adder_tmp_{file_name_prefix}.v",
              f"adder_nangate45_{file_name_prefix}.sdc",
              batch_idx,
              global_vars.flow_type
          ))
  
      if not global_vars.openroad_path.endswith('/'):
          global_vars.openroad_path += '/'
  
      tcl_script = f"adder_nangate45_{file_name_prefix}.tcl"
  
      # ------------------------------
      # NEW LOGIC: 30 sec timeout, retry 3 times
      # ------------------------------
      MAX_RETRIES = 3
      TIMEOUT = 30
      output = None
  
      for attempt in range(1, MAX_RETRIES + 1):
          try:
              output = subprocess.check_output(
                  ['openroad', tcl_script],
                  cwd=global_vars.openroad_path,
                  timeout=TIMEOUT,
                  stderr=subprocess.STDOUT
              ).decode('utf-8')
  
              # Try extracting results
              area, wslack, power, note = self.extract_results(output)
  
              # If valid result ? break
              if note is not None:
                  break
  
          except subprocess.TimeoutExpired:
              print(f"[OpenROAD] Timeout on attempt {attempt} for {file_name_prefix}")
  
          except Exception as e:
              print(f"[OpenROAD] Error on attempt {attempt}: {e}")
  
          # If failed extraction, loop continues to retry
  
      else:
          # ------------------------------------------------
          # All 3 attempts failed ? give up on graph safely
          # ------------------------------------------------
          print(f"[OpenROAD] FAILED after {MAX_RETRIES} attempts ? skipping graph {file_name_prefix}")
          self.delay = 1e5
          self.area = 1e5
          self.power = 1e5
          return 1e5, 1e5, 1e5
  
      # ------------------------------
      # Extraction succeeded
      # ------------------------------
      delay = global_vars.CLOCK_PERIOD_TARGET - wslack
      self.delay = delay
      self.area = area
      self.power = power
  
      # Cache it
      global_vars.result_cache[hash_idx] = {
          "delay": delay,
          "area": area,
          "power": power
      }
  
      # Clean temp files
      for path in [
          yosys_file_name,
          tcl_path,
          sdc_file_path,
          verilog_file_path
      ]:
          if os.path.exists(path):
              os.remove(path)
  
      return delay, area, power

            
    # Run OpenROAD to perform place and route on the synthesized Verilog code (Taken from ArithTreeRL)
    #def run_openroad(self, batch_idx: int = 0):

        #file_name_prefix = self.verilog_file_name.split(".")[0]
        
        # Check to see if results are cached
        #hash_idx = file_name_prefix.split("_")[-1]
        #if hash_idx in global_vars.result_cache:
            #delay = global_vars.result_cache[hash_idx]["delay"]
            #area = global_vars.result_cache[hash_idx]["area"]
            #power = global_vars.result_cache[hash_idx]["power"]
            #global_vars.cache_hit += 1
            #self.delay = delay
            #self.area = area
            #self.power = power
            #return delay, area, power
        
        # Copy the Yosys output to the OpenROAD test directory
        #verilog_file_path = "{}adder_tmp_{}.v".format(global_vars.openroad_path, file_name_prefix)
        #yosys_file_name = os.path.join(global_vars.output_dir, "run_yosys_mid", self.verilog_file_name.split(".")[0] + "_yosys.v")
        #shutil.copyfile(yosys_file_name, verilog_file_path)
        
        #sdc_file_path = "{}adder_nangate45_{}.sdc".format(global_vars.openroad_path, file_name_prefix)
        #fopen_sdc = open(sdc_file_path, "w")
        #fopen_sdc.write(global_vars.sdc_format)
        #fopen_sdc.close()
        #fopen_tcl = open("{}adder_nangate45_{}.tcl".format(global_vars.openroad_path, file_name_prefix), "w")
        #fopen_tcl.write(global_vars.openroad_tcl.format("adder_tmp_{}.v".format(file_name_prefix), 
            #"adder_nangate45_{}.sdc".format(file_name_prefix), batch_idx, global_vars.flow_type))
        #fopen_tcl.close()
        
        # Ensure openroad_path ends with '/' for consistent path handling
        #if not global_vars.openroad_path.endswith('/'):
            #global_vars.openroad_path = global_vars.openroad_path + '/'
            
        #tcl_script = "adder_nangate45_{}.tcl".format(file_name_prefix)
        #command = "openroad {}".format(tcl_script)
    
        # print("COMMAND: {}".format(command))
        #try:
            #output = subprocess.check_output(
                #['openroad', "{}adder_nangate45_{}.tcl".format("", file_name_prefix)], 
                #cwd=global_vars.openroad_path,
                #timeout=600,
                #stderr=subprocess.STDOUT
            #).decode('utf-8')
        #except subprocess.TimeoutExpired:
            #raise RuntimeError(f"OpenROAD timed out after 600 seconds for {file_name_prefix}")
        
        #note = None
        #retry = 0
        #area, wslack, power, note = self.extract_results(output)
        #while note is None and retry < 3:
            #try:
                #output = subprocess.check_output(
                    #['openroad', tcl_script], 
                    #cwd=global_vars.openroad_path,
                    #timeout=600,
                    #stderr=subprocess.STDOUT
                #).decode('utf-8')
            #except subprocess.TimeoutExpired:
                #raise RuntimeError(f"OpenROAD retry {retry+1} timed out after 600 seconds for {file_name_prefix}")
            #area, wslack, power, note = self.extract_results(output)
            #retry += 1
        #if os.path.exists(yosys_file_name):
            #os.remove(yosys_file_name)
        #if os.path.exists("{}adder_nangate45_{}.tcl".format(global_vars.openroad_path, 
                #file_name_prefix)):
            #os.remove("{}adder_nangate45_{}.tcl".format(global_vars.openroad_path, file_name_prefix))
        #if os.path.exists("{}adder_nangate45_{}.sdc".format(global_vars.openroad_path, 
                #file_name_prefix)):
            #os.remove("{}adder_nangate45_{}.sdc".format(global_vars.openroad_path, file_name_prefix))
        #if os.path.exists("{}adder_tmp_{}.v".format(global_vars.openroad_path, 
                #file_name_prefix)):
            #os.remove("{}adder_tmp_{}.v".format(global_vars.openroad_path,file_name_prefix))
        #delay = global_vars.CLOCK_PERIOD_TARGET - wslack
        # TODO: Removed the multiplier of 1000 to adjust these units to be in ns
        # delay *= 1000
        #self.delay = delay
        #self.area = area
        #self.power = power
        #global_vars.result_cache[hash_idx] = {"delay": delay, "area": area, "power": power}
        #return delay, area, power
    
    # Parse reports from OpenROAD and return area, timing, and power metrics (Adapted from ArithTreeRL)
    def extract_results(self, openroad_output):
        lines = openroad_output.split("\n")[-15:]
        area = -100.0
        wslack = -100.0
        power = 0.0
        note = None
        for line in lines:
            if not line.startswith("result:") and not line.startswith("Total"):
                continue
            # print(line, flush=True)
            if "design_area" in line:
                area = float(line.split(" = ")[-1])
            elif "worst_slack" in line:
                wslack = float(line.split(" = ")[-1])
                note = lines
            elif "Total" in line:
                power = float(line.split()[-2])

        return area, wslack, power, note
    

def evaluate_job(args):
  b, current_states, best_action, action_x, action_y = args
  
#   print(f"[Batch {b}] Starting evaluation")
  
  action_type = best_action[b].item()
  x = action_x[b].item()
  y = action_y[b].item()
  
  next_state = current_states[b].evaluate_next_state(action_type, x, y, b)
  
#   with _lock:
#     print(f"[Batch {b}] Finished evaluation: {next_state.verilog_file_name}")
  
  return next_state
    
# Evaluate the next state metrics for each batch element
# TODO: should be performed in parallel instead of sequentially
def evaluate_next_state_sequential(current_states: List[Graph_State], best_action: torch.Tensor, action_x: torch.Tensor, action_y: torch.Tensor, batch_size: int):
    next_states: List[Graph_State] = []
    for b in range(batch_size):
        action_type = best_action[b].item()
        x = action_x[b].item()
        y = action_y[b].item()
        
        next_state = current_states[b].evaluate_next_state(action_type, x, y, b)
        next_states.append(next_state)
        
    return next_states
    
def evaluate_next_state_parallel(current_states: List["Graph_State"], best_action: torch.Tensor, action_x: torch.Tensor, action_y: torch.Tensor, batch_size: int):
  # Move tensors to CPU to avoid CUDA issues in multiprocessing
  best_action_cpu = best_action.cpu()
  action_x_cpu = action_x.cpu()
  action_y_cpu = action_y.cpu()
  args = [(b, current_states, best_action_cpu, action_x_cpu, action_y_cpu) for b in range (batch_size)]

  num_workers = max(1, min(os.cpu_count() - 1, batch_size))
#   print(f"Starting evaluation with {num_workers} worker(s) for {batch_size} batch elements...")
  
  with Pool(processes=num_workers) as pool:
    next_states = pool.map(evaluate_job, args)

  return next_states

def evaluate_next_state_batch(current_states: List["Graph_State"], best_action: torch.Tensor, action_x: torch.Tensor, action_y: torch.Tensor, batch_size: int):
    if global_vars.disable_parallel_evaluation or batch_size == 1:
        return evaluate_next_state_sequential(current_states, best_action, action_x, action_y, batch_size)
    else:
        return evaluate_next_state_parallel(current_states, best_action, action_x, action_y, batch_size)