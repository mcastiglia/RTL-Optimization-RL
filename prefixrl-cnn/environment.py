import numpy as np
import copy
import os
import global_vars
import math
import hashlib
import subprocess
import shutil

step_num = 0

class Graph_State(object):
    def __init__(self, level, n, size, nodelist, levellist, minlist, step_num, action, reward, level_bound_delta):
        self.current_value = 0.0
        self.current_round_index = 0
        self.n = n
        self.cumulative_choices = []
        self.level = level
        self.nodelist = nodelist
        self.levellist = levellist
        self.fanoutlist = np.zeros((self.n, self.n), dtype=np.int8)
        self.minlist = minlist
        self.reward = reward
        self.size = size
        self.delay = None
        self.area = None
        self.level_bound_delta = level_bound_delta
        self.level_bound = int(math.log2(n) + 1 + level_bound_delta)
        
        # Check if the number of nodes is correct
        assert (self.nodelist.sum() - self.n) == self.size

        upper_triangle_mask = np.triu(np.ones((self.n, self.n), dtype = np.int8),
            k = 1)
        self.prob = np.ones((2, self.n, self.n), dtype = np.int8)
        self.prob[0] = np.where(self.nodelist >= 1.0, 0, self.prob[0])
        self.prob[0] = np.where(upper_triangle_mask >= 1.0, 0, self.prob[0])
        self.prob[1] = np.where(self.minlist <= 0.0, 0, self.prob[1])
        self.prob[1] = np.where(upper_triangle_mask >= 1.0, 0, self.prob[1])

        self.available_choicelist = []
        cnt = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.prob[1, i, j] == 1:
                    self.available_choicelist.append(self.n **2 +i* self.n+j)
                    cnt += 1
        for i in range(self.n):
            for j in range(self.n):
                if self.prob[0, i, j] == 1:
                    self.available_choicelist.append(i* self.n+j)
                    cnt += 1
                    
        self.choice_count = cnt
        self.action = action
        self.step_num = step_num
        
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
            assert self.minlist[x, y] == 1
            assert self.nodelist[x, y] == 1
            next_nodelist[x, y] = 0
            next_nodelist, next_minlist = self.legalize(next_nodelist, next_minlist)
            
        next_levellist = self.update_levellist(next_nodelist, next_levellist)
        
        return next_nodelist, next_minlist, next_levellist
    
    # Algorithm 1 (Legalize) from PrefixRL
    def legalize(self, nodelist, minlist):
        minlist = copy.deepcopy(nodelist)
        for i in range(self.n):
            minlist[i, 0] = 0
            minlist[i, i] = 0
        for m in range(self.n-1, 0, -1):
            prev_l = m
            for l in range(m-1, -1, -1):
                if nodelist[m, l] == 1:
                    nodelist[prev_l-1, l] = 1
                    minlist[prev_l-1, l] = 0
                    prev_l = l
        return nodelist, minlist

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

    # Update the available choice list of adding nodes based on nodelist and minlist (Taken from ArithTreeRL)
    def update_available_choice(self):
        upper_triangle_mask = np.triu(np.ones((self.n, self.n)), k = 1)
        prob = np.ones((2, self.n, self.n), dtype = np.int8)
        prob[0] = np.where(self.nodelist >= 1.0, 0, prob[0])
        prob[0] = np.where(upper_triangle_mask >= 1.0, 0, prob[0])
        prob[1] = np.where(self.minlist <= 0.0, 0, prob[1])
        prob[1] = np.where(upper_triangle_mask >= 1.0, 0, prob[1])
        
        self.available_choicelist = []
        cnt = 0
        for i in range(self.n):
            for j in range(self.n):
                if prob[1, i, j] == 1:
                    self.available_choicelist.append(self.n ** 2 + i * self.n + j)
                    cnt += 1
                    
        self.choice_count = cnt

    # Randomly select an action from the available choice list and return the next state (Adapted from ArithTreeRL)
    # TODO: Selection decision should come not from a random choice, but from the choice of the DQN
    def get_random_next_state(self):
        try_step = 0
        min_metric = 1e10
        while self.choice_count > 0 and try_step < 4:
            sample_prob = np.ones((self.choice_count))
            choice_idx = np.random.choice(self.choice_count, size = 1, replace=False, 
                    p = sample_prob/sample_prob.sum())[0]
            random_choice = self.available_choicelist[choice_idx]
            action_type = random_choice // (self.n ** 2)
            x = (random_choice % (self.n ** 2)) // self.n
            y = (random_choice % (self.n ** 2)) % self.n


            next_nodelist, next_minlist, next_levellist = self.modify_nodelist(action_type,x,y)
            
            # next_nodelist = copy.deepcopy(self.nodelist)
            # next_minlist = np.zeros((self.n, self.n))
            # next_levellist = np.zeros((self.n, self.n))
        
            # if action_type == 0:
            #     assert next_nodelist[x, y] == 0
            #     next_nodelist[x, y] = 1
            #     next_nodelist, next_minlist = legalize(self.n, next_nodelist, next_minlist)
            # elif action_type == 1:
            #     assert self.minlist[x, y] == 1
            #     assert self.nodelist[x, y] == 1
            #     next_nodelist[x, y] = 0
            #     next_nodelist, next_minlist = legalize(self.n, next_nodelist, next_minlist)
            # next_levellist = update_levellist(self.n, next_nodelist, next_levellist)
            
            next_level = next_levellist.max()
            next_size = next_nodelist.sum() - self.n
            next_step_num = self.step_num + 1
            action = random_choice
            reward = 0

            next_state = Graph_State(next_level, self.n, next_size, next_nodelist,
                next_levellist, next_minlist, next_step_num, action, reward, self.level_bound_delta)
            
            next_state.output_verilog()
            next_state.run_yosys()
            delay, area, power = next_state.run_openroad()
            global_step += 1
            # print("delay = {}, area = {}".format(delay, area))

            next_state.delay = delay
            next_state.area = area
            next_state.power = power
            next_state.update_fanout_map()
            fanout = next_state.fanout_map.max()
            
            print("try_step = {}".format(try_step))
            try_step += 1
            flog.write("{}\t{:.2f}\t{:.2f}\t{}\t{}\t{}\t{}\t{}\t{}\t{:d}\t{:.2f}\n".format(
                        next_state.verilog_file_name.split(".")[0], 
                        next_state.delay, next_state.area, next_state.power, 
                        int(next_state.level), int(next_state.size), fanout,
                        global_step, global_vars.cache_hit,
                        0, time.time() - start_time))
            record_num += 1
            flog.flush()
            
            print("record_num : {}/{}".format(record_num, args.step))
            
            # Exit the program if the step count is reached
            if record_num >= global_vars.step_count:
                sys.exit()
                
            # Use area + delay unless the initial adder type is serial
            candidate_min_metric = next_state.area + (next_state.delay if global_vars.initial_adder_type != 0 else 0)
            current_metric = self.area + (self.delay if global_vars.initial_adder_type != 0 else 0)
             
            # Update the best next state if the updated area/delay metric is less than the current min metric
            if candidate_min_metric < min_metric:
                best_next_state = copy.deepcopy(next_state)
                min_metric = candidate_min_metric
                    
            # If the updated area/delay metric is greater than the current metric, remove the action from the available choice list
            if candidate_min_metric > current_metric:
                self.available_choicelist.remove(random_choice)
                self.choice_count -=1
                assert self.choice_count == len(self.available_choicelist)
                # continue
            
            self.cumulative_choices.append(action)
            return next_state
        
        return best_next_state
    
    # Get the next state from the action and coordinates (x, y)
    def get_next_state(self, action, action_coord):
        next_nodelist, next_minlist, next_levellist = self.modify_nodelist(not action, action_coord[0], action_coord[1])
        next_level = next_levellist.max()
        next_size = next_nodelist.sum() - self.n
        next_step_num = self.step_num + 1
        action = action
        reward = 0
        
        next_state = Graph_State(next_level, self.n, next_size, next_nodelist,
            next_levellist, next_minlist, next_step_num, action, reward, self.level_bound_delta)
        
        next_state.output_verilog()
        next_state.run_yosys()
        delay, area, power = next_state.run_openroad()
        # global_step += 1
        # print("delay = {}, area = {}".format(delay, area))

        next_state.delay = delay
        next_state.area = area
        next_state.power = power
        next_state.update_fanoutlist()
        fanout = next_state.fanoutlist.max()
    
        return next_state
    
    # return best_next_state
  
    # Output the nodelist as ASCIIart to a file (Taken from ArithTreeRL)
    def output_nodelist(self):
        verilog_mid_dir = os.path.join(global_vars.output_dir, "run_verilog_mid")
        if not os.path.exists(verilog_mid_dir):
            os.mkdir(verilog_mid_dir)
        fdot_save = open(os.path.join(verilog_mid_dir, "adder_{}b_{}_{}_{}.log".format(self.n, 
                int(self.levellist.max()), int(self.nodelist.sum()-self.n),
                self.hash_value)), 'w')
        for i in range(self.n):
            for j in range(self.n):
                fdot_save.write("{}".format(str(int(self.nodelist[i, j]))))
            fdot_save.write("\n")
        fdot_save.write("\n")
        fdot_save.close()

    # Output the nodelist as Verilog code to a file (Taken from ArithTreeRL)
    def output_verilog(self,file_name = None):
        verilog_mid_dir = os.path.join(global_vars.output_dir, "run_verilog_mid")
        if not os.path.exists(verilog_mid_dir):
            os.mkdir(verilog_mid_dir)
            
        # Create a unique hash identifier for each adder state
        rep_int = self.get_represent_int()
        self.hash_value = hashlib.md5(str(rep_int).encode()).hexdigest()
        self.output_nodelist()
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
        src_file_path = os.path.join(global_vars.output_dir, "run_verilog_mid", self.verilog_file_name)

        yosys_script_dir = os.path.join(global_vars.output_dir, "run_yosys_script")
        if not os.path.exists(yosys_script_dir):
            os.mkdir(yosys_script_dir)
        yosys_script_file_name = os.path.join(yosys_script_dir, 
            "{}.ys".format(file_name_prefix))
        fopen = open(yosys_script_file_name, "w")
        fopen.write(global_vars.yosys_script_format.format(src_file_path, global_vars.openroad_path, dst_file_name))
        fopen.close()
        _ = subprocess.check_output(["yosys {}".format(yosys_script_file_name)], shell= True)
        if not global_vars.save_verilog:
            os.remove(src_file_path)
    
    # Run OpenROAD to perform place and route on the synthesized Verilog code (Taken from ArithTreeRL)
    def run_openroad(self):

        file_name_prefix = self.verilog_file_name.split(".")[0]
        
        # Check to see if results are cached
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
        
        # Copy the Yosys output to the OpenROAD test directory
        verilog_file_path = "{}adder_tmp_{}.v".format(global_vars.openroad_path, file_name_prefix)
        yosys_file_name = os.path.join(global_vars.output_dir, "run_yosys_mid", self.verilog_file_name.split(".")[0] + "_yosys.v")
        shutil.copyfile(yosys_file_name, verilog_file_path)
        
        sdc_file_path = "{}adder_nangate45_{}.sdc".format(global_vars.openroad_path, file_name_prefix)
        fopen_sdc = open(sdc_file_path, "w")
        fopen_sdc.write(global_vars.sdc_format)
        fopen_sdc.close()
        fopen_tcl = open("{}adder_nangate45_{}.tcl".format(global_vars.openroad_path, file_name_prefix), "w")
        fopen_tcl.write(global_vars.openroad_tcl.format("adder_tmp_{}.v".format(file_name_prefix), 
            "adder_nangate45_{}.sdc".format(file_name_prefix)))
        fopen_tcl.close()
        
        # Ensure openroad_path ends with '/' for consistent path handling
        if not global_vars.openroad_path.endswith('/'):
            global_vars.openroad_path = global_vars.openroad_path + '/'
            
        tcl_script = "adder_nangate45_{}.tcl".format(file_name_prefix)
        command = "openroad {}".format(tcl_script)
    
        # print("COMMAND: {}".format(command))
        output = subprocess.check_output(['openroad',
            "{}adder_nangate45_{}.tcl".format("", file_name_prefix)], 
            cwd=global_vars.openroad_path).decode('utf-8')
        note = None
        retry = 0
        area, wslack, power, note = self.extract_results(output)
        while note is None and retry < 3:
            output = subprocess.check_output(['openroad', tcl_script], 
            cwd=global_vars.openroad_path).decode('utf-8')
            area, wslack, power, note = self.extract_results(output)
            retry += 1
        if os.path.exists(yosys_file_name):
            os.remove(yosys_file_name)
        if os.path.exists("{}adder_nangate45_{}.tcl".format(global_vars.openroad_path, 
                file_name_prefix)):
            os.remove("{}adder_nangate45_{}.tcl".format(global_vars.openroad_path, file_name_prefix))
        if os.path.exists("{}adder_nangate45_{}.sdc".format(global_vars.openroad_path, 
                file_name_prefix)):
            os.remove("{}adder_nangate45_{}.sdc".format(global_vars.openroad_path, file_name_prefix))
        if os.path.exists("{}adder_tmp_{}.v".format(global_vars.openroad_path, 
                file_name_prefix)):
            os.remove("{}adder_tmp_{}.v".format(global_vars.openroad_path,file_name_prefix))
        delay = global_vars.CLOCK_PERIOD_TARGET - wslack
        delay *= 1000
        self.delay = delay
        self.area = area
        self.power = power
        global_vars.result_cache[hash_idx] = {"delay": delay, "area": area, "power": power}
        return delay, area, power
    
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
            print("line", line)
            if "design_area" in line:
                area = float(line.split(" = ")[-1])
            elif "worst_slack" in line:
                wslack = float(line.split(" = ")[-1])
                note = lines
            elif "Total" in line:
                power = float(line.split()[-2])

        return area, wslack, power, note