import argparse
import time
import os
import global_vars
from init_states import init_graph
from q_network import PrefixRL_DQN, build_features, build_action_masks, apply_action_masks, scalarize_q, argmax_action, get_best_action, TrainingConfig, train
from plotting_utils import print_section_header, print_title_banner, GRAPH_TO_GATES_TITLE, SEPARATOR, print_info_formatted

def parse_arguments():
    global args
    
    parser = argparse.ArgumentParser(description='Prefix graph to adder netlist conversion tool')
    parser.add_argument('-n','--input_bitwidth', type = int, required=True, help="Input bitwidth for the adder")
    parser.add_argument('--adder_type', type = int, required=True, help="Initial starting state for prefix graph (0: serial, 1: sklansky, 2: brent-kung)")
    parser.add_argument('-b', '--batch_size', type = int, default = 192, help="Batch size for RL training")
    parser.add_argument('--num_steps', type = int, default = 5000, help="Number of training steps per episode")
    parser.add_argument('--num_episodes', type = int, default = 100, help="Number of training episodes")
    parser.add_argument('--w_scalar', type = float, default = 0.5, help="Weight scalar for area and delay (w_area = w_scalar, w_delay = 1 - w_scalar)")
    parser.add_argument('--openroad_path', type = str, default = '../OpenROAD/prefix-flow/')
    parser.add_argument('--flow_type', type = str, default = 'fast_flow', choices=['fast_flow', 'full_flow'], help="Flow type for OpenROAD (fast_flow or full_flow)")
    parser.add_argument('--output_dir', type = str, default = 'out/', help='Output directory for generated files')
    parser.add_argument('--save_verilog', action = 'store_true', default = False, help="Save the generated Verilog files")
    parser.add_argument('--disable_parallel_evaluation', action = 'store_true', default = False, help="Enable parallel synthesis and PnR for next state evaluation")
    parser.add_argument('--restore_from', type=str, default=None, help="Path to checkpoint file to restore from")
    
    args = parser.parse_args()
    
    print(SEPARATOR)
    print_section_header("ARGUMENT CONFIGURATION")
    print(SEPARATOR)
    print_info_formatted("Input bitwidth", str(args.input_bitwidth))
    print_info_formatted("Adder type", "RCA" if args.adder_type == 0 else "Sklansky" if args.adder_type == 1 else "Brent-Kung")
    print_info_formatted("Number of training steps per episode", str(args.num_steps))
    print_info_formatted("Number of training episodes", str(args.num_episodes))
    print_info_formatted("Weight scalar for area and delay", str(args.w_scalar))
    print_info_formatted("Batch size", str(args.batch_size))
    print_info_formatted("OpenROAD path", args.openroad_path)
    print_info_formatted("Flow type", args.flow_type)
    print_info_formatted("Output directory", args.output_dir)
    print_info_formatted("Save Verilog", str(args.save_verilog))
    print_info_formatted("Parallel evaluation", "Disabled" if args.disable_parallel_evaluation else "Enabled")
    print_info_formatted("Restore from", args.restore_from if args.restore_from else "None")
    print(SEPARATOR)

    global_vars.initial_adder_type = args.adder_type
    global_vars.openroad_path = args.openroad_path
    global_vars.flow_type = args.flow_type
    global_vars.output_dir = args.output_dir
    global_vars.n = args.input_bitwidth # number of nodes in the adder
    global_vars.num_steps = args.num_steps
    global_vars.num_episodes = args.num_episodes
    global_vars.w_scalar = args.w_scalar
    global_vars.batch_size = args.batch_size
    global_vars.save_verilog = args.save_verilog
    global_vars.disable_parallel_evaluation = args.disable_parallel_evaluation
    global_vars.restore_from = args.restore_from
    strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    
    # Create output directory structure
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, "adder_training_log")):
        os.mkdir(os.path.join(args.output_dir, "adder_training_log"))
    if not os.path.exists(os.path.join(args.output_dir, "adder_training_log/adder_{}b".format(args.input_bitwidth))):
        os.mkdir(os.path.join(args.output_dir, "adder_training_log/adder_{}b".format(args.input_bitwidth)))
    global_vars.synthesis_log = open(os.path.join(args.output_dir, "adder_training_log/adder_{}b/adder_{}b_openroad_type{}_{}.csv".format(args.input_bitwidth, 
        args.input_bitwidth, args.adder_type, strftime)), "w")
    global_vars.training_log = open(os.path.join(args.output_dir, "adder_training_log/adder_{}b/adder_{}b_training_{}.csv".format(args.input_bitwidth, 
        args.input_bitwidth, strftime)), "w")
    global_vars.synthesis_log.write("verilog_file_name,delay,area,power,level,size,fanout,cache_hit,time\n")
    global_vars.training_log.write("timestamp,episode,step,action,action_x,action_y,reward,bellman_target,expected_q,expected_q_next,loss\n")
    global_vars.start_time = time.time()
    
    return args

def log_initial_states():
    serial_adder = init_graph(global_vars.n, 0)
    serial_adder.output_verilog()
    serial_adder.output_feature_list("nodelist", serial_adder.nodelist)
    serial_adder.output_feature_list("levellist", serial_adder.levellist)
    serial_adder.output_feature_list("minlist", serial_adder.minlist)
    serial_adder.output_feature_list("fanoutlist", serial_adder.fanoutlist)
    serial_adder.run_yosys()
    delay,area,power = serial_adder.run_openroad()
    serial_adder.plot_prefix_graph()
    print("Serial adder delay: ", delay)
    print("Serial adder area: ", area)
    print("Serial adder power: ", power)
    
    sklansky_adder = init_graph(global_vars.n, 1)
    sklansky_adder.output_verilog()
    sklansky_adder.output_feature_list("nodelist", sklansky_adder.nodelist)
    sklansky_adder.output_feature_list("levellist", sklansky_adder.levellist)
    sklansky_adder.output_feature_list("minlist", sklansky_adder.minlist)
    sklansky_adder.output_feature_list("fanoutlist", sklansky_adder.fanoutlist)
    sklansky_adder.run_yosys()
    delay,area,power = sklansky_adder.run_openroad()
    sklansky_adder.plot_prefix_graph()
    print("Sklansky adder delay: ", delay)
    print("Sklansky adder area: ", area)
    print("Sklansky adder power: ", power)
    
    brent_kung_adder = init_graph(global_vars.n, 2)
    brent_kung_adder.output_verilog()
    brent_kung_adder.output_feature_list("nodelist", brent_kung_adder.nodelist)
    brent_kung_adder.output_feature_list("levellist", brent_kung_adder.levellist)
    brent_kung_adder.output_feature_list("minlist", brent_kung_adder.minlist)
    brent_kung_adder.output_feature_list("fanoutlist", brent_kung_adder.fanoutlist)
    brent_kung_adder.run_yosys()
    delay,area,power = brent_kung_adder.run_openroad()
    brent_kung_adder.plot_prefix_graph()
    print("Brent-Kung adder delay: ", delay)
    print("Brent-Kung adder area: ", area)
    print("Brent-Kung adder power: ", power)
    
def main():
    global args
    print(SEPARATOR)
    print_title_banner(GRAPH_TO_GATES_TITLE)
    args = parse_arguments()
    
    print_section_header("RL TRAINING")
    print(SEPARATOR)
    
    # Run RL training process
    train(TrainingConfig(), global_vars.restore_from)

if __name__ == "__main__":
    main()