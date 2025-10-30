import argparse
import time
import os
import global_vars
from init_states import init_graph
from q_network import PrefixRL_DQN, build_features, build_action_masks, apply_action_masks, scalarize_q, argmax_action, get_best_action

def parse_arguments():
    global args, INPUT_BIT, flog
    
    parser = argparse.ArgumentParser(description='Prefix graph to adder netlist conversion tool')
    parser.add_argument('-n','--input_bitwidth', type = int, required=True, help="Input bitwidth for the adder")
    parser.add_argument('--adder_type', type = int, required=True, help="Initial starting state for prefix graph (0: serial, 1: sklansky, 2: brent-kung)")
    parser.add_argument('-b', '--batch_size', type = int, default = 4, help="Batch size for RL training")
    parser.add_argument('--step_count', type = int, default = 1666, help="Maximum number of optimization steps for RL training")
    parser.add_argument('--openroad_path', type = str, default = '../OpenROAD/prefix-flow/')
    parser.add_argument('--output_dir', type = str, default = 'out/', help='Output directory for generated files')
    parser.add_argument('--save_verilog', action = 'store_true', default = False, help="Save the generated Verilog files")
    parser.add_argument('--mode', type = str, choices=['generate', 'train'], default='generate', help="Mode: generate initial Verilog or run training")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("ARGUMENT CONFIGURATION")
    print("=" * 50)
    print(f"Input bitwidth: {args.input_bitwidth}")
    print(f"Adder type: {args.adder_type}")
    print(f"Step limit: {args.step_count}")
    print(f"OpenROAD path: {args.openroad_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Save Verilog: {args.save_verilog}")
    print(f"Mode: {args.mode}")
    print("=" * 50)

    global_vars.initial_adder_type = args.adder_type
    global_vars.step_count = args.step_count
    global_vars.openroad_path = args.openroad_path
    global_vars.output_dir = args.output_dir
    global_vars.n = args.input_bitwidth
    global_vars.save_verilog = args.save_verilog
    
    strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    
    # Create output directory structure
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, "adder_training_log")):
        os.mkdir(os.path.join(args.output_dir, "adder_training_log"))
    if not os.path.exists(os.path.join(args.output_dir, "adder_training_log/adder_{}b".format(args.input_bitwidth))):
        os.mkdir(os.path.join(args.output_dir, "adder_training_log/adder_{}b".format(args.input_bitwidth)))
    global_vars.flog = open(os.path.join(args.output_dir, "adder_training_log/adder_{}b/adder_{}b_openroad_type{}_{}.log".format(args.input_bitwidth, 
        args.input_bitwidth, args.adder_type, strftime)), "w")

    global_vars.start_time = time.time()
    
    return args

# TODO: Currently, main only initializes the state space, forward
def main():
    global args
    args = parse_arguments()
    
    # Generate Verilog file or run RL traing process
    if args.mode == 'generate':
        generate_initial_verilog()
    elif args.mode == 'train':
        init_state = init_graph(args.input_bitwidth, args.adder_type)
        init_state.output_verilog()
        init_state.output_nodelist()
        init_state.run_yosys()
        delay,area,power = init_state.run_openroad()
        # print(f"Delay: {delay}, Area: {area}, Power: {power}")
        
        features = build_features(
            init_state.nodelist, 
            init_state.minlist, 
            init_state.levellist, 
            init_state.fanoutlist, 
            batch_size=args.batch_size
        )
        
        model = PrefixRL_DQN()
        
        q = model(features)
        print("q.shape = ", q.shape)
        print("q_area_add = ", q[:,0])
        print("q_area_del = ", q[:,1])
        print("q_delay_add = ", q[:,2])
        print("q_delay_del = ", q[:,3])
        
        action_masks = build_action_masks(args.input_bitwidth, init_state.nodelist, init_state.minlist)
        q_masked = apply_action_masks(q, action_masks)
        print("q_masked.shape = ", q_masked.shape)
        print("q_area_add_masked = ", q_masked[:,0])
        print("q_area_del_masked = ", q_masked[:,1])
        print("q_delay_add_masked = ", q_masked[:,2])
        print("q_delay_del_masked = ", q_masked[:,3])
        
        scores = scalarize_q(q_masked, w_area=1.0, w_delay=1.0)
        best_is_add, best_vals = argmax_action(q_masked, w_area=1.0, w_delay=1.0)
        print("scores.shape = ", scores.shape)
        print("best_is_add.shape = ", best_is_add.shape)
        print("best_vals.shape = ", best_vals.shape)
        print("Scores: ", scores)
        print("Best action: ", best_is_add)
        print("Best values: ", best_vals)
        
        best_action_idx, best_action = get_best_action(q_masked, w_area=1.0, w_delay=1.0)
        print("Best action index: ", best_action_idx)
        print("Best action: ", best_action)
        
        if (best_action.size(0) == 1):
            action = best_action[0].item()
            action_idx = best_action_idx[0].tolist()
        else:
            raise ValueError("TODO: edit program for batch size > 1")
            
        next_state = init_state.get_next_state(action, action_idx)
        
        reward = (init_state.area - next_state.area, init_state.delay - next_state.delay)
        print("init_state.area = ", init_state.area)
        print("next_state.area = ", next_state.area)
        print("init_state.delay = ", init_state.delay)
        print("next_state.delay = ", next_state.delay)
        print("Reward for action {} at node ({}, {}): area={}, delay={}".format(action, action_idx[0], action_idx[1], reward[0], reward[1]))
        

if __name__ == "__main__":
    main()