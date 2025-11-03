### Global Variable Declarations ###
flog = ""
start_time = {}
initial_adder_type = None
result_cache = {}
cache_hit = 0
step_count = 1666
openroad_path = "OpenROAD/prefix-flow/"
output_dir = "out/"
save_verilog = False
n = 8
num_steps = 5000
num_episodes = 100
w_scalar = 0.5
batch_size = 192

### Verilog Cell Definitions ###
BLACK_CELL = '''module BLACK (
\tinput gik, pik, gkj, pkj,
\toutput gij, pij
);
assign pij = pik & pkj;
assign gij = gik | (pik & gkj);
endmodule
'''

GREY_CELL = '''module GREY(
\tinput gik, pik, gkj,
\toutput gij
);
assign gij = gik | (pik & gkj);
endmodule
'''

### Script Definitions ###
yosys_script_format = \
'''read -sv {}
hierarchy -top adder_top
flatten
proc; techmap; opt;
abc -fast -liberty {}NangateOpenCellLibrary_typical.lib
write_verilog {}
'''

CLOCK_PERIOD_TARGET = 3.0

sdc_format = \
f'''create_clock [get_ports clk] -name core_clock -period {CLOCK_PERIOD_TARGET}
set_all_input_output_delays
'''

openroad_tcl = \
'''source "helpers.tcl"
source "flow_helpers.tcl"
source "Nangate45/Nangate45.vars"
set design "adder"
set top_module "adder_top"
set synth_verilog "{}"
set sdc_file "{}"
set die_area {{0 0 80 80}}
set core_area {{0 0 80 80}}
set batch_index "{}"
source -echo "fast_flow.tcl"
'''
