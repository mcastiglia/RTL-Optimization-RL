flog = ""
start_time = {}
initial_adder_type = None

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

yosys_script_format = \
'''read -sv {}
hierarchy -top adder_top
flatten
proc; techmap; opt;
abc -fast -liberty {}/NangateOpenCellLibrary_typical.lib
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
source -echo "fast_flow.tcl"
'''