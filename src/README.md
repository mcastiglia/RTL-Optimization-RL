# RTL Optimization RL - Graph to Gates Tool

A reinforcement learning-based tool for optimizing prefix adders by converting graph representations to gate-level netlists and synthesizing them using Yosys and OpenROAD.

## Setup

1. **Enter the containerized environment:**
   ```bash
   apptainer shell --bind "$PWD":/workspace ./openroad.sif
   ```

2. **Navigate to the project top level directory:**

## Usage

### Command Line Interface

```bash
python3 src/graph-to-gates.py [OPTIONS]
```

### Required Arguments

- `-n INPUT_BITWIDTH, --input_bitwidth INPUT_BITWIDTH`
  - Bit width of the adder to generate (e.g., 16, 32, 64)

- `--adder_type ADDER_TYPE`
  - Type of initial adder configuration:
    - `0`: Normal (Linear) adder
    - `1`: Sklansky adder  
    - `2`: Brent-Kung adder

### Optional Arguments

- `--openroad_path OPENROAD_PATH` (default: '/home')
  - Path to OpenROAD prefix-flow directory containing technology libraries

- `--synth`
  - Enable synthesis and physical design analysis

- `--output_dir OUTPUT_DIR` (default: 'out')
  - Directory for generated files and results

- `--mode {generate,train}` (default: 'generate')
  - Operation mode:
    - `generate`: Create initial Verilog files only
    - `train`: Run full MCTS optimization

- `--step_count STEP_COUNT` (default: 1666)
  - Maximum number of optimization steps for MCTS training

## Examples

### Generate 32-bit Sklansky Adder
```bash
python3 src/graph-to-gates.py -n 32 --adder_type 1 --openroad_path=/OpenROAD/prefix-flow/ --synth
```

### Generate 16-bit Normal Adder (Verilog only)
```bash
python3 src/graph-to-gates.py -n 16 --adder_type 0 --mode generate
```

### Run Training on 64-bit Brent-Kung Adder
```bash
python3 src/graph-to-gates.py -n 64 --adder_type 2 --mode train --synth --step_count 1000
```

## Output Structure

The tool generates the following directory structure:

```
output_dir/
├── run_verilog_mid/          # Intermediate Verilog files
├── run_yosys_script/         # Generated Yosys synthesis scripts
├── run_yosys_mid/           # Synthesized netlists
├── run_openroad_mid/       # OpenROAD physical design files
└── adder_parc_log/         # Training logs and results
    └── adder_Nb/           # N-bit adder specific logs
```

## Adder Types

### Normal (Linear) Adder (type 0)
- Traditional ripple-carry adder structure
- Simple but higher delay for large bit widths

### Sklansky Adder (type 1)  
- Parallel prefix adder with logarithmic depth
- Good balance of area and delay

### Brent-Kung Adder (type 2)
- Parallel prefix adder with reduced area
- More area-efficient than Sklansky

## Directory Structure
```
src/
├── graph-to-gates.py       # Main entry point
├── global_vars.py          # Global variables and templates
├── state_class.py          # State management for MCTS
├── node_class.py           # MCTS node implementation
├── init_graphs.py          # Initial adder graph generation
└── README.md              # This file
```

