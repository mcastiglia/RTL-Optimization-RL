# OpenROAD Apptainer Image Usage Instructions 

from `/OpenROAD/` change script permissions

`chmod +x scripts/*.sh`

build the image

`scripts/build_sif.sh --def openroad.def --out openroad.sif`

Add git submodule for OpenROAD flow example scripts

`git submodule update`

Checkout older version of repo, as master branch is incompatible with version of OpenROAD

`cd /OpenROAD-flow-scripts/`

`git checkout 659f54e2`

`cd ..`

run OpenROAD (example)

`apptainer shell --bind "$PWD":/workspace ./openroad.sif`

`cd /workspace/OpenROAD-flow-scripts/flow/`

`make DESIGN_CONFIG=./designs/nangate45/aes/config.mk`

This will perform the RTL-to-GDS flow for an AES core

The resulting design files will be under OpenROAD/OpenROAD-flow-scripts/flow/results/nangate45/aes/base

The generated reports at each step in the flow will be under OpenROAD/OpenROAD-flow-scripts/flow/reports/nangate45/aes/base
