#!/bin/sh

# Reproducing experimental results
# Process data and train BT models.
./experiment/train.sh

# FFA result reproduction. The logs are stored in ```../logs/cut/```.
mkdir -p ../logs/cut
./experiment/ffa_maxp.sh
./experiment/ffa_mcxp.sh 
./experiment/ffa_switch.sh

# Parse JIT logs and compute metrics. The statistics are stored in ```../stats/```.
mkdir -p ../stats/
python ./other/parse_logs.py

# Integrate metrics and generate plots. Plots are stored in ```../plots```.
python ./other/metric.py & python ./other/visualize.py
