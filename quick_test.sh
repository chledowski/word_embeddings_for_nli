#!/usr/bin/env bash
# Set up an environment and reproduces one of experiments.

set -e

source env.sh.default
scripts/download.sh
python3 run.py --command=train --config=training_configs/esim-snli0.01.json --savedir=results/esim-snli0.01
python3 run.py --command=evaluate --model-path=results/esim-snli0.01