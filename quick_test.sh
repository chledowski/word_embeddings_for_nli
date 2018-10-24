#!/usr/bin/env bash
# Set up an environment and reproduces one of experiments.

set -e

source env.sh.default
scripts/download.sh
python3 run.py --command=train --config=training_configs/esim-doti-mnli0.01.json --savedir=results/esim-doti-mnli0.01
python3 run.py --command=evaluate --model-path=results/esim-doti-mnli0.01