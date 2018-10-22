#!/usr/bin/env bash

set -e

source env.sh.default
scripts/download.sh
python3 run.py --command=train --config=training_configs/esim-doti-mnli0.01.json --savedir=results/esim-doti-mnli
python3 run.py --command=evaluate --model-path=results/esim-doti-mnli