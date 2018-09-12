#!/usr/bin/env bash

#set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

python src/scripts/train_eval/train.py \
    esim \
    results/elmotest \
    --use_elmo