#!/usr/bin/env bash

set -e

MODEL=$1
DUMP_ONLY=$2
RUN_MODULO=$3
MODULO=$4

for i in {1..1}
    do
    if [[ $(($i % $MODULO)) == $(($RUN_MODULO)) ]]; then
        echo "EPOCH ${i}..."
        if [ "${DUMP_ONLY}" = 1 ]; then
            echo "DUMP_ONLY mode on"
            python3 src/scripts/train_eval/restorer.py  \
                --model-name=${MODEL} \
                --top-k=100000 \
                --model-epoch=${i} \
                --dump-only
        else
            python3 src/scripts/train_eval/restorer.py  \
                --model-name=${MODEL} \
                --top-k=100000 \
                --restorer-num-epochs=5000 \
                --model-epoch=${i}
        fi
    fi
    done