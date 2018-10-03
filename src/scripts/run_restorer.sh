#!/usr/bin/env bash

set -e

DUMP_ONLY=$1
RUN_MODULO=$2
MODULO=$3

for i in {7..20}
    do
    if [[ $(($i % $MODULO)) == $(($RUN_MODULO)) ]]; then
        echo "EPOCH ${i}..."
        if [ "${DUMP_ONLY}" = 1 ]; then
            echo "DUMP_ONLY mode on"
            python3 src/scripts/train_eval/restorer.py  \
                --model-name=esim_to_restore \
                --top-k=100000 \
                --model-epoch=${i} \
                --dump-only
        else
            python3 src/scripts/train_eval/restorer.py  \
                --model-name=esim_to_restore \
                --top-k=100000 \
                --restorer-num-epochs=5000 \
                --model-epoch=${i}
        fi
    fi
    done