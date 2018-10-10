#!/usr/bin/env bash

set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

MODEL=$1 # esim, esim-kim, esim-elmo
DATASET=$2 # dataset name {SNLI, MNLI}
RUN_MODULO=$3
MODULO=$4
START_FROM=$5

SEED=9
FRACTION=0.1

printf "${GREEN}Dataset = ${DATASET}${NC}\n"

declare -a NAMES=(
    "gcc840" # mod_drop
    "gcc840_snli_gcc840_fq_12_q" # mod_drop
    "gcc840" # mod_drop
    "gcc840_snli_gcc840_fq_12_q" # mod_drop
)

declare -a CONNECTION=(
    "mod_drop"
    "mod_drop"
    "mod_drop"
    "mod_drop"
)

declare -a NORMALIZE=(
    "normalize"
    "normalize"
    "no_normalize"
    "no_normalize"
)

for (( i=${START_FROM}; i<${#NAMES[@]}; i++ ));
do
    if [[ $(($i % $MODULO)) == $(($RUN_MODULO)) ]]; then
        NAME="${MODEL}_${DATASET}_${CONNECTION[$i]}_${NAMES[$i]}_frac${FRACTION}_mgr"
        RESULTS_DIR="results/${NAME}"

        printf "${GREEN}Running $NAME...${NC}\n"
        printf "${GREEN}Results dir: $RESULTS_DIR${NC}\n"
        mkdir -p $RESULTS_DIR
        python src/scripts/train_eval/train.py $MODEL $RESULTS_DIR \
            --embedding_name="gcc840" \
            --embedding_second_name=${NAMES[$i]} \
            --residual_embedding.active \
            --residual_embedding.type=${CONNECTION[$i]} \
            --residual_embedding.${NORMALIZE[$i]} \
            --dataset=${DATASET} \
            --seed=${SEED} \
            --train_on_fraction=${FRACTION}

        python src/scripts/train_eval/evaluate.py --model-name="${NAME}"
    fi
done