#!/usr/bin/env bash

set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

MODEL=$1 # esim, esim-kim, esim-elmo
DATASET=$2 # dataset name {SNLI, MNLI}
RUN_MODULO=$3
MODULO=$4

SEED=9

printf "${GREEN}Dataset = ${DATASET}${NC}\n"

declare -a NAMES=(
    "gcc840_snli_fq_2"
    "gcc840_snli_fq_12"
    "gcc840_snli_gcc840_fq_2"
    "gcc840_snli_gcc840_fq_12"
    "gcc840_snli_gcc840_fq_2_q"
    "gcc840_snli_gcc840_fq_12_q"
)

for (( i=0; i<${#NAMES[@]}; i++ ));
do
    if [[ $(($i % $MODULO)) == $(($RUN_MODULO)) ]]; then
        NAME="${MODEL}_${DATASET}_${NAMES[$i]}_mgr"
        RESULTS_DIR="results/${NAME}"

        printf "${GREEN}Running $NAME...${NC}\n"
        printf "${GREEN}Results dir: $RESULTS_DIR${NC}\n"
        mkdir -p $RESULTS_DIR
        python src/scripts/train_eval/train.py $MODEL $RESULTS_DIR \
            --embedding_name=${NAMES[$i]} \
            --dataset=${DATASET} \
            --seed=${SEED}
        python src/scripts/train_eval/evaluate.py --model-name="${NAME}"
    fi
done