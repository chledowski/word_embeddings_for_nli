#!/usr/bin/env bash

set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

RUN_MODULO=$1
MODULO=$2 # how many runs at once
START_FROM=0
EMBEDDING=$3
FRAC=$4

printf "${GREEN}Modulo $MODULO == $RUN_MODULO${NC}\n"
printf "${GREEN}CUDA: $CUDA_VISIBLE_DEVICES${NC}\n"
printf "${GREEN}START FROM: ${START_FROM}${NC}\n"
printf "${GREEN}EMBEDDING: ${EMBEDDING}${NC}\n"

declare -a NAMES=(
    "knowledge_after_lstm_best_${EMBEDDING}_both_al0.1_dot_frac${FRAC}"
    "knowledge_after_lstm_best_${EMBEDDING}_al1.0_dot_frac${FRAC}"
)

declare -a KNOWLEDGE=(
    "dot"
    "dot"
)

declare -a FIRST_EMBEDDING=(
    "${EMBEDDING}"
    "gcc840"
)

declare -a LAMBDA=(
    "0.1"
    "1.0"
)


for (( i=${START_FROM}; i<${#NAMES[@]}; i++ ));
do
    if [[ $(($i % $MODULO)) == $(($RUN_MODULO)) ]]; then
        NAME=${NAMES[$i]}
        RESULTS_DIR="results/$NAME"

        printf "${GREEN}Running $NAME...${NC}\n"
        printf "${GREEN}Results dir: $RESULTS_DIR${NC}\n"
        mkdir -p $RESULTS_DIR
        python src/scripts/train_eval/train.py esim ${RESULTS_DIR} \
            --embedding_name=${FIRST_EMBEDDING[$i]} \
            --embedding_second_name=${EMBEDDING} \
            --dataset=snli \
            --train_on_fraction=${FRAC} \
            --knowledge_after_lstm=${KNOWLEDGE[$i]} \
            --i_lambda=${LAMBDA[$i]} \
            --seed=1234 \
            --no_save_model
        python src/scripts/train_eval/evaluate.py --model-name=${NAME}
        # To save memory
        rm ${RESULTS_DIR}/best_model.h5
    fi
done