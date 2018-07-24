#!/usr/bin/env bash

set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

RUN_MODULO=$1
MODULO=$2 # how many runs at once
START_FROM=0
FRAC="0.05"

# For example, if you want to run on two devices, execute:
# src/scripts/reproduce.sh esim snli gcc840 0 2
# src/scripts/reproduce.sh esim snli gcc840 1 2
# in two terminals with different CUDA device exported.

printf "${GREEN}Modulo $MODULO == $RUN_MODULO${NC}\n"
printf "${GREEN}CUDA: $CUDA_VISIBLE_DEVICES${NC}\n"
printf "${GREEN}START FROM: ${START_FROM}${NC}\n"

declare -a NAMES=(
    "repro_kim_masked_glove_i_al0.002_frac${FRAC}"
    "repro_kim_masked_glove_i_al0.01_frac${FRAC}"
    "repro_kim_masked_glove_i_al0.02_frac${FRAC}"
    "repro_kim_masked_glove_i_al0.1_frac${FRAC}"
    "repro_kim_masked_glove_i_al0.2_frac${FRAC}"
    "repro_kim_masked_glove_i_al1.0_frac${FRAC}"
    "repro_kim_masked_glove_i_al5.0_frac${FRAC}"
    "repro_kim_masked_glove_i_al10.0_frac${FRAC}"
    "repro_kim_masked_glove_frac${FRAC}"
)

declare -a EMBEDDINGS=(
    "gcc840"
    "gcc840"
    "gcc840"
    "gcc840"
    "gcc840"
    "gcc840"
    "gcc840"
    "gcc840"
    "gcc840"
)

declare -a USEITRICK=(
    "0"
    "1"
    "1"
    "1"
    "1"
    "1"
    "1"
    "1"
    "1"
)

declare -a LAMBDA=(
    "0"
    "0.002"
    "0.01"
    "0.02"
    "0.1"
    "0.2"
    "1.0"
    "5.0"
    "10.0"
)


for (( i=${START_FROM}; i<${#NAMES[@]}; i++ ));
do
    if [[ $(($i % $MODULO)) == $(($RUN_MODULO)) ]]; then
        NAME="forgetting_${NAMES[$i]}"
        RESULTS_DIR="results/$NAME"

        printf "${GREEN}Running $NAME...${NC}\n"
        printf "${GREEN}Results dir: $RESULTS_DIR${NC}\n"
        mkdir -p $RESULTS_DIR
        python src/scripts/train_eval/train.py esim ${RESULTS_DIR} \
            --embedding_name=${EMBEDDINGS[$i]} \
            --dataset=snli \
            --train_on_fraction=${FRAC} \
            --useitrick=${USEITRICK[$i]} \
            --lambda=${LAMBDA[$i]}
#        python src/scripts/train_eval/evaluate.py --model-name=${NAME}
    fi
done