#!/usr/bin/env bash


#!/usr/bin/env bash

set -e
set -x

GREEN='\033[1;32m'
NC='\033[0m' # No Color

RUN_MODULO=$1
MODULO=$2 # how many runs at once
START_FROM=0

# For example, if you want to run on two devices, execute:
# src/scripts/reproduce.sh esim snli gcc840 0 2
# src/scripts/reproduce.sh esim snli gcc840 1 2
# in two terminals with different CUDA device exported.

printf "${GREEN}Modulo $MODULO == $RUN_MODULO${NC}\n"
printf "${GREEN}CUDA: $CUDA_VISIBLE_DEVICES${NC}\n"
printf "${GREEN}START FROM: ${START_FROM}${NC}\n"

declare -a FRACS=(
    "0.001"
    "0.001"

    "0.01"
    "0.01"

    "0.1"
    "0.1"
)

declare -a NAMES=(
    "esim_elmo_comparison_no_elmo"
    "esim_elmo_comparison_elmo"

    "esim_elmo_comparison_no_elmo"
    "esim_elmo_comparison_elmo"

    "esim_elmo_comparison_no_elmo"
    "esim_elmo_comparison_elmo"
)

declare -a MODELS=(
    "esim"
    "esim-elmo"

    "esim"
    "esim-elmo"

    "esim"
    "esim-elmo"
)

for (( i=${START_FROM}; i<${#NAMES[@]}; i++ ));
do
    if [[ $(($i % $MODULO)) == $(($RUN_MODULO)) ]]; then
        NAME="${NAMES[$i]}_frac${FRACS[$i]}"
        RESULTS_DIR="results/$NAME"

        printf "${GREEN}Running $NAME...${NC}\n"
        printf "${GREEN}Results dir: $RESULTS_DIR${NC}\n"
        mkdir -p $RESULTS_DIR

        python src/scripts/train_eval/train.py ${MODELS[$i]} ${RESULTS_DIR} \
            --train_on_fraction=${FRACS[$i]} --seed=9
    fi
done