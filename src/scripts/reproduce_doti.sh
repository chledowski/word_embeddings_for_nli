#!/usr/bin/env bash

set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

DATASET=$1 # dataset name {SNLI, MNLI}
RUN_MODULO=$2
MODULO=$3
START_FROM=$4

SEED=9

printf "${GREEN}Dataset = ${DATASET}${NC}\n"

declare -a FRACS=(
    "0.001"
    "0.001"
    "0.001"
    "0.01"
    "0.01"
    "0.01"
#    "0.1"
#    "0.1"
#    "0.1"
)

declare -a EMBEDDINGS=(
    "gcc840"
    "gcc840_snli_gcc840_fq_12_q"
    "gcc840"
    "gcc840"
    "gcc840_snli_gcc840_fq_12_q"
    "gcc840"
    "gcc840"
    "gcc840_snli_gcc840_fq_12_q"
    "gcc840"
)

declare -a PARAMS=(
    "--knowledge_after_lstm=dot"
    "--knowledge_after_lstm=dot"
    "--knowledge_after_lstm=none"
    "--knowledge_after_lstm=dot"
    "--knowledge_after_lstm=dot"
    "--knowledge_after_lstm=none"
    "--knowledge_after_lstm=dot"
    "--knowledge_after_lstm=dot"
    "--knowledge_after_lstm=none"
)

declare -a MODELS=(
    "esim"
    "esim"
    "esim-kim"
    "esim"
    "esim"
    "esim-kim"
    "esim"
    "esim"
    "esim-kim"
)

declare -a ATTENTION_LAMBDA=(
    "0"
    "0"
    "20"
    "0"
    "0"
    "8"
    "0"
    "0"
    "2"
)

declare -a I_LAMBDA=(
    "5"
    "5"
    "1"
    "5"
    "5"
    "1"
    "0.1"
    "0.1"
    "1"
)

for (( i=${START_FROM}; i<${#FRACS[@]}; i++ ));
do
    if [[ $(($i % $MODULO)) == $(($RUN_MODULO)) ]]; then
        NAME="${MODELS[$i]}_${DATASET}_${EMBEDDINGS[$i]}_frac${FRACS[$i]}_mgr"
        RESULTS_DIR="results/${NAME}"

        printf "${GREEN}Running $NAME...${NC}\n"
        printf "${GREEN}Results dir: $RESULTS_DIR${NC}\n"
#        mkdir -p $RESULTS_DIR
#        python src/scripts/train_eval/train.py ${MODELS[$i]} $RESULTS_DIR \
#            --embedding_name="gcc840" \
#            --embedding_second_name=${EMBEDDINGS[$i]} \
#            --dataset=${DATASET} \
#            --seed=${SEED} \
#            --i_lambda=${I_LAMBDA[$i]} \
#            --a_lambda=${ATTENTION_LAMBDA[$i]} \
#            --train_on_fraction=${FRACS[$i]} \
#            ${PARAMS[i]}

        python src/scripts/train_eval/evaluate.py --model-name="${NAME}"
    fi
done