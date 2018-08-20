#!/usr/bin/env bash

set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

RUN_MODULO=$1
MODULO=$2 # how many runs at once
START_FROM=0
FRAC="0.04"

# For example, if you want to run on two devices, execute:
# src/scripts/reproduce.sh esim snli gcc840 0 2
# src/scripts/reproduce.sh esim snli gcc840 1 2
# in two terminals with different CUDA device exported.

printf "${GREEN}Modulo $MODULO == $RUN_MODULO${NC}\n"
printf "${GREEN}CUDA: $CUDA_VISIBLE_DEVICES${NC}\n"
printf "${GREEN}START FROM: ${START_FROM}${NC}\n"

declare -a NAMES=(
    "knowledge_after_lstm_glove_al0.002_dot_frac${FRAC}"
    "knowledge_after_lstm_glove_al0.01_dot_frac${FRAC}"
    "knowledge_after_lstm_glove_al0.1_dot_frac${FRAC}"
    "knowledge_after_lstm_glove_al1.0_dot_frac${FRAC}"
    "knowledge_after_lstm_glove_al5.0_dot_frac${FRAC}"

    "knowledge_after_lstm_glove_al0.002_euc_frac${FRAC}"
    "knowledge_after_lstm_glove_al0.01_euc_frac${FRAC}"
    "knowledge_after_lstm_glove_al0.1_euc_frac${FRAC}"
    "knowledge_after_lstm_glove_al1.0_euc_frac${FRAC}"
    "knowledge_after_lstm_glove_al5.0_euc_frac${FRAC}"

    "knowledge_after_lstm_retro_glove_al0.0_dot_frac${FRAC}"
    "knowledge_after_lstm_retro_glove_al0.01_dot_frac${FRAC}"
    "knowledge_after_lstm_retro_glove_al0.1_dot_frac${FRAC}"
    "knowledge_after_lstm_retro_glove_al1.0_dot_frac${FRAC}"
    "knowledge_after_lstm_retro_glove_al5.0_dot_frac${FRAC}"

    "knowledge_after_lstm_retro_glove_al0.0_euc_frac${FRAC}"
    "knowledge_after_lstm_retro_glove_al0.01_euc_frac${FRAC}"
    "knowledge_after_lstm_retro_glove_al0.1_euc_frac${FRAC}"
    "knowledge_after_lstm_retro_glove_al1.0_euc_frac${FRAC}"
    "knowledge_after_lstm_retro_glove_al5.0_euc_frac${FRAC}"
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
    "gcc840"

    "gcc840_mnli_gcc840_fq_12_q"
    "gcc840_mnli_gcc840_fq_12_q"
    "gcc840_mnli_gcc840_fq_12_q"
    "gcc840_mnli_gcc840_fq_12_q"
    "gcc840_mnli_gcc840_fq_12_q"

    "gcc840_mnli_gcc840_fq_12_q"
    "gcc840_mnli_gcc840_fq_12_q"
    "gcc840_mnli_gcc840_fq_12_q"
    "gcc840_mnli_gcc840_fq_12_q"
    "gcc840_mnli_gcc840_fq_12_q"
)

declare -a KNOWLEDGE=(
    "dot"
    "dot"
    "dot"
    "dot"
    "dot"

    "euc"
    "euc"
    "euc"
    "euc"
    "euc"

    "dot"
    "dot"
    "dot"
    "dot"
    "dot"

    "euc"
    "euc"
    "euc"
    "euc"
    "euc"
)

declare -a LAMBDA=(
    "0.002"
    "0.01"
    "0.1"
    "1.0"
    "5.0"

    "0.002"
    "0.01"
    "0.1"
    "1.0"
    "5.0"

    "0.0"
    "0.01"
    "0.1"
    "1.0"
    "5.0"

    "0.0"
    "0.01"
    "0.1"
    "1.0"
    "5.0"
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
            --embedding_name=${EMBEDDINGS[$i]} \
            --embedding_second_name="gcc840_mnli_gcc840_fq_12_q" \
            --dataset=snli \
            --train_on_fraction=${FRAC} \
            --knowledge_after_lstm=${KNOWLEDGE[$i]} \
            --i_lambda=${LAMBDA[$i]} \
            --no_save_model
        python src/scripts/train_eval/evaluate.py --model-name=${NAME}
        # To save memory
        rm ${RESULTS_DIR}/best_model.h5
    fi
done