#!/usr/bin/env bash

# run like src/scripts/reproduce_lear.sh esim snli wiki

set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

MODEL=$1 # model name {esim, cbow, blstm}
DATASET=$2 # dataset name {SNLI, MNLI}
EMBEDDING=$3 # initial embedding name {wiki etc}

printf "${GREEN}Model = ${MODEL}${NC}\n"
printf "${GREEN}Dataset = ${DATASET}${NC}\n"
printf "${GREEN}Embedding = ${EMBEDDING}${NC}\n"

# python src/scripts/preprocess_data/embedding_file_change_format.py --convert-to txt --h5 ${EMBEDDING} --txt wv_pre_lear
# python src/scripts/retrofitting/lear/code/lear.py
# python src/scripts/preprocess_data/embedding_file_change_format.py --convert-to h5 --h5 lear_${EMBEDDING}_norm --txt wv_after_lear
# python src/scripts/retrofitting/lear/code/lear_no_norm.py
# python src/scripts/preprocess_data/embedding_file_change_format.py --convert-to h5 --h5 lear_${EMBEDDING} --txt wv_after_lear

declare -a NAMES=(

    "${EMBEDDING}_lear_sum_q_norm"
    "${EMBEDDING}_lear_sum_norm"
    "${EMBEDDING}_lear_norm"
    "${EMBEDDING}_lear_sum_q"
    "${EMBEDDING}_lear_sum"
    "${EMBEDDING}_lear"
    "${EMBEDDING}_norm"

)

declare -a RETRO_ARGS=(

    " --save-embedding  --normalize-wv-only --sum --q --second-embedding lear_${EMBEDDING}_norm" # EMBEDDING_lear_q
    " --save-embedding  --normalize-wv-only --sum --second-embedding lear_${EMBEDDING}_norm" # EMBEDDING_lear
    " --save-embedding  --normalize-wv-only --second-embedding lear_${EMBEDDING}_norm" # lear_on_EMBEDDING

    " --save-embedding  --normalize-wv-only --sum --q --second-embedding lear_${EMBEDDING}" # EMBEDDING_lear_q
    " --save-embedding  --normalize-wv-only --sum --second-embedding lear_${EMBEDDING}" # EMBEDDING_lear
    " --save-embedding  --normalize-wv-only --second-embedding lear_${EMBEDDING}" # lear_on_EMBEDDING

    " --save-embedding --normalize-wv-only" # lear_on_EMBEDDING

)

for (( i=0; i<${#NAMES[@]}; i++ ));
do

        NAME="${MODEL}_${DATASET}_${NAMES[$i]}"
        RESULTS_DIR="results/$NAME"

        printf "${GREEN}Running $NAME...${NC}\n"
        printf "${GREEN}Results dir: $RESULTS_DIR${NC}\n"
        python src/scripts/retrofitting/retrofitting.py ${RETRO_ARGS[$i]} --save-text $NAME --embedding ${EMBEDDING}
        mkdir -p $RESULTS_DIR
        python src/scripts/train_eval/train.py $MODEL $RESULTS_DIR --embedding_name=$NAME --dataset=$DATASET
        python src/scripts/train_eval/evaluate.py --model-name=$NAME --embedding-name=$NAME

done