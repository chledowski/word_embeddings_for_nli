#!/usr/bin/env bash

# run like src/scripts/reproduce_lear.sh esim snli wiki

set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

MODEL=$1 # model name {esim, cbow, blstm}
DATASET=$2 # dataset name {SNLI, MNLI}
EMBEDDING=$3 # initial embedding name {wiki etc}

printf "${GREEN}Dataset = ${DATASET}${NC}\n"
printf "${GREEN}Model = ${MODEL}${NC}\n"
printf "${GREEN}Embedding = ${EMBEDDING}${NC}\n"

python src/scripts/preprocess_data/embedding_file_change_format.py --convert-to txt --h5 $EMBEDDING --txt wv_pre_lear
python src/scripts/retrofitting/lear/code/lear.py
python src/scripts/preprocess_data/embedding_file_change_format.py --convert-to h5 --h5 lear_on_$EMBEDDING --txt wv_after_lear

declare -a NAMES=(

    "${EMBEDDING}_lear_q"
    "${EMBEDDING}_lear"

    "lear_on_${EMBEDDING}"
    
)

declare -a RETRO_ARGS=(

    " --save-embedding --sum --q " # EMBEDDING_lear_q
    " --save-embedding --sum " # EMBEDDING_lear

    " --save-embedding " # lear_on_EMBEDDING

)

for (( i=0; i<${#NAMES[@]}; i++ ));
do

        NAME="${MODEL}_${DATASET}_${NAMES[$i]}"
        RESULTS_DIR="results/$NAME"

        printf "${GREEN}Running $NAME...${NC}\n"
        printf "${GREEN}Results dir: $RESULTS_DIR${NC}\n"
        python src/scripts/retrofitting/retrofitting.py ${RETRO_ARGS[$i]} --save-text $NAME --second-embedding lear_on_$EMBEDDING --embedding $EMBEDDING
        mkdir -p $RESULTS_DIR
        python src/scripts/train_eval/train_$MODEL.py root $RESULTS_DIR --embedding_name=$NAME --dataset=$DATASET
        python src/scripts/train_eval/evaluate.py --model-name=$NAME

done