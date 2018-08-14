#!/usr/bin/env bash

#set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

RUN_MODULO=$1
MODULO=$2 # how many runs at once

printf "${GREEN}Modulo $MODULO == $RUN_MODULO${NC}\n"
printf "${GREEN}CUDA: $CUDA_VISIBLE_DEVICES${NC}\n"

declare -a NAMES=(
    "curve-kim-train-emb-0.008"
    "curve-kim-train-emb-0.04"
    "curve-kim-train-emb-0.2"
    "curve-kim-train-emb-1.0"
    "curve-kim-0.008"
    "curve-kim-0.04"
    "curve-kim-0.2"
    "curve-kim-1.0"
    "curve-esim-0.008"
    "curve-esim-0.04"
    "curve-esim-0.2"
)

declare -a ARGS=(
    "--train_on_fraction=0.008 --useitrick=1 --train_embeddings=1"
    "--train_on_fraction=0.04 --useitrick=1 --train_embeddings=1"
    "--train_on_fraction=0.2 --useitrick=1 --train_embeddings=1"
    "--train_on_fraction=1.0 --useitrick=1 --train_embeddings=1"
    "--train_on_fraction=0.008 --useitrick=1 --train_embeddings=0"
    "--train_on_fraction=0.04 --useitrick=1 --train_embeddings=0"
    "--train_on_fraction=0.2 --useitrick=1 --train_embeddings=0"
    "--train_on_fraction=1.0 --useitrick=1 --train_embeddings=0"
    "--train_on_fraction=0.008 --useitrick=0 --train_embeddings=0"
    "--train_on_fraction=0.04 --useitrick=0 --train_embeddings=0"
    "--train_on_fraction=0.2 --useitrick=0 --train_embeddings=0"
)

# python src/scripts/preprocess_data/fetch_embeddings.py --embeddings wiki

for (( i=0; i<${#NAMES[@]}; i++ ));
do
    if [[ $(($i % $MODULO)) == $(($RUN_MODULO)) ]]; then
        NAME="${NAMES[$i]}"
        RESULTS_DIR="results/$NAME"

        printf "${GREEN}Running $NAME...${NC}\n"
        printf "${GREEN}Results dir: $RESULTS_DIR${NC}\n"
        mkdir -p $RESULTS_DIR
        python src/scripts/train_eval/train.py esim $RESULTS_DIR --dataset=snli ${ARGS}
        python src/scripts/train_eval/evaluate.py --model-name=$NAME
    fi
done