#!/usr/bin/env bash

set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

MODEL=$1 # model name {esim, cbow, blstm}
DATASET=$2 # dataset name {SNLI, MNLI}
EMBEDDING=$3 # emb name {gwiki6, gcc42, gcc840}
EMBEDDING_2=$4 # emb name 


printf "${GREEN}Dataset = ${DATASET}${NC}\n"
printf "${GREEN}Model = ${MODEL}${NC}\n"
printf "${GREEN}Embedding = ${EMBEDDING}${NC}\n"
printf "${GREEN}Embedding_2 = ${EMBEDDING_2}${NC}\n"

NAME_1="${MODEL}_${EMBEDDING}_plus_${EMBEDDING_2}_${DATASET}"
NAME_2="${MODEL}_${EMBEDDING}_plus_${EMBEDDING_2}_${DATASET}_q"
RESULTS_DIR_1="results/$NAME_1"
RESULTS_DIR_2="results/$NAME_2"

printf "${GREEN}Running $NAME...${NC}\n"
python src/scripts/retrofitting/retrofitting.py --save-embedding --sum --q --save-text=${NAME_1} --embedding=$EMBEDDING --second-embedding=$EMBEDDING_2  --normalize-wv-only
python src/scripts/retrofitting/retrofitting.py --save-embedding --sum --q --save-text=${NAME_2} --embedding=$EMBEDDING --second-embedding=$EMBEDDING_2  --normalize-wv-only

python src/scripts/train_eval/train.py $MODEL $RESULTS_DIR_1 --embedding_name=$NAME_1 --dataset=$DATASET
python src/scripts/train_eval/evaluate.py --model-name=$NAME_1 --embedding-name=$NAME_1
python src/scripts/train_eval/train.py $MODEL $RESULTS_DIR_2 --embedding_name=$NAME_2 --dataset=$DATASET
python src/scripts/train_eval/evaluate.py --model-name=$NAME_2 --embedding-name=$NAME_2
