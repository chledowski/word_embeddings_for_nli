#!/usr/bin/env bash

set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

MODEL=$1 # model name {esim, cbow, blstm}
DATASET=$2 # dataset name {SNLI, MNLI}
EMBEDDING=$3 # emb namme {gwiki6, gcc42, gcc840}
RUN_MODULO=$4
MODULO=$5 # how many runs at once


# For example, if you want to run on two devices, execute:
# src/scripts/reproduce.sh esim snli gcc840 0 2
# src/scripts/reproduce.sh esim snli gcc840 1 2
# in two terminals with different CUDA device exported.


printf "${GREEN}Dataset = ${DATASET}${NC}\n"
printf "${GREEN}Model = ${MODEL}${NC}\n"
printf "${GREEN}Embedding = ${EMBEDDING}${NC}\n"
printf "${GREEN}Modulo $MODULO == $RUN_MODULO${NC}\n"
printf "${GREEN}CUDA: $CUDA_VISIBLE_DEVICES${NC}\n"

declare -a NAMES=(
    "${EMBEDDING}"

    "cokim_${EMBEDDING}_fq_12_q"
#    "cokim_${EMBEDDING}_fq_2_q"
    "cokim_${EMBEDDING}_fq_12"
#    "cokim_${EMBEDDING}_fq_2"
#    "cokim_fq_12"
#    "cokim_fq_2"

    "${EMBEDDING}_fq_12_q"
    "${EMBEDDING}_fq_2_q"
    "${EMBEDDING}_fq_12"
    "${EMBEDDING}_fq_2"
#    "fq_12"
#    "fq_2"

#    "kim_${EMBEDDING}_fq_12_q"
#    "kim_${EMBEDDING}_fq_2_q"
#    "kim_${EMBEDDING}_fq_12"
#    "kim_${EMBEDDING}_fq_2"
#    "kim_fq_12"
#    "kim_fq_2"
)

declare -a RETRO_ARGS=(
    " --save-embedding --second-embedding=${EMBEDDING}" # wiki

    " --save-embedding --sum --q --retrofitting --lexicon-name=cokim" # cokim_wiki_fq_12_q
#    " --save-embedding --sum --q --retrofitting --lexicon-name=cokim --losses 2 --losses-2 2" # cokim_wiki_fq_2_q
    " --save-embedding --sum --retrofitting --lexicon-name=cokim" # cokim_wiki_fq_12
#    " --save-embedding --sum --retrofitting --lexicon-name=cokim --losses 2 --losses-2 2" # cokim_wiki_fq_2
#    " --save-embedding --retrofitting --lexicon-name=cokim" # cokim_fq_12
#    " --save-embedding --retrofitting --lexicon-name=cokim --losses 2 --losses-2 2" # cokim_fq_2

    " --save-embedding --sum --q --retrofitting" # wiki_fq_12_q
    " --save-embedding --sum --q --retrofitting --losses 2 --losses-2 2" # wiki_fq_2_q
    " --save-embedding --sum --retrofitting" # wiki_fq_12
    " --save-embedding --sum --retrofitting --losses 2 --losses-2 2" # wiki_fq_2
#    " --save-embedding --retrofitting" # fq_12
#    " --save-embedding --retrofitting --losses 2 --losses-2 2" # fq_2

#    " --save-embedding --sum --q --retrofitting --lexicon-name=kim" # kim_wiki_fq_12_q
#    " --save-embedding --sum --q --retrofitting --lexicon-name=kim --losses 2 --losses-2 2" # kim_wiki_fq_2_q
#    " --save-embedding --sum --retrofitting --lexicon-name=kim" # kim_wiki_fq_12
#    " --save-embedding --sum --retrofitting --lexicon-name=kim --losses 2 --losses-2 2" # kim_wiki_fq_2
#    " --save-embedding --retrofitting --lexicon-name=kim" # kim_fq_12
#    " --save-embedding --retrofitting --lexicon-name=kim --losses 2 --losses-2 2" # kim_fq_2
)

# python src/scripts/preprocess_data/fetch_embeddings.py --embeddings wiki

for (( i=0; i<${#NAMES[@]}; i++ ));
do
    if [[ $(($i % $MODULO)) == $(($RUN_MODULO)) ]]; then
        NAME="${MODEL}_${EMBEDDING}_${DATASET}_${NAMES[$i]}"
        RESULTS_DIR="results/$NAME"

        printf "${GREEN}Running $NAME...${NC}\n"
        printf "${GREEN}Results dir: $RESULTS_DIR${NC}\n"
        python src/scripts/retrofitting/retrofitting.py ${RETRO_ARGS[$i]} --save-text=$NAME --embedding=$EMBEDDING
        mkdir -p $RESULTS_DIR
        python src/scripts/train_eval/train_$MODEL.py root $RESULTS_DIR --embedding_name=$NAME --dataset=$DATASET
        python src/scripts/train_eval/evaluate.py --model-name=$NAME
    fi
done