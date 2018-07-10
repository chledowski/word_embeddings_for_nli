#!/usr/bin/env bash

set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

MODEL=$1 # model name {esim, cbow, blstm}
DATASET=$2 # dataset name {SNLI, MNLI}
RUN_MODULO=$3
MODULO=$4 # how many runs at once


# For example, if you want to run on two devices, execute:
# src/scripts/reproduce.sh cbow snli 0 2
# src/scripts/reproduce.sh cbow snli 1 2
# in two terminals with different CUDA device exported.


printf "${GREEN}Dataset = ${DATASET}${NC}\n"
printf "${GREEN}Model = ${MODEL}${NC}\n"
printf "${GREEN}Modulo $MODULO == $RUN_MODULO${NC}\n"
printf "${GREEN}CUDA: $CUDA_VISIBLE_DEVICES${NC}\n"

declare -a NAMES=(
    "wiki"

    "cokim_wiki_fq_12_q"
    "cokim_wiki_fq_2_q"
    "cokim_wiki_fq_12"
    "cokim_wiki_fq_2"
    "cokim_fq_12"
    "cokim_fq_2"

    "wiki_fq_12_q"
    "wiki_fq_2_q"
    "wiki_fq_12"
    "wiki_fq_2"
    "fq_12"
    "fq_2"

#    "kim_wiki_fq_12_q"
#    "kim_wiki_fq_2_q"
#    "kim_wiki_fq_12"
#    "kim_wiki_fq_2"
#    "kim_fq_12"
#    "kim_fq_2"
)

declare -a RETRO_ARGS=(
    " --second-embedding=wiki" # wiki

    " --save-embedding --sum --q --retrofitting --lexicon-name=cokim" # cokim_wiki_fq_12_q
    " --save-embedding --sum --q --retrofitting --lexicon-name=cokim --losses 2 --losses-2 2" # cokim_wiki_fq_2_q
    " --save-embedding --sum --retrofitting --lexicon-name=cokim" # cokim_wiki_fq_12
    " --save-embedding --sum --retrofitting --lexicon-name=cokim --losses 2 --losses-2 2" # cokim_wiki_fq_2
    " --save-embedding --retrofitting --lexicon-name=cokim" # cokim_fq_12
    " --save-embedding --retrofitting --lexicon-name=cokim --losses 2 --losses-2 2" # cokim_fq_2

    " --save-embedding --sum --q --retrofitting" # wiki_fq_12_q
    " --save-embedding --sum --q --retrofitting --losses 2 --losses-2 2" # wiki_fq_2_q
    " --save-embedding --sum --retrofitting" # wiki_fq_12
    " --save-embedding --sum --retrofitting --losses 2 --losses-2 2" # wiki_fq_2
    " --save-embedding --retrofitting" # fq_12
    " --save-embedding --retrofitting --losses 2 --losses-2 2" # fq_2

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
        NAME="${MODEL}_${DATASET}_${NAMES[$i]}"
        RESULTS_DIR="results/$NAME"

        printf "${GREEN}Running $NAME...${NC}\n"
        printf "${GREEN}Results dir: $RESULTS_DIR${NC}\n"
        python src/scripts/retrofitting/retrofitting.py ${RETRO_ARGS[$i]} --save-text=$NAME
        mkdir -p $RESULTS_DIR
        python src/scripts/train_eval/train_$MODEL.py root $RESULTS_DIR --embedding_name=$NAME --dataset=$DATASET
        python src/scripts/train_eval/evaluate.py --model-name=$NAME
    fi
done