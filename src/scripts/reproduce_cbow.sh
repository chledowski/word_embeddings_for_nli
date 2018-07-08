#!/usr/bin/env bash

set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

MODULO=12
RUN_MODULO=$1
printf "${GREEN}Running experiments modulo $MODULO == $RUN_MODULO${NC}\n"
printf "${GREEN}CUDA: $CUDA_VISIBLE_DEVICES${NC}\n"

# python src/scripts/fetch_data/fetch_embeddings.py --embeddings wiki

DATASET="snli"
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
    " --save-embedding --second-embedding=wiki" # wiki

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

for (( i=0; i<${#NAMES[@]}; i++ ));
do
    if [[ $(($i % $MODULO)) == $(($RUN_MODULO)) ]]; then
        NAME="${DATASET}_${NAMES[$i]}"
        RESULTS_DIR="results/$NAME"

        printf "${GREEN}Running $NAME...${NC}\n"
        printf "${GREEN}Results dir: $RESULTS_DIR${NC}\n"
        python src/scripts/retrofitting/retrofitting.py ${RETRO_ARGS[$i]} --save-text=$NAME
        mkdir -p $RESULTS_DIR
        python src/scripts/train_eval/train_cbow.py root $RESULTS_DIR --embedding_name=$NAME
        python src/scripts/train_eval/evaluate.py --model-name=$NAME
    fi
done

# -----
#
#python src/scripts/retrofitting/retrofitting.py --evaluate --save-text=wiki --second-embedding=wiki
#mkdir -p results/wiki/
#python src/scripts/train_eval/train_cbow.py wiki results/wiki/ --embedding_name=wiki
#python src/scripts/train_eval/evaluate_cbow.py --model-name wiki
#
#python src/scripts/retrofitting/retrofitting.py --save-text=wiki_fq_12_q --evaluate --save-embedding --sum --q --retrofitting
#python src/scripts/retrofitting/retrofitting.py --save-text=wiki_fq_2_q --evaluate --save-embedding --sum --q --retrofitting --losses 2 --losses-2 2
#python src/scripts/retrofitting/retrofitting.py --save-text=wiki_fq_12 --evaluate --save-embedding --sum --retrofitting
#python src/scripts/retrofitting/retrofitting.py --save-text=wiki_fq_2 --evaluate --save-embedding --sum --retrofitting --losses 2 --losses-2 2
#python src/scripts/retrofitting/retrofitting.py --save-text=fq_12 --evaluate --save-embedding --retrofitting
#python src/scripts/retrofitting/retrofitting.py --save-text=fq_2 --evaluate --save-embedding --retrofitting --losses 2 --losses-2 2
#mkdir -p results/wiki_fq_12_q/
#mkdir -p results/wiki_fq_2_q/
#mkdir -p results/wiki_fq_12/
#mkdir -p results/wiki_fq_2/
#mkdir -p results/fq_12/
#mkdir -p results/fq_2/
#python src/scripts/train_eval/train_cbow.py wiki results/wiki_fq_12_q/ --embedding_name=wiki_fq_12_q
#python src/scripts/train_eval/train_cbow.py wiki results/wiki_fq_2_q/ --embedding_name=wiki_fq_2_q
#python src/scripts/train_eval/train_cbow.py wiki results/wiki_fq_12/ --embedding_name=wiki_fq_12
#python src/scripts/train_eval/train_cbow.py wiki results/wiki_fq_2/ --embedding_name=wiki_fq_2
#python src/scripts/train_eval/train_cbow.py wiki results/fq_12/ --embedding_name=fq_12
#python src/scripts/train_eval/train_cbow.py wiki results/fq_2/ --embedding_name=fq_2
#python src/scripts/train_eval/evaluate_cbow.py --model-name wiki_fq_12_q
#python src/scripts/train_eval/evaluate_cbow.py --model-name wiki_fq_2_q
#python src/scripts/train_eval/evaluate_cbow.py --model-name wiki_fq_12
#python src/scripts/train_eval/evaluate_cbow.py --model-name wiki_fq_2
#python src/scripts/train_eval/evaluate_cbow.py --model-name fq_12
#python src/scripts/train_eval/evaluate_cbow.py --model-name fq_2
#
#
#
#python src/scripts/retrofitting/retrofitting.py --save-text=kim_wiki_fq_12_q --evaluate --save-embedding --sum --q --retrofitting --lexicon-name=kim
#python src/scripts/retrofitting/retrofitting.py --save-text=kim_wiki_fq_2_q --evaluate --save-embedding --sum --q --retrofitting --lexicon-name=kim --losses 2 --losses-2 2
#python src/scripts/retrofitting/retrofitting.py --save-text=kim_wiki_fq_12 --evaluate --save-embedding --sum --retrofitting --lexicon-name=kim
#python src/scripts/retrofitting/retrofitting.py --save-text=kim_wiki_fq_2 --evaluate --save-embedding --sum --retrofitting --lexicon-name=kim --losses 2 --losses-2 2
#python src/scripts/retrofitting/retrofitting.py --save-text=kim_fq_12 --evaluate --save-embedding --retrofitting --lexicon-name=kim
#python src/scripts/retrofitting/retrofitting.py --save-text=kim_fq_2 --evaluate --save-embedding --retrofitting --lexicon-name=kim --losses 2 --losses-2 2
#mkdir -p results/kim_wiki_fq_12_q/
#mkdir -p results/kim_wiki_fq_2_q/
#mkdir -p results/kim_wiki_fq_12/
#mkdir -p results/kim_wiki_fq_2/
#mkdir -p results/kim_fq_12/
#mkdir -p results/kim_fq_2/
#python src/scripts/train_eval/train_cbow.py wiki results/kim_wiki_fq_12_q/ --embedding_name=kim_wiki_fq_12_q
#python src/scripts/train_eval/train_cbow.py wiki results/kim_wiki_fq_2_q/ --embedding_name=kim_wiki_fq_2_q
#python src/scripts/train_eval/train_cbow.py wiki results/kim_wiki_fq_12/ --embedding_name=kim_wiki_fq_12
#python src/scripts/train_eval/train_cbow.py wiki results/kim_wiki_fq_2/ --embedding_name=kim_wiki_fq_2
#python src/scripts/train_eval/train_cbow.py wiki results/kim_fq_12/ --embedding_name=kim_fq_12
#python src/scripts/train_eval/train_cbow.py wiki results/kim_fq_2/ --embedding_name=kim_fq_2
#python src/scripts/train_eval/evaluate_cbow.py --model-name kim_wiki_fq_12_q
#python src/scripts/train_eval/evaluate_cbow.py --model-name kim_wiki_fq_2_q
#python src/scripts/train_eval/evaluate_cbow.py --model-name kim_wiki_fq_12
#python src/scripts/train_eval/evaluate_cbow.py --model-name kim_wiki_fq_2
#python src/scripts/train_eval/evaluate_cbow.py --model-name kim_fq_12
#python src/scripts/train_eval/evaluate_cbow.py --model-name kim_fq_2
#
#python src/scripts/retrofitting/retrofitting.py --save-text=cokim_wiki_fq_12_q --evaluate --save-embedding --sum --q --retrofitting --lexicon-name=cokim
#python src/scripts/retrofitting/retrofitting.py --save-text=cokim_wiki_fq_2_q --evaluate --save-embedding --sum --q --retrofitting --lexicon-name=cokim --losses 2 --losses-2 2
#python src/scripts/retrofitting/retrofitting.py --save-text=cokim_wiki_fq_12 --evaluate --save-embedding --sum --retrofitting --lexicon-name=cokim
#python src/scripts/retrofitting/retrofitting.py --save-text=cokim_wiki_fq_2 --evaluate --save-embedding --sum --retrofitting --lexicon-name=cokim --losses 2 --losses-2 2
#python src/scripts/retrofitting/retrofitting.py --save-text=cokim_fq_12 --evaluate --save-embedding --retrofitting --lexicon-name=cokim
#python src/scripts/retrofitting/retrofitting.py --save-text=cokim_fq_2 --evaluate --save-embedding --retrofitting --lexicon-name=cokim --losses 2 --losses-2 2
#mkdir -p results/cokim_wiki_fq_12_q/
#mkdir -p results/cokim_wiki_fq_2_q/
#mkdir -p results/cokim_wiki_fq_12/
#mkdir -p results/cokim_wiki_fq_2/
#mkdir -p results/cokim_fq_12/
#mkdir -p results/cokim_fq_2/
#python src/scripts/train_eval/train_cbow.py wiki results/cokim_wiki_fq_12_q/ --embedding_name=cokim_wiki_fq_12_q
#python src/scripts/train_eval/train_cbow.py wiki results/cokim_wiki_fq_2_q/ --embedding_name=cokim_wiki_fq_2_q
#python src/scripts/train_eval/train_cbow.py wiki results/cokim_wiki_fq_12/ --embedding_name=cokim_wiki_fq_12
#python src/scripts/train_eval/train_cbow.py wiki results/cokim_wiki_fq_2/ --embedding_name=cokim_wiki_fq_2
#python src/scripts/train_eval/train_cbow.py wiki results/cokim_fq_12/ --embedding_name=cokim_fq_12
#python src/scripts/train_eval/train_cbow.py wiki results/cokim_fq_2/ --embedding_name=cokim_fq_2
#python src/scripts/train_eval/evaluate_cbow.py --model-name cokim_wiki_fq_12_q
#python src/scripts/train_eval/evaluate_cbow.py --model-name cokim_wiki_fq_2_q
#python src/scripts/train_eval/evaluate_cbow.py --model-name cokim_wiki_fq_12
#python src/scripts/train_eval/evaluate_cbow.py --model-name cokim_wiki_fq_2
#python src/scripts/train_eval/evaluate_cbow.py --model-name cokim_fq_12
#python src/scripts/train_eval/evaluate_cbow.py --model-name cokim_fq_2
