#!/usr/bin/env bash

python src/scripts/fetch_data/fetch_embeddings.py --embeddings wiki

python src/scripts/retrofitting/retrofitting.py --evaluate --save-text=wiki --second-embedding=wiki
mkdir -p results/wiki/
python src/scripts/train_eval/train_cbow.py wiki results/wiki/ --embedding_name=wiki
python src/scripts/train_eval/evaluate_cbow.py --model-name wiki

python src/scripts/retrofitting/retrofitting.py --save-text=wiki_fq_12_q --evaluate --save-embedding --sum --q --retrofitting
python src/scripts/retrofitting/retrofitting.py --save-text=wiki_fq_2_q --evaluate --save-embedding --sum --q --retrofitting --losses 2 --losses-2 2
python src/scripts/retrofitting/retrofitting.py --save-text=wiki_fq_12 --evaluate --save-embedding --sum --retrofitting
python src/scripts/retrofitting/retrofitting.py --save-text=wiki_fq_2 --evaluate --save-embedding --sum --retrofitting --losses 2 --losses-2 2
python src/scripts/retrofitting/retrofitting.py --save-text=fq_12 --evaluate --save-embedding --retrofitting
python src/scripts/retrofitting/retrofitting.py --save-text=fq_2 --evaluate --save-embedding --retrofitting --losses 2 --losses-2 2
mkdir -p results/wiki_fq_12_q/
mkdir -p results/wiki_fq_2_q/
mkdir -p results/wiki_fq_12/
mkdir -p results/wiki_fq_2/
mkdir -p results/fq_12/
mkdir -p results/fq_2/
python src/scripts/train_eval/train_cbow.py wiki results/wiki_fq_12_q/ --embedding_name=wiki_fq_12_q
python src/scripts/train_eval/train_cbow.py wiki results/wiki_fq_2_q/ --embedding_name=wiki_fq_2_q
python src/scripts/train_eval/train_cbow.py wiki results/wiki_fq_12/ --embedding_name=wiki_fq_12
python src/scripts/train_eval/train_cbow.py wiki results/wiki_fq_2/ --embedding_name=wiki_fq_2
python src/scripts/train_eval/train_cbow.py wiki results/fq_12/ --embedding_name=fq_12
python src/scripts/train_eval/train_cbow.py wiki results/fq_2/ --embedding_name=fq_2
python src/scripts/train_eval/evaluate_cbow.py --model-name wiki_fq_12_q
python src/scripts/train_eval/evaluate_cbow.py --model-name wiki_fq_2_q
python src/scripts/train_eval/evaluate_cbow.py --model-name wiki_fq_12
python src/scripts/train_eval/evaluate_cbow.py --model-name wiki_fq_2
python src/scripts/train_eval/evaluate_cbow.py --model-name fq_12
python src/scripts/train_eval/evaluate_cbow.py --model-name fq_2

python src/scripts/retrofitting/retrofitting.py --save-text=kim_wiki_fq_12_q --evaluate --save-embedding --sum --q --retrofitting --lexicon-name=kim
python src/scripts/retrofitting/retrofitting.py --save-text=kim_wiki_fq_2_q --evaluate --save-embedding --sum --q --retrofitting --lexicon-name=kim --losses 2 --losses-2 2
python src/scripts/retrofitting/retrofitting.py --save-text=kim_wiki_fq_12 --evaluate --save-embedding --sum --retrofitting --lexicon-name=kim
python src/scripts/retrofitting/retrofitting.py --save-text=kim_wiki_fq_2 --evaluate --save-embedding --sum --retrofitting --lexicon-name=kim --losses 2 --losses-2 2
python src/scripts/retrofitting/retrofitting.py --save-text=kim_fq_12 --evaluate --save-embedding --retrofitting --lexicon-name=kim
python src/scripts/retrofitting/retrofitting.py --save-text=kim_fq_2 --evaluate --save-embedding --retrofitting --lexicon-name=kim --losses 2 --losses-2 2
mkdir -p results/kim_wiki_fq_12_q/
mkdir -p results/kim_wiki_fq_2_q/
mkdir -p results/kim_wiki_fq_12/
mkdir -p results/kim_wiki_fq_2/
mkdir -p results/kim_fq_12/
mkdir -p results/kim_fq_2/
python src/scripts/train_eval/train_cbow.py wiki results/kim_wiki_fq_12_q/ --embedding_name=kim_wiki_fq_12_q
python src/scripts/train_eval/train_cbow.py wiki results/kim_wiki_fq_2_q/ --embedding_name=kim_wiki_fq_2_q
python src/scripts/train_eval/train_cbow.py wiki results/kim_wiki_fq_12/ --embedding_name=kim_wiki_fq_12
python src/scripts/train_eval/train_cbow.py wiki results/kim_wiki_fq_2/ --embedding_name=kim_wiki_fq_2
python src/scripts/train_eval/train_cbow.py wiki results/kim_fq_12/ --embedding_name=kim_fq_12
python src/scripts/train_eval/train_cbow.py wiki results/kim_fq_2/ --embedding_name=kim_fq_2
python src/scripts/train_eval/evaluate_cbow.py --model-name kim_wiki_fq_12_q
python src/scripts/train_eval/evaluate_cbow.py --model-name kim_wiki_fq_2_q
python src/scripts/train_eval/evaluate_cbow.py --model-name kim_wiki_fq_12
python src/scripts/train_eval/evaluate_cbow.py --model-name kim_wiki_fq_2
python src/scripts/train_eval/evaluate_cbow.py --model-name kim_fq_12
python src/scripts/train_eval/evaluate_cbow.py --model-name kim_fq_2

python src/scripts/retrofitting/retrofitting.py --save-text=cokim_wiki_fq_12_q --evaluate --save-embedding --sum --q --retrofitting --lexicon-name=cokim
python src/scripts/retrofitting/retrofitting.py --save-text=cokim_wiki_fq_2_q --evaluate --save-embedding --sum --q --retrofitting --lexicon-name=cokim --losses 2 --losses-2 2
python src/scripts/retrofitting/retrofitting.py --save-text=cokim_wiki_fq_12 --evaluate --save-embedding --sum --retrofitting --lexicon-name=cokim
python src/scripts/retrofitting/retrofitting.py --save-text=cokim_wiki_fq_2 --evaluate --save-embedding --sum --retrofitting --lexicon-name=cokim --losses 2 --losses-2 2
python src/scripts/retrofitting/retrofitting.py --save-text=cokim_fq_12 --evaluate --save-embedding --retrofitting --lexicon-name=cokim
python src/scripts/retrofitting/retrofitting.py --save-text=cokim_fq_2 --evaluate --save-embedding --retrofitting --lexicon-name=cokim --losses 2 --losses-2 2
mkdir -p results/cokim_wiki_fq_12_q/
mkdir -p results/cokim_wiki_fq_2_q/
mkdir -p results/cokim_wiki_fq_12/
mkdir -p results/cokim_wiki_fq_2/
mkdir -p results/cokim_fq_12/
mkdir -p results/cokim_fq_2/
python src/scripts/train_eval/train_cbow.py wiki results/cokim_wiki_fq_12_q/ --embedding_name=cokim_wiki_fq_12_q
python src/scripts/train_eval/train_cbow.py wiki results/cokim_wiki_fq_2_q/ --embedding_name=cokim_wiki_fq_2_q
python src/scripts/train_eval/train_cbow.py wiki results/cokim_wiki_fq_12/ --embedding_name=cokim_wiki_fq_12
python src/scripts/train_eval/train_cbow.py wiki results/cokim_wiki_fq_2/ --embedding_name=cokim_wiki_fq_2
python src/scripts/train_eval/train_cbow.py wiki results/cokim_fq_12/ --embedding_name=cokim_fq_12
python src/scripts/train_eval/train_cbow.py wiki results/cokim_fq_2/ --embedding_name=cokim_fq_2
python src/scripts/train_eval/evaluate_cbow.py --model-name cokim_wiki_fq_12_q
python src/scripts/train_eval/evaluate_cbow.py --model-name cokim_wiki_fq_2_q
python src/scripts/train_eval/evaluate_cbow.py --model-name cokim_wiki_fq_12
python src/scripts/train_eval/evaluate_cbow.py --model-name cokim_wiki_fq_2
python src/scripts/train_eval/evaluate_cbow.py --model-name cokim_fq_12
python src/scripts/train_eval/evaluate_cbow.py --model-name cokim_fq_2
