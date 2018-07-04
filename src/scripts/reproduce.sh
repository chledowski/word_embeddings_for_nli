#!/usr/bin/env bash

#python src/scripts/fetch_data/fetch_embeddings.py --embeddings wiki
#python src/scripts/retrofitting/retrofitting.py --evaluate --save-text=wiki --second-embedding=wiki
#python src/scripts/retrofitting/retrofitting.py --save-text=wiki_fq_12_q --evaluate --save-embedding --sum --q --retrofitting
#python src/scripts/retrofitting/retrofitting.py --save-text=wiki_fq_2_q --evaluate --save-embedding --sum --q --retrofitting
#python src/scripts/retrofitting/retrofitting.py --save-text=wiki_fq_12 --evaluate --save-embedding --sum --retrofitting
#python src/scripts/retrofitting/retrofitting.py --save-text=wiki_fq_2 --evaluate --save-embedding --sum --retrofitting
#python src/scripts/retrofitting/retrofitting.py --save-text=fq_12 --evaluate --save-embedding --retrofitting
#python src/scripts/retrofitting/retrofitting.py --save-text=fq_2 --evaluate --save-embedding --retrofitting

#python src/scripts/retrofitting/retrofitting.py --save-text=kim_wiki_fq_12_q --evaluate --save-embedding --sum --q --retrofitting --lexicon-name=kim
#python src/scripts/retrofitting/retrofitting.py --save-text=kim_wiki_fq_2_q --evaluate --save-embedding --sum --q --retrofitting --lexicon-name=kim
#python src/scripts/retrofitting/retrofitting.py --save-text=kim_wiki_fq_12 --evaluate --save-embedding --sum --retrofitting --lexicon-name=kim
#python src/scripts/retrofitting/retrofitting.py --save-text=kim_wiki_fq_2 --evaluate --save-embedding --sum --retrofitting --lexicon-name=kim
#python src/scripts/retrofitting/retrofitting.py --save-text=kim_fq_12 --evaluate --save-embedding --retrofitting --lexicon-name=kim
#python src/scripts/retrofitting/retrofitting.py --save-text=kim_fq_2 --evaluate --save-embedding --retrofitting --lexicon-name=kim

#mkdir -p results/wiki/
#mkdir -p results/wiki_fq_12_q/
#mkdir -p results/wiki_fq_2_q/
#mkdir -p results/wiki_fq_12/
#mkdir -p results/wiki_fq_2/
#mkdir -p results/fq_12/
#mkdir -p results/fq_2/
#
#python src/scripts/train_eval/train_esim.py wiki results/wiki/ --embedding_name=wiki
#python src/scripts/train_eval/train_esim.py wiki results/wiki_fq_12_q/ --embedding_name=wiki_fq_12_q
#python src/scripts/train_eval/train_esim.py wiki results/wiki_fq_2_q/ --embedding_name=wiki_fq_2_q
#python src/scripts/train_eval/train_esim.py wiki results/wiki_fq_12/ --embedding_name=wiki_fq_12
#python src/scripts/train_eval/train_esim.py wiki results/wiki_fq_2/ --embedding_name=wiki_fq_2
#python src/scripts/train_eval/train_esim.py wiki results/fq_12/ --embedding_name=fq_12
#python src/scripts/train_eval/train_esim.py wiki results/fq_2/ --embedding_name=fq_2
#
#python src/scripts/train_eval/evaluate_esim.py --model-name wiki
#python src/scripts/train_eval/evaluate_esim.py --model-name wiki_fq_12_q
#python src/scripts/train_eval/evaluate_esim.py --model-name wiki_fq_2_q
#python src/scripts/train_eval/evaluate_esim.py --model-name wiki_fq_12
#python src/scripts/train_eval/evaluate_esim.py --model-name wiki_fq_2
#python src/scripts/train_eval/evaluate_esim.py --model-name fq_12
#python src/scripts/train_eval/evaluate_esim.py --model-name fq_2
python src/scripts/retrofitting/retrofitting.py  --sum --q --second-embedding=fq_12 --lexicon-name=kim