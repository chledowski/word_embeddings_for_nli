#!/usr/bin/env bash

set -e

GREEN='\033[1;32m'
NC='\033[0m' # No Color

CUDA=$1

export THEANO_FLAGS="mode=FAST_RUN,device=cuda${CUDA},cuda.root=/usr/local/cuda-8.0,floatX=float32,optimizer_including=cudnn,warn_float64=warn"

printf "${GREEN}THEANO_FLAGS: $THEANO_FLAGS${NC}\n"
printf "${GREEN}CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES${NC}\n"
printf "${GREEN}Running RETRO${NC}\n"

python src/scripts/retrofitting/retrofitting.py \
    --embedding=gcc840 \
    --save-embedding \
    --retrofitting \
    --sum \
    --save-text=gcc840_snli_gcc840_fq_12

python3 -u src/models/kim/scripts/kim/train.py \
    --embedding=cokim_CC840_12_q \
    --model=kim_cokim_CC840_12_q


