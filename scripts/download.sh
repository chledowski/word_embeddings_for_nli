#!/usr/bin/env bash

cd "$(dirname "$0")"

if [ -z ${DATA_DIR+x} ]; then
    echo "DATA_DIR is unset. Run env.sh.* script"
    exit 1
fi

# SNLI
./gdown.pl https://drive.google.com/file/d/1H00Lc0co_lYljXiBQtkzekATbPz7vZuU/ ${DATA_DIR}/snli.zip
unzip ${DATA_DIR}/snli.zip -d ${DATA_DIR}/
rm ${DATA_DIR}/snli.zip

# MNLI
./gdown.pl https://drive.google.com/file/d/1crajBtYles_yeVdtXtcx9nfDoLLpiqB9/ ${DATA_DIR}/mnli.zip
unzip ${DATA_DIR}/mnli.zip -d ${DATA_DIR}/
rm ${DATA_DIR}/mnli.zip

# GloVe
./gdown.pl https://drive.google.com/file/d/1nnTJ-5B_19czIFVHPIVwUVXHgYGjAHFO/ ${DATA_DIR}/gcc840.zip
mkdir -p ${DATA_DIR}/embeddings
unzip ${DATA_DIR}/gcc840.zip -d ${DATA_DIR}/
rm ${DATA_DIR}/gcc840.zip