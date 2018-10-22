#!/usr/bin/env bash

cd "$(dirname "$0")"

if [ -z ${DATA_DIR+x} ]; then
    echo "DATA_DIR is unset. Run env.sh.* script"
    exit 1
fi

# SNLI
if [ ! -d ${DATA_DIR}/snli ]; then
    ./gdown.pl https://drive.google.com/file/d/1H00Lc0co_lYljXiBQtkzekATbPz7vZuU/ ${DATA_DIR}/snli.zip
    unzip ${DATA_DIR}/snli.zip -d ${DATA_DIR}/
    rm ${DATA_DIR}/snli.zip
fi

# MNLI
if [ ! -d ${DATA_DIR}/mnli ]; then
    ./gdown.pl https://drive.google.com/file/d/1crajBtYles_yeVdtXtcx9nfDoLLpiqB9/ ${DATA_DIR}/mnli.zip
    unzip ${DATA_DIR}/mnli.zip -d ${DATA_DIR}/
    rm ${DATA_DIR}/mnli.zip
fi

# GloVe
if [ ! -f ${DATA_DIR}/embeddings/gcc840.h5 ]; then
    ./gdown.pl https://drive.google.com/file/d/1nnTJ-5B_19czIFVHPIVwUVXHgYGjAHFO/ ${DATA_DIR}/gcc840.zip
    mkdir -p ${DATA_DIR}/embeddings
    unzip ${DATA_DIR}/gcc840.zip -d ${DATA_DIR}/
    rm ${DATA_DIR}/gcc840.zip
fi

# WordNet features
if [ ! -f ${DATA_DIR}/wordnet_features.pkl ]; then
    ./gdown.pl https://drive.google.com/file/d/1oHRQapAFr1EvN7a4mWlrRMp5rUf_gFbW ${DATA_DIR}/wordnet_features.zip
    mkdir -p ${DATA_DIR}/embeddings
    unzip ${DATA_DIR}/wordnet_features.zip -d ${DATA_DIR}/
    rm ${DATA_DIR}/wordnet_features.zip
fi

# Lexicons
if [ ! -d ${DATA_DIR}/lexicons ]; then
    ./gdown.pl https://drive.google.com/file/d/1PlDt0xOgle1xxCY6Ury3vRQh4I2oo1xI ${DATA_DIR}/lexicons.zip
    mkdir -p ${DATA_DIR}/lexicons
    unzip ${DATA_DIR}/lexicons.zip -d ${DATA_DIR}/
    rm ${DATA_DIR}/lexicons.zip
fi